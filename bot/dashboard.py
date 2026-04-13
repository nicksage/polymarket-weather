"""
dashboard.py — Highest-temperature market analysis dashboard

Two tabs:
    Tab 1 — All Contracts
        Every discovered highest-temperature event and every outcome range,
        with city, date, temperature range, market price, model probability,
        edge, and EV.  Includes a distribution chart (model vs. market) per event.

    Tab 2 — Trade Signals
        Only the outcome ranges where the model identifies sufficient edge
        (|model_prob - market_price| >= EDGE_THRESHOLD) to justify a trade.
        Filterable by city, EV, date range.

Run with:
    cd ~/polymarket-weather/bot
    streamlit run dashboard.py --server.port 8501
"""

import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

from config import DB_PATH, EDGE_THRESHOLD

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Temp Arb Dashboard",
    page_icon="🌡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loaders (cached, auto-refresh every 60 s)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_events() -> pd.DataFrame:
    """Load all temp_events rows from the latest scan."""
    try:
        conn = sqlite3.connect(DB_PATH)
        ts_row = conn.execute("SELECT MAX(scan_timestamp) FROM temp_events").fetchone()
        if not ts_row or not ts_row[0]:
            conn.close()
            return pd.DataFrame()
        ts = ts_row[0]
        df = pd.read_sql(
            "SELECT * FROM temp_events WHERE scan_timestamp = ? ORDER BY city, date",
            conn, params=(ts,)
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_outcomes(signals_only: bool = False) -> pd.DataFrame:
    """Load all temp_outcomes rows from the latest scan, joined with event context."""
    try:
        conn = sqlite3.connect(DB_PATH)
        ts_row = conn.execute("SELECT MAX(scan_timestamp) FROM temp_events").fetchone()
        if not ts_row or not ts_row[0]:
            conn.close()
            return pd.DataFrame()
        ts = ts_row[0]
        where = "AND o.is_signal = 1" if signals_only else ""
        df = pd.read_sql(
            f"""
            SELECT
                o.*,
                e.city,
                e.date,
                e.event_title,
                e.forecast_mu_c,
                e.forecast_sigma_c,
                e.clim_mu_c,
                e.clim_sigma_c,
                e.forecast_mu_display,
                e.display_unit,
                e.days_ahead,
                e.market_overround,
                e.model_probs_sum,
                e.normalization_warning,
                e.lat,
                e.lon
            FROM temp_outcomes o
            JOIN temp_events e ON o.event_row_id = e.id
            WHERE o.scan_timestamp = ?
            {where}
            ORDER BY o.ev DESC
            """,
            conn, params=(ts,)
        )
        conn.close()
        _US_LAT = (15.0, 72.0)
        _US_LON = (-180.0, -60.0)
        df["is_us"] = (
            df["lat"].between(_US_LAT[0], _US_LAT[1]) &
            df["lon"].between(_US_LON[0], _US_LON[1])
        )
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_positions() -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql(
            "SELECT * FROM positions ORDER BY entry_time DESC", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_scan_timestamp() -> str:
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT MAX(scan_timestamp) FROM temp_events").fetchone()
        conn.close()
        return row[0] if row and row[0] else "No scans yet"
    except Exception:
        return "Error reading DB"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _highlight_side(val):
    if val == "YES": return "color: #2ca02c; font-weight: bold"
    if val == "NO":  return "color: #d62728; font-weight: bold"
    return ""

def _fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v * 100:.1f}%"

def _fmt_ev(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.3f}"

def _fmt_temp(v, unit) -> str:
    if v is None or pd.isna(v):
        return "—"
    sym = "°F" if unit == "fahrenheit" else "°C"
    return f"{v:.1f}{sym}"

def _signal_badge(side) -> str:
    if side == "YES":
        return "🟢 BUY YES"
    elif side == "NO":
        return "🔴 BUY NO"
    return "—"

def _fmt_entered(ts) -> str:
    """Format an ISO timestamp as 'HH:MM:SS AM/PM MM-DD-YYYY'."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.strftime("%I:%M:%S %p %m-%d-%Y")
    except Exception:
        return str(ts)[:19]


def _is_missing(v) -> bool:
    """Return True for Python None or pandas/numpy NaN."""
    if v is None:
        return True
    try:
        return pd.isna(v)
    except (TypeError, ValueError):
        return False


def _range_label(low, high, unit) -> str:
    sym = "°F" if unit == "fahrenheit" else "°C"
    lo = None if _is_missing(low)  else float(low)
    hi = None if _is_missing(high) else float(high)
    if lo is None and hi is not None:
        return f"≤{hi:.0f}{sym}"
    if hi is None and lo is not None:
        return f"≥{lo:.0f}{sym}"
    if lo is not None and hi is not None:
        if lo == hi:
            return f"{lo:.0f}{sym}"
        return f"{lo:.0f}–{hi:.0f}{sym}"
    return "?"


# ---------------------------------------------------------------------------
# Tab 3 — event detail modal (must be defined at module level)
# ---------------------------------------------------------------------------

@st.dialog("Temperature Ranges", width="large")
def _show_event_modal(city: str, date_str: str, rows: list[dict]) -> None:
    """Full-detail popup showing every outcome bin for one event."""
    title = (
        rows[0].get("event_title") or f"Highest temperature in {city} on {date_str}"
        if rows else f"{city} — {date_str}"
    )
    st.markdown(f"### {title}")

    # Forecast summary metrics
    if rows:
        r0       = rows[0]
        ev_unit  = r0.get("display_unit", r0.get("unit", "celsius"))
        sym      = "°F" if ev_unit == "fahrenheit" else "°C"
        mu_disp  = r0.get("forecast_mu_display")
        sigma_c  = r0.get("forecast_sigma_c")
        clim_mu  = r0.get("clim_mu_c")

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            if not _is_missing(mu_disp):
                st.metric("Forecast Avg", f"{float(mu_disp):.1f}{sym}")
        with mc2:
            if not _is_missing(sigma_c):
                sd = float(sigma_c) if ev_unit == "celsius" else float(sigma_c) * 9 / 5
                st.metric("Uncertainty (σ)", f"{sd:.1f}{sym}")
        with mc3:
            if not _is_missing(clim_mu):
                cd = float(clim_mu) if ev_unit == "celsius" else float(clim_mu) * 9 / 5 + 32
                st.metric("Historical Avg", f"{cd:.1f}{sym}")

    st.divider()

    # Sort rows by temperature descending (≥23°C at top, ≤13°C at bottom).
    # Sort key: use range_low when present (exact and open-high bins), else range_high.
    def _modal_sort_key(r):
        lo = None if _is_missing(r.get("range_low"))  else float(r.get("range_low"))
        hi = None if _is_missing(r.get("range_high")) else float(r.get("range_high"))
        return lo if lo is not None else (hi if hi is not None else 0.0)

    rows_sorted = sorted(rows, key=_modal_sort_key, reverse=True)

    # Full outcome table
    table_rows = []
    for r in rows_sorted:
        u      = r.get("unit", "celsius")
        mdl    = r.get("model_prob")
        is_sig = bool(r.get("is_signal"))
        # Prefer stored yes_price/no_price; fall back to market_price / 1-market_price
        yes_p  = r.get("yes_price") or r.get("market_price")
        no_p   = r.get("no_price")
        if _is_missing(no_p) and not _is_missing(yes_p):
            no_p = 1.0 - float(yes_p)
        # No model prob = 1 - yes model prob (binary contract identity)
        no_mdl = (1.0 - float(mdl)) if not _is_missing(mdl) else None
        edge   = r.get("edge")
        ev     = r.get("ev")
        table_rows.append({
            "Range":            _range_label(r.get("range_low"), r.get("range_high"), u),
            "Yes Price":        _fmt_pct(yes_p),
            "No Price":         _fmt_pct(no_p),
            "Yes Model Prob":   _fmt_pct(mdl),
            "No Model Prob":    _fmt_pct(no_mdl),
            # Edge: decimal gap between model prob and market price (e.g. +0.075 = model
            # thinks YES is 7.5pp more likely than the market does)
            "Edge":             _fmt_ev(edge),
            # EV: expected profit per $1 wagered on the recommended side
            "EV ($/dollar)":    _fmt_ev(ev),
            "Signal":           _signal_badge(r.get("recommended_side")) if is_sig else "—",
            "Volume":           f"${r.get('volume_usd', 0):,.0f}",
        })

    df_modal = pd.DataFrame(table_rows)
    st.dataframe(
        df_modal, width="stretch", hide_index=True,
        height=min(120 + len(df_modal) * 38, 520),
    )


def _render_event_card(city: str, date_str: str, grp: pd.DataFrame) -> None:
    """Render a single event summary card for the Market Overview grid."""
    unit     = grp["unit"].iloc[0] if not grp.empty else "celsius"
    ev_unit  = grp["display_unit"].iloc[0] if "display_unit" in grp.columns else unit
    days_out = int(grp["days_ahead"].iloc[0]) if "days_ahead" in grp.columns else 0
    title_val = grp["event_title"].iloc[0] if "event_title" in grp.columns else None
    title    = (
        title_val if not _is_missing(title_val)
        else f"Highest temperature in {city} on {date_str}"
    )
    total_vol = grp["volume_usd"].sum() if "volume_usd" in grp.columns else 0
    n_signals = int(grp["is_signal"].sum()) if "is_signal" in grp.columns else 0

    # Top 2 outcomes by model probability (parseable only)
    parseable = grp[grp["model_prob"].notna()].copy()
    top2      = parseable.sort_values("model_prob", ascending=False).head(2)

    # Unique stable key
    card_key = f"card_{''.join(c if c.isalnum() else '_' for c in f'{city}_{date_str}')}"

    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"{days_out}d out  ·  ${total_vol:,.0f} Vol.")

        st.divider()

        if top2.empty:
            st.caption("No model data available")
        else:
            for _, row in top2.iterrows():
                rl  = _range_label(row.get("range_low"), row.get("range_high"), unit)
                mkt = row.get("market_price")
                mdl = row.get("model_prob")

                c_lbl, c_mkt, c_mdl = st.columns([2, 1, 1])
                with c_lbl:
                    st.markdown(f"**{rl}**")
                with c_mkt:
                    st.markdown(f"Market Probability: {_fmt_pct(mkt)}")
                with c_mdl:
                    if not _is_missing(mdl) and not _is_missing(mkt):
                        color = "green" if float(mdl) > float(mkt) else "red"
                        # Green = model thinks YES is underpriced (edge to buy YES)
                        # Red   = model thinks YES is overpriced (edge to buy NO)
                        st.markdown(f":{color}[Model Probability: {_fmt_pct(mdl)}]")
                    else:
                        st.markdown("Model Probability: —")

        st.divider()

        cf_sig, cf_btn = st.columns([2, 1])
        with cf_sig:
            if n_signals:
                st.markdown(f"✨ **{n_signals}** signal(s)")
            else:
                st.caption("No signals")
        with cf_btn:
            if st.button("Details", key=card_key):
                _show_event_modal(city, date_str, grp.to_dict("records"))


# ---------------------------------------------------------------------------
# Header + metrics
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    div[data-testid="stButton"] > button {
        background-color: #c0392b;
        color: white;
        border: none;
    }
    div[data-testid="stButton"] > button:hover {
        background-color: #e74c3c;
        color: white;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌡 Highest-Temperature Arbitrage Dashboard")

last_scan = load_scan_timestamp()
st.caption(f"Last scan: {last_scan}  |  DB: {DB_PATH}  |  Auto-refresh: 60s")

events_df    = load_events()
outcomes_df  = load_outcomes(signals_only=False)
signals_df   = load_outcomes(signals_only=True)
positions_df = load_positions()

n_events   = len(events_df)   if not events_df.empty   else 0
n_outcomes = len(outcomes_df) if not outcomes_df.empty else 0
n_signals  = len(signals_df)  if not signals_df.empty  else 0
total_pnl  = positions_df["pnl"].sum() if not positions_df.empty and "pnl" in positions_df else 0
open_count = len(positions_df[positions_df["status"] == "open"]) if not positions_df.empty else 0

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Events Tracked",  n_events)
m2.metric("Outcome Ranges",  n_outcomes)
m3.metric("Trade Signals",   n_signals)
m4.metric("Open Positions",  open_count)
m5.metric("Cumulative P&L",  f"${total_pnl:.2f}")

st.divider()

# ===========================================================================
# Main tabs
# ===========================================================================
_tab_contract, _tab_position = st.tabs(["Contract Data", "Position Data"])

# ===========================================================================
# TAB 1 — Contract Data
# ===========================================================================
with _tab_contract:

    if outcomes_df.empty:
        st.info("No data yet. Run the bot or wait for the next scan.")
    else:
        # ── Derive filter bounds ─────────────────────────────────────────────
        _all_cities  = sorted(outcomes_df["city"].dropna().unique().tolist())
        _all_dates   = sorted(outcomes_df["date"].dropna().unique().tolist())

        _sigma_vals  = outcomes_df["forecast_sigma_c"].dropna()
        _sigma_min   = float(round(_sigma_vals.min(), 1)) if not _sigma_vals.empty else 0.0
        _sigma_max   = float(round(_sigma_vals.max(), 1)) if not _sigma_vals.empty else 10.0
        if _sigma_min == _sigma_max:
            _sigma_max = _sigma_min + 1.0

        # ── Filter panel ─────────────────────────────────────────────────────
        with st.expander("Filters", expanded=True):
            r1c1, r1c2, r1c3, r1c4 = st.columns([3, 1, 2, 1])

            with r1c1:
                sel_cities = st.multiselect(
                    "City", _all_cities, default=_all_cities, key="f_cities",
                )

            _contracts_mode_current = st.session_state.get("f_contracts", "All Contracts")
            signals_only_preview = (_contracts_mode_current == "Signals Only")

            with r1c2:
                if signals_only_preview:
                    sel_sides = st.multiselect(
                        "Side", ["YES", "NO"], default=["YES", "NO"], key="f_side",
                    )
                else:
                    st.multiselect(
                        "Side", ["YES", "NO"], default=["YES", "NO"],
                        key="f_side_disabled", disabled=True,
                        help="Switch to 'Signals Only' to filter by side.",
                    )
                    sel_sides = ["YES", "NO"]

            with r1c3:
                sel_dates = st.multiselect(
                    "Date", _all_dates, default=_all_dates, key="f_dates",
                )

            with r1c4:
                view_mode = st.radio(
                    "View", ["Overview", "List View"], index=0,
                    key="f_view", horizontal=True,
                )
                contracts_mode = st.radio(
                    "Contracts", ["All Contracts", "Signals Only"], index=0,
                    key="f_contracts", horizontal=True,
                )
                signals_only = (contracts_mode == "Signals Only")
                geo_mode = st.radio(
                    "Geography", ["Global", "US Only"], index=0,
                    key="f_geo", horizontal=True,
                )

            r3c1, r3c2, r3c3 = st.columns(3)
            with r3c1:
                min_ev = st.slider(
                    "Min EV ($/dollar)", min_value=0.0, max_value=0.50,
                    value=0.0, step=0.01, format="%.2f", key="f_ev",
                )
            with r3c2:
                min_liq = st.slider(
                    "Min Liquidity ($)", min_value=0, max_value=10_000,
                    value=0, step=100, key="f_liq",
                )
            with r3c3:
                sigma_range = st.slider(
                    "Uncertainty σ range",
                    min_value=_sigma_min, max_value=_sigma_max,
                    value=(_sigma_min, _sigma_max),
                    step=0.1, format="%.1f", key="f_sigma",
                )

        # ── Apply filters ────────────────────────────────────────────────────
        _base = outcomes_df.copy()

        if sel_cities:
            _base = _base[_base["city"].isin(sel_cities)]
        if sel_dates:
            _base = _base[_base["date"].isin(sel_dates)]

        _base = _base[_base["liquidity_usd"].fillna(0) >= min_liq]
        _base = _base[
            _base["forecast_sigma_c"].fillna(_sigma_min).between(sigma_range[0], sigma_range[1])
        ]

        if geo_mode == "US Only" and "is_us" in _base.columns:
            _base = _base[_base["is_us"] == True]

        if signals_only:
            _base = _base[_base["is_signal"] == 1]
            _base = _base[_base["recommended_side"].isin(sel_sides)]
            if min_ev > 0:
                _base = _base[_base["ev"].fillna(0) >= min_ev]

        if _base.empty:
            st.info("No outcomes match the current filters.")
        else:
            # ── Status line ──────────────────────────────────────────────────
            n_filtered_events  = _base.groupby(["city", "date"]).ngroups
            n_filtered_signals = int(_base["is_signal"].sum())
            st.caption(
                f"{n_filtered_events} event(s)  ·  {len(_base)} outcome(s)  ·  "
                f"{n_filtered_signals} signal(s) match current filters"
            )

            # ── Overview View ────────────────────────────────────────────────
            if view_mode == "Overview":
                _event_groups = sorted(
                    _base.groupby(["city", "date"]),
                    key=lambda x: x[0][0].lower(),
                )
                _COLS = 3
                for _i in range(0, len(_event_groups), _COLS):
                    _chunk     = _event_groups[_i : _i + _COLS]
                    _grid_cols = st.columns(_COLS)
                    for _j, ((_city, _date), _grp) in enumerate(_chunk):
                        with _grid_cols[_j]:
                            _render_event_card(_city, _date, _grp)

            # ── List View ────────────────────────────────────────────────────
            else:
                _list_df = _base.copy()
                _list_df["_sort_low"] = pd.to_numeric(_list_df["range_low"], errors="coerce")
                _list_df = _list_df.sort_values(
                    ["city", "date", "_sort_low"], ascending=[True, True, True], na_position="first"
                ).drop(columns=["_sort_low"])

                display_rows = []
                for _, row in _list_df.iterrows():
                    unit    = row.get("unit", "celsius")
                    ev_unit = row.get("display_unit", unit)
                    sigma_c = row.get("forecast_sigma_c")
                    sigma_disp = (
                        None if _is_missing(sigma_c)
                        else (float(sigma_c) * 9 / 5 if ev_unit == "fahrenheit" else float(sigma_c))
                    )
                    sym = "°F" if ev_unit == "fahrenheit" else "°C"
                    display_rows.append({
                        "City":            row.get("city", ""),
                        "Date":            row.get("date", ""),
                        "Days Out":        int(row.get("days_ahead", 0)),
                        "Range":           _range_label(row.get("range_low"), row.get("range_high"), unit),
                        "Side":            row.get("recommended_side", "—") if row.get("is_signal") else "—",
                        "Yes Market Prob": _fmt_pct(row.get("yes_price") or row.get("market_price")),
                        "No Market Prob":  _fmt_pct(row.get("no_price")),
                        "Yes Model Prob":  _fmt_pct(row.get("model_prob")),
                        "No Model Prob":   _fmt_pct((1.0 - float(row["model_prob"])) if not _is_missing(row.get("model_prob")) else None),
                        "Edge":            _fmt_ev(row.get("edge")),
                        "EV ($/dollar)":   _fmt_ev(row.get("ev")),
                        "Kelly $":         f"${row.get('kelly_size', 0):.2f}" if row.get("is_signal") else "—",
                        "Liquidity":       f"${row.get('liquidity_usd', 0):,.0f}",
                        "Fcst Avg":        _fmt_temp(row.get("forecast_mu_display"), ev_unit),
                        "Uncertainty":     f"{sigma_disp:.1f}{sym}" if sigma_disp is not None else "—",
                        "Signal":          _signal_badge(row.get("recommended_side")) if row.get("is_signal") else "—",
                        "Norm Warn":       "⚠️" if row.get("normalization_warning") else "",
                    })

                display_df = pd.DataFrame(display_rows)
                st.dataframe(
                    display_df.style.map(_highlight_side, subset=["Side"]),
                    width="stretch",
                    height=min(80 + len(display_df) * 36, 600),
                )

                st.divider()

                st.subheader("Model vs. Market Distribution")
                st.caption(
                    "Blue = Market Price · Orange = Model Probability.  "
                    "Expand a city to see its full distribution."
                )

                if signals_only:
                    _chart_events = _base[["city", "date"]].drop_duplicates()
                    _chart_df = outcomes_df.merge(_chart_events, on=["city", "date"], how="inner")
                    if sel_cities:
                        _chart_df = _chart_df[_chart_df["city"].isin(sel_cities)]
                    if sel_dates:
                        _chart_df = _chart_df[_chart_df["date"].isin(sel_dates)]
                else:
                    _chart_df = _base

                _chart_groups = sorted(
                    _chart_df.groupby(["city", "date"]),
                    key=lambda x: (x[0][1], x[0][0]),
                )

                for (city, date_str), grp in _chart_groups:
                    grp     = grp.copy()
                    unit    = grp["unit"].iloc[0] if not grp.empty else "celsius"
                    sym     = "°F" if unit == "fahrenheit" else "°C"
                    ev_unit = grp["display_unit"].iloc[0] if "display_unit" in grp.columns else unit

                    mu_disp   = grp["forecast_mu_display"].iloc[0] if "forecast_mu_display" in grp.columns else None
                    sigma_c   = grp["forecast_sigma_c"].iloc[0]    if "forecast_sigma_c"    in grp.columns else None
                    clim_mu   = grp["clim_mu_c"].iloc[0]           if "clim_mu_c"           in grp.columns else None
                    overround = grp["market_overround"].iloc[0]    if "market_overround"    in grp.columns else None
                    days_out  = int(grp["days_ahead"].iloc[0])     if "days_ahead"          in grp.columns else 0

                    def _chart_sort(r):
                        lo = None if _is_missing(r.range_low)  else float(r.range_low)
                        hi = None if _is_missing(r.range_high) else float(r.range_high)
                        return lo if lo is not None else (hi if hi is not None else 0.0)
                    grp = grp.sort_values(by=grp.columns[0], key=lambda _: grp.apply(_chart_sort, axis=1), ascending=False)

                    labels   = [_range_label(r.range_low, r.range_high, unit) for _, r in grp.iterrows()]
                    mkt_vals = [v if not _is_missing(v) else 0.0 for v in grp["market_price"].tolist()]
                    mdl_vals = [v if not _is_missing(v) else 0.0 for v in grp["model_prob"].tolist()]
                    n_sig    = int(grp["is_signal"].sum())

                    with st.expander(
                        f"**{city}** — {date_str}  ({days_out}d out)  "
                        f"| Forecast: {_fmt_temp(mu_disp, ev_unit)}  "
                        f"| Overround: {_fmt_pct(overround)}"
                        + (f"  | ✨ {n_sig} signal(s)" if n_sig else ""),
                        expanded=(days_out <= 3),
                    ):
                        col_chart, col_stats = st.columns([3, 1])
                        with col_chart:
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                name="Market Price", x=labels,
                                y=[v * 100 for v in mkt_vals],
                                marker_color="#4c78a8", opacity=0.85,
                            ))
                            fig.add_trace(go.Bar(
                                name="Model Probability", x=labels,
                                y=[v * 100 for v in mdl_vals],
                                marker_color="#f58518", opacity=0.85,
                            ))
                            fig.update_layout(
                                barmode="group",
                                xaxis_title=f"Temperature Range ({sym})",
                                yaxis_title="Probability (%)",
                                yaxis=dict(ticksuffix="%"),
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                                height=350, margin=dict(t=40, b=40),
                            )
                            st.plotly_chart(fig, width="stretch")

                        with col_stats:
                            st.markdown("**Forecast**")
                            if not _is_missing(mu_disp):
                                st.markdown(
                                    f"**Forecast Avg:** {_fmt_temp(mu_disp, ev_unit)}",
                                    help="Model's predicted mean max temperature (ECMWF + GFS blend)."
                                )
                            if not _is_missing(sigma_c):
                                sd  = float(sigma_c) if ev_unit == "celsius" else float(sigma_c) * 9 / 5
                                st.markdown(
                                    f"**Uncertainty (σ):** {sd:.1f}{sym}",
                                    help="Spread of ensemble members — larger = less confident."
                                )
                            if not _is_missing(clim_mu):
                                cd = float(clim_mu) if ev_unit == "celsius" else float(clim_mu) * 9 / 5 + 32
                                st.markdown(
                                    f"Historical Avg: {_fmt_temp(cd, ev_unit)}",
                                    help="10-year ERA5 climatological average for this date and location."
                                )
                            st.markdown("**Market**")
                            if not _is_missing(overround):
                                vig = (float(overround) - 1) * 100
                                col = "red" if vig > 8 else "orange" if vig > 4 else "green"
                                st.markdown(f"Overround: :{col}[{vig:.1f}%]")

                if signals_only and not _base.empty:
                    st.divider()
                    st.subheader("Signal Analysis")

                    col_ev, col_bbl = st.columns(2)
                    with col_ev:
                        fig_ev = px.histogram(
                            _base, x="ev", nbins=20,
                            color="recommended_side",
                            color_discrete_map={"YES": "#2ca02c", "NO": "#d62728"},
                            labels={"ev": "EV ($/dollar)", "recommended_side": "Side"},
                            title="Expected Value Distribution",
                        )
                        fig_ev.add_vline(x=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_ev, width="stretch")

                    with col_bbl:
                        _bbl = _base[_base["kelly_size"].fillna(0) > 0].copy()
                        if not _bbl.empty:
                            fig_bbl = px.scatter(
                                _bbl, x="edge", y="liquidity_usd",
                                size="kelly_size", color="recommended_side",
                                color_discrete_map={"YES": "#2ca02c", "NO": "#d62728"},
                                hover_data=["city", "date", "ev"],
                                labels={
                                    "edge": "Edge (model − market)",
                                    "liquidity_usd": "Liquidity ($)",
                                    "recommended_side": "Side",
                                },
                                title="Edge vs. Liquidity  (bubble size = Kelly $)",
                            )
                            fig_bbl.add_vline(x=0, line_dash="dash", line_color="gray")
                            st.plotly_chart(fig_bbl, width="stretch")


# ===========================================================================
# TAB 2 — Position Data
# ===========================================================================
with _tab_position:

    # ── P&L and Model Calibration ────────────────────────────────────────────
    with st.expander("P&L and Model Calibration", expanded=False):
        col_pnl, col_cal = st.columns(2)
        with col_pnl:
            st.subheader("Cumulative P&L")
            if not positions_df.empty and "pnl" in positions_df.columns:
                closed = positions_df[positions_df["status"] == "closed"].copy()
                if not closed.empty:
                    closed = closed.sort_values("exit_time")
                    closed["cumulative_pnl"] = closed["pnl"].cumsum()
                    fig_pnl = px.line(
                        closed, x="exit_time", y="cumulative_pnl",
                        labels={"exit_time": "Date", "cumulative_pnl": "Cumulative P&L ($)"},
                    )
                    fig_pnl.update_traces(line_color="#00CC96")
                    st.plotly_chart(fig_pnl, width="stretch")
                else:
                    st.info("No closed positions yet")
            else:
                st.info("No position data available")

        with col_cal:
            st.subheader("Model Calibration")
            st.caption("Average actual outcome vs. model probability bucket.")
            try:
                conn   = sqlite3.connect(DB_PATH)
                sig_df = pd.read_sql(
                    "SELECT * FROM signals WHERE executed = 1 AND outcome IS NOT NULL", conn
                )
                conn.close()
                if not sig_df.empty and "model_p" in sig_df.columns:
                    sig_df["p_bucket"] = pd.cut(sig_df["model_p"], bins=10, labels=False)
                    cal = sig_df.groupby("p_bucket").agg(
                        avg_model_p=("model_p", "mean"),
                        avg_outcome=("outcome", "mean"),
                    ).reset_index()
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Bar(
                        x=cal["avg_model_p"], y=cal["avg_outcome"],
                        name="Actual Frequency", marker_color="steelblue",
                    ))
                    fig_cal.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1], name="Perfect Calibration",
                        line=dict(color="red", dash="dash"),
                    ))
                    fig_cal.update_layout(
                        xaxis_title="Model Probability", yaxis_title="Actual Outcome Rate",
                    )
                    st.plotly_chart(fig_cal, width="stretch")
                else:
                    st.info("Not enough resolved trades for calibration chart")
            except Exception:
                st.info("Not enough resolved trades for calibration chart")

    # ── Helper: build open-position rows ────────────────────────────────────
    def _build_open_rows(df):
        rows = []
        for _, p in df.iterrows():
            entry  = p.get("entry_price")
            curr   = p.get("current_price")
            unreal = p.get("unrealized_pnl")
            shares = p.get("shares")
            _unit  = p.get("unit", "celsius")
            _range = _range_label(p.get("range_low"), p.get("range_high"), _unit) if (
                not _is_missing(p.get("range_low")) or not _is_missing(p.get("range_high"))
            ) else (p.get("question") or "")[:30]
            rows.append({
                "City":           p.get("city", ""),
                "Date":           p.get("date", ""),
                "Local Time":     p.get("local_time") or "—",
                "Question":       p.get("question") or "",
                "Side":           p.get("side", ""),
                "Fill":           p.get("fill_status", "filled"),
                "Range":          _range,
                "Size ($)":       f"${p.get('size_usdc', 0):.2f}",
                "Shares":         f"{shares:.2f}" if shares else "—",
                "Entered":        _fmt_entered(p.get("entry_time")),
                "Entry Price":    f"{entry:.4f}" if entry else "—",
                "Current Price":  f"{curr:.4f}" if curr else "—",
                "Unrealized P&L": f"${unreal:+.4f}" if unreal is not None else "—",
                "Model Prob":     _fmt_pct(p.get("model_prob")),
                "Market Prob":    _fmt_pct(p.get("market_prob")),
                "EV":             _fmt_ev(p.get("ev")),
            })
        return rows

    def _build_closed_rows(df):
        rows = []
        for _, p in df.iterrows():
            _unit  = p.get("unit", "celsius")
            _range = _range_label(p.get("range_low"), p.get("range_high"), _unit) if (
                not _is_missing(p.get("range_low")) or not _is_missing(p.get("range_high"))
            ) else (p.get("question") or "")[:30]
            rows.append({
                "City":         p.get("city", ""),
                "Date":         p.get("date", ""),
                "Question":     (p.get("question") or "")[:40],
                "Side":         p.get("side", ""),
                "Range":        _range,
                "Size ($)":     f"${p.get('size_usdc', 0):.2f}",
                "Entry":        f"{p.get('entry_price', 0):.4f}",
                "Exit":         f"{p.get('exit_price', 0):.4f}" if p.get("exit_price") is not None else "—",
                "Realized P&L": f"${p.get('pnl', 0):+.4f}" if p.get("pnl") is not None else "—",
                "Entered":      _fmt_entered(p.get("entry_time")),
                "Closed":       _fmt_entered(p.get("exit_time")),
            })
        return rows

    def _color_pnl(val):
        if isinstance(val, str) and val.startswith("$"):
            try:
                v = float(val.replace("$", "").replace("+", ""))
                return "color: #2ca02c" if v > 0 else ("color: #d62728" if v < 0 else "")
            except ValueError:
                pass
        return ""

    # Partition positions into paper vs live
    _all_open = pd.DataFrame()
    _all_closed = pd.DataFrame()
    if not positions_df.empty:
        _base_open = positions_df[positions_df["status"] == "open"].copy()
        if "fill_status" in positions_df.columns:
            _base_open = _base_open[_base_open["fill_status"] != "cancelled"]
        _base_closed = positions_df[positions_df["status"] == "closed"].copy()
        if "fill_status" in positions_df.columns:
            _base_closed = _base_closed[_base_closed["fill_status"] != "cancelled"]

        _paper_open  = _base_open[_base_open["is_paper"] == 1]  if "is_paper" in _base_open.columns  else _base_open
        _live_open   = _base_open[_base_open["is_paper"] == 0]  if "is_paper" in _base_open.columns  else pd.DataFrame()
        _paper_closed = _base_closed[_base_closed["is_paper"] == 1] if "is_paper" in _base_closed.columns else _base_closed
        _live_closed  = _base_closed[_base_closed["is_paper"] == 0] if "is_paper" in _base_closed.columns else pd.DataFrame()
    else:
        _paper_open = _live_open = _paper_closed = _live_closed = pd.DataFrame()

    # ── Paper Trades ─────────────────────────────────────────────────────────
    st.subheader("Paper Trades")

    with st.expander("Open Positions — Paper", expanded=True):
        if not _paper_open.empty:
            _df = pd.DataFrame(_build_open_rows(_paper_open))
            st.dataframe(
                _df.style.map(_highlight_side, subset=["Side"]).map(_color_pnl, subset=["Unrealized P&L"]),
                width="stretch", height=min(80 + len(_df) * 36, 500),
                column_config={"Question": st.column_config.TextColumn("Question", width="large")},
            )
            _tu = _paper_open["unrealized_pnl"].fillna(0).sum() if "unrealized_pnl" in _paper_open.columns else 0
            _td = _paper_open["size_usdc"].fillna(0).sum()
            _pc = int((_paper_open["fill_status"] == "pending").sum()) if "fill_status" in _paper_open.columns else 0
            _pm1, _pm2, _pm3, _pm4 = st.columns([2, 2, 2, 1])
            _pm1.metric("Total Deployed", f"${_td:.2f}")
            _pm2.metric("Unrealized P&L", f"${_tu:+.4f}")
            if _pc:
                _pm3.metric("Pending Fills", _pc)
            with _pm4:
                if st.button("🗑 Delete All", key="del_paper_open",
                             help="Delete all open paper trade records from the database"):
                    import sqlite3 as _sl
                    _c = _sl.connect(DB_PATH)
                    _c.execute("DELETE FROM positions WHERE is_paper = 1 AND status = 'open'")
                    _c.commit()
                    _c.close()
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.info("No open paper positions")

    with st.expander("Closed Positions — Paper", expanded=False):
        if not _paper_closed.empty:
            _df = pd.DataFrame(_build_closed_rows(_paper_closed))
            st.dataframe(
                _df.style.map(_highlight_side, subset=["Side"]).map(_color_pnl, subset=["Realized P&L"]),
                width="stretch", height=min(80 + len(_df) * 36, 400),
                column_config={"Question": st.column_config.TextColumn("Question", width="large")},
            )
            _tr = _paper_closed["pnl"].fillna(0).sum()
            _cr1, _cr2 = st.columns([3, 1])
            with _cr1:
                st.metric("Total Realized P&L", f"${_tr:+.4f}")
            with _cr2:
                if st.button("🗑 Delete All", key="del_paper_closed",
                             help="Delete all closed paper trade records from the database"):
                    import sqlite3 as _sl
                    _c = _sl.connect(DB_PATH)
                    _c.execute("DELETE FROM positions WHERE is_paper = 1 AND status = 'closed'")
                    _c.commit()
                    _c.close()
                    st.cache_data.clear()
                    st.rerun()
        else:
            st.info("No closed paper positions yet")

        if not positions_df.empty and "fill_status" in positions_df.columns:
            _cancelled_paper = positions_df[
                (positions_df["fill_status"] == "cancelled") &
                (positions_df.get("is_paper", pd.Series(dtype=int)) == 1)
            ].copy() if "is_paper" in positions_df.columns else pd.DataFrame()
            if not _cancelled_paper.empty:
                st.markdown("**Cancelled Paper Orders**")
                st.caption(f"{len(_cancelled_paper)} order(s) were cancelled before fill")
                _can_rows = [{
                    "City": p.get("city", ""), "Date": p.get("date", ""),
                    "Side": p.get("side", ""), "Size ($)": f"${p.get('size_usdc', 0):.2f}",
                    "Reason": p.get("cancelled_reason", ""),
                    "Time": (p.get("exit_time") or "")[:16],
                } for _, p in _cancelled_paper.iterrows()]
                st.dataframe(pd.DataFrame(_can_rows), width="stretch", hide_index=True)

    st.divider()

    # ── Live Trades ───────────────────────────────────────────────────────────
    st.subheader("Live Trades")

    with st.expander("Open Positions — Live", expanded=True):
        if not _live_open.empty:
            _df = pd.DataFrame(_build_open_rows(_live_open))
            st.dataframe(
                _df.style.map(_highlight_side, subset=["Side"]).map(_color_pnl, subset=["Unrealized P&L"]),
                width="stretch", height=min(80 + len(_df) * 36, 500),
                column_config={"Question": st.column_config.TextColumn("Question", width="large")},
            )
            _tu = _live_open["unrealized_pnl"].fillna(0).sum() if "unrealized_pnl" in _live_open.columns else 0
            _td = _live_open["size_usdc"].fillna(0).sum()
            _pc = int((_live_open["fill_status"] == "pending").sum()) if "fill_status" in _live_open.columns else 0
            _lm1, _lm2, _lm3 = st.columns(3)
            _lm1.metric("Total Deployed", f"${_td:.2f}")
            _lm2.metric("Unrealized P&L", f"${_tu:+.4f}")
            if _pc:
                _lm3.metric("Pending Fills", _pc)
        else:
            st.info("No open live positions")

    with st.expander("Closed Positions — Live", expanded=False):
        if not _live_closed.empty:
            _df = pd.DataFrame(_build_closed_rows(_live_closed))
            st.dataframe(
                _df.style.map(_highlight_side, subset=["Side"]).map(_color_pnl, subset=["Realized P&L"]),
                width="stretch", height=min(80 + len(_df) * 36, 400),
                column_config={"Question": st.column_config.TextColumn("Question", width="large")},
            )
            _tr = _live_closed["pnl"].fillna(0).sum()
            st.metric("Total Realized P&L", f"${_tr:+.4f}")

            if not positions_df.empty and "fill_status" in positions_df.columns:
                _cancelled_live = positions_df[
                    (positions_df["fill_status"] == "cancelled") &
                    (positions_df["is_paper"] == 0)
                ].copy() if "is_paper" in positions_df.columns else pd.DataFrame()
                if not _cancelled_live.empty:
                    st.markdown("**Cancelled Live Orders**")
                    st.caption(f"{len(_cancelled_live)} order(s) were cancelled before fill")
                    _can_rows = [{
                        "City": p.get("city", ""), "Date": p.get("date", ""),
                        "Side": p.get("side", ""), "Size ($)": f"${p.get('size_usdc', 0):.2f}",
                        "Reason": p.get("cancelled_reason", ""),
                        "Time": (p.get("exit_time") or "")[:16],
                    } for _, p in _cancelled_live.iterrows()]
                    st.dataframe(pd.DataFrame(_can_rows), width="stretch", hide_index=True)
        else:
            st.info("No closed live positions yet")

    # ── Refresh button ───────────────────────────────────────────────────────
    if st.button("🔄 Refresh Data"):
        with st.spinner("Running monitor loop…"):
            try:
                import sys as _sys, os as _os
                _sys.path.insert(0, _os.path.dirname(__file__))
                from db import backfill_position_coords
                from monitor import run_monitor_loop
                backfill_position_coords()
                run_monitor_loop()
            except Exception as _e:
                st.warning(f"Monitor run failed: {_e}")
        st.cache_data.clear()
        for _k in ("f_dates", "f_side", "f_side_disabled", "f_view", "f_contracts", "f_geo"):
            st.session_state.pop(_k, None)
        st.rerun()
