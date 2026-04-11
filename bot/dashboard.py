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
                e.normalization_warning
            FROM temp_outcomes o
            JOIN temp_events e ON o.event_row_id = e.id
            WHERE o.scan_timestamp = ?
            {where}
            ORDER BY o.ev DESC
            """,
            conn, params=(ts,)
        )
        conn.close()
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
                    st.markdown(f"Mkt: {_fmt_pct(mkt)}")
                with c_mdl:
                    if not _is_missing(mdl) and not _is_missing(mkt):
                        color = "green" if float(mdl) > float(mkt) else "red"
                        st.markdown(f":{color}[Mdl: {_fmt_pct(mdl)}]")
                    else:
                        st.markdown("Mdl: —")

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
# Top-level header
# ---------------------------------------------------------------------------

st.title("🌡 Highest-Temperature Arbitrage Dashboard")

last_scan = load_scan_timestamp()
st.caption(f"Last scan: {last_scan}  |  DB: {DB_PATH}  |  Auto-refresh: 60s")

# Global metrics
events_df   = load_events()
outcomes_df = load_outcomes(signals_only=False)
signals_df  = load_outcomes(signals_only=True)
positions_df = load_positions()

n_events   = len(events_df) if not events_df.empty else 0
n_outcomes = len(outcomes_df) if not outcomes_df.empty else 0
n_signals  = len(signals_df) if not signals_df.empty else 0
total_pnl  = positions_df["pnl"].sum() if not positions_df.empty and "pnl" in positions_df else 0
open_count = (
    len(positions_df[positions_df["status"] == "open"])
    if not positions_df.empty else 0
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Events Tracked",     n_events)
c2.metric("Outcome Ranges",     n_outcomes)
c3.metric("Trade Signals",      n_signals)
c4.metric("Open Positions",     open_count)
c5.metric("Cumulative P&L",     f"${total_pnl:.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3 = st.tabs([
    f"All Contracts  ({n_outcomes})",
    f"Trade Signals  ({n_signals}) ✨",
    "Market Overview  🗂",
])


# ===========================================================================
# TAB 1 — ALL CONTRACTS
# ===========================================================================
with tab1:
    if outcomes_df.empty:
        st.info("No data yet. Run the bot or wait for the next scan.")
    else:
        # Sidebar-style filters inside the tab
        with st.expander("Filters", expanded=False):
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                available_cities = sorted(outcomes_df["city"].dropna().unique().tolist())
                selected_cities = st.multiselect(
                    "Cities", available_cities, default=available_cities, key="t1_cities"
                )
            with col_f2:
                available_dates = sorted(outcomes_df["date"].dropna().unique().tolist())
                selected_dates = st.multiselect(
                    "Dates", available_dates, default=available_dates, key="t1_dates"
                )

        mask = (
            outcomes_df["city"].isin(selected_cities) &
            outcomes_df["date"].isin(selected_dates)
        )
        filtered = outcomes_df[mask].copy()

        # Build a flat display table
        display_rows = []
        for _, row in filtered.iterrows():
            unit = row.get("unit", "celsius")
            # Display temp in event's unit (inherits from city convention)
            ev_unit = row.get("display_unit", unit)
            display_rows.append({
                "City":         row.get("city", ""),
                "Date":         row.get("date", ""),
                "Days Out":     int(row.get("days_ahead", 0)),
                "Range":        _range_label(row.get("range_low"), row.get("range_high"), unit),
                "Market Price": _fmt_pct(row.get("market_price")),
                "Model Prob":   _fmt_pct(row.get("model_prob")),
                "Edge":         _fmt_ev(row.get("edge")),
                "EV":           _fmt_ev(row.get("ev")),
                "Signal":       _signal_badge(row.get("recommended_side")) if row.get("is_signal") else "—",
                "Liquidity":    f"${row.get('liquidity_usd', 0):,.0f}",
                "Norm Warn":    "⚠️" if row.get("normalization_warning") else "",
            })

        display_df = pd.DataFrame(display_rows)

        st.subheader("All Outcome Ranges")
        st.dataframe(
            display_df,
            width="stretch",
            height=400,
        )

        st.divider()

        # Distribution charts — one per event
        st.subheader("Model vs. Market Distribution")
        st.markdown(
            "*For each event, bars show **Market Price** (blue) and **Model Probability** (orange) "
            "for each temperature range.  The dashed line is the model's normal distribution curve.*"
        )

        # Group by city+date to render one chart per event
        event_groups = filtered.groupby(["city", "date"])

        for (city, date_str), grp in event_groups:
            grp = grp.copy()
            unit = grp["unit"].iloc[0] if not grp.empty else "celsius"
            sym  = "°F" if unit == "fahrenheit" else "°C"
            ev_unit = grp["display_unit"].iloc[0] if "display_unit" in grp.columns else unit

            mu_c    = grp["forecast_mu_c"].iloc[0] if "forecast_mu_c" in grp.columns else None
            sigma_c = grp["forecast_sigma_c"].iloc[0] if "forecast_sigma_c" in grp.columns else None
            clim_mu = grp["clim_mu_c"].iloc[0] if "clim_mu_c" in grp.columns else None
            mu_disp = grp["forecast_mu_display"].iloc[0] if "forecast_mu_display" in grp.columns else None
            overround = grp["market_overround"].iloc[0] if "market_overround" in grp.columns else None
            days_out = int(grp["days_ahead"].iloc[0]) if "days_ahead" in grp.columns else 0

            labels   = [_range_label(r.range_low, r.range_high, unit) for _, r in grp.iterrows()]
            mkt_vals = grp["market_price"].tolist()
            mdl_vals = grp["model_prob"].tolist()

            with st.expander(
                f"**{city}** — {date_str}  ({days_out}d out)  "
                f"| Forecast: {_fmt_temp(mu_disp, ev_unit)}  "
                f"| Overround: {_fmt_pct(overround)}",
                expanded=(days_out <= 3),
            ):
                col_chart, col_stats = st.columns([3, 1])

                with col_chart:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name="Market Price",
                        x=labels,
                        y=[v * 100 for v in mkt_vals],
                        marker_color="#4c78a8",
                        opacity=0.85,
                    ))
                    fig.add_trace(go.Bar(
                        name="Model Probability",
                        x=labels,
                        y=[v * 100 for v in mdl_vals],
                        marker_color="#f58518",
                        opacity=0.85,
                    ))
                    fig.update_layout(
                        barmode="group",
                        xaxis_title=f"Temperature Range ({sym})",
                        yaxis_title="Probability (%)",
                        yaxis=dict(ticksuffix="%"),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                        height=350,
                        margin=dict(t=40, b=40),
                    )
                    st.plotly_chart(fig, width="stretch")

                with col_stats:
                    st.markdown("**Forecast**")
                    if mu_disp is not None:
                        st.markdown(
                            f"**Forecast Avg:** {_fmt_temp(mu_disp, ev_unit)}",
                            help="The model's predicted mean maximum temperature for this day, "
                                 "blended from ECMWF + GFS ensemble members."
                        )
                    if sigma_c is not None:
                        sigma_disp = sigma_c if ev_unit == "celsius" else sigma_c * 9 / 5
                        sym = "°F" if ev_unit == "fahrenheit" else "°C"
                        st.markdown(
                            f"**Uncertainty (σ):** {sigma_disp:.1f}{sym}",
                            help="Standard deviation of the forecast distribution — how spread out "
                                 "the ensemble members are. Larger = less certain forecast."
                        )
                    if clim_mu is not None:
                        clim_disp = clim_mu if ev_unit == "celsius" else clim_mu * 9 / 5 + 32
                        st.markdown(
                            f"Historical Avg: {_fmt_temp(clim_disp, ev_unit)}",
                            help="The 10-year ERA5 climatological average for this calendar date "
                                 "and location. Used as a Bayesian prior at longer forecast horizons."
                        )

                    st.markdown("**Market**")
                    if overround is not None:
                        vig_pct = (overround - 1) * 100
                        color = "red" if vig_pct > 8 else "orange" if vig_pct > 4 else "green"
                        st.markdown(f"Overround: :{color}[{vig_pct:.1f}%]")

                    n_sig = int(grp["is_signal"].sum())
                    if n_sig:
                        st.markdown(f"**{n_sig} signal(s) ✨**")


# ===========================================================================
# TAB 2 — TRADE SIGNALS
# ===========================================================================
with tab2:
    if signals_df.empty:
        st.info(
            f"No signals above edge threshold ({EDGE_THRESHOLD*100:.0f}%). "
            "Either no edge exists in current markets or the bot hasn't run yet."
        )
    else:
        # Filters
        with st.expander("Filters", expanded=True):
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                sig_cities = sorted(signals_df["city"].dropna().unique().tolist())
                sel_cities = st.multiselect("City", sig_cities, default=sig_cities, key="t2_city")
            with col_s2:
                sides = ["YES", "NO"]
                sel_sides = st.multiselect("Side", sides, default=sides, key="t2_side")
            with col_s3:
                min_ev = st.slider(
                    "Min EV", min_value=0.0, max_value=0.50,
                    value=0.03, step=0.01, format="%.2f", key="t2_ev"
                )
            with col_s4:
                min_liq = st.slider(
                    "Min Liquidity ($)", min_value=0, max_value=10000,
                    value=500, step=100, key="t2_liq"
                )

        mask2 = (
            signals_df["city"].isin(sel_cities) &
            signals_df["recommended_side"].isin(sel_sides) &
            (signals_df["ev"] >= min_ev) &
            (signals_df["liquidity_usd"] >= min_liq)
        )
        sig_filtered = signals_df[mask2].copy()

        if sig_filtered.empty:
            st.info("No signals match the current filters.")
        else:
            st.success(f"**{len(sig_filtered)} signal(s)** match current filters")

            sig_rows = []
            for _, row in sig_filtered.iterrows():
                unit = row.get("unit", "celsius")
                ev_unit = row.get("display_unit", unit)
                mu_disp = row.get("forecast_mu_display")
                sigma_c = row.get("forecast_sigma_c")
                sigma_disp = None if sigma_c is None else (sigma_c * 9 / 5 if ev_unit == "fahrenheit" else sigma_c)
                sig_rows.append({
                    "City":          row.get("city", ""),
                    "Date":          row.get("date", ""),
                    "Days Out":      int(row.get("days_ahead", 0)),
                    "Range":         _range_label(row.get("range_low"), row.get("range_high"), unit),
                    "Side":          row.get("recommended_side", ""),
                    "Market Price":  _fmt_pct(row.get("market_price")),
                    "Model Prob":    _fmt_pct(row.get("model_prob")),
                    "Edge":          _fmt_ev(row.get("edge")),
                    "EV":            _fmt_ev(row.get("ev")),
                    "Kelly $":       f"${row.get('kelly_size', 0):.2f}",
                    "Liquidity":     f"${row.get('liquidity_usd', 0):,.0f}",
                    "Fcst μ":        _fmt_temp(mu_disp, ev_unit),
                    "Fcst σ":        f"{sigma_disp:.1f}{'°F' if ev_unit == 'fahrenheit' else '°C'}" if sigma_disp else "—",
                    "Norm Warn":     "⚠️" if row.get("normalization_warning") else "",
                    "Contract ID":   (row.get("contract_id") or "")[:12],
                })

            sig_display = pd.DataFrame(sig_rows)

            # Color-code the Side column
            def _highlight_side(val):
                if val == "YES":
                    return "color: #2ca02c; font-weight: bold"
                if val == "NO":
                    return "color: #d62728; font-weight: bold"
                return ""

            st.dataframe(
                sig_display.style.map(_highlight_side, subset=["Side"]),
                width="stretch",
                height=min(50 + len(sig_display) * 36, 600),
            )

            # EV distribution chart
            st.subheader("Signal EV Distribution")
            fig_ev = px.histogram(
                sig_filtered,
                x="ev",
                nbins=20,
                color="recommended_side",
                color_discrete_map={"YES": "#2ca02c", "NO": "#d62728"},
                labels={"ev": "Expected Value", "recommended_side": "Side"},
                title="Expected Value Distribution of Signals",
            )
            fig_ev.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_ev, width="stretch")

            # Bubble chart: edge vs. liquidity
            st.subheader("Edge vs. Liquidity")
            fig_bubble = px.scatter(
                sig_filtered,
                x="edge",
                y="liquidity_usd",
                size="kelly_size",
                color="recommended_side",
                color_discrete_map={"YES": "#2ca02c", "NO": "#d62728"},
                hover_data=["city", "date", "question", "ev"],
                labels={
                    "edge": "Edge (model − market)",
                    "liquidity_usd": "Liquidity ($)",
                    "recommended_side": "Side",
                },
                title="Signal Edge vs. Liquidity  (bubble size = Kelly $)",
            )
            fig_bubble.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_bubble, width="stretch")


# ===========================================================================
# TAB 3 — MARKET OVERVIEW
# ===========================================================================
with tab3:
    if outcomes_df.empty:
        st.info("No data yet. Run the bot or wait for the next scan.")
    else:
        # ── Top row: date selector pinned to the right ───────────────────────
        col_hdr, _spacer, col_date = st.columns([3, 2, 1])
        with col_hdr:
            st.subheader("Market Overview")
        with col_date:
            _all_dates_ov = sorted(outcomes_df["date"].dropna().unique().tolist())
            sel_date_ov   = st.selectbox(
                "Date", _all_dates_ov, index=0, key="t3_date",
                label_visibility="collapsed",
            )

        # ── Filters expander (matches Tab 1 / Tab 2 pattern) ────────────────
        with st.expander("Filters", expanded=True):
            ov_col1, ov_col2, ov_col3, ov_col4 = st.columns(4)
            with ov_col1:
                _ov_all_cities = sorted(outcomes_df["city"].dropna().unique().tolist())
                ov_sel_cities  = st.multiselect(
                    "City", _ov_all_cities, default=_ov_all_cities, key="ov_city",
                )
            with ov_col2:
                ov_signals_only = st.checkbox(
                    "Signals only", value=False, key="ov_signals_only",
                )
            with ov_col3:
                ov_min_ev = st.slider(
                    "Min EV", min_value=0.0, max_value=0.30,
                    value=0.0, step=0.01, format="%.2f", key="ov_min_ev",
                )
            with ov_col4:
                ov_min_liq = st.slider(
                    "Min Liquidity ($)", min_value=0, max_value=10_000,
                    value=500, step=100, key="ov_min_liq",
                )

        # ── Apply filters ────────────────────────────────────────────────────
        ov = outcomes_df.copy()
        ov = ov[ov["date"] == sel_date_ov]
        if ov_sel_cities:
            ov = ov[ov["city"].isin(ov_sel_cities)]
        ov = ov[ov["liquidity_usd"].fillna(0) >= ov_min_liq]
        if ov_min_ev > 0:
            ov = ov[ov["ev"].fillna(0) >= ov_min_ev]
        if ov_signals_only:
            _sig_events = (
                ov[ov["is_signal"] == 1][["city", "date"]].drop_duplicates()
            )
            ov = ov.merge(_sig_events, on=["city", "date"], how="inner")

        if ov.empty:
            st.info("No events match the current filters for this date.")
        else:
            _event_groups = sorted(
                ov.groupby(["city", "date"]),
                key=lambda x: x[0][0].lower(),
            )
            st.caption(f"{len(_event_groups)} event(s) on {sel_date_ov}")

            _COLS = 3
            for _i in range(0, len(_event_groups), _COLS):
                _chunk     = _event_groups[_i : _i + _COLS]
                _grid_cols = st.columns(_COLS)
                for _j, ((_city, _date), _grp) in enumerate(_chunk):
                    with _grid_cols[_j]:
                        _render_event_card(_city, _date, _grp)


st.divider()

# ---------------------------------------------------------------------------
# Bottom panel — P&L and model calibration (retained from prior version)
# ---------------------------------------------------------------------------
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
        st.markdown("*Average actual outcome vs. model probability bucket.*")
        try:
            conn = sqlite3.connect(DB_PATH)
            sig_df = pd.read_sql(
                "SELECT * FROM signals WHERE executed = 1 AND outcome IS NOT NULL", conn
            )
            conn.close()
            if not sig_df.empty and "model_p" in sig_df.columns:
                sig_df["p_bucket"] = pd.cut(sig_df["model_p"], bins=10, labels=False)
                cal = sig_df.groupby("p_bucket").agg(
                    avg_model_p=("model_p", "mean"),
                    avg_outcome=("outcome", "mean"),
                    count=("outcome", "count"),
                ).reset_index()
                fig_cal = go.Figure()
                fig_cal.add_trace(go.Bar(
                    x=cal["avg_model_p"], y=cal["avg_outcome"],
                    name="Actual Frequency", marker_color="steelblue",
                ))
                fig_cal.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    name="Perfect Calibration",
                    line=dict(color="red", dash="dash"),
                ))
                fig_cal.update_layout(
                    xaxis_title="Model Probability",
                    yaxis_title="Actual Outcome Rate",
                )
                st.plotly_chart(fig_cal, width="stretch")
            else:
                st.info("Not enough resolved trades for calibration chart")
        except Exception:
            st.info("Not enough resolved trades for calibration chart")

# ---------------------------------------------------------------------------
# Open positions table
# ---------------------------------------------------------------------------
with st.expander("Open Positions", expanded=False):
    if not positions_df.empty:
        open_pos = positions_df[positions_df["status"] == "open"]
        if not open_pos.empty:
            st.dataframe(open_pos, width="stretch")
        else:
            st.info("No open positions")
    else:
        st.info("No position data")

# ---------------------------------------------------------------------------
# Refresh button
# ---------------------------------------------------------------------------
if st.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
