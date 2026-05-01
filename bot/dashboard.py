"""
dashboard.py — Highest-temperature market analysis dashboard

Two tabs:
    Tab 1 — All Contracts
        Every discovered highest-temperature event and every outcome range,
        with city, date, temperature range, market price, model probability,
        edge, and EV.  Includes a distribution chart (model vs. market) per event.

    Tab 2 — Trade Signals
        Only the outcome ranges that qualify as signals under the active strategy.
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
from datetime import datetime, timezone
from functools import lru_cache
from streamlit_autorefresh import st_autorefresh

from config import DB_PATH


def _connect_db(timeout: float = 30.0):
    """Open a SQLite connection with WAL + busy_timeout pragmas.

    Mirrors db._get_conn's pragmas so the dashboard's read connections
    don't lock against the bot's writes.  Without WAL the dashboard's
    60-second auto-refresh would frequently collide with monitor cycles
    causing `sqlite3.OperationalError: database is locked` (the user-
    reported error from 2026-04-30).  WAL mode allows N readers + 1
    writer concurrently — exactly our access pattern.
    """
    conn = sqlite3.connect(DB_PATH, timeout=timeout)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except Exception:
        pass
    return conn


# ---------------------------------------------------------------------------
# Local-time helper (cached per-rounded-coord; uses timezonefinder + zoneinfo)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=512)
def _resolve_zoneinfo(lat_round: float, lon_round: float):
    from zoneinfo import ZoneInfo
    try:
        from timezonefinder import TimezoneFinder
        name = TimezoneFinder().timezone_at(lat=lat_round, lng=lon_round)
        return ZoneInfo(name) if name else timezone.utc
    except Exception:
        return timezone.utc


def _local_time_str(lat, lon, when_utc: datetime | None = None) -> str:
    """Formatted local time at (lat, lon).  Returns '' on missing/invalid."""
    try:
        if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
            return ""
        if when_utc is None:
            when_utc = datetime.now(timezone.utc)
        tz = _resolve_zoneinfo(round(float(lat), 2), round(float(lon), 2))
        return when_utc.astimezone(tz).strftime("%a %H:%M %Z")
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Polymarket Weather Bot",
    page_icon="🌡",
    layout="wide",
)

_REFRESH_INTERVAL_MS = 300_000  # 5 min — was 60s.  See top-of-file note.

# Cache TTL for DB-backed loaders.  Deliberately SHORTER than the refresh
# interval so every autorefresh hits a warm cache and renders instantly,
# but the cached data is fresh enough to still feel current.  At 240s
# (4 min) with 300s refresh, the cache is always populated by the time
# the autorefresh fires the next render.
_CACHE_TTL_SHORT = 240
_CACHE_TTL_LONG  = 600

# Hide autorefresh bar + center tabs + add metric padding
st.markdown("""<style>
    iframe[title="streamlit_autorefresh.st_autorefresh"] { display: none; }
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    [data-testid="stMetric"] {
        padding: 20px 0;
        text-align: center;
    }
    [data-testid="stMetricLabel"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    [data-testid="stMetricValue"] {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    th { text-align: center !important; }
    .stDataFrame th, .stDataFrame td { text-align: center !important; }
    [data-testid="stDataFrameResizable"] th { text-align: center !important; }
</style>""", unsafe_allow_html=True)
st_autorefresh(interval=_REFRESH_INTERVAL_MS, limit=None, key="global_autorefresh")

# ---------------------------------------------------------------------------
# Data loaders (cached, auto-refresh every 15 s — synced with autorefresh)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=_CACHE_TTL_SHORT)
def load_events() -> pd.DataFrame:
    """Load the most recent scan row for each event (city+date)."""
    try:
        conn = _connect_db()
        df = pd.read_sql(
            """
            SELECT e.* FROM temp_events e
            INNER JOIN (
                SELECT event_id, MAX(scan_timestamp) AS max_ts
                FROM temp_events
                GROUP BY event_id
            ) latest ON e.event_id = latest.event_id
                    AND e.scan_timestamp = latest.max_ts
            WHERE e.date >= date('now')
            ORDER BY e.city, e.date
            """,
            conn,
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=_CACHE_TTL_SHORT)
def load_outcomes(signals_only: bool = False) -> pd.DataFrame:
    """Load the most recent outcomes for each event, joined with event context."""
    try:
        conn = _connect_db()
        where = "AND o.is_signal = 1" if signals_only else ""
        df = pd.read_sql(
            f"""
            SELECT
                o.*,
                e.city,
                e.date,
                e.event_title,
                e.event_id,
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
            INNER JOIN (
                SELECT event_id, MAX(scan_timestamp) AS max_ts
                FROM temp_events
                GROUP BY event_id
            ) latest ON e.event_id = latest.event_id
                    AND e.scan_timestamp = latest.max_ts
            WHERE e.date >= date('now')
            {where}
            ORDER BY o.ev DESC
            """,
            conn,
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


@st.cache_data(ttl=_CACHE_TTL_SHORT)
def load_positions() -> pd.DataFrame:
    try:
        conn = _connect_db()
        df = pd.read_sql(
            "SELECT * FROM positions ORDER BY entry_time DESC", conn
        )
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_scan_timestamp() -> str:
    try:
        conn = _connect_db()
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

def _fmt_date_mmddyyyy(d) -> str:
    """Format a YYYY-MM-DD date string as MM-DD-YYYY."""
    if not d:
        return ""
    try:
        parts = str(d)[:10].split("-")
        if len(parts) == 3:
            return f"{parts[1]}-{parts[2]}-{parts[0]}"
    except Exception:
        pass
    return str(d)[:10]


def _fmt_local_time(lt) -> str:
    """Reformat the local_time string (stored as 'HH:MM MM-DD-YYYY') to MM-DD-YY HH:MM:SS."""
    if not lt:
        return ""
    try:
        # Parse the stored format: "HH:MM MM-DD-YYYY"
        dt = datetime.strptime(str(lt).strip(), "%H:%M %m-%d-%Y")
        return dt.strftime("%m-%d-%y %H:%M:%S")
    except Exception:
        return str(lt)


def _fmt_entered(ts) -> str:
    """
    Format an ISO timestamp as 'HH:MM:SS AM/PM MM-DD-YYYY' in US Central time.
    Handles both UTC-stored legacy entries and Chicago-stored new entries:
    any tz-aware ISO string is converted to America/Chicago before formatting.
    Naive strings are assumed to already be local and displayed as-is.
    """
    if not ts:
        return "—"
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(ZoneInfo("America/Chicago"))
        return dt.strftime("%m-%d-%y %H:%M:%S")
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


def _safe_str(v) -> str:
    """Coerce a possibly-NaN/None value from a pandas row dict into a clean
    string.  pandas reads SQL NULLs as float NaN — the idiom
    `(value or "").upper()` blows up because NaN is truthy AND has no
    .upper().  Use this helper anywhere we read string-ish columns from
    a DataFrame iterrows() pass."""
    if _is_missing(v):
        return ""
    return str(v)


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

    # Show local time at the city, if we have coords on any row
    if rows:
        _r0 = rows[0]
        _lat = _r0.get("lat"); _lon = _r0.get("lon")
        _lt = _local_time_str(_lat, _lon)
        if _lt:
            # Also surface the ML decision_hour fold (diagnostic) when present
            _fold_bits = ""
            for _r in rows:
                _fold = _r.get("ml_decision_hour")
                if _fold is not None and not _is_missing(_fold):
                    _fold_bits = f"  ·  ML fold: {int(_fold):02d}:00"
                    break
            st.caption(f"Local: {_lt}{_fold_bits}")

    # Forecast summary metrics
    if rows:
        r0       = rows[0]
        ev_unit  = r0.get("display_unit", r0.get("unit", "celsius"))
        sym      = "°F" if ev_unit == "fahrenheit" else "°C"
        mu_disp  = r0.get("forecast_mu_display")
        sigma_c  = r0.get("forecast_sigma_c")
        clim_mu  = r0.get("clim_mu_c")

        # Fetch individual ECMWF / GFS predictions from forecast_runs
        _ecmwf_mu = None
        _gfs_mu = None
        try:
            _event_id = r0.get("event_id") if "event_id" in r0 else None
            if _event_id:
                from db import get_latest_forecast_run
                _ecmwf_run = get_latest_forecast_run(_event_id, "ecmwf")
                _gfs_run = get_latest_forecast_run(_event_id, "gfs")
                if _ecmwf_run and _ecmwf_run.get("forecast_mu_c") is not None:
                    _ecmwf_mu = float(_ecmwf_run["forecast_mu_c"])
                    if ev_unit == "fahrenheit":
                        _ecmwf_mu = _ecmwf_mu * 9 / 5 + 32
                if _gfs_run and _gfs_run.get("forecast_mu_c") is not None:
                    _gfs_mu = float(_gfs_run["forecast_mu_c"])
                    if ev_unit == "fahrenheit":
                        _gfs_mu = _gfs_mu * 9 / 5 + 32
        except Exception:
            pass

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        with mc1:
            if _ecmwf_mu is not None:
                st.metric("ECMWF", f"{_ecmwf_mu:.1f}{sym}")
        with mc2:
            if _gfs_mu is not None:
                st.metric("GFS", f"{_gfs_mu:.1f}{sym}")
        with mc3:
            if not _is_missing(mu_disp):
                st.metric("Blended Avg", f"{float(mu_disp):.1f}{sym}")
        with mc4:
            if not _is_missing(sigma_c):
                sd = float(sigma_c) if ev_unit == "celsius" else float(sigma_c) * 9 / 5
                st.metric("Uncertainty", f"{sd:.1f}{sym}")
        with mc5:
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

    # Load open positions for this event to show held status
    _open_pos_by_contract: dict[str, dict] = {}
    try:
        _pos_conn = _connect_db()
        _pos_conn.row_factory = sqlite3.Row
        _pos_rows = _pos_conn.execute(
            "SELECT contract_id, side, size_usdc, target_size_usdc, unrealized_pnl "
            "FROM positions WHERE city = ? AND date = ? AND status = 'open' "
            "AND fill_status = 'filled'",
            (city, date_str),
        ).fetchall()
        _pos_conn.close()
        for _p in _pos_rows:
            _open_pos_by_contract[_p["contract_id"]] = dict(_p)
    except Exception:
        pass

    # Full outcome table
    table_rows = []
    for r in rows_sorted:
        u      = r.get("unit", "celsius")
        mdl    = r.get("model_prob")
        yes_p  = r.get("yes_price") or r.get("market_price")
        no_p   = r.get("no_price")
        if _is_missing(no_p) and not _is_missing(yes_p):
            no_p = 1.0 - float(yes_p)
        no_mdl = (1.0 - float(mdl)) if not _is_missing(mdl) else None
        edge   = r.get("edge")
        ev     = r.get("ev")

        # Position status: show HELD YES/NO if we have an open position,
        # otherwise show the strategy's signal recommendation
        cid = r.get("contract_id")
        held = _open_pos_by_contract.get(cid)
        if held:
            size = float(held.get("size_usdc") or 0)
            target = held.get("target_size_usdc")
            side_label = held["side"]
            if target and float(target) > size:
                status = f"HELD {side_label} ${size:.0f}/${float(target):.0f}"
            else:
                status = f"HELD {side_label} ${size:.0f}"
        else:
            # Show signal based on active strategy's criteria, not edge-disagreement
            from config import ACTIVE_STRATEGY, ALLOWED_SIDES as _allowed
            _mdl_p = float(mdl) if not _is_missing(mdl) else 0
            _show_signal = False
            _sig_side = ""

            if ACTIVE_STRATEGY == "top_bin_value":
                from strategies.top_bin_value import tbv_qualifies_as_signal
                _mkt_p = float(yes_p) if not _is_missing(yes_p) else 0
                if tbv_qualifies_as_signal(_mdl_p, _mkt_p, city=city,
                        forecast_sigma_c=float(sigma_c) if not _is_missing(sigma_c) else None):
                    _sig_side = "YES"
                    _show_signal = True
            elif bool(r.get("is_signal")):
                _sig_side = r.get("recommended_side", "")
                _show_signal = True

            if _show_signal:
                if (_allowed == "yes" and _sig_side == "NO") or \
                   (_allowed == "no" and _sig_side == "YES"):
                    status = "--"
                else:
                    status = _signal_badge(_sig_side)
            else:
                status = "--"

        # P&L for held positions
        _pnl_val = None
        if held:
            _pnl_val = float(held.get("unrealized_pnl") or 0)

        ml_p = r.get("ml_bin_prob")
        table_rows.append({
            "Range":            _range_label(r.get("range_low"), r.get("range_high"), u),
            "Yes Market Prob":  _fmt_pct(yes_p),
            "Yes Model Prob":   _fmt_pct(mdl),
            "ML Bin Prob":      _fmt_pct(ml_p),
            "No Market Prob":   _fmt_pct(no_p),
            "No Model Prob":    _fmt_pct(no_mdl),
            "Status":           status,
            "P&L":              _pnl_val,
            "Volume":           f"${r.get('volume_usd', 0):,.0f}",
        })

    df_modal = pd.DataFrame(table_rows)
    # Hide the ML column entirely when no row has data (e.g., D>0 events
    # or scans before the pooled model is loaded).  _fmt_pct returns "—"
    # for None.
    if "ML Bin Prob" in df_modal.columns and (df_modal["ML Bin Prob"] == "—").all():
        df_modal = df_modal.drop(columns=["ML Bin Prob"])

    def _color_modal_pnl(val):
        if val is None or pd.isna(val):
            return ""
        if val > 0:
            return "color: #2ecc71"
        elif val < 0:
            return "color: #e74c3c"
        return ""

    st.dataframe(
        df_modal.style.map(
            _color_modal_pnl, subset=["P&L"]
        ).format({"P&L": lambda v: f"${v:+,.2f}" if pd.notna(v) and v is not None else ""}),
        width="stretch", hide_index=True,
        height=min(120 + len(df_modal) * 38, 520),
    )


def _render_event_card(city: str, date_str: str, grp: pd.DataFrame) -> None:
    """Render a single event summary card for the Market Overview grid."""
    unit     = grp["unit"].iloc[0] if not grp.empty else "celsius"
    ev_unit  = grp["display_unit"].iloc[0] if "display_unit" in grp.columns else unit
    _da_raw = grp["days_ahead"].iloc[0] if "days_ahead" in grp.columns else 0
    days_out = int(_da_raw) if _da_raw is not None and not (isinstance(_da_raw, float) and _da_raw != _da_raw) else 0
    title_val = grp["event_title"].iloc[0] if "event_title" in grp.columns else None
    title    = (
        title_val if not _is_missing(title_val)
        else f"Highest temperature in {city} on {date_str}"
    )
    total_vol = grp["volume_usd"].sum() if "volume_usd" in grp.columns else 0
    _lat = grp["lat"].iloc[0] if "lat" in grp.columns else None
    _lon = grp["lon"].iloc[0] if "lon" in grp.columns else None
    local_now = _local_time_str(_lat, _lon)

    # Signal count based on active strategy's criteria
    from config import ACTIVE_STRATEGY as _card_strategy
    if _card_strategy == "top_bin_value" and "model_prob" in grp.columns:
        from strategies.top_bin_value import tbv_qualifies_as_signal as _tbv_card
        _card_city = grp["city"].iloc[0] if "city" in grp.columns else None
        n_signals = sum(
            _tbv_card(
                float(r.get("model_prob") or 0) if not _is_missing(r.get("model_prob")) else 0,
                float(r.get("yes_price") or r.get("market_price") or 0) if not _is_missing(r.get("yes_price") or r.get("market_price")) else 0,
                city=_card_city,
                forecast_sigma_c=float(r.get("forecast_sigma_c")) if not _is_missing(r.get("forecast_sigma_c")) else None,
            ) for _, r in grp.iterrows()
        )
    else:
        n_signals = int(grp["is_signal"].sum()) if "is_signal" in grp.columns else 0

    # Top 4 outcomes — strategy-aware:
    #   top_bin_value: filter to rows with model_prob, sort by market_price
    #     (preserves prior behavior — model_prob is needed to display)
    #   anything else (incl. market_price_value): no model_prob filter, just
    #     sort by market_price.  Captures majority of market view either way.
    if _card_strategy == "top_bin_value" and "model_prob" in grp.columns:
        candidates = grp[grp["model_prob"].notna()].copy()
    else:
        candidates = grp.copy()
    top4 = candidates.sort_values("market_price", ascending=False).head(4)

    # Detect ML bin-prob availability — only show the ML column when we
    # actually have data (D=0 events with the pooled model loaded).
    has_ml = (
        "ml_bin_prob" in top4.columns
        and top4["ml_bin_prob"].notna().any()
    )

    # Unique stable key
    card_key = f"card_{''.join(c if c.isalnum() else '_' for c in f'{city}_{date_str}')}"

    with st.container(border=True):
        st.markdown(f"### {title}")
        # Header line: days-out · volume · local time at the city
        _header_bits = [f"{days_out}d out", f"${total_vol:,.0f} Vol."]
        if local_now:
            _header_bits.append(f"Local: {local_now}")
        st.caption("  ·  ".join(_header_bits))

        st.divider()

        if top4.empty:
            st.caption("No bin data available")
        else:
            # Column headers — adjust layout when ML column is shown
            if has_ml:
                cols = st.columns([2, 1, 1, 1])
                cols[0].caption("Range")
                cols[1].caption("Market Prob")
                cols[2].caption("Model Prob")
                cols[3].caption("ML Prob")
            else:
                cols = st.columns([2, 1, 1])
                cols[0].caption("Range")
                cols[1].caption("Market Prob")
                cols[2].caption("Model Prob")

            for _, row in top4.iterrows():
                rl   = _range_label(row.get("range_low"), row.get("range_high"), unit)
                mkt  = row.get("market_price")
                mdl  = row.get("model_prob")
                ml_p = row.get("ml_bin_prob") if has_ml else None

                if has_ml:
                    rcols = st.columns([2, 1, 1, 1])
                    rcols[0].markdown(f"**{rl}**")
                    rcols[1].markdown(_fmt_pct(mkt))
                    rcols[2].markdown(_fmt_pct(mdl))
                    rcols[3].markdown(_fmt_pct(ml_p))
                else:
                    rcols = st.columns([2, 1, 1])
                    rcols[0].markdown(f"**{rl}**")
                    rcols[1].markdown(_fmt_pct(mkt))
                    rcols[2].markdown(_fmt_pct(mdl))

        st.divider()

        cf_sig, cf_spacer, cf_btn = st.columns([2, 2, 1])
        with cf_sig:
            if n_signals:
                st.markdown(f"**{n_signals}** signal(s)")
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

st.markdown("<h1 style='text-align: center;'>🌡 Polymarket Weather Bot</h1>", unsafe_allow_html=True)

# Manual refresh control — autorefresh fires every 5 min, but the operator
# can force an immediate cache-clear-and-rerun here.  Centered under the
# title, narrow column so the button doesn't dominate the header.
_refresh_left, _refresh_mid, _refresh_right = st.columns([3, 1, 3])
with _refresh_mid:
    if st.button("Refresh now", width="stretch", help=(
        f"Auto-refresh runs every {_REFRESH_INTERVAL_MS // 60000} minutes. "
        f"Click to clear cached data and re-render immediately."
    )):
        _do_light_refresh()

last_scan = load_scan_timestamp()

events_df    = load_events()
outcomes_df  = load_outcomes(signals_only=False)
signals_df   = load_outcomes(signals_only=True)
positions_df = load_positions()

n_events   = len(events_df)   if not events_df.empty   else 0
n_outcomes = len(outcomes_df) if not outcomes_df.empty else 0
# Signal count based on active strategy
from config import ACTIVE_STRATEGY as _top_strategy
if _top_strategy == "top_bin_value" and not outcomes_df.empty and "model_prob" in outcomes_df.columns:
    import os
    from strategies.top_bin_value import tbv_qualifies_as_signal as _tbv_q

    def _tbv_check_row(r):
        _mp_v = float(r.get("model_prob") or 0) if not _is_missing(r.get("model_prob")) else 0
        _yp_v = float(r.get("yes_price") or r.get("market_price") or 0) if not _is_missing(r.get("yes_price") or r.get("market_price")) else 0
        _sc_v = float(r.get("forecast_sigma_c")) if not _is_missing(r.get("forecast_sigma_c")) else None
        return _tbv_q(_mp_v, _yp_v, city=r.get("city"), forecast_sigma_c=_sc_v)

    n_signals = sum(_tbv_check_row(r) for _, r in outcomes_df.iterrows())
else:
    n_signals = len(signals_df) if not signals_df.empty else 0
n_cities   = outcomes_df["city"].nunique() if not outcomes_df.empty else 0

# Partition positions into paper vs live (needed by all tabs)
if not positions_df.empty:
    _base_open = positions_df[positions_df["status"] == "open"].copy()
    if "fill_status" in positions_df.columns:
        _base_open = _base_open[_base_open["fill_status"] != "cancelled"]
    _base_closed = positions_df[positions_df["status"] == "closed"].copy()
    if "fill_status" in positions_df.columns:
        _base_closed = _base_closed[_base_closed["fill_status"] != "cancelled"]

    _paper_open   = _base_open[_base_open["is_paper"] == 1]   if "is_paper" in _base_open.columns   else _base_open
    _live_open    = _base_open[_base_open["is_paper"] == 0]   if "is_paper" in _base_open.columns   else pd.DataFrame()
    _paper_closed = _base_closed[_base_closed["is_paper"] == 1] if "is_paper" in _base_closed.columns else _base_closed
    _live_closed  = _base_closed[_base_closed["is_paper"] == 0] if "is_paper" in _base_closed.columns else pd.DataFrame()
else:
    _paper_open = _live_open = _paper_closed = _live_closed = pd.DataFrame()

# ---------------------------------------------------------------------------
# Refresh helpers — defined before tabs so any tab can trigger a refresh
# ---------------------------------------------------------------------------
def _do_refresh(spinner_key: str):
    msgs: list[tuple[str, str]] = []  # (level, text) — survives st.rerun() via session_state

    with st.spinner("Refreshing data…"):
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(__file__))

        # Step 1 — ensure DB schema is current (adds forecast_sigma_c column if missing)
        try:
            from db import init_db
            init_db()
            msgs.append(("info", "DB schema verified"))
        except Exception as _e:
            msgs.append(("error", f"DB init failed: {_e}"))

        # Step 2 — backfill coords and sigma independently so one failure doesn't
        #           block the other
        try:
            from db import backfill_position_coords
            n_coords = backfill_position_coords()
            if n_coords:
                msgs.append(("info", f"Backfilled lat/lon for {n_coords} position(s)"))
        except Exception as _e:
            msgs.append(("error", f"Coord backfill failed: {_e}"))

        try:
            from db import backfill_position_sigma
            n_sigma = backfill_position_sigma()
            if n_sigma:
                msgs.append(("info", f"Backfilled uncertainty for {n_sigma} position(s)"))
        except Exception as _e:
            msgs.append(("error", f"Sigma backfill failed: {_e}"))

        # Step 3 — run the full monitor loop
        try:
            from monitor import run_monitor_loop
            summary = run_monitor_loop()
            msgs.append((
                "success",
                f"Monitor complete — cancelled={summary.get('cancelled', 0)} "
                f"resolved={summary.get('closed', 0)} "
                f"pnl_updated={summary.get('updated', 0)}"
            ))
        except Exception as _e:
            msgs.append(("error", f"Monitor loop failed: {_e}"))

    # Persist messages across st.rerun() via session_state
    st.session_state["_refresh_msgs"] = msgs

    st.cache_data.clear()
    for _k in ("f_dates", "f_side", "f_side_disabled", "f_view", "f_contracts", "f_geo"):
        st.session_state.pop(_k, None)
    st.rerun()


def _show_refresh_messages():
    """Display any messages left by the last _do_refresh call, then clear them."""
    msgs = st.session_state.pop("_refresh_msgs", [])
    for level, text in msgs:
        if level == "success":
            st.success(text)
        elif level == "error":
            st.error(text)
        else:
            st.info(text)


def _do_light_refresh():
    """
    Lightweight refresh — clears cached DB queries and reruns so the
    on-screen metrics reflect the latest positions table.  Does NOT
    run init_db, backfills, or the monitor loop.
    """
    st.cache_data.clear()
    st.rerun()


# ===========================================================================
# Main tabs
# ===========================================================================
_tab_contract, _tab_paper, _tab_live, _tab_activity, _tab_accuracy = st.tabs(
    ["Contract Data", "Paper Trade Data", "Live Trade Data", "Activity", "Forecast Accuracy"]
)

# ===========================================================================
# TAB 1 — Contract Data
# ===========================================================================
with _tab_contract:

    _show_refresh_messages()

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
        r1c1, r1c2, r1c3 = st.columns([3, 2, 1])
        sel_sides = ["YES", "NO"]

        with r1c1:
            sel_cities = st.multiselect(
                "City", _all_cities, default=_all_cities, key="f_cities",
            )

        with r1c2:
            sel_dates = st.multiselect(
                "Date", _all_dates, default=_all_dates, key="f_dates",
            )

        with r1c3:
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


        # ── Apply filters ────────────────────────────────────────────────────
        _base = outcomes_df.copy()

        if sel_cities:
            _base = _base[_base["city"].isin(sel_cities)]
        if sel_dates:
            _base = _base[_base["date"].isin(sel_dates)]

        if geo_mode == "US Only" and "is_us" in _base.columns:
            _base = _base[_base["is_us"] == True]

        # Signals-only filter: use active strategy's criteria
        if signals_only:
            if _top_strategy == "top_bin_value":
                _keep = _base.apply(lambda r: _tbv_check_row(r), axis=1)
                _base = _base[_keep]
            else:
                _base = _base[_base["is_signal"] == 1]
                _base = _base[_base["recommended_side"].isin(sel_sides)]

        if _base.empty:
            st.info("No outcomes match the current filters.")
        else:
            # ── Status line ──────────────────────────────────────────────────
            n_filtered_events  = _base.groupby(["city", "date"]).ngroups
            if _top_strategy == "top_bin_value":
                n_filtered_signals = sum(_tbv_check_row(r) for _, r in _base.iterrows())
            else:
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

                # Load open positions for status display
                _list_positions: dict[str, dict] = {}
                try:
                    _lp_conn = _connect_db()
                    _lp_conn.row_factory = sqlite3.Row
                    for _lp in _lp_conn.execute(
                        "SELECT contract_id, side, size_usdc FROM positions "
                        "WHERE status='open' AND fill_status='filled'"
                    ).fetchall():
                        _list_positions[_lp["contract_id"]] = dict(_lp)
                    _lp_conn.close()
                except Exception:
                    pass

                display_rows = []
                for _, row in _list_df.iterrows():
                    unit    = row.get("unit", "celsius")
                    ev_unit = row.get("display_unit", unit)
                    sigma_c = row.get("forecast_sigma_c")
                    sigma_disp = (
                        None if _is_missing(sigma_c)
                        else (float(sigma_c) * 9 / 5 if ev_unit == "fahrenheit" else float(sigma_c))
                    )
                    sym = "F" if ev_unit == "fahrenheit" else "C"

                    # Status: same logic as the popup — held > signal > empty
                    _cid = row.get("contract_id")
                    _held = _list_positions.get(_cid) if _cid else None
                    if _held:
                        _status = f"HELD {_held['side']} ${float(_held['size_usdc']):.0f}"
                    else:
                        _mdl_p = float(row.get("model_prob")) if not _is_missing(row.get("model_prob")) else 0
                        _mkt_p_lv = float(row.get("yes_price") or row.get("market_price") or 0) if not _is_missing(row.get("yes_price") or row.get("market_price")) else 0
                        if _top_strategy == "top_bin_value":
                            if _tbv_check_row(row):
                                _status = _signal_badge("YES")
                            else:
                                _status = "--"
                        elif row.get("is_signal"):
                            _sig_s = row.get("recommended_side", "")
                            from config import ALLOWED_SIDES as _lv_allowed
                            if (_lv_allowed == "yes" and _sig_s == "NO") or \
                               (_lv_allowed == "no" and _sig_s == "YES"):
                                _status = "--"
                            else:
                                _status = _signal_badge(_sig_s)
                        else:
                            _status = "--"

                    display_rows.append({
                        "City":            row.get("city", ""),
                        "Date":            row.get("date", ""),
                        "Days Out":        int(row.get("days_ahead") or 0) if not _is_missing(row.get("days_ahead")) else 0,
                        "Range":           _range_label(row.get("range_low"), row.get("range_high"), unit),
                        "Yes Market Prob": _fmt_pct(row.get("yes_price") or row.get("market_price")),
                        "No Market Prob":  _fmt_pct(row.get("no_price")),
                        "Yes Model Prob":  _fmt_pct(row.get("model_prob")),
                        "No Model Prob":   _fmt_pct((1.0 - float(row["model_prob"])) if not _is_missing(row.get("model_prob")) else None),
                        "Edge":            _fmt_ev(row.get("edge")),
                        "EV ($/dollar)":   _fmt_ev(row.get("ev")),
                        "Liquidity":       f"${row.get('liquidity_usd', 0):,.0f}",
                        "Fcst Avg":        _fmt_temp(row.get("forecast_mu_display"), ev_unit),
                        "Uncertainty":     f"{sigma_disp:.1f}{sym}" if sigma_disp is not None else "--",
                        "Status":          _status,
                    })

                display_df = pd.DataFrame(display_rows)
                st.dataframe(
                    display_df,
                    width="stretch",
                    hide_index=True,
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
                    _da2 = grp["days_ahead"].iloc[0] if "days_ahead" in grp.columns else 0
                    days_out  = int(_da2) if _da2 is not None and not (isinstance(_da2, float) and _da2 != _da2) else 0

                    def _chart_sort(r):
                        lo = None if _is_missing(r.range_low)  else float(r.range_low)
                        hi = None if _is_missing(r.range_high) else float(r.range_high)
                        return lo if lo is not None else (hi if hi is not None else 0.0)
                    grp = grp.sort_values(by=grp.columns[0], key=lambda _: grp.apply(_chart_sort, axis=1), ascending=False)

                    labels   = [_range_label(r.range_low, r.range_high, unit) for _, r in grp.iterrows()]
                    mkt_vals = [v if not _is_missing(v) else 0.0 for v in grp["market_price"].tolist()]
                    mdl_vals = [v if not _is_missing(v) else 0.0 for v in grp["model_prob"].tolist()]
                    if _top_strategy == "top_bin_value" and "model_prob" in grp.columns:
                        from strategies.top_bin_value import tbv_qualifies_as_signal as _tbv_ch
                        _ch_city = grp["city"].iloc[0] if "city" in grp.columns else None
                        n_sig = sum(
                            _tbv_ch(
                                float(r.get("model_prob") or 0) if not _is_missing(r.get("model_prob")) else 0,
                                float(r.get("yes_price") or r.get("market_price") or 0) if not _is_missing(r.get("yes_price") or r.get("market_price")) else 0,
                                city=_ch_city,
                                forecast_sigma_c=float(r.get("forecast_sigma_c")) if not _is_missing(r.get("forecast_sigma_c")) else None,
                            ) for _, r in grp.iterrows()
                        )
                    else:
                        n_sig = int(grp["is_signal"].sum())

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
                            # Individual model predictions
                            try:
                                _eid = grp["event_id"].iloc[0] if "event_id" in grp.columns else None
                                if _eid:
                                    from db import get_latest_forecast_run
                                    _er = get_latest_forecast_run(_eid, "ecmwf")
                                    _gr = get_latest_forecast_run(_eid, "gfs")
                                    if _er and _er.get("forecast_mu_c") is not None:
                                        _em = float(_er["forecast_mu_c"])
                                        if ev_unit == "fahrenheit":
                                            _em = _em * 9 / 5 + 32
                                        st.markdown(f"**ECMWF:** {_em:.1f}{sym}")
                                    if _gr and _gr.get("forecast_mu_c") is not None:
                                        _gm = float(_gr["forecast_mu_c"])
                                        if ev_unit == "fahrenheit":
                                            _gm = _gm * 9 / 5 + 32
                                        st.markdown(f"**GFS:** {_gm:.1f}{sym}")
                            except Exception:
                                pass
                            if not _is_missing(mu_disp):
                                st.markdown(
                                    f"**Blended Avg:** {_fmt_temp(mu_disp, ev_unit)}",
                                    help="Weighted blend of ECMWF (65%) + GFS (35%) + bias correction."
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
# Position helpers (shared by Paper Trade Data and Live Trade Data tabs)
# ===========================================================================

@st.cache_data(ttl=_CACHE_TTL_SHORT)
def _load_latest_model_probs() -> dict[str, float]:
    """Latest model probability per contract_id from the most recent scan.
    Cached so repeated dashboard renders within the TTL window don't
    re-hit the DB."""
    out: dict[str, float] = {}
    try:
        c = _connect_db()
        ts = c.execute("SELECT MAX(scan_timestamp) FROM temp_outcomes").fetchone()[0]
        if ts:
            for r in c.execute(
                "SELECT contract_id, model_prob FROM temp_outcomes "
                "WHERE scan_timestamp = ? AND model_prob IS NOT NULL",
                (ts,),
            ).fetchall():
                out[r[0]] = float(r[1])
        c.close()
    except Exception:
        pass
    return out


@st.cache_data(ttl=_CACHE_TTL_SHORT)
def _load_latest_volumes() -> dict[str, float]:
    """Latest volume_usd per contract_id, scoped to the most recent scan
    only.  The previous version did a full table scan of temp_outcomes
    (no scan_timestamp filter, no LIMIT) and pulled every historical row
    into Python — the slowest single query in the dashboard render path
    once temp_outcomes accumulated tens of thousands of rows."""
    out: dict[str, float] = {}
    try:
        c = _connect_db()
        ts = c.execute("SELECT MAX(scan_timestamp) FROM temp_outcomes").fetchone()[0]
        if ts:
            for r in c.execute(
                "SELECT contract_id, volume_usd FROM temp_outcomes "
                "WHERE scan_timestamp = ? AND volume_usd IS NOT NULL",
                (ts,),
            ).fetchall():
                out[r[0]] = float(r[1])
        c.close()
    except Exception:
        pass
    return out


@st.cache_data(ttl=_CACHE_TTL_SHORT)
def _load_ledger_aggregates(pos_ids_tuple: tuple) -> dict[int, tuple[float, float]]:
    """Per-position (committed_usdc, filled_usdc) summed from the
    position_orders ledger.  Mirrors db.get_committed_usdc semantics:
    partial fills count as still-committing (intended_usdc), not filled-only.

    Cached.  Argument is a tuple (hashable) of position ids; the cache
    key changes only when the open-position set changes — within a single
    refresh window, called multiple times for free.
    """
    if not pos_ids_tuple:
        return {}
    out: dict[int, tuple[float, float]] = {}
    try:
        c = _connect_db()
        ph = ",".join(["?"] * len(pos_ids_tuple))
        for pid, committed, filled in c.execute(f"""
            SELECT position_id,
                SUM(CASE
                    WHEN status = 'filled' AND filled_usdc >= intended_usdc * 0.99
                        THEN filled_usdc
                    WHEN status = 'filled' AND filled_usdc < intended_usdc * 0.99
                        THEN intended_usdc
                    WHEN status IN ('pending','live','matched','partial')
                        THEN intended_usdc
                    ELSE 0
                END) AS committed,
                SUM(filled_usdc) AS filled
            FROM position_orders
            WHERE position_id IN ({ph})
              AND role IN ('entry', 'topup')
            GROUP BY position_id
        """, list(pos_ids_tuple)).fetchall():
            out[int(pid)] = (float(committed or 0), float(filled or 0))
        c.close()
    except Exception:
        pass
    return out


def _build_open_rows(df):
    # All three lookups are cached — first render in a refresh window pays
    # the DB cost; subsequent renders within _CACHE_TTL_SHORT are free.
    _current_model_probs = _load_latest_model_probs()
    _volumes = _load_latest_volumes()
    _ledger_aggs: dict[int, tuple[float, float]] = {}
    if not df.empty:
        try:
            _pos_ids = tuple(sorted(int(pid) for pid in df["id"].dropna().tolist()))
            if _pos_ids:
                _ledger_aggs = _load_ledger_aggregates(_pos_ids)
        except Exception:
            pass
    _ledger_committed = {pid: c for pid, (c, _f) in _ledger_aggs.items()}
    _ledger_filled    = {pid: f for pid, (_c, f) in _ledger_aggs.items()}

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

        _cid = p.get("contract_id")
        _curr_model = _current_model_probs.get(_cid)
        _curr_market = float(curr) if curr else None
        _vol = _volumes.get(_cid)
        _pid = int(p.get("id")) if p.get("id") is not None else None
        _committed = _ledger_committed.get(_pid)
        _filled_lg = _ledger_filled.get(_pid)
        _target    = float(p.get("target_size_usdc") or 0)

        # Lifecycle (Phase 9): on-chain confirmation stage of the entry
        # trade.  Healthy live positions show 'confirmed'; legacy/paper
        # rows pre-dating Phase 9 show NULL → blank.  Anything stuck on
        # 'matched' or 'mined' here means the on-chain side hasn't
        # finalized — worth investigating.
        _trade_status = _safe_str(p.get("trade_status")).lower()
        _is_paper_row = bool(p.get("is_paper", 0))
        if _is_paper_row:
            _lifecycle = "—"  # paper trades have no on-chain lifecycle
        elif not _trade_status:
            _lifecycle = ""   # legacy or pre-Phase-9
        else:
            _lifecycle = _trade_status.upper()

        # Filled / Committed / Target — surfaces partial-fill state +
        # in-flight resting orders so the operator can see at a glance
        # when a position has unfilled orders pulling more capital.
        # Falls back to the legacy size_usdc when ledger data isn't
        # available (e.g., pre-Phase-B rows that didn't get backfilled).
        _legacy_size = float(p.get("size_usdc") or 0)
        _f = _filled_lg if _filled_lg is not None else _legacy_size
        _c = _committed if _committed is not None else _legacy_size
        rows.append({
            "Strategy":          p.get("strategy") or "top_bin_value",
            "City":              p.get("city", ""),
            "Date":              _fmt_date_mmddyyyy(p.get("date")),
            "Local Time":        _fmt_local_time(p.get("local_time")),
            "Side":              p.get("side", ""),
            "Range":             _range,
            "Filled":            f"${_f:.2f}",
            "Committed":         f"${_c:.2f}",
            "Target":            f"${_target:.2f}" if _target > 0 else "",
            "Shares":            f"{shares:.2f}" if shares else "",
            "Entered":           _fmt_entered(p.get("entry_time")),
            "Lifecycle":         _lifecycle,
            "Entry Price":       f"{entry:.4f}" if entry else "",
            "Current Price":     f"{curr:.4f}" if curr else "",
            "Peak Price":        f"{float(p.get('peak_price')):.4f}" if not _is_missing(p.get("peak_price")) else "",
            "SL Price":          f"{float(p.get('stop_loss_price')):.4f}" if not _is_missing(p.get("stop_loss_price")) else "",
            "Unrealized P&L":    f"${unreal:+.4f}" if unreal is not None else "",
            "Entry Market Prob": _fmt_pct(p.get("market_prob")),
            "Curr Market Prob":  _fmt_pct(_curr_market),
            "Volume":            f"${_vol:,.0f}" if _vol else "",
        })
    return rows


def _color_lifecycle(val):
    """Color the Lifecycle column: confirmed=green, mined/matched=amber,
    failed/retrying=red, blank/dash=neutral."""
    if not isinstance(val, str):
        return ""
    v = val.upper().strip()
    if v == "CONFIRMED":
        return "color: #2ca02c"
    if v in ("MATCHED", "MINED"):
        return "color: #ff9900"
    if v in ("FAILED", "RETRYING"):
        return "color: #d62728"
    return ""


# ---------------------------------------------------------------------------
# Live-tab system health strip (Dashboard #3)
# ---------------------------------------------------------------------------

def _render_live_health_strip() -> None:
    """Render the live-mode health strip at the top of the Live Trade tab.

    Reads the most recent monitor_health row.  Surfaces:
      * WS connection state (🟢 connected / 🔴 down / paper)
      * Wallet balance vs effective bankroll cap
      * On-chain reconciliation drift counts
      * Time since last monitor cycle

    No-op (renders nothing) when monitor_health table is empty (fresh
    install before first monitor cycle).
    """
    try:
        conn = _connect_db()
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM monitor_health ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
    except Exception:
        return
    if row is None:
        st.caption("System health: waiting for first monitor cycle…")
        return

    h = dict(row)

    # --- WS status ---
    ws_state = h.get("ws_running")
    if ws_state is None:
        ws_label = "📝 PAPER MODE"
        ws_color = "#888"
    elif ws_state == 1:
        ws_label = "🟢 CONNECTED"
        ws_color = "#2ca02c"
    else:
        ws_label = "🔴 DISCONNECTED"
        ws_color = "#d62728"

    # --- Wallet vs bankroll ---
    wb = h.get("wallet_balance_usdc")
    eb = h.get("effective_bankroll_usdc")
    if wb is not None and eb is not None:
        wallet_value = f"${wb:,.2f}"
        # Bankroll-bound (wallet capped) vs config-bound — eb is the
        # smaller, so equality means wallet has plenty.
        bound_marker = "⚠️ wallet-capped" if (wb - eb) < 5.0 else ""
        bankroll_value = f"${eb:,.2f} {bound_marker}".strip()
    else:
        wallet_value = "—"
        bankroll_value = "—"

    # --- Drift ---
    drift_total = (
        int(h.get("drift_orphan_db") or 0)
        + int(h.get("drift_share_drift") or 0)
        + int(h.get("drift_orphan_chain") or 0)
    )
    drift_value = f"{drift_total}" if drift_total > 0 else "0"
    drift_help = (
        f"orphan_db={h.get('drift_orphan_db') or 0} | "
        f"share_drift={h.get('drift_share_drift') or 0} | "
        f"orphan_chain={h.get('drift_orphan_chain') or 0}"
    )

    # --- Last monitor cycle ---
    last_at = h.get("recorded_at") or ""
    try:
        # Parse ISO; compute minutes ago.  Fall back to raw string.
        dt = datetime.fromisoformat(last_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        mins_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
        if mins_ago < 60:
            cycle_value = f"{mins_ago:.0f} min ago"
        else:
            cycle_value = f"{mins_ago/60:.1f} hr ago"
        # Monitor runs every 30 min; if last cycle > 90 min ago, flag
        cycle_stale = mins_ago > 90
    except Exception:
        cycle_value = last_at[:16] or "—"
        cycle_stale = False

    # --- Render the strip ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div style='font-size:0.85em;color:#888'>WebSocket</div>"
            f"<div style='font-weight:600;color:{ws_color}'>{ws_label}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='font-size:0.85em;color:#888'>Wallet / Bankroll</div>"
            f"<div style='font-weight:600'>{wallet_value} → {bankroll_value}</div>",
            unsafe_allow_html=True,
        )
    with c3:
        drift_color = "#d62728" if drift_total > 0 else "#888"
        st.markdown(
            f"<div style='font-size:0.85em;color:#888'>On-chain Drift</div>"
            f"<div style='font-weight:600;color:{drift_color}' title='{drift_help}'>"
            f"{drift_value}</div>",
            unsafe_allow_html=True,
        )
    with c4:
        cycle_color = "#d62728" if cycle_stale else "#888"
        st.markdown(
            f"<div style='font-size:0.85em;color:#888'>Last Monitor Cycle</div>"
            f"<div style='font-weight:600;color:{cycle_color}'>{cycle_value}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# In-flight orders section (Dashboard #2)
# ---------------------------------------------------------------------------

def _render_in_flight_orders(positions_df) -> None:
    """Surface orders that are currently in flight — pending buys,
    exiting positions, in-flight top-ups.  These are invisible in the
    Open / Closed tables today, which means a stuck order is invisible
    until it eventually resolves.

    Renders three small tables.  If all three are empty, the section
    collapses to a single one-line caption.
    """
    if positions_df is None or positions_df.empty:
        st.caption("In-flight orders: none")
        return

    df = positions_df.copy()

    # --- Pending BUYS (limit orders awaiting fill) ---
    pending_buys = df[
        (df.get("fill_status") == "pending")
        & (df.get("status") == "open")
    ] if "fill_status" in df.columns else pd.DataFrame()

    # --- Exiting positions (sell ladder in progress) ---
    exiting = df[df.get("status") == "exiting"] if "status" in df.columns else pd.DataFrame()

    # --- In-flight top-ups (parent position has pending_topup_order_id) ---
    if "pending_topup_order_id" in df.columns:
        topups = df[df["pending_topup_order_id"].notna()
                    & (df["pending_topup_order_id"] != "")]
    else:
        topups = pd.DataFrame()

    if pending_buys.empty and exiting.empty and topups.empty:
        st.caption("In-flight orders: none")
        return

    if not pending_buys.empty:
        st.markdown("**Pending Buys** — limit orders placed but not yet matched")
        rows = []
        for _, p in pending_buys.iterrows():
            ts = _safe_str(p.get("trade_status")).upper() or "—"
            rows.append({
                "City":         p.get("city", ""),
                "Date":         _fmt_date_mmddyyyy(p.get("date")),
                "Side":         p.get("side", ""),
                "Limit Price":  f"{float(p.get('entry_price') or 0):.4f}",
                "Size ($)":     f"${p.get('size_usdc', 0):.2f}",
                "Lifecycle":    ts,
                "Order ID":     _safe_str(p.get("order_id"))[:12],
                "Placed":       _fmt_entered(p.get("entry_time")),
            })
        st.dataframe(
            pd.DataFrame(rows).style.map(_color_lifecycle, subset=["Lifecycle"]),
            width="stretch", hide_index=True,
        )

    if not exiting.empty:
        st.markdown("**Exiting Positions** — sell ladder in progress")
        rows = []
        for _, p in exiting.iterrows():
            ts = _safe_str(p.get("exit_trade_status")).upper() or "—"
            retry = int(p.get("exit_retry_count") or 0)
            rows.append({
                "City":            p.get("city", ""),
                "Date":            _fmt_date_mmddyyyy(p.get("date")),
                "Side":            p.get("side", ""),
                "Entry":           f"{float(p.get('entry_price') or 0):.4f}",
                "Intended Exit":   f"{float(p.get('exit_intended_price') or 0):.4f}",
                "Reason":          _safe_str(p.get("exit_reason")),
                "Rung":             retry,
                "Lifecycle":       ts,
                "Order ID":        _safe_str(p.get("exit_order_id"))[:12],
            })
        st.dataframe(
            pd.DataFrame(rows).style.map(_color_lifecycle, subset=["Lifecycle"]),
            width="stretch", hide_index=True,
        )

    if not topups.empty:
        st.markdown("**In-Flight Top-ups** — adds to existing positions awaiting fill")
        rows = []
        for _, p in topups.iterrows():
            rows.append({
                "City":            p.get("city", ""),
                "Date":            _fmt_date_mmddyyyy(p.get("date")),
                "Side":            p.get("side", ""),
                "Add ($)":         f"${float(p.get('pending_topup_amount_usdc') or 0):.2f}",
                "Limit Price":     f"{float(p.get('pending_topup_intended_price') or 0):.4f}",
                "Parent Size":     f"${p.get('size_usdc', 0):.2f}",
                "Order ID":        _safe_str(p.get("pending_topup_order_id"))[:12],
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


@st.cache_data(ttl=_CACHE_TTL_SHORT)
def _load_order_ledger_rows(pos_ids_tuple: tuple) -> list[dict]:
    """Cached fetch of position_orders rows for a fixed set of positions.
    The tuple arg is hashable so Streamlit can key on it; same set of
    open positions across renders means a free read."""
    if not pos_ids_tuple:
        return []
    try:
        conn = _connect_db()
        conn.row_factory = sqlite3.Row
        ph = ",".join(["?"] * len(pos_ids_tuple))
        rows = [dict(r) for r in conn.execute(f"""
            SELECT po.position_id, po.role, po.intended_usdc, po.filled_usdc,
                   po.intended_shares, po.filled_shares, po.limit_price,
                   po.fill_price, po.status, po.trade_status, po.fee_usdc,
                   po.created_at, po.cancelled_reason, po.order_id,
                   p.city, p.date
            FROM position_orders po
            JOIN positions p ON po.position_id = p.id
            WHERE po.position_id IN ({ph})
            ORDER BY po.position_id DESC, po.id ASC
        """, list(pos_ids_tuple)).fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _render_order_ledger(positions_df) -> None:
    """Render the per-position order ledger (Phase B, 2026-04-30).

    Shows EVERY individual CLOB order placed for every open live
    position — entry buys, top-ups, and exits.  Each row includes
    intended size + filled size + status, so the operator can see at
    a glance which orders are partial / resting / fully filled.

    This is the truth-of-state view: the legacy `positions.size_usdc`
    aggregates can be misleading on partial fills (the user-reported
    bug from 2026-04-30 where Busan/Jeddah/Lagos showed $10 size but
    really had $19+ committed across an entry + a top-up).
    """
    if positions_df is None or positions_df.empty:
        return
    try:
        pids = tuple(sorted(int(pid) for pid in positions_df["id"].dropna().tolist()))
    except Exception:
        pids = ()
    if not pids:
        return

    rows_raw = _load_order_ledger_rows(pids)

    if not rows_raw:
        return

    feed_rows = []
    for r in rows_raw:
        intended = float(r.get("intended_usdc") or 0)
        filled   = float(r.get("filled_usdc") or 0)
        resting  = max(0.0, intended - filled) if r.get("status") in (
            "pending", "live", "matched", "partial"
        ) else 0.0
        feed_rows.append({
            "Pos":      r["position_id"],
            "Market":   f"{r.get('city', '')[:12]} {r.get('date', '')}",
            "Role":     r.get("role", "").upper(),
            "Intended": f"${intended:.2f}",
            "Filled":   f"${filled:.2f}",
            "Resting":  f"${resting:.2f}" if resting > 0 else "",
            "Status":   r.get("status", "").upper(),
            "Lifecycle": _safe_str(r.get("trade_status")).upper() or "—",
            "Limit":    f"{float(r.get('limit_price') or 0):.4f}",
            "Fill":     f"{float(r.get('fill_price') or 0):.4f}" if r.get("fill_price") else "",
            "Fee":      f"${float(r.get('fee_usdc') or 0):.4f}" if r.get("fee_usdc") else "",
            "Order ID": _safe_str(r.get("order_id"))[:14],
        })

    def _color_status(val):
        if not isinstance(val, str):
            return ""
        v = val.upper()
        if v == "FILLED":           return "color: #2ca02c"
        if v == "PARTIAL":          return "color: #ff9900; font-weight:600"
        if v in ("CANCELLED", "FAILED"): return "color: #888"
        if v in ("PENDING", "LIVE", "MATCHED"): return "color: #1f77b4"
        return ""

    st.subheader("Order Ledger — every CLOB order per open position")
    st.caption(
        "Tracks each individual order placed for the open positions.  "
        "Multiple orders can sum toward one position's target (e.g. an "
        "entry + a top-up).  'Resting' = unfilled portion still on the "
        "book.  Status colors: 🟢 filled · 🟠 partial · 🔵 pending/live · ⬜ cancelled/failed."
    )
    df = pd.DataFrame(feed_rows)
    st.dataframe(
        df.style.map(_color_status, subset=["Status"]),
        width="stretch", hide_index=True,
        height=min(80 + len(df) * 32, 500),
    )


def _build_closed_rows(df):
    rows = []
    for _, p in df.iterrows():
        _unit  = p.get("unit", "celsius")
        _range = _range_label(p.get("range_low"), p.get("range_high"), _unit) if (
            not _is_missing(p.get("range_low")) or not _is_missing(p.get("range_high"))
        ) else (p.get("question") or "")[:30]
        # Show fee total only when something was captured (live trades).
        # Paper trades + legacy rows have NULL/0 here and we suppress.
        _entry_fees = float(p.get("entry_fees") or 0)
        _exit_fees  = float(p.get("exit_fees") or 0)
        _total_fees = _entry_fees + _exit_fees
        _gross_pnl  = p.get("pnl")
        _net_pnl    = p.get("pnl_net")
        rows.append({
            "Strategy":      p.get("strategy") or "top_bin_value",
            "City":          p.get("city", ""),
            "Date":          _fmt_date_mmddyyyy(p.get("date")),
            "Side":          p.get("side", ""),
            "Range":         _range,
            "Size ($)":      f"${p.get('size_usdc', 0):.2f}",
            "Gross P&L":     f"${_gross_pnl:+.4f}" if _gross_pnl is not None else "",
            # Show fees explicitly for any position that actually traded
            # (gross_pnl != None), even when fees are $0.0000 — explicit
            # zero is more honest than blank for closed positions where
            # we've verified the fees are genuinely zero.  Cancelled-before-
            # fill positions (gross_pnl is None) still show blank.
            "Fees":          f"${_total_fees:.4f}" if _gross_pnl is not None else "",
            "Net P&L":       (
                f"${_net_pnl:+.4f}" if _net_pnl is not None
                else (f"${_gross_pnl:+.4f}" if _gross_pnl is not None else "")
            ),
            "Entry":         f"{p.get('entry_price', 0):.4f}",
            "Exit":          f"{p.get('exit_price', 0):.4f}" if p.get("exit_price") is not None else "",
            "Entered":       _fmt_entered(p.get("entry_time")),
            "Closed":        _fmt_entered(p.get("exit_time")),
            "Exit Reason":   p.get("exit_reason") or "",
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


def _compute_trade_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the summary stats table with columns:
    Metric | Open Yes | Open No | Closed Yes | Closed No | Total Open | Total Closed | Grand Total

    Open columns use unrealized_pnl; closed columns use realized pnl.
    Win/loss breakdown rows show '—' for open columns (unresolved trades).
    """
    _METRICS = [
        "Trade Count",
        "Win Count",
        "Loss Count",
        "Win Rate %",
        "Avg P&L $",
        "Avg Return %",
        "Avg Win $",
        "Avg Win %",
        "Avg Loss $",
        "Avg Loss %",
        "Win Total $",
        "Loss Total $",
        "Total P&L $",
    ]
    _COL_NAMES = [
        "Open Yes", "Open No", "Closed Yes", "Closed No",
        "Total Open", "Total Closed", "Grand Total",
    ]

    if df.empty:
        return pd.DataFrame(
            [{"Metric": m, **{c: "—" for c in _COL_NAMES}} for m in _METRICS]
        )

    _open = df[df["status"] == "open"].copy()
    if "fill_status" in _open.columns:
        _open = _open[_open["fill_status"] != "cancelled"]

    _closed = df[df["status"] == "closed"].copy()
    if "fill_status" in _closed.columns:
        _closed = _closed[_closed["fill_status"] != "cancelled"]

    def _pnl_fmt(v):
        if pd.isna(v):
            return "—"
        return f"${v:,.2f}" if v >= 0 else f"-${abs(v):,.2f}"

    def _open_col(sub: pd.DataFrame) -> list:
        """Stats for an open subset — P&L is unrealized. Win/loss based on current return."""
        n = len(sub)
        if n == 0:
            return ["0"] + ["—"] * 12

        unreal = sub["unrealized_pnl"].fillna(0) if "unrealized_pnl" in sub.columns else pd.Series([0.0] * n)
        size   = sub["size_usdc"].replace(0, float("nan")) if "size_usdc" in sub.columns else pd.Series([float("nan")] * n)

        avg_pnl = unreal.mean()
        avg_ret = (unreal / size * 100).mean()

        wins   = sub[unreal > 0]
        losses = sub[unreal < 0]
        win_unreal   = unreal[unreal > 0]
        loss_unreal  = unreal[unreal < 0]
        win_size     = size[unreal > 0]
        loss_size    = size[unreal < 0]

        win_rate = f"{len(wins) / n * 100:.1f}%"

        avg_win_p = (win_unreal / win_size * 100).mean()
        avg_win_p_fmt = f"+{avg_win_p:.1f}%" if not pd.isna(avg_win_p) else "—"

        avg_loss_p = (loss_unreal / loss_size * 100).mean()
        avg_loss_p_fmt = f"{avg_loss_p:.1f}%" if not pd.isna(avg_loss_p) else "—"

        return [
            str(n),                                                        # Trade Count
            str(len(wins)),                                                # Win Count
            str(len(losses)),                                              # Loss Count
            win_rate,                                                      # Win Rate %
            _pnl_fmt(avg_pnl),                                            # Avg P&L $
            f"{avg_ret:+.1f}%" if not pd.isna(avg_ret) else "—",         # Avg Return %
            _pnl_fmt(win_unreal.mean())  if not wins.empty   else "—",    # Avg Win $
            avg_win_p_fmt,                                                 # Avg Win %
            _pnl_fmt(loss_unreal.mean()) if not losses.empty else "—",    # Avg Loss $
            avg_loss_p_fmt,                                                # Avg Loss %
            _pnl_fmt(win_unreal.sum())   if not wins.empty   else "—",    # Win Total $
            _pnl_fmt(loss_unreal.sum())  if not losses.empty else "—",    # Loss Total $
            _pnl_fmt(unreal.sum()),                                        # Total P&L $
        ]

    def _closed_col(sub: pd.DataFrame) -> list:
        """Stats for a closed subset — P&L is realized."""
        n = len(sub)
        if n == 0:
            return ["0"] + ["—"] * 12

        pnl  = sub["pnl"].fillna(0)
        size = sub["size_usdc"].replace(0, float("nan")) if "size_usdc" in sub.columns else pd.Series([float("nan")] * n)
        wins   = sub[sub["pnl"] > 0]
        losses = sub[sub["pnl"] < 0]

        win_rate = f"{len(wins) / n * 100:.1f}%"
        avg_pnl  = pnl.mean()
        avg_ret  = (pnl / size * 100).mean()

        if not wins.empty and "size_usdc" in wins.columns:
            avg_win_p = (wins["pnl"] / wins["size_usdc"].replace(0, float("nan")) * 100).mean()
            avg_win_p_fmt = f"+{avg_win_p:.1f}%" if not pd.isna(avg_win_p) else "—"
        else:
            avg_win_p_fmt = "—"

        if not losses.empty and "size_usdc" in losses.columns:
            avg_loss_p = (losses["pnl"] / losses["size_usdc"].replace(0, float("nan")) * 100).mean()
            avg_loss_p_fmt = f"{avg_loss_p:.1f}%" if not pd.isna(avg_loss_p) else "—"
        else:
            avg_loss_p_fmt = "—"

        return [
            str(n),                                                              # Trade Count
            str(len(wins)),                                                      # Win Count
            str(len(losses)),                                                    # Loss Count
            win_rate,                                                            # Win Rate %
            _pnl_fmt(avg_pnl),                                                  # Avg P&L $
            f"{avg_ret:+.1f}%" if not pd.isna(avg_ret) else "—",               # Avg Return %
            _pnl_fmt(wins["pnl"].mean())   if not wins.empty   else "—",        # Avg Win $
            avg_win_p_fmt,                                                       # Avg Win %
            _pnl_fmt(losses["pnl"].mean()) if not losses.empty else "—",        # Avg Loss $
            avg_loss_p_fmt,                                                      # Avg Loss %
            _pnl_fmt(wins["pnl"].sum())    if not wins.empty   else "—",        # Win Total $
            _pnl_fmt(losses["pnl"].sum())  if not losses.empty else "—",        # Loss Total $
            _pnl_fmt(pnl.sum()),                                                 # Total P&L $
        ]

    def _grand_total_col() -> list:
        """Combines open (unrealized) + closed (realized) for grand totals."""
        n = len(_open) + len(_closed)
        if n == 0:
            return ["0"] + ["—"] * 12

        # Combined P&L series (unrealized for open, realized for closed)
        real   = _closed["pnl"].fillna(0)
        unreal = _open["unrealized_pnl"].fillna(0) if "unrealized_pnl" in _open.columns else pd.Series([0.0] * len(_open))
        n_cl   = len(_closed)
        all_pnl  = pd.concat([real, unreal], ignore_index=True)
        all_size = pd.concat([
            _closed["size_usdc"] if "size_usdc" in _closed.columns else pd.Series([float("nan")] * n_cl),
            _open["size_usdc"]   if "size_usdc" in _open.columns   else pd.Series([float("nan")] * len(_open)),
        ], ignore_index=True).replace(0, float("nan"))

        avg_pnl = all_pnl.mean()
        avg_ret = (all_pnl / all_size * 100).mean()

        # Win/loss across all: closed uses pnl, open uses unrealized_pnl
        wins_mask   = all_pnl > 0
        losses_mask = all_pnl < 0
        win_rate    = f"{wins_mask.sum() / n * 100:.1f}%"

        win_pnl  = all_pnl[wins_mask]
        loss_pnl = all_pnl[losses_mask]
        win_size  = all_size[wins_mask]
        loss_size = all_size[losses_mask]

        avg_win_p = (win_pnl / win_size * 100).mean()
        avg_win_p_fmt = f"+{avg_win_p:.1f}%" if not pd.isna(avg_win_p) else "—"

        avg_loss_p = (loss_pnl / loss_size * 100).mean()
        avg_loss_p_fmt = f"{avg_loss_p:.1f}%" if not pd.isna(avg_loss_p) else "—"

        return [
            str(n),                                                             # Trade Count
            str(int(wins_mask.sum())),                                          # Win Count
            str(int(losses_mask.sum())),                                        # Loss Count
            win_rate,                                                           # Win Rate %
            _pnl_fmt(avg_pnl),                                                 # Avg P&L $
            f"{avg_ret:+.1f}%" if not pd.isna(avg_ret) else "—",              # Avg Return %
            _pnl_fmt(win_pnl.mean())  if not win_pnl.empty  else "—",         # Avg Win $
            avg_win_p_fmt,                                                      # Avg Win %
            _pnl_fmt(loss_pnl.mean()) if not loss_pnl.empty else "—",         # Avg Loss $
            avg_loss_p_fmt,                                                     # Avg Loss %
            _pnl_fmt(win_pnl.sum())   if not win_pnl.empty  else "—",         # Win Total $
            _pnl_fmt(loss_pnl.sum())  if not loss_pnl.empty else "—",         # Loss Total $
            _pnl_fmt(all_pnl.sum()),                                           # Total P&L $
        ]

    cols_data = [
        _open_col(_open[_open["side"] == "YES"]),
        _open_col(_open[_open["side"] == "NO"]),
        _closed_col(_closed[_closed["side"] == "YES"]),
        _closed_col(_closed[_closed["side"] == "NO"]),
        _open_col(_open),
        _closed_col(_closed),
        _grand_total_col(),
    ]

    rows = []
    for i, metric in enumerate(_METRICS):
        row = {"Metric": metric}
        for col_name, col_vals in zip(_COL_NAMES, cols_data):
            row[col_name] = col_vals[i]
        rows.append(row)
    return pd.DataFrame(rows)


def _sigma_bucket(sigma) -> str:
    """Assign a human-readable uncertainty tier to a forecast_sigma_c value (°C)."""
    if _is_missing(sigma):
        return "Unknown"
    s = float(sigma)
    if s < 1.5:
        return "Low (<1.5°C)"
    if s < 2.5:
        return "Medium (1.5–2.5°C)"
    if s < 4.0:
        return "High (2.5–4.0°C)"
    return "Very High (≥4.0°C)"


_SIGMA_BUCKET_ORDER = [
    "Low (<1.5°C)",
    "Medium (1.5–2.5°C)",
    "High (2.5–4.0°C)",
    "Very High (≥4.0°C)",
    "Unknown",
]


def _render_pnl_by_date(df: pd.DataFrame) -> None:
    """Bar chart of realized + unrealized P&L grouped by contract date."""
    if df.empty:
        st.info("No trade data yet.")
        return

    import plotly.graph_objects as go

    # Use 'date' (contract date) for grouping; fall back to exit_time date for closed
    _df = df.copy()
    if "date" not in _df.columns or _df["date"].isna().all():
        st.info("No date data available for P&L breakdown.")
        return

    # Compute P&L per position: use pnl for closed, unrealized_pnl for open
    def _get_pnl(row):
        if row.get("status") == "closed" and pd.notna(row.get("pnl")):
            return float(row["pnl"])
        if pd.notna(row.get("unrealized_pnl")):
            return float(row["unrealized_pnl"])
        return 0.0

    _df["_pnl"] = _df.apply(_get_pnl, axis=1)
    _grouped = _df.groupby("date")["_pnl"].sum().sort_index()

    if _grouped.empty:
        st.info("No P&L data to display.")
        return

    _colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in _grouped.values]
    _cumulative = _grouped.cumsum()

    _col_bar, _col_line = st.columns(2)

    with _col_bar:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=_grouped.index,
            y=_grouped.values,
            marker_color=_colors,
            text=[f"${v:+,.0f}" for v in _grouped.values],
            textposition="outside",
        ))
        fig.update_layout(
            title="Daily P&L",
            xaxis_title="Contract Date",
            yaxis_title="P&L ($)",
            height=350,
            margin=dict(t=40, b=40),
        )
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
        st.plotly_chart(fig, width="stretch")

    with _col_line:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=_cumulative.index,
            y=_cumulative.values,
            mode="lines+markers",
            name="Cumulative P&L",
            line=dict(color="#3498db", width=3),
            marker=dict(size=6),
            text=[f"${v:+,.0f}" for v in _cumulative.values],
            textposition="top center",
        ))
        fig2.update_layout(
            title="Cumulative P&L",
            xaxis_title="Contract Date",
            yaxis_title="Cumulative P&L ($)",
            height=350,
            margin=dict(t=40, b=40),
        )
        fig2.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
        st.plotly_chart(fig2, width="stretch")

    # Summary line
    _total = _grouped.sum()
    _winning_dates = (_grouped > 0).sum()
    _losing_dates = (_grouped < 0).sum()
    st.caption(
        f"Total: ${_total:+,.2f} | "
        f"Winning dates: {_winning_dates} | "
        f"Losing dates: {_losing_dates}"
    )


def _render_sigma_breakdown(df: pd.DataFrame) -> None:
    """
    Render a return-by-uncertainty-tier breakdown table and bar chart.
    Uses realized P&L for closed positions and unrealized for open.
    """
    if df.empty:
        st.info("No position data available for uncertainty breakdown.")
        return

    if "forecast_sigma_c" not in df.columns or df["forecast_sigma_c"].isna().all():
        st.info("No uncertainty data recorded yet. Run the monitor loop to backfill existing positions.")
        return

    df = df.copy()
    df["_bucket"] = df["forecast_sigma_c"].apply(_sigma_bucket)

    # Derive a unified P&L column: realized for closed, unrealized for open
    def _effective_pnl(row):
        if row.get("status") == "closed":
            return row.get("pnl") or 0.0
        return row.get("unrealized_pnl") or 0.0

    df["_eff_pnl"]  = df.apply(_effective_pnl, axis=1)
    df["_eff_ret"]  = df["_eff_pnl"] / df["size_usdc"].replace(0, float("nan")) * 100

    # Filter to non-cancelled
    if "fill_status" in df.columns:
        df = df[df["fill_status"] != "cancelled"]

    rows = []
    for bucket in _SIGMA_BUCKET_ORDER:
        sub = df[df["_bucket"] == bucket]
        if sub.empty:
            continue
        closed = sub[sub["status"] == "closed"]
        wins   = closed[closed["pnl"] > 0]
        n      = len(sub)
        n_cl   = len(closed)
        win_rate = f"{len(wins) / n_cl * 100:.1f}%" if n_cl > 0 else "—"
        avg_ret  = sub["_eff_ret"].mean()
        total    = sub["_eff_pnl"].sum()
        rows.append({
            "Uncertainty Tier": bucket,
            "Trades":           n,
            "Closed":           n_cl,
            "Win Rate":         win_rate,
            "Avg Return %":     f"{avg_ret:+.1f}%" if not pd.isna(avg_ret) else "—",
            "Total P&L $":      f"${total:,.2f}" if total >= 0 else f"-${abs(total):,.2f}",
        })

    if not rows:
        st.info("No data to display.")
        return

    sigma_summary = pd.DataFrame(rows)
    st.dataframe(sigma_summary, hide_index=True, width="stretch")

    # Bar chart — avg return % by tier
    chart_rows = [r for r in rows if r["Avg Return %"] not in ("—",)]
    if chart_rows:
        chart_df = pd.DataFrame({
            "Tier":       [r["Uncertainty Tier"] for r in chart_rows],
            "Avg Return": [float(r["Avg Return %"].replace("%", "").replace("+", "")) for r in chart_rows],
        })
        fig = px.bar(
            chart_df, x="Tier", y="Avg Return",
            color="Avg Return",
            color_continuous_scale=["#d62728", "#aec7e8", "#2ca02c"],
            color_continuous_midpoint=0,
            labels={"Tier": "Uncertainty Tier", "Avg Return": "Avg Return (%)"},
            title="Avg Return % by Forecast Uncertainty Tier",
        )
        fig.update_layout(
            yaxis_ticksuffix="%",
            coloraxis_showscale=False,
            height=320,
            margin=dict(t=40, b=40),
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig, width="stretch")


# (moved above tab blocks so it can be called from any tab)

    st.divider()
    _cc1, _cc2, _cc3 = st.columns([2, 1, 2])
    with _cc2:
        if st.button("Refresh Data", key="refresh_contract", type="primary"):
            _do_light_refresh()


# ===========================================================================
# TAB 2 — Paper Trade Data
# ===========================================================================
with _tab_paper:

    _show_refresh_messages()

    # Strategy filter (applied to ALL content in this tab)
    _paper_all_raw = (
        positions_df[positions_df["is_paper"] == 1].copy()
        if not positions_df.empty and "is_paper" in positions_df.columns
        else pd.DataFrame()
    )
    _strat_options = ["All"]
    if not _paper_all_raw.empty and "strategy" in _paper_all_raw.columns:
        _strat_options += sorted(_paper_all_raw["strategy"].dropna().unique().tolist())
    _sel_strategy = st.selectbox("Strategy", _strat_options, key="paper_strategy_filter")

    if _sel_strategy != "All" and not _paper_all_raw.empty and "strategy" in _paper_all_raw.columns:
        _paper_all = _paper_all_raw[_paper_all_raw["strategy"] == _sel_strategy].copy()
    else:
        _paper_all = _paper_all_raw

    _paper_open = _paper_all[_paper_all["status"] == "open"].copy() if not _paper_all.empty else pd.DataFrame()
    if not _paper_open.empty and "fill_status" in _paper_open.columns:
        _paper_open = _paper_open[_paper_open["fill_status"] != "cancelled"]
    _paper_closed = _paper_all[_paper_all["status"] == "closed"].copy() if not _paper_all.empty else pd.DataFrame()
    if not _paper_closed.empty and "fill_status" in _paper_closed.columns:
        _paper_closed = _paper_closed[_paper_closed["fill_status"] != "cancelled"]

    _paper_capital = _paper_open["size_usdc"].sum() if not _paper_open.empty and "size_usdc" in _paper_open.columns else 0

    _paper_all_pnl = []
    if not _paper_open.empty and "unrealized_pnl" in _paper_open.columns:
        _paper_all_pnl.extend(_paper_open["unrealized_pnl"].dropna().tolist())
    if not _paper_closed.empty and "pnl" in _paper_closed.columns:
        _paper_all_pnl.extend(_paper_closed["pnl"].dropna().tolist())
    _p_winning = sum(1 for p in _paper_all_pnl if float(p) > 0)
    _p_losing = sum(1 for p in _paper_all_pnl if float(p) < 0)
    _p_max_profit = max((float(p) for p in _paper_all_pnl), default=0)
    _p_total_pnl = sum(float(p) for p in _paper_all_pnl)

    _p_closed_count = len(_paper_closed) if not _paper_closed.empty else 0
    _pm1, _pm2, _pm3, _pm4, _pm5, _pm6, _pm7 = st.columns(7)
    _pm1.metric("Open Trades", len(_paper_open))
    _pm2.metric("Closed Trades", _p_closed_count)
    _pm3.metric("Capital Deployed", f"${_paper_capital:,.2f}")
    _pm4.metric("Winning Trades", _p_winning)
    _pm5.metric("Losing Trades", _p_losing)
    _pm6.metric("Max Profit", f"${_p_max_profit:,.2f}")
    _pm7.metric("Total P&L", f"${_p_total_pnl:+,.2f}")

    _paper_stats = _compute_trade_stats(_paper_all)
    st.dataframe(
        _paper_stats,
        hide_index=True, width="stretch",
        height=38 + len(_paper_stats) * 35,
    )
    st.divider()

    st.subheader("Return by Date")
    _render_pnl_by_date(_paper_all)
    st.divider()

    # ── Open Positions ────────────────────────────────────────────────────────
    st.subheader("Open Positions")
    if not _paper_open.empty:
        _df = pd.DataFrame(_build_open_rows(_paper_open))
        _df = _df.sort_values(["City", "Date", "Range"], ascending=True, na_position="last")
        st.dataframe(
            _df.style
                .map(_highlight_side, subset=["Side"])
                .map(_color_pnl, subset=["Unrealized P&L"])
                .map(_color_lifecycle, subset=["Lifecycle"]),
            width="stretch", hide_index=True, height=min(80 + len(_df) * 36, 500),
            column_config={"Question": st.column_config.TextColumn("Question", width="large")},
        )
    else:
        st.info("No open paper positions")

    st.divider()

    # ── Closed Positions ──────────────────────────────────────────────────────
    st.subheader("Closed Positions")
    if not _paper_closed.empty:
        _df = pd.DataFrame(_build_closed_rows(_paper_closed))
        _df = _df.sort_values(["City", "Date", "Range"], ascending=True, na_position="last")
        st.dataframe(
            _df.style
                .map(_highlight_side, subset=["Side"])
                .map(_color_pnl, subset=["Gross P&L", "Net P&L"]),
            width="stretch", hide_index=True, height=min(80 + len(_df) * 36, 400),
            column_config={"Question": st.column_config.TextColumn("Question", width="large")},
        )
    else:
        st.info("No closed paper positions yet")

    # ── Cancelled Orders ──────────────────────────────────────────────────────
    if not positions_df.empty and "fill_status" in positions_df.columns:
        _cancelled_paper = positions_df[
            (positions_df["fill_status"] == "cancelled") &
            (positions_df.get("is_paper", pd.Series(dtype=int)) == 1)
        ].copy() if "is_paper" in positions_df.columns else pd.DataFrame()
        if not _cancelled_paper.empty:
            st.divider()
            st.subheader("Cancelled Orders")
            st.caption(f"{len(_cancelled_paper)} order(s) were cancelled before fill")
            _can_rows = [{
                "City": p.get("city", ""), "Date": p.get("date", ""),
                "Side": p.get("side", ""), "Size ($)": f"${p.get('size_usdc', 0):.2f}",
                "Reason": p.get("cancelled_reason", ""),
                "Time": (p.get("exit_time") or "")[:16],
            } for _, p in _cancelled_paper.iterrows()]
            st.dataframe(pd.DataFrame(_can_rows), width="stretch", hide_index=True)

    st.divider()
    _pc1, _pc2, _pc3 = st.columns([2, 1, 2])
    with _pc2:
        if st.button("Refresh Data", key="refresh_paper", type="primary"):
            _do_refresh("refresh_paper")


# ===========================================================================
# TAB 3 — Live Trade Data
# ===========================================================================
with _tab_live:

    _show_refresh_messages()

    # System health strip — WS state, wallet/bankroll, drift, last cycle.
    # Reads from the monitor_health table populated each monitor cycle.
    _render_live_health_strip()
    st.divider()

    # Strategy filter (applied to ALL content in this tab)
    _live_all_raw = (
        positions_df[positions_df["is_paper"] == 0].copy()
        if not positions_df.empty and "is_paper" in positions_df.columns
        else pd.DataFrame()
    )
    _live_strat_options = ["All"]
    if not _live_all_raw.empty and "strategy" in _live_all_raw.columns:
        _live_strat_options += sorted(_live_all_raw["strategy"].dropna().unique().tolist())
    _sel_live_strategy = st.selectbox("Strategy", _live_strat_options, key="live_strategy_filter")

    if _sel_live_strategy != "All" and not _live_all_raw.empty and "strategy" in _live_all_raw.columns:
        _live_all = _live_all_raw[_live_all_raw["strategy"] == _sel_live_strategy].copy()
    else:
        _live_all = _live_all_raw

    _live_open = _live_all[_live_all["status"] == "open"].copy() if not _live_all.empty else pd.DataFrame()
    if not _live_open.empty and "fill_status" in _live_open.columns:
        _live_open = _live_open[_live_open["fill_status"] != "cancelled"]
    _live_closed = _live_all[_live_all["status"] == "closed"].copy() if not _live_all.empty else pd.DataFrame()
    if not _live_closed.empty and "fill_status" in _live_closed.columns:
        _live_closed = _live_closed[_live_closed["fill_status"] != "cancelled"]

    _live_capital = _live_open["size_usdc"].sum() if not _live_open.empty and "size_usdc" in _live_open.columns else 0

    _live_all_pnl = []
    if not _live_open.empty and "unrealized_pnl" in _live_open.columns:
        _live_all_pnl.extend(_live_open["unrealized_pnl"].dropna().tolist())
    if not _live_closed.empty and "pnl" in _live_closed.columns:
        _live_all_pnl.extend(_live_closed["pnl"].dropna().tolist())
    _l_winning = sum(1 for p in _live_all_pnl if float(p) > 0)
    _l_losing = sum(1 for p in _live_all_pnl if float(p) < 0)
    _l_max_profit = max((float(p) for p in _live_all_pnl), default=0)
    _l_total_pnl = sum(float(p) for p in _live_all_pnl)

    _l_closed_count = len(_live_closed) if not _live_closed.empty else 0
    _lm1, _lm2, _lm3, _lm4, _lm5, _lm6, _lm7 = st.columns(7)
    _lm1.metric("Open Trades", len(_live_open))
    _lm2.metric("Closed Trades", _l_closed_count)
    _lm3.metric("Capital Deployed", f"${_live_capital:,.2f}")
    _lm4.metric("Winning Trades", _l_winning)
    _lm5.metric("Losing Trades", _l_losing)
    _lm6.metric("Max Profit", f"${_l_max_profit:,.2f}")
    _lm7.metric("Total P&L", f"${_l_total_pnl:+,.2f}")

    _live_stats = _compute_trade_stats(_live_all)
    st.dataframe(
        _live_stats,
        hide_index=True, width="stretch",
        height=38 + len(_live_stats) * 35,
    )
    st.divider()

    st.subheader("Return by Date")
    _render_pnl_by_date(_live_all)
    st.divider()

    # ── In-flight Orders (pending buys, exiting, top-ups) ────────────────────
    st.subheader("In-Flight Orders")
    _render_in_flight_orders(_live_all)
    st.divider()

    # ── Open Positions ────────────────────────────────────────────────────────
    st.subheader("Open Positions")
    if not _live_open.empty:
        _df = pd.DataFrame(_build_open_rows(_live_open))
        _df = _df.sort_values(["City", "Date", "Range"], ascending=True, na_position="last")
        st.dataframe(
            _df.style
                .map(_highlight_side, subset=["Side"])
                .map(_color_pnl, subset=["Unrealized P&L"])
                .map(_color_lifecycle, subset=["Lifecycle"]),
            width="stretch", hide_index=True, height=min(80 + len(_df) * 36, 500),
            column_config={"Question": st.column_config.TextColumn("Question", width="large")},
        )
        pass
    else:
        st.info("No open live positions")

    st.divider()

    # ── Order Ledger (Phase B) — every CLOB order per open position ──────────
    _render_order_ledger(_live_open)

    st.divider()

    # ── Closed Positions ──────────────────────────────────────────────────────
    st.subheader("Closed Positions")
    if not _live_closed.empty:
        _df = pd.DataFrame(_build_closed_rows(_live_closed))
        _df = _df.sort_values(["City", "Date", "Range"], ascending=True, na_position="last")
        st.dataframe(
            _df.style
                .map(_highlight_side, subset=["Side"])
                .map(_color_pnl, subset=["Gross P&L", "Net P&L"]),
            width="stretch", hide_index=True, height=min(80 + len(_df) * 36, 400),
            column_config={"Question": st.column_config.TextColumn("Question", width="large")},
        )
    else:
        st.info("No closed live positions yet")

    # ── Cancelled Orders ──────────────────────────────────────────────────────
    if not positions_df.empty and "fill_status" in positions_df.columns:
        _cancelled_live = positions_df[
            (positions_df["fill_status"] == "cancelled") &
            (positions_df["is_paper"] == 0)
        ].copy() if "is_paper" in positions_df.columns else pd.DataFrame()
        if not _cancelled_live.empty:
            st.divider()
            st.subheader("Cancelled Orders")
            st.caption(f"{len(_cancelled_live)} order(s) were cancelled before fill")
            _can_rows = [{
                "City": p.get("city", ""), "Date": p.get("date", ""),
                "Side": p.get("side", ""), "Size ($)": f"${p.get('size_usdc', 0):.2f}",
                "Reason": p.get("cancelled_reason", ""),
                "Time": (p.get("exit_time") or "")[:16],
            } for _, p in _cancelled_live.iterrows()]
            st.dataframe(pd.DataFrame(_can_rows), width="stretch", hide_index=True)

    st.divider()
    _lc1, _lc2, _lc3 = st.columns([2, 1, 2])
    with _lc2:
        if st.button("Refresh Data", key="refresh_live", type="primary"):
            _do_refresh("refresh_live")

# ===========================================================================
# TAB 4 — Activity
# ===========================================================================
with _tab_activity:

    _show_refresh_messages()

    st.markdown(
        "Critical bot actions in chronological order — buys, sells, fills, "
        "cancellations, on-chain failures, drift, WebSocket events, system "
        "lifecycle.  Same content as `bot/logs/activity.log` (see droplet "
        "for raw file)."
    )

    # ---- Filters ----
    _act_categories_available = []
    try:
        _act_conn = _connect_db()
        _act_categories_available = [
            r[0] for r in _act_conn.execute(
                "SELECT DISTINCT category FROM activity_log ORDER BY category"
            ).fetchall()
        ]
        _act_conn.close()
    except Exception:
        pass

    _act_f1, _act_f2, _act_f3, _act_f4 = st.columns([2, 2, 1, 1])
    with _act_f1:
        _sel_categories = st.multiselect(
            "Category", _act_categories_available,
            default=_act_categories_available,
            key="activity_filter_category",
        )
    with _act_f2:
        _sel_levels = st.multiselect(
            "Level", ["INFO", "WARN", "ERROR"],
            default=["INFO", "WARN", "ERROR"],
            key="activity_filter_level",
        )
    with _act_f3:
        _act_window = st.selectbox(
            "Window",
            ["Last 1h", "Last 6h", "Last 24h", "Last 7d", "All"],
            index=2,
            key="activity_filter_window",
        )
    with _act_f4:
        _act_limit = st.selectbox(
            "Max rows", [50, 100, 200, 500, 1000],
            index=2,
            key="activity_filter_limit",
        )

    # ---- Window → since_iso ----
    _window_hours = {
        "Last 1h": 1, "Last 6h": 6, "Last 24h": 24,
        "Last 7d": 24 * 7, "All": None,
    }[_act_window]
    if _window_hours is None:
        _since_iso = None
    else:
        _since_iso = (
            datetime.now(timezone.utc).timestamp() - _window_hours * 3600
        )
        _since_iso = datetime.fromtimestamp(_since_iso, tz=timezone.utc).isoformat()

    # ---- Query ----
    _act_rows: list[dict] = []
    try:
        _where_parts = []
        _args: list = []
        if _sel_categories:
            _ph = ",".join(["?"] * len(_sel_categories))
            _where_parts.append(f"category IN ({_ph})")
            _args.extend(_sel_categories)
        if _sel_levels:
            _ph = ",".join(["?"] * len(_sel_levels))
            _where_parts.append(f"level IN ({_ph})")
            _args.extend(_sel_levels)
        if _since_iso:
            _where_parts.append("timestamp >= ?")
            _args.append(_since_iso)
        _where_sql = ("WHERE " + " AND ".join(_where_parts)) if _where_parts else ""
        _act_conn = _connect_db()
        _act_conn.row_factory = sqlite3.Row
        _act_rows = [dict(r) for r in _act_conn.execute(
            f"SELECT * FROM activity_log {_where_sql} "
            f"ORDER BY id DESC LIMIT ?",
            tuple(_args) + (int(_act_limit),),
        ).fetchall()]
        _act_conn.close()
    except Exception as _e:
        st.warning(f"Could not load activity log: {_e}")

    # ---- Per-category counters ----
    _act_counts: dict[str, int] = {}
    for _r in _act_rows:
        _c = _r.get("category", "?")
        _act_counts[_c] = _act_counts.get(_c, 0) + 1
    _err_count = sum(1 for r in _act_rows if r.get("level") == "ERROR")
    _warn_count = sum(1 for r in _act_rows if r.get("level") == "WARN")

    _hm1, _hm2, _hm3, _hm4 = st.columns(4)
    _hm1.metric("Total events", len(_act_rows))
    _hm2.metric("Errors", _err_count)
    _hm3.metric("Warnings", _warn_count)
    _hm4.metric("Unique categories", len(_act_counts))

    if _act_counts:
        _cat_chips = " · ".join(f"**{c}** ({n})" for c, n in sorted(_act_counts.items()))
        st.caption(_cat_chips)

    st.divider()

    # ---- Activity feed table ----
    if not _act_rows:
        st.info("No activity matches the current filters.")
    else:
        def _fmt_ts(ts: str) -> str:
            try:
                _dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if _dt.tzinfo is None:
                    _dt = _dt.replace(tzinfo=timezone.utc)
                return _dt.astimezone(ZoneInfo("America/Chicago")).strftime(
                    "%m-%d %H:%M:%S"
                )
            except Exception:
                return ts[:19]

        _feed_rows = []
        for _r in _act_rows:
            _feed_rows.append({
                "When":     _fmt_ts(_r.get("timestamp") or ""),
                "Level":    _r.get("level", ""),
                "Category": _r.get("category", ""),
                # Cast to string consistently — pyarrow infers int64 from
                # the populated rows and chokes on empty strings for events
                # without a position_id (SYSTEM startup, WS connect, etc.).
                # Stringification keeps the column homogeneous.
                "Pos":      str(_r.get("position_id")) if _r.get("position_id") is not None else "",
                "Message":  _r.get("message", ""),
            })

        def _color_level(val):
            if not isinstance(val, str):
                return ""
            if val == "ERROR":
                return "color: #d62728; font-weight:600"
            if val == "WARN":
                return "color: #ff9900; font-weight:600"
            return "color: #888"

        _feed_df = pd.DataFrame(_feed_rows)
        st.dataframe(
            _feed_df.style.map(_color_level, subset=["Level"]),
            width="stretch", hide_index=True,
            height=min(80 + len(_feed_df) * 32, 700),
            column_config={
                "Message": st.column_config.TextColumn("Message", width="large"),
            },
        )

    st.divider()
    _ac1, _ac2, _ac3 = st.columns([2, 1, 2])
    with _ac2:
        if st.button("Refresh Activity", key="refresh_activity", type="primary"):
            _do_refresh("refresh_activity")

# ===========================================================================
# TAB 5 — Forecast Accuracy
# ===========================================================================
with _tab_accuracy:

    _show_refresh_messages()

    # ---- City Accuracy Ranking Table ----
    st.subheader("City Forecast Accuracy Ranking")
    st.caption(
        "Composite score (0-100) combining MAE, error stability, % within 1-2C, "
        "and tail risk.  Higher = more accurate and tradeable.  Rebuilt daily."
    )

    @st.cache_data(ttl=_CACHE_TTL_SHORT)
    def _load_city_accuracy():
        conn = _connect_db()
        return pd.read_sql(
            "SELECT * FROM city_forecast_accuracy ORDER BY accuracy_score DESC",
            conn,
        )

    _acc_df = _load_city_accuracy()
    if not _acc_df.empty:
        # Get list of cities with active Polymarket contracts
        @st.cache_data(ttl=_CACHE_TTL_SHORT)
        def _get_active_market_cities():
            conn = _connect_db()
            rows = conn.execute(
                "SELECT DISTINCT city FROM forecast_runs "
                "WHERE date >= DATE('now') AND city IS NOT NULL"
            ).fetchall()
            return {r[0].lower() for r in rows}

        _active_cities = _get_active_market_cities()

        # ---- Filter ----
        _only_active = st.checkbox("Active markets only", value=True,
                                   key="acc_active_only")

        # Apply filter
        _filtered = _acc_df.copy()
        if _only_active:
            _filtered = _filtered[_filtered["city"].str.lower().isin(_active_cities)]

        st.caption(f"Showing {len(_filtered)} of {len(_acc_df)} cities")

        # Format for display
        _acc_cols = [
            "city", "accuracy_score", "mae_c", "rmse_c", "bias_c",
            "error_std_c", "max_error_c",
            "pct_within_1c", "pct_within_2c",
            "pct_underpredicted", "pct_overpredicted",
        ]
        _acc_names = [
            "City", "Score", "MAE", "RMSE", "Bias",
            "Error Std", "Max Error",
            "< 1C", "< 2C",
            "% Under", "% Over",
        ]
        if "avg_uncertainty_c" in _filtered.columns:
            _acc_cols.append("avg_uncertainty_c")
            _acc_names.append("Avg Uncertainty")
        _display_acc = _filtered[_acc_cols].copy()
        _display_acc.columns = _acc_names
        for col in ["< 1C", "< 2C", "% Under", "% Over"]:
            _display_acc[col] = (_display_acc[col] * 100).round(0).astype(int).astype(str) + "%"
        _float_cols = ["Score", "MAE", "RMSE", "Bias", "Error Std", "Max Error"]
        if "Avg Uncertainty" in _display_acc.columns:
            _float_cols.append("Avg Uncertainty")
        for col in _float_cols:
            _display_acc[col] = _display_acc[col].apply(
                lambda v: f"{v:.2f}" if pd.notna(v) else "--"
            )

        st.dataframe(
            _display_acc,
            hide_index=True, width="stretch",
            height=min(80 + len(_display_acc) * 36, 500),
        )
    else:
        _only_active = False
        _active_cities = set()
        st.info("No city accuracy data yet. Run the bot for at least one bias update cycle.")

    st.divider()

    # ---- Per-City Forecast vs Actual Chart ----
    st.subheader("Forecast vs Actual High Temperature")
    st.caption(
        "Compares the model's blended forecast (ECMWF + GFS) against the "
        "actual observed daily high from Visual Crossing station data.  "
        "Only shows dates where both a forecast and an observation exist."
    )

    @st.cache_data(ttl=_CACHE_TTL_SHORT)
    def _load_accuracy_data():
        """Load forecast predictions paired with actual observations."""
        conn = _connect_db()
        # Use historical_forecasts_previous_runs (model predictions at lead=3)
        # paired with historical_observed_daily (actual tmax).
        # Also include temp_events forecast_mu for recent dates.
        # Use JOINs instead of correlated subqueries for performance.
        # Pre-filter historical_forecasts to latest lead per (city, date, model)
        # using a CTE, then join once.
        df = pd.read_sql("""
            WITH ecmwf_latest AS (
                SELECT city, date, forecast_tempmax_c,
                       ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY lead_days ASC) rn
                FROM historical_forecasts_previous_runs
                WHERE model = 'ecmwf_ifs025'
                  AND date >= DATE('now', '-30 days')
            ),
            gfs_latest AS (
                SELECT city, date, forecast_tempmax_c,
                       ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY lead_days ASC) rn
                FROM historical_forecasts_previous_runs
                WHERE model = 'gfs_global'
                  AND date >= DATE('now', '-30 days')
            )
            SELECT
                h.city,
                h.date,
                h.tempmax_c AS actual_c,
                e.forecast_tempmax_c AS ecmwf_c,
                g.forecast_tempmax_c AS gfs_c
            FROM historical_observed_daily h
            LEFT JOIN ecmwf_latest e
              ON LOWER(e.city) = LOWER(h.city) AND e.date = h.date AND e.rn = 1
            LEFT JOIN gfs_latest g
              ON LOWER(g.city) = LOWER(h.city) AND g.date = h.date AND g.rn = 1
            WHERE h.tempmax_c IS NOT NULL
              AND h.date >= DATE('now', '-30 days')
            ORDER BY h.city, h.date
        """, conn)
        if df.empty:
            return df
        # Compute blended (65/35)
        mask = df["ecmwf_c"].notna() & df["gfs_c"].notna()
        df.loc[mask, "blended_c"] = df.loc[mask, "ecmwf_c"] * 0.65 + df.loc[mask, "gfs_c"] * 0.35
        df.loc[~mask & df["ecmwf_c"].notna(), "blended_c"] = df.loc[~mask & df["ecmwf_c"].notna(), "ecmwf_c"]
        df.loc[~mask & df["gfs_c"].notna(), "blended_c"] = df.loc[~mask & df["gfs_c"].notna(), "gfs_c"]
        df["error_c"] = df["actual_c"] - df["blended_c"]
        return df

    @st.cache_data(ttl=_CACHE_TTL_SHORT)
    def _load_future_forecast_data():
        """Load forecast data for future dates (no actual temp yet).

        Uses temp_events.forecast_mu_c for the bias-corrected blended average,
        and forecast_runs for the raw per-model values (ECMWF/GFS before bias).
        """
        conn = _connect_db()
        df = pd.read_sql("""
            SELECT
                te.city,
                te.date,
                te.forecast_mu_c AS blended_c,
                fr_e.forecast_mu_c AS ecmwf_c,
                fr_g.forecast_mu_c AS gfs_c
            FROM (
                SELECT city, date, forecast_mu_c,
                       ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY scan_timestamp DESC) rn
                FROM temp_events
                WHERE date >= DATE('now')
            ) te
            LEFT JOIN (
                SELECT city, date, forecast_mu_c,
                       ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY pulled_at DESC) rn
                FROM forecast_runs WHERE source = 'ecmwf'
            ) fr_e ON LOWER(fr_e.city) = LOWER(te.city) AND fr_e.date = te.date AND fr_e.rn = 1
            LEFT JOIN (
                SELECT city, date, forecast_mu_c,
                       ROW_NUMBER() OVER (PARTITION BY city, date ORDER BY pulled_at DESC) rn
                FROM forecast_runs WHERE source = 'gfs'
            ) fr_g ON LOWER(fr_g.city) = LOWER(te.city) AND fr_g.date = te.date AND fr_g.rn = 1
            WHERE te.rn = 1
            ORDER BY te.city, te.date
        """, conn)
        if df.empty:
            return df
        # Fill blended from raw if temp_events didn't have it
        mask_no_blend = df["blended_c"].isna()
        mask_both = df["ecmwf_c"].notna() & df["gfs_c"].notna()
        df.loc[mask_no_blend & mask_both, "blended_c"] = (
            df.loc[mask_no_blend & mask_both, "ecmwf_c"] * 0.65 +
            df.loc[mask_no_blend & mask_both, "gfs_c"] * 0.35
        )
        # Unbiased blend: always computed from raw ECMWF/GFS (no bias correction)
        df["unbiased_blend_c"] = None
        mask_ub = df["ecmwf_c"].notna() & df["gfs_c"].notna()
        df.loc[mask_ub, "unbiased_blend_c"] = (
            df.loc[mask_ub, "ecmwf_c"] * 0.65 +
            df.loc[mask_ub, "gfs_c"] * 0.35
        )
        df.loc[~mask_ub & df["ecmwf_c"].notna(), "unbiased_blend_c"] = df.loc[~mask_ub & df["ecmwf_c"].notna(), "ecmwf_c"]
        df.loc[~mask_ub & df["gfs_c"].notna(), "unbiased_blend_c"] = df.loc[~mask_ub & df["gfs_c"].notna(), "gfs_c"]
        df["actual_c"] = None
        df["error_c"] = None
        return df

    # Build city list from both data sources for the shared selector
    @st.cache_data(ttl=_CACHE_TTL_SHORT)
    def _get_all_accuracy_cities():
        conn = _connect_db()
        c1 = pd.read_sql("SELECT DISTINCT city FROM historical_observed_daily WHERE tempmax_c IS NOT NULL", conn)
        c2 = pd.read_sql("SELECT DISTINCT city FROM forecast_runs", conn)
        all_c = pd.concat([c1, c2])["city"].str.title().dropna().unique().tolist()
        return sorted(all_c)

    _all_acc_cities = _get_all_accuracy_cities()
    if not _all_acc_cities:
        st.info("No forecast or observation data available yet.")
    else:
        # Filter city dropdown by active markets if the checkbox is checked
        if _only_active:
            _chart_cities = sorted([c for c in _all_acc_cities if c.lower() in _active_cities])
        else:
            _chart_cities = _all_acc_cities
        sel_city = st.selectbox("Select City", _chart_cities, key="accuracy_city")

        import plotly.graph_objects as go
        from datetime import date as _date_cls

        future_df = _load_future_forecast_data()
        if not future_df.empty:
            future_df["city"] = future_df["city"].str.title()
        _fut = future_df[future_df["city"] == sel_city].copy() if not future_df.empty else pd.DataFrame()

        # ---- Chart 0: Unbiased Upcoming Forecast ----
        st.markdown("#### Unbiased Upcoming Forecast")
        st.caption("Raw ECMWF + GFS blend (65/35) without bias correction applied.")
        if not _fut.empty:
            _fut_ub = _fut.copy()
            _fut_ub["date"] = pd.to_datetime(_fut_ub["date"])
            _fut_ub = _fut_ub.sort_values("date")
            fig_ub = go.Figure()
            if _fut_ub["unbiased_blend_c"].notna().any():
                fig_ub.add_trace(go.Scatter(
                    x=_fut_ub["date"], y=_fut_ub["unbiased_blend_c"],
                    mode="lines+markers", name="Unbiased Blend",
                    line=dict(color="#9b59b6", width=3),
                    marker=dict(size=6),
                ))
            if _fut_ub["ecmwf_c"].notna().any():
                fig_ub.add_trace(go.Scatter(
                    x=_fut_ub["date"], y=_fut_ub["ecmwf_c"],
                    mode="lines+markers", name="ECMWF (raw)",
                    line=dict(color="#e74c3c", width=2, dash="dot"),
                    marker=dict(size=4),
                ))
            if _fut_ub["gfs_c"].notna().any():
                fig_ub.add_trace(go.Scatter(
                    x=_fut_ub["date"], y=_fut_ub["gfs_c"],
                    mode="lines+markers", name="GFS (raw)",
                    line=dict(color="#f39c12", width=2, dash="dot"),
                    marker=dict(size=4),
                ))
            fig_ub.update_layout(
                title=f"{sel_city} - Unbiased Forecast (no bias correction)",
                xaxis_title="Date", yaxis_title="Temperature (C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=300, margin=dict(t=60, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig_ub, width="stretch")

            # Show bias delta
            if _fut_ub["blended_c"].notna().any() and _fut_ub["unbiased_blend_c"].notna().any():
                _bias_delta = (_fut_ub["blended_c"] - _fut_ub["unbiased_blend_c"]).mean()
                if pd.notna(_bias_delta):
                    _dir = "warmer" if _bias_delta > 0 else "cooler"
                    st.caption(
                        f"Average bias correction for {sel_city}: "
                        f"{_bias_delta:+.2f}C ({_dir} than raw models)"
                    )
        else:
            st.info(f"No upcoming forecast data for {sel_city}")

        st.divider()

        # ---- Chart 1: Upcoming Forecast (bias-corrected) ----
        st.markdown("#### Upcoming Forecast (Bias-Corrected)")

        if not _fut.empty:
            _fut["date"] = pd.to_datetime(_fut["date"])
            _fut = _fut.sort_values("date")
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(
                x=_fut["date"], y=_fut["blended_c"],
                mode="lines+markers", name="Blended Forecast",
                line=dict(color="#3498db", width=3),
                marker=dict(size=6),
            ))
            if _fut["ecmwf_c"].notna().any():
                fig_f.add_trace(go.Scatter(
                    x=_fut["date"], y=_fut["ecmwf_c"],
                    mode="lines+markers", name="ECMWF",
                    line=dict(color="#e74c3c", width=2, dash="dot"),
                    marker=dict(size=4),
                ))
            if _fut["gfs_c"].notna().any():
                fig_f.add_trace(go.Scatter(
                    x=_fut["date"], y=_fut["gfs_c"],
                    mode="lines+markers", name="GFS",
                    line=dict(color="#f39c12", width=2, dash="dot"),
                    marker=dict(size=4),
                ))
            fig_f.update_layout(
                title=f"{sel_city} - Forecast for Upcoming Days",
                xaxis_title="Date", yaxis_title="Temperature (C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=350, margin=dict(t=60, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig_f, width="stretch")
        else:
            st.info(f"No upcoming forecast data for {sel_city}")

        # ---- Contract Prices + Open Positions per date ----
        if not _fut.empty:
            @st.cache_data(ttl=_CACHE_TTL_SHORT)
            def _load_contracts_for_city(city_name):
                conn = _connect_db()
                conn.row_factory = sqlite3.Row
                # Latest outcomes per event for this city (using most recent
                # scan per date, not a single global scan timestamp)
                outcomes = conn.execute("""
                    SELECT o.contract_id, o.range_low, o.range_high, o.unit,
                           o.yes_price, o.no_price, o.model_prob,
                           e.date, e.display_unit
                    FROM temp_outcomes o
                    JOIN temp_events e ON o.event_row_id = e.id
                    WHERE LOWER(e.city) = LOWER(?)
                      AND e.id IN (
                          SELECT id FROM temp_events
                          WHERE LOWER(city) = LOWER(?)
                          GROUP BY date
                          HAVING scan_timestamp = MAX(scan_timestamp)
                      )
                    ORDER BY e.date, o.range_low DESC
                """, (city_name, city_name)).fetchall()
                # Open positions for this city
                positions = conn.execute("""
                    SELECT contract_id, side, size_usdc, target_size_usdc,
                           range_low, range_high, date, unit, unrealized_pnl
                    FROM positions
                    WHERE LOWER(city) = LOWER(?) AND status = 'open'
                      AND fill_status = 'filled'
                    ORDER BY date, range_low DESC
                """, (city_name,)).fetchall()
                # Group by date
                by_date_o = {}
                for r in outcomes:
                    d = r["date"]
                    by_date_o.setdefault(d, []).append(dict(r))
                by_date_p = {}
                for r in positions:
                    d = r["date"]
                    by_date_p.setdefault(d, []).append(dict(r))
                return by_date_o, by_date_p

            _contracts_by_date, _positions_by_date = _load_contracts_for_city(sel_city)

            # Get the dates from the forecast chart
            _forecast_dates = sorted(_fut["date"].dt.strftime("%Y-%m-%d").unique().tolist())

            if _forecast_dates and (_contracts_by_date or _positions_by_date):
                st.markdown("#### Contract Prices & Positions")
                _date_cols = st.columns(len(_forecast_dates))
                for _di, _fd in enumerate(_forecast_dates):
                    with _date_cols[_di]:
                        st.markdown(f"**{_fd}**")
                        _outcomes = _contracts_by_date.get(_fd, [])
                        _pos_list = _positions_by_date.get(_fd, [])
                        _pos_by_cid = {p["contract_id"]: p for p in _pos_list}

                        if _outcomes:
                            _rows = []
                            for _o in _outcomes:
                                _u = _o.get("unit", "celsius")
                                _sym = "F" if _u == "fahrenheit" else "C"
                                _lo = _o.get("range_low")
                                _hi = _o.get("range_high")
                                if _lo is not None and _hi is not None and _lo == _hi:
                                    _label = f"{int(_lo)}{_sym}"
                                elif _lo is not None and _hi is not None:
                                    _label = f"{int(_lo)}-{int(_hi)}{_sym}"
                                elif _lo is not None:
                                    _label = f">={int(_lo)}{_sym}"
                                else:
                                    _label = f"<={int(_hi)}{_sym}" if _hi else "?"

                                _yp = float(_o.get("yes_price") or 0)
                                _mp = float(_o.get("model_prob") or 0)

                                _cid = _o.get("contract_id")
                                _held = _pos_by_cid.get(_cid)
                                if _held:
                                    _sz = float(_held.get("size_usdc") or 0)
                                    _pnl = float(_held.get("unrealized_pnl") or 0)
                                    _pos_str = f"HELD {_held['side']}"
                                    _pnl_val = _pnl
                                else:
                                    _pos_str = ""
                                    _pnl_val = None

                                _rows.append({
                                    "Range": _label,
                                    "Market Prob": f"{_yp*100:.0f}%",
                                    "Model Prob": f"{_mp*100:.0f}%",
                                    "Position": _pos_str,
                                    "P&L": _pnl_val,
                                })
                            _contract_df = pd.DataFrame(_rows)

                            def _color_pnl_cell(val):
                                if val is None or pd.isna(val):
                                    return ""
                                if val > 0:
                                    return "color: #2ecc71"
                                elif val < 0:
                                    return "color: #e74c3c"
                                return ""

                            st.dataframe(
                                _contract_df.style.map(
                                    _color_pnl_cell, subset=["P&L"]
                                ).format({"P&L": lambda v: f"${v:+,.2f}" if pd.notna(v) and v is not None else ""}),
                                hide_index=True, width="stretch",
                                height=min(60 + len(_rows) * 35, 400),
                            )
                        else:
                            st.caption("No contract data")

        st.divider()

        # ---- Chart 2: Historical Forecast vs Actual (slower query) ----
        st.markdown("#### Historical Forecast vs Actual")
        accuracy_df = _load_accuracy_data()
        if not accuracy_df.empty:
            accuracy_df["city"] = accuracy_df["city"].str.title()
        _hist = accuracy_df[accuracy_df["city"] == sel_city].copy() if not accuracy_df.empty else pd.DataFrame()
        _hist = _hist.dropna(subset=["actual_c", "blended_c"]) if not _hist.empty else _hist

        if not _hist.empty:
            _hist["date"] = pd.to_datetime(_hist["date"])
            _hist = _hist.sort_values("date")

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=_hist["date"], y=_hist["actual_c"],
                mode="lines+markers", name="Actual High",
                line=dict(color="#2ecc71", width=3),
                marker=dict(size=6),
            ))
            fig_h.add_trace(go.Scatter(
                x=_hist["date"], y=_hist["blended_c"],
                mode="lines+markers", name="Blended Forecast",
                line=dict(color="#3498db", width=2, dash="dash"),
                marker=dict(size=5),
            ))
            if _hist["ecmwf_c"].notna().any():
                fig_h.add_trace(go.Scatter(
                    x=_hist["date"], y=_hist["ecmwf_c"],
                    mode="lines", name="ECMWF",
                    line=dict(color="#e74c3c", width=1, dash="dot"),
                    opacity=0.6,
                ))
            if _hist["gfs_c"].notna().any():
                fig_h.add_trace(go.Scatter(
                    x=_hist["date"], y=_hist["gfs_c"],
                    mode="lines", name="GFS",
                    line=dict(color="#f39c12", width=1, dash="dot"),
                    opacity=0.6,
                ))
            fig_h.update_layout(
                title=f"{sel_city} - Forecast vs Actual (Last 30 Days)",
                xaxis_title="Date", yaxis_title="Temperature (C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                height=400, margin=dict(t=60, b=40),
                hovermode="x unified",
            )
            st.plotly_chart(fig_h, width="stretch")

            # Summary stats
            _errors = _hist["error_c"].dropna()
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("MAE", f"{_errors.abs().mean():.2f}C")
            with sc2:
                st.metric("RMSE", f"{(_errors ** 2).mean() ** 0.5:.2f}C")
            with sc3:
                st.metric("Bias", f"{_errors.mean():+.2f}C",
                          help="Positive = model too cold, Negative = model too warm")
            with sc4:
                st.metric("Days", str(len(_errors)))

            # Error bar chart
            st.divider()
            st.markdown("**Daily Forecast Error (Actual - Blended)**")
            fig2 = go.Figure()
            _ev = _errors.tolist()
            fig2.add_trace(go.Bar(
                x=_hist["date"], y=_ev,
                marker_color=["#2ecc71" if e >= 0 else "#e74c3c" for e in _ev],
            ))
            fig2.update_layout(
                xaxis_title="Date", yaxis_title="Error (C)",
                height=250, margin=dict(t=20, b=40),
            )
            fig2.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
            st.plotly_chart(fig2, width="stretch")
        else:
            st.info(f"No historical forecast+actual data for {sel_city}")

    st.divider()
    _ac1, _ac2, _ac3 = st.columns([2, 1, 2])
    with _ac2:
        if st.button("Refresh Data", key="refresh_accuracy", type="primary"):
            _do_refresh("refresh_accuracy")
