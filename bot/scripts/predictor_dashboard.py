"""
predictor_dashboard.py — Live dashboard for the intraday predictor.

Reads from paper_predictor_signals + live_predictor_orders and renders a
self-contained HTML page with:

  * Top mode toggle (paper / live / both)
  * KPI strip — counts respect the mode toggle
  * Per-city panels — one for every US city, showing today's event with
    EVERY bin's our_p, market_p, edge, action, entry price, and P&L
  * Today's signals table — filterable, sortable, grouped by city

Configured entirely via .env (no CLI flags needed for normal operation):
  DASHBOARD_PORT       (default 8082)
  DASHBOARD_WATCH_SEC  (default 30, auto-regenerates HTML)
  DASHBOARD_DAYS       (default 1, lookback for the table)

Service install:  sudo bash deploy/setup_dashboard_systemd.sh
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
import os
import socketserver
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_BOT_DIR)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

# Force python-dotenv to load .env BEFORE config.py imports.  This
# matters when run via systemd, whose EnvironmentFile= parser does NOT
# strip inline comments — so a line like "MAX_OPEN_POSITIONS=0  # cap"
# arrives as the literal string '0  # cap' and int() chokes.  Loading
# with override=True via python-dotenv (which DOES strip comments)
# rewrites the polluted vars before config.py touches them.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"), override=True)
except ImportError:
    pass

from config import DB_PATH  # type: ignore
try:
    from station_meta import CITY_STATIONS  # type: ignore
except Exception:
    CITY_STATIONS = {}
try:
    from scripts.find_nearby_stations import US_CITY_STATES  # type: ignore
except Exception:
    US_CITY_STATES = {}

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("predictor_dash")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _ensure_tables(db: str) -> None:
    """Create predictor tables if missing — lets the dashboard run before
    the bot's first scan has populated them."""
    try:
        from scheduled_predictor import ensure_schema  # type: ignore
        ensure_schema()
    except Exception as e:
        log.debug(f"ensure_schema import failed: {e}")


# Hard cap on rows pulled into the dashboard per regenerate.  At ~121
# rows/scan × 30 scans/hour, today alone hits ~87K rows.  The dashboard
# filters by event_date=today in JS anyway, so we narrow at the SQL
# layer — but cap as a safety net in case event_date is sloppy.
DASHBOARD_ROW_HARD_CAP = int(os.getenv("DASHBOARD_ROW_HARD_CAP", "30000"))


def load_signals(db: str, since_utc: datetime) -> list[dict]:
    """Load ONLY what the dashboard actually displays.

    Two row sets, UNION'd:
      1. The most-recent scan per (event_date) — gives "current state"
         for each city panel.  Typically ~120 rows.
      2. All BUY rows (PAPER_BUY / LIVE_BUY) in the date window —
         needed for the trades view + buy attribution in panels.
         Typically ~5-30 rows.

    Total: ~150 rows vs the previous 30,000.  This is a ~150x reduction
    in HTML size and JSON parse time on the browser side.

    The old implementation loaded EVERY scan's rows (after the "always
    re-evaluate" fix), bloating to ~80K+ rows/day = ~50MB JSON.  The
    dashboard browser doesn't need any of the historical scans — they
    just slow things down.
    """
    if not os.path.exists(db):
        return []
    _ensure_tables(db)
    since_date = since_utc.date()
    today_date = datetime.now(timezone.utc).date()
    dates_wanted = []
    d = since_date
    while d <= today_date:
        dates_wanted.append(d.isoformat())
        d = d + timedelta(days=1)
    placeholders = ",".join("?" for _ in dates_wanted)

    cols = (
        "scanned_at_utc, mode, city, settlement_station, "
        "event_date, event_id, contract_id, yes_token_id, bin_label, "
        "bin_range_low, our_prob, market_prob, edge, liquidity_usd, "
        "action, gate_blocked_by, recommended_stake_usd, "
        "observed_max_c, observed_peak_hour, forecast_high_c, "
        "forecast_peak_hour, wind_octant, market_closed"
    )

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            # Row set 1: latest scan per event_date.  We grab the max
            # scanned_at_utc per event_date and join back to get all rows
            # from that exact timestamp.
            latest = conn.execute(
                f"""
                SELECT {cols} FROM paper_predictor_signals
                WHERE event_date IN ({placeholders})
                  AND scanned_at_utc IN (
                    SELECT MAX(scanned_at_utc)
                    FROM paper_predictor_signals
                    WHERE event_date IN ({placeholders})
                    GROUP BY event_date
                  )
                """,
                (*dates_wanted, *dates_wanted),
            ).fetchall()
            # Row set 2: all BUY rows in the window (regardless of scan
            # timestamp).  De-dup'd against row set 1 by id-less merge
            # (using contract_id + scanned_at_utc as effective primary key).
            buys = conn.execute(
                f"""
                SELECT {cols} FROM paper_predictor_signals
                WHERE event_date IN ({placeholders})
                  AND action IN ('PAPER_BUY', 'LIVE_BUY')
                """,
                tuple(dates_wanted),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning(f"paper_predictor_signals not readable: {e}")
            return []

    seen = set()
    out: list[dict] = []
    for r in list(latest) + list(buys):
        d = dict(r)
        key = (d["contract_id"], d["scanned_at_utc"])
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    # Sort newest-first to match prior ordering expectations in JS
    out.sort(key=lambda d: d["scanned_at_utc"], reverse=True)
    log.info(f"load_signals: {len(out)} rows ({len(latest)} latest-scan + "
              f"{len(buys)} buy rows, de-duped)")
    return out


def load_live_orders(db: str, since_utc: datetime) -> list[dict]:
    if not os.path.exists(db):
        return []
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM live_predictor_orders WHERE placed_at_utc >= ? "
                "ORDER BY placed_at_utc DESC", (since_utc.isoformat(),)
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# City detail — drilldown shown when the operator clicks a city in the
# Analysis table.  Time-series of temperature + model_p + market_p for
# the bought bin AND the winning bin, plus context (bin distribution at
# final scan, sunset/peak hour markers, summary card data).
# ---------------------------------------------------------------------------

def _load_city_detail(conn: sqlite3.Connection, *, city: str,
                       event_date: str, bought_contract_id: str | None,
                       winning_contract_id: str | None,
                       ) -> dict:
    """Build the detail-view blob for one (city, event_date) event.

    Returns a dict with:
      summary:       headline metadata for the summary card
      timeseries:    one row per scan with shared context
                       (scanned_at_utc, observed_max_c, forecast_high_c,
                        mu_c, sigma_c)
      bought_series: per-scan our_prob/market_prob for the bought bin
      winning_series: same for the winning bin
      bin_distribution: final-scan probability vector across ALL bins
                         (used for the bar-chart drawer at the bottom)
      forecast_peak_hour, sunset_hour: scalar markers (best-effort)
    """
    out: dict = {
        "summary":            {},
        "timeseries":         [],
        "bought_series":      [],
        "winning_series":     [],
        "bin_distribution":   [],
        "forecast_peak_hour": None,
        "sunset_hour":        None,
    }

    # ---- Per-scan shared context (mu, sigma, observed/forecast highs) ----
    # We pick ONE bin per scan to avoid replicating shared context.
    # Use MIN(bin_label) as a stable selector; the shared columns
    # (observed_max_c, mu_c, sigma_c, forecast_high_c) are bin-invariant.
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT
                scanned_at_utc,
                observed_max_c,
                forecast_high_c,
                forecast_peak_hour,
                mu_c,
                sigma_c
            FROM paper_predictor_signals
            WHERE city = ?
              AND event_date = ?
            GROUP BY scanned_at_utc
            ORDER BY scanned_at_utc ASC
            """,
            (city, event_date),
        ).fetchall()
        out["timeseries"] = [
            {
                "t":              r["scanned_at_utc"],
                "observed_max_c": r["observed_max_c"],
                "forecast_high_c": r["forecast_high_c"],
                "mu_c":           r["mu_c"],
                "sigma_c":        r["sigma_c"],
            }
            for r in rows
        ]
        # First non-null forecast_peak_hour wins (it's per-scan but
        # usually stable across the day).
        for r in rows:
            if r["forecast_peak_hour"] is not None:
                out["forecast_peak_hour"] = int(r["forecast_peak_hour"])
                break
    except sqlite3.OperationalError as e:
        log.warning(f"_load_city_detail shared-context query failed: {e}")

    # ---- Per-bin series — bought + winning ----
    def _bin_series(contract_id: str | None) -> list[dict]:
        if not contract_id:
            return []
        try:
            rows = conn.execute(
                """
                SELECT scanned_at_utc, our_prob, market_prob
                FROM paper_predictor_signals
                WHERE city = ?
                  AND event_date = ?
                  AND contract_id = ?
                ORDER BY scanned_at_utc ASC
                """,
                (city, event_date, contract_id),
            ).fetchall()
            return [
                {
                    "t":           r["scanned_at_utc"],
                    "our_prob":    r["our_prob"],
                    "market_prob": r["market_prob"],
                }
                for r in rows
            ]
        except sqlite3.OperationalError:
            return []

    out["bought_series"]  = _bin_series(bought_contract_id)
    out["winning_series"] = _bin_series(winning_contract_id)

    # ---- Final-scan distribution: every bin's our_p + market_p ----
    # Tells the operator "at the moment the market resolved, what did
    # we think vs what did the market think across the full bin range."
    try:
        latest = conn.execute(
            "SELECT MAX(scanned_at_utc) FROM paper_predictor_signals "
            "WHERE city = ? AND event_date = ?",
            (city, event_date),
        ).fetchone()
        latest_ts = latest[0] if latest else None
        if latest_ts:
            rows = conn.execute(
                """
                SELECT
                    bin_label, bin_range_low, bin_range_high, unit,
                    our_prob, market_prob, contract_id
                FROM paper_predictor_signals
                WHERE city = ? AND event_date = ?
                  AND scanned_at_utc = ?
                ORDER BY bin_range_low ASC
                """,
                (city, event_date, latest_ts),
            ).fetchall()
            out["bin_distribution"] = [
                {
                    "bin_label":    r["bin_label"],
                    "range_low":    r["bin_range_low"],
                    "range_high":   r["bin_range_high"],
                    "unit":         r["unit"],
                    "our_prob":     r["our_prob"],
                    "market_prob":  r["market_prob"],
                    "contract_id":  r["contract_id"],
                    "is_bought":    (r["contract_id"] == bought_contract_id),
                    "is_winning":   (r["contract_id"] == winning_contract_id),
                }
                for r in rows
            ]
    except sqlite3.OperationalError as e:
        log.warning(f"_load_city_detail distribution query failed: {e}")

    return out


def _load_city_details_for_purchases(conn: sqlite3.Connection,
                                        purchases: list[dict]) -> dict:
    """Loop through unique (city, event_date) pairs in the purchases
    set and build a detail blob for each.  Bought/winning contract_ids
    come straight from the already-loaded purchase row, so this only
    adds one query batch per event (not per purchase row)."""
    details: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for p in purchases:
        city = p.get("city") or ""
        date = p.get("event_date") or ""
        if not city or not date:
            continue
        key = (city, date)
        if key in seen:
            continue
        seen.add(key)
        bought_cid  = p.get("contract_id")
        # winning_contract_id isn't selected into purchases yet — derive
        # by re-using the win_lo / win_hi pair to find the matching
        # contract_id from the latest scan.
        winning_cid = None
        win_lo, win_hi = p.get("win_lo"), p.get("win_hi")
        if win_lo is not None and win_hi is not None:
            try:
                latest_ts_row = conn.execute(
                    "SELECT MAX(scanned_at_utc) FROM paper_predictor_signals "
                    "WHERE city = ? AND event_date = ?",
                    (city, date),
                ).fetchone()
                latest_ts = latest_ts_row[0] if latest_ts_row else None
                if latest_ts:
                    row = conn.execute(
                        """SELECT contract_id FROM paper_predictor_signals
                           WHERE city = ? AND event_date = ?
                             AND scanned_at_utc = ?
                             AND bin_range_low  = ?
                             AND bin_range_high = ?
                             LIMIT 1""",
                        (city, date, latest_ts, win_lo, win_hi),
                    ).fetchone()
                    if row:
                        winning_cid = row[0]
            except sqlite3.OperationalError:
                pass

        details[f"{city}||{date}"] = _load_city_detail(
            conn,
            city                = city,
            event_date          = date,
            bought_contract_id  = bought_cid,
            winning_contract_id = winning_cid,
        )
    return details


# ---------------------------------------------------------------------------
# Analysis tab — purchases & outcomes (the readable headline)
# ---------------------------------------------------------------------------
# The headline product is one denormalized row per purchased bin showing:
#   - What we bought (city, date, bin, side, stake)
#   - What the model said at purchase time (forecast_high, mu, sigma, our_p)
#   - What actually won (the bin whose market_prob settled at >= 0.99)
#   - Realized P&L
#
# Winning bin is derived from paper_predictor_signals: the bin in the
# latest scan of (city, event_date) with market_prob >= 0.99 is the bin
# Polymarket resolved YES.  Falls back to event_resolutions if available.

def load_analysis_data(db: str, lookback_days: int = 30) -> dict:
    """Compose every analysis dataset in one pass.  Returns a dict the
    JS can iterate over; empty lists for tables that have no rows.

    Every query is wrapped in try/except so missing tables (fresh DB)
    don't break the whole tab.
    """
    # Headline KPIs are computed in JS from the filtered `purchases` set
    # so they react to the Paper / Live / Both mode toggle.  Python no
    # longer pre-aggregates them.
    out: dict = {
        "lookback_days":       int(lookback_days),
        "pipeline_today":      {"signals": 0, "orders": 0, "positions": 0},
        "purchases":           [],
        "calibration_buckets": [],
        "stuck_positions":     [],
        # City detail blob — keyed by "<city>||<event_date>", populated
        # AFTER the purchases query so we only emit detail for events
        # we actually have purchases on.  Loaded by load_city_details.
        "city_details":        {},
    }
    if not os.path.exists(db):
        return out

    days = max(1, int(lookback_days))

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row

        # -------- Pipeline coverage today (compact health KPI) --------
        try:
            sig = conn.execute(
                "SELECT COUNT(*) FROM paper_predictor_signals "
                "WHERE event_date = date('now') AND action = 'LIVE_BUY'"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            sig = 0
        try:
            ord_n = conn.execute(
                "SELECT COUNT(*) FROM live_predictor_orders "
                "WHERE substr(placed_at_utc, 1, 10) = date('now')"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            ord_n = 0
        try:
            pos = conn.execute(
                "SELECT COUNT(*) FROM positions "
                "WHERE date = date('now') AND COALESCE(is_paper, 0) = 0 "
                "  AND status IN ('open', 'closed')"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            pos = 0
        out["pipeline_today"] = {
            "signals": int(sig), "orders": int(ord_n), "positions": int(pos)
        }

        # -------- Purchases + outcomes (the headline) -----------------
        # Three CTEs:
        #   purchases    — every live position with the LIVE_BUY signal
        #                   row that produced it, model context joined
        #   latest_scan  — most-recent scan_at_utc per (city, event_date)
        #   winners      — the bin in that latest scan with market_prob
        #                   >= 0.99 (Polymarket's resolution signal)
        # Final select joins purchases LEFT JOIN winners — pending events
        # come through with NULL winner fields.
        try:
            rows = conn.execute(
                f"""
                WITH
                -- One signal row per (contract, event_date, action).
                -- Without this dedupe, the LEFT JOIN below multiplies
                -- each positions row by the number of LIVE_BUY/PAPER_BUY
                -- scans for that contract (the bot writes one signal row
                -- per scan, so an actively-traded bin can have 30+
                -- LIVE_BUY rows — same positions row would appear 30 times).
                -- We pick the EARLIEST buy signal as the "model state at
                -- the moment the buy decision first triggered."
                first_buy_signal AS (
                    SELECT * FROM (
                        SELECT
                            s.contract_id, s.event_date, s.action,
                            s.our_prob, s.market_prob,
                            s.forecast_high_c, s.mu_c, s.sigma_c,
                            s.observed_max_c, s.data_quality_flag,
                            s.scanned_at_utc,
                            ROW_NUMBER() OVER (
                                PARTITION BY s.contract_id, s.event_date, s.action
                                ORDER BY s.scanned_at_utc ASC
                            ) AS rn
                        FROM paper_predictor_signals s
                        WHERE s.action IN ('LIVE_BUY', 'PAPER_BUY')
                          AND s.event_date >= date('now', '-{days} days')
                    ) WHERE rn = 1
                ),
                -- Aggregate every positions row that shares the same
                -- (city, event_date, contract_id, side, is_paper) into
                -- ONE logical purchase.  An "initial buy + topup +
                -- re-entry-after-cancel" sequence collapses to one row.
                -- size sums; entry_price weight-averages; pnl sums; the
                -- earliest entry_time wins for display.  status uses a
                -- precedence (any closed > any open > any pending).
                positions_agg AS (
                    SELECT
                        p.city            AS city,
                        p.date            AS event_date,
                        p.contract_id     AS contract_id,
                        p.side            AS side,
                        COALESCE(p.is_paper, 0) AS is_paper,
                        -- Bin: per-contract these are constant; MAX is
                        -- a no-op pickup so the grouping still works.
                        MAX(p.range_low)  AS bought_lo,
                        MAX(p.range_high) AS bought_hi,
                        MAX(p.unit)       AS bought_unit,
                        -- Stake: total deployed across all rows
                        ROUND(SUM(COALESCE(p.size_usdc, 0)), 2) AS stake_usd,
                        -- Entry price: weighted average by stake
                        ROUND(
                          SUM(COALESCE(p.size_usdc, 0) * COALESCE(p.entry_price, 0))
                          / NULLIF(SUM(COALESCE(p.size_usdc, 0)), 0),
                          3
                        ) AS entry_px,
                        -- Exit: same for every row of a resolved market,
                        -- so MAX is identity (NULL when not resolved).
                        ROUND(MAX(p.exit_price), 3) AS exit_px,
                        -- PNL: sum across all rows (initial + topup
                        -- share the same outcome but persist as
                        -- separate position_orders rows with their own
                        -- pnl values for that slice of the position).
                        ROUND(SUM(COALESCE(p.pnl_net, p.pnl, 0)), 2) AS pnl,
                        -- Status precedence: closed beats open beats
                        -- pending.  CASE expression assigns sort order
                        -- and we pick the max.
                        CASE MAX(
                          CASE p.status
                            WHEN 'closed' THEN 3
                            WHEN 'open'   THEN 2
                            ELSE 1
                          END
                        )
                          WHEN 3 THEN 'closed'
                          WHEN 2 THEN 'open'
                          ELSE        'pending'
                        END AS status,
                        -- Fill status precedence: filled > pending > cancelled
                        CASE MAX(
                          CASE p.fill_status
                            WHEN 'filled'    THEN 3
                            WHEN 'pending'   THEN 2
                            ELSE 1
                          END
                        )
                          WHEN 3 THEN 'filled'
                          WHEN 2 THEN 'pending'
                          ELSE        'cancelled'
                        END AS fill_status,
                        MIN(p.entry_time) AS entry_time,
                        MAX(p.exit_time)  AS exit_time,
                        COUNT(*)          AS n_rows
                    FROM positions p
                    WHERE p.date >= date('now', '-{days} days')
                    GROUP BY p.city, p.date, p.contract_id, p.side,
                             COALESCE(p.is_paper, 0)
                ),
                purchases AS (
                    -- Final shape: aggregated purchases LEFT JOINed to
                    -- the first-buy signal for model context.
                    SELECT
                        pa.city, pa.event_date, pa.contract_id,
                        pa.side, pa.is_paper, pa.status, pa.fill_status,
                        pa.stake_usd, pa.entry_px, pa.exit_px, pa.pnl,
                        pa.bought_lo, pa.bought_hi, pa.bought_unit,
                        pa.entry_time, pa.exit_time, pa.n_rows,
                        s.our_prob        AS at_buy_our_p,
                        s.market_prob     AS at_buy_mkt_p,
                        ROUND(s.forecast_high_c, 2) AS at_buy_fc_high_c,
                        ROUND(s.mu_c,            2) AS at_buy_mu_c,
                        ROUND(s.sigma_c,         2) AS at_buy_sigma_c,
                        ROUND(s.observed_max_c,  2) AS at_buy_obs_max_c,
                        s.data_quality_flag       AS dq_flag
                    FROM positions_agg pa
                    LEFT JOIN first_buy_signal s
                      ON s.contract_id = pa.contract_id
                     AND s.event_date  = pa.event_date
                     AND ((pa.is_paper = 0 AND s.action = 'LIVE_BUY')
                       OR (pa.is_paper = 1 AND s.action = 'PAPER_BUY'))
                ),
                latest_scan AS (
                    SELECT city, event_date,
                           MAX(scanned_at_utc) AS max_ts
                    FROM paper_predictor_signals
                    WHERE event_date >= date('now', '-{days} days')
                    GROUP BY city, event_date
                ),
                winners AS (
                    SELECT
                        s.city,
                        s.event_date,
                        s.bin_range_low  AS win_lo,
                        s.bin_range_high AS win_hi,
                        s.unit           AS win_unit,
                        s.market_prob    AS win_mkt_p
                    FROM paper_predictor_signals s
                    JOIN latest_scan ls
                      ON ls.city = s.city
                     AND ls.event_date = s.event_date
                     AND ls.max_ts = s.scanned_at_utc
                    WHERE s.market_prob >= 0.99
                )
                SELECT pu.*,
                       w.win_lo, w.win_hi, w.win_unit, w.win_mkt_p
                FROM purchases pu
                LEFT JOIN winners w
                       ON w.city = pu.city AND w.event_date = pu.event_date
                ORDER BY pu.event_date DESC, pu.entry_time DESC, pu.city ASC
                LIMIT 500
                """
            ).fetchall()
            out["purchases"] = [dict(r) for r in rows]
        except sqlite3.OperationalError as _e:
            log.warning(f"purchases query failed: {_e}")

        # -------- City details (drilldown blob for click-through) -----
        # Only build details for (city, event_date) pairs that appear
        # in the purchases set — caps the JSON payload.
        try:
            out["city_details"] = _load_city_details_for_purchases(
                conn, out["purchases"]
            )
        except Exception as _e:
            log.warning(f"city_details build failed: {_e}")

        # Headline KPIs are computed in JS so they react to the
        # Paper / Live / Both mode toggle.  See computePurchaseKpis().

        # -------- Calibration buckets (kept for model-tuning) ---------
        try:
            rows = conn.execute(
                f"""
                WITH joined AS (
                    SELECT
                        p.id, p.exit_price, s.our_prob
                    FROM positions p
                    JOIN paper_predictor_signals s
                      ON s.contract_id = p.contract_id
                     AND s.action = 'LIVE_BUY'
                     AND s.event_date = p.date
                    WHERE COALESCE(p.is_paper, 0) = 0
                      AND p.status = 'closed'
                      AND p.exit_price IS NOT NULL
                      AND p.date >= date('now', '-{days} days')
                )
                SELECT
                    CASE
                        WHEN our_prob < 0.3 THEN '0.0-0.3'
                        WHEN our_prob < 0.5 THEN '0.3-0.5'
                        WHEN our_prob < 0.7 THEN '0.5-0.7'
                        WHEN our_prob < 0.9 THEN '0.7-0.9'
                        ELSE                       '0.9-1.0'
                    END                       AS conf_bucket,
                    COUNT(*)                  AS n,
                    ROUND(AVG(our_prob), 3)   AS avg_model_p,
                    ROUND(AVG(exit_price), 3) AS actual_win_rate,
                    ROUND(AVG(exit_price) - AVG(our_prob), 3) AS calibration_gap
                FROM joined
                GROUP BY conf_bucket
                ORDER BY conf_bucket
                """
            ).fetchall()
            out["calibration_buckets"] = [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass

        # -------- Stuck positions (compact warning if any) ------------
        try:
            rows = conn.execute(
                """
                SELECT
                    id, city, date, side,
                    ROUND(size_usdc, 2)   AS stake_usd,
                    ROUND(entry_price, 3) AS entry_px,
                    status, fill_status, entry_time,
                    ROUND(julianday('now') - julianday(date), 1) AS days_past_event
                FROM positions
                WHERE COALESCE(is_paper, 0) = 0
                  AND status = 'open'
                  AND date < date('now')
                ORDER BY date ASC
                LIMIT 50
                """
            ).fetchall()
            out["stuck_positions"] = [dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass

    return out


# ---------------------------------------------------------------------------
# Polymarket data API — authoritative LIVE positions / P&L
# ---------------------------------------------------------------------------

POLYMARKET_DATA_API = "https://data-api.polymarket.com"


def _get_proxy_address() -> str | None:
    """Find the wallet/proxy address that holds our Polymarket positions."""
    for var in ("POLYMARKET_FUNDER_ADDRESS", "POLYMARKET_PROXY_ADDRESS",
                 "POLY_PROXY", "BROWSER_ADDRESS", "PROXY_ADDRESS",
                 "WALLET_ADDRESS"):
        v = os.getenv(var)
        if v and isinstance(v, str) and v.lower().startswith("0x") and len(v) == 42:
            return v
    return None


def fetch_live_positions() -> list[dict]:
    """Hit Polymarket's data API for our current positions (the authoritative
    source for LIVE P&L).  Returns list of dicts with at minimum:
        asset      — token ID (matches paper_predictor_signals.yes_token_id)
        size       — current shares held
        avgPrice   — average entry price (actual fills, not market_prob)
        curPrice   — current market mid-price
        cashPnl    — dollar P&L per Polymarket's own calculation
        percentPnl — % P&L

    Empty list on failure (network, missing wallet address, etc.) so the
    JS falls back to the formula-based PAPER P&L estimate.
    """
    addr = _get_proxy_address()
    if not addr:
        log.debug("no proxy address in env — skipping live positions fetch")
        return []
    try:
        import urllib.request, urllib.parse
        params = {"user": addr, "sizeThreshold": "0.01",
                   "limit": "500", "sortBy": "CURRENT"}
        url = f"{POLYMARKET_DATA_API}/positions?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "polymarket-weather/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        log.info(f"fetched {len(data)} live positions from Polymarket data API")
        return data
    except Exception as e:
        log.warning(f"Polymarket positions API failed: {e} — JS will fall back to formula")
        return []


# ---------------------------------------------------------------------------
# Static city list (US only — what the predictor trades)
# ---------------------------------------------------------------------------

_TZ_SHORT = {
    "America/New_York":    "ET",
    "America/Chicago":     "CT",
    "America/Denver":      "MT",
    "America/Phoenix":     "MST",   # AZ — no DST
    "America/Los_Angeles": "PT",
    "America/Anchorage":   "AKT",
    "Pacific/Honolulu":    "HT",
}


def _us_cities() -> list[dict]:
    us = (list(US_CITY_STATES.keys()) if US_CITY_STATES
           else [c for c, m in CITY_STATIONS.items() if m[0].startswith("K")])
    out = []
    for city in sorted(us):
        s = CITY_STATIONS.get(city)
        if not s:
            continue
        icao, _net, tz, _lat, _lon = s
        tz_label = _TZ_SHORT.get(tz, tz.split("/")[-1].replace("_", " "))
        # tz_str: full IANA name for client-side time conversion
        # (Analysis tab uses it to render entry_time in city-local time).
        out.append({"city": city, "station": icao,
                       "tz_label": tz_label, "tz_str": tz})
    return out


# ---------------------------------------------------------------------------
# HTML / CSS / JS
# ---------------------------------------------------------------------------

DASHBOARD_CSS = """
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, "Segoe UI", sans-serif;
       margin: 0; background: #0f172a; color: #e2e8f0; font-size: 13px; }

header { background: linear-gradient(90deg, #1e293b, #0f172a);
         padding: 14px 24px; border-bottom: 1px solid #334155;
         display: flex; align-items: center; justify-content: space-between; }
header h1 { margin: 0; font-size: 18px; font-weight: 600; color: white; }
header .meta { font-family: monospace; font-size: 11px; color: #94a3b8; text-align: right; }

/* Mode toggle in the header */
.mode-toggle { display: inline-flex; background: #0f172a; border-radius: 6px;
               overflow: hidden; border: 1px solid #334155; }
.mode-toggle button { background: transparent; border: none;
                       color: #94a3b8; padding: 6px 14px; cursor: pointer;
                       font-size: 12px; font-weight: 600; }
.mode-toggle button:hover { background: #273449; color: white; }
.mode-toggle button.active.paper { background: #1e3a8a; color: #dbeafe; }
.mode-toggle button.active.live  { background: #7f1d1d; color: #fef2f2; }
.mode-toggle button.active.both  { background: #4338ca; color: white; }
.mode-toggle button.active.view  { background: #475569; color: white; }

/* KPI strip */
.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
        gap: 8px; padding: 12px 24px; background: #0f172a; }
.kpi { background: #1e293b; padding: 10px 14px; border-radius: 6px;
       border-left: 3px solid #4338ca; }
.kpi .label { font-size: 9px; color: #94a3b8; text-transform: uppercase;
              letter-spacing: 0.5px; font-weight: 600; }
.kpi .val { font-size: 22px; font-weight: 700; margin-top: 4px;
            font-family: monospace; color: white; }
.kpi .sub { font-size: 10px; color: #94a3b8; margin-top: 2px; font-family: monospace; }
.kpi.buy { border-left-color: #22c55e; }
.kpi.buy .val { color: #4ade80; }
.kpi.live { border-left-color: #f59e0b; }
.kpi.live .val { color: #fbbf24; }
.kpi.pos { border-left-color: #22c55e; }
.kpi.pos .val { color: #4ade80; }
.kpi.neg { border-left-color: #ef4444; }
.kpi.neg .val { color: #f87171; }

.section-title { padding: 18px 24px 6px; font-size: 11px; font-weight: 700;
                  color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }

/* === Per-city panels === */
.city-grid { display: grid; gap: 14px; padding: 6px 24px 18px;
             grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); }
.city-panel { background: #1e293b; border-radius: 8px; overflow: hidden;
               border: 1px solid #334155; display: flex; flex-direction: column; }
.city-panel.has-position { border-color: #22c55e; }
.city-panel.has-live { border-color: #f59e0b; }
.city-panel.no-data { opacity: 0.55; }

.city-header { display: flex; justify-content: space-between; align-items: center;
                background: #0f172a; padding: 10px 14px; border-bottom: 1px solid #334155; }
.city-header .name { font-size: 16px; font-weight: 700; color: white; }
.city-header .station { font-family: monospace; color: #94a3b8; font-size: 11px;
                         margin-top: 2px; }
.city-header .now { text-align: right; font-family: monospace; font-size: 11px;
                     color: #94a3b8; }
.city-header .now .temp { color: #fbbf24; font-size: 16px; font-weight: 700;
                           font-family: monospace; }

.city-stats { padding: 8px 14px; background: #1a2438; font-size: 11px;
               color: #94a3b8; display: grid; grid-template-columns: 1fr 1fr 1fr;
               gap: 8px; border-bottom: 1px solid #334155; font-family: monospace; }
.city-stats .v { color: #e2e8f0; font-weight: 600; }
.city-stats .v.hot { color: #fbbf24; }

table.bins { width: 100%; border-collapse: collapse; font-size: 12px; }
table.bins th, table.bins td { padding: 6px 12px; text-align: right;
                                 border-bottom: 1px solid #273449; }
table.bins th { background: #0f172a; color: #64748b; font-weight: 600;
                 font-size: 10px; text-transform: uppercase; letter-spacing: 0.3px; }
table.bins th:first-child, table.bins td:first-child { text-align: left; }
table.bins td.bin-label { font-weight: 700; color: white; font-family: monospace; }
table.bins td.bin-label.top { color: #fbbf24; }
table.bins td.action { text-align: center; }
table.bins td.action .pill { display: inline-block; padding: 1px 6px;
                              border-radius: 10px; font-size: 9px; font-weight: 700; }
.pill.PAPER_BUY { background: #22c55e; color: white; }
.pill.LIVE_BUY  { background: #f59e0b; color: white; }
.pill.RESOLVED  { background: #475569; color: #e2e8f0; }
.pill.idle      { background: #334155; color: #94a3b8; }
.pnl.pos { color: #4ade80; font-weight: 700; }
.pnl.neg { color: #f87171; font-weight: 700; }
.resolved-count { color: #94a3b8; font-size: 11px; margin-left: 4px; }

table.bins tr.bought { background: rgba(34, 197, 94, 0.08); }
table.bins tr.bought.live { background: rgba(245, 158, 11, 0.10); }
table.bins tr.bought.resolved {
  background: rgba(71, 85, 105, 0.18); color: #cbd5e1;
}
table.bins tr.bought.resolved .pnl { opacity: 0.85; }
table.bins tr.top-p { box-shadow: inset 3px 0 0 #fbbf24; }

.city-empty { padding: 24px; text-align: center; color: #64748b; font-size: 12px;
               font-style: italic; }

/* Event sub-section header (only shown when a city has >1 event) */
.event-header { display: flex; justify-content: space-between; align-items: center;
                background: #273449; padding: 6px 14px; font-size: 11px;
                color: #cbd5e1; border-top: 1px solid #334155; }
.event-header .event-date { font-weight: 700; color: white; font-family: monospace; }
.event-header .event-stats { color: #94a3b8; font-family: monospace; }

/* === Signals table === */
.filters { background: #1e293b; padding: 10px 24px; display: flex;
           gap: 16px; flex-wrap: wrap; align-items: center; font-size: 12px;
           border-bottom: 1px solid #334155; position: sticky; top: 0; z-index: 10; }
.filters label { font-weight: 600; color: #cbd5e1; margin-right: 4px; }
.filters select, .filters input { padding: 4px 8px; font-size: 12px;
           background: #0f172a; color: white; border: 1px solid #475569;
           border-radius: 4px; }
.filters .count { color: #94a3b8; font-family: monospace; margin-left: auto; }

table.signals { width: calc(100% - 48px); margin: 12px 24px; background: #1e293b;
                 border-collapse: collapse; border-radius: 6px; overflow: hidden;
                 font-size: 12px; }
table.signals th, table.signals td { padding: 7px 10px; text-align: left;
                                       border-bottom: 1px solid #273449; }
table.signals th { background: #0f172a; color: #94a3b8; font-weight: 600;
                    font-size: 10px; text-transform: uppercase;
                    letter-spacing: 0.5px; cursor: pointer; user-select: none; }
table.signals th:hover { background: #1e293b; color: white; }
table.signals th.sorted-asc::after { content: ' ▲'; color: #818cf8; }
table.signals th.sorted-desc::after { content: ' ▼'; color: #818cf8; }
table.signals td.num { font-family: monospace; text-align: right; }
table.signals td.tstamp { font-family: monospace; font-size: 11px; color: #94a3b8; }

/* City-break: thicker top border when city changes */
table.signals tr.city-break td { border-top: 3px solid #475569; }

table.signals tr.LIVE_BUY  { background: rgba(245, 158, 11, 0.18);
                              border-left: 4px solid #f59e0b; }
table.signals tr.PAPER_BUY { background: rgba(34, 197, 94, 0.10); }
table.signals tr.AVOID     { background: rgba(239, 68, 68, 0.06); color: #94a3b8; }
table.signals tr.SKIP      { color: #64748b; }
table.signals tr:hover     { background: #273449 !important; }

table.signals td .pill {
  display: inline-block; padding: 2px 7px; border-radius: 10px;
  font-size: 10px; font-weight: 700;
}
table.signals .pill.LIVE_BUY  { background: #f59e0b; color: white; }
table.signals .pill.PAPER_BUY { background: #22c55e; color: white; }
table.signals .pill.SKIP      { background: #475569; color: #cbd5e1; }
table.signals .pill.AVOID     { background: #ef4444; color: white; }

.edge { font-family: monospace; font-weight: 700; }
.edge.pos { color: #4ade80; }
.edge.neg { color: #f87171; }

.empty { text-align: center; color: #64748b; padding: 32px 0; font-size: 13px; }

/* Live orders table */
table.live-orders { width: calc(100% - 48px); margin: 8px 24px 16px;
                     background: #1e293b; border-collapse: collapse;
                     border-radius: 6px; overflow: hidden; font-size: 12px; }
table.live-orders th, table.live-orders td { padding: 7px 12px;
                     border-bottom: 1px solid #273449; }
table.live-orders th { background: #0f172a; color: #94a3b8; font-weight: 600;
                       font-size: 10px; text-transform: uppercase; }
.pill.placed { background: #22c55e; color: white; }
.pill.filled { background: #16a34a; color: white; }
.pill.failed { background: #ef4444; color: white; }
.pill.error  { background: #dc2626; color: white; }
.pill.skip   { background: #475569; color: #cbd5e1; }
"""


def build_dashboard(signals: list[dict], live_orders: list[dict],
                     generated_at_utc: str,
                     auto_refresh_sec: int | None = None,
                     live_positions: list[dict] | None = None,
                     analysis: dict | None = None) -> str:
    """Build the full HTML page.  All data filtering / aggregation happens
    in JS so the mode toggle is instant and reuses one data blob."""
    cities_meta = _us_cities()
    sig_json    = json.dumps(signals, default=str, separators=(",", ":"))
    live_json   = json.dumps(live_orders, default=str, separators=(",", ":"))
    cities_json = json.dumps(cities_meta, default=str, separators=(",", ":"))
    positions_json = json.dumps(live_positions or [], default=str,
                                  separators=(",", ":"))
    analysis_json  = json.dumps(analysis or {}, default=str,
                                  separators=(",", ":"))
    # Dashboard's default mode mirrors the bot's PREDICTOR_MODE so a fresh
    # browser (no localStorage) shows what the bot is actually doing.
    # localStorage still wins when present — the user's manual toggle takes
    # priority over the env default.
    default_mode = (os.getenv("PREDICTOR_MODE") or "paper").lower().strip()
    if default_mode not in ("paper", "live", "both"):
        default_mode = "paper"

    # No meta refresh — the JS does a silent fetch + re-render every
    # auto_refresh_sec, preserving sort order, filter state, and scroll
    # position.  No blank flash on update.
    refresh_badge = (f'<span style="color:#94a3b8;font-size:10px">auto-refresh: {auto_refresh_sec}s</span>'
                      if auto_refresh_sec else "")
    refresh_sec_js = int(auto_refresh_sec or 0)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Predictor Dashboard</title>
<style>{DASHBOARD_CSS}</style>
<!-- Chart.js for the city-detail drilldown line charts -->
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
</head><body>

<header>
  <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap">
    <h1>Predictor Dashboard</h1>
    <div class="mode-toggle">
      <button id="mode-paper" class="active paper">Paper</button>
      <button id="mode-live">Live</button>
      <button id="mode-both">Both</button>
    </div>
    <div class="mode-toggle view-toggle">
      <button id="view-signals" class="active view">Signals</button>
      <button id="view-trades" class="view">Trades</button>
      <button id="view-analysis" class="view">Analysis</button>
    </div>
  </div>
  <div class="meta">generated {generated_at_utc} {refresh_badge}<br>
    <span id="hdr-counts">—</span>
  </div>
</header>

<div class="kpis" id="kpis"></div>

<div class="section-title" id="panels-title">Per-city panels</div>
<div class="filters">
  <div><label>Date</label><select id="f-date"></select></div>
  <div><label>City</label><select id="f-city"><option value="">All</option></select></div>
  <div><label>Action</label><select id="f-action">
    <option value="">All</option>
    <option>PAPER_BUY</option><option>LIVE_BUY</option>
    <option>SKIP</option><option>AVOID</option>
  </select></div>
  <div><label>Min edge</label>
    <input id="f-edge" type="number" step="0.05" value="-1" style="width:70px"></div>
  <div><label>Buys only</label>
    <input id="f-buys" type="checkbox"></div>
  <div><label>Latest scan only</label>
    <input id="f-latest" type="checkbox" checked></div>
  <div class="count" id="count">—</div>
</div>
<div class="city-grid" id="city-grid"></div>

<div class="section-title" id="signals-title">All signals</div>

<table class="signals" id="sig-table">
  <thead><tr>
    <th data-key="scanned_at_utc">Scanned (UTC)</th>
    <th data-key="city">City</th>
    <th data-key="bin_label">Bin</th>
    <th data-key="our_prob">Our P</th>
    <th data-key="market_prob">Mkt P</th>
    <th data-key="edge">Edge</th>
    <th data-key="liquidity_usd">Liquidity</th>
    <th data-key="recommended_stake_usd">Stake</th>
    <th data-key="action">Action</th>
    <th data-key="gate_blocked_by">Reason</th>
  </tr></thead>
  <tbody id="sig-tbody"></tbody>
</table>

<table class="signals" id="trades-table" style="display:none">
  <thead><tr>
    <th data-key="scanned_at_utc">Placed (UTC)</th>
    <th data-key="city">City</th>
    <th data-key="bin_label">Bin</th>
    <th data-key="action">Mode</th>
    <th data-key="entry_price">Entry</th>
    <th data-key="current_price">Now</th>
    <th data-key="stake">Stake</th>
    <th data-key="shares">Shares</th>
    <th data-key="current_value">Value</th>
    <th data-key="pnl_usd">P&L $</th>
    <th data-key="pnl_pct">P&L %</th>
    <th data-key="live_status">Order status</th>
    <th data-key="live_order_id">Order ID</th>
  </tr></thead>
  <tbody id="trades-tbody"></tbody>
</table>

<!-- ===================== ANALYSIS TAB ===================== -->
<div id="analysis-section" style="display:none">

  <!-- Headline KPI strip -->
  <div class="section-title">
    Purchases &amp; outcomes
    (last <span id="an-lookback-1">30</span> days)
  </div>

  <!-- In-tab filter row.  Mode comes from the top-level toggle
       (Paper / Live / Both); date here is independent so the operator
       can drill into a single event_date without leaving the tab. -->
  <div class="filters" style="margin-bottom:8px">
    <div>
      <label>Date</label>
      <select id="an-f-date">
        <option value="">All dates (window)</option>
      </select>
    </div>
    <div class="count" id="an-count">--</div>
  </div>

  <div class="kpis" id="an-headline-kpis"></div>

  <!-- THE headline table: one row per purchased bin.  Shows what we
       bought, what the model said at purchase time, and what won. -->
  <div style="overflow-x:auto;margin-bottom:24px">
    <table class="signals" id="an-purchases-table">
      <thead><tr>
        <th>Date</th>
        <th>City</th>
        <th>Purchased (local)</th>
        <th>Mode</th>
        <th>Bought bin</th>
        <th>Side</th>
        <th>Stake</th>
        <th>Entry $</th>
        <th>At-buy forecast high</th>
        <th>At-buy &mu;</th>
        <th>At-buy &sigma;</th>
        <th>Model P</th>
        <th>Mkt P</th>
        <th>Winning bin</th>
        <th>Result</th>
        <th>P&amp;L</th>
      </tr></thead>
      <tbody id="an-purchases-tbody"></tbody>
    </table>
  </div>

  <!-- Compact pipeline health KPI strip (one-glance check) -->
  <div class="section-title">Pipeline (today)</div>
  <div style="color:#94a3b8;font-size:12px;margin-bottom:8px">
    Signals &rarr; orders &rarr; positions chain for today.  Large
    gaps mean writes are dropping somewhere in the pipeline.
  </div>
  <div class="kpis" id="an-pipeline-kpis"></div>
  <div id="an-pipeline-alert"></div>

  <!-- Calibration (kept — useful for tuning the model) -->
  <div class="section-title" style="margin-top:24px">
    Model calibration
  </div>
  <div style="color:#94a3b8;font-size:12px;margin-bottom:8px">
    Bucket closed positions by the model's confidence at buy time.  If
    well-calibrated, avg_model_p &asymp; actual_win_rate within each
    bucket.  Gap colored: green &lt; 5pp, yellow 5-10pp, red &gt;= 10pp.
  </div>
  <table class="signals" id="an-cal-table" style="max-width:780px">
    <thead><tr>
      <th>Bucket</th><th>N</th><th>Avg model_p</th>
      <th>Actual win rate</th><th>Gap</th>
    </tr></thead>
    <tbody id="an-cal-tbody"></tbody>
  </table>

  <!-- Stuck-position alert (only visible when non-empty) -->
  <div id="an-stuck-summary" style="margin-top:24px"></div>
  <table class="signals" id="an-stuck-table" style="display:none">
    <thead><tr>
      <th>Pos</th><th>City</th><th>Event Date</th><th>Side</th>
      <th>Stake</th><th>Entry</th><th>Status</th><th>Fill</th>
      <th>Entry Time</th><th>Days Past</th>
    </tr></thead>
    <tbody id="an-stuck-tbody"></tbody>
  </table>

</div>

<!-- ===================== CITY-DETAIL DRILLDOWN ===================== -->
<!-- Shown when the user clicks a city link in the Analysis table.
     Driven by location.hash = "#detail/<city>/<event_date>". -->
<div id="detail-section" style="display:none;padding:0 24px 24px">

  <div style="margin:18px 0">
    <button id="detail-back"
            style="background:#334155;color:#e2e8f0;border:1px solid #475569;
                   padding:6px 14px;border-radius:6px;cursor:pointer;
                   font-size:12px;font-weight:600">
      &larr; Back to Analysis
    </button>
    <span id="detail-title"
          style="margin-left:14px;font-size:16px;font-weight:600">
    </span>
  </div>

  <!-- Summary card: what we bought, result, P&L -->
  <div class="kpis" id="detail-summary"></div>

  <!-- Chart 1: bought bin -->
  <div class="section-title" style="margin-top:24px">
    Bought bin — <span id="detail-bought-label">--</span>
  </div>
  <div style="color:#94a3b8;font-size:12px;margin-bottom:8px">
    Temperature (left axis) and probabilities (right axis) over the day.
    Dashed line is our model's probability; solid line is the market's
    implied probability.
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:12px;
              margin-bottom:24px">
    <div style="position:relative;height:560px">
      <canvas id="detail-chart-bought"></canvas>
    </div>
  </div>

  <!-- Chart 2: winning bin -->
  <div class="section-title" style="margin-top:24px">
    Winning bin — <span id="detail-winning-label">--</span>
  </div>
  <div style="color:#94a3b8;font-size:12px;margin-bottom:8px">
    Same view as above, anchored on the bin that Polymarket settled YES.
    Useful for comparing "what we bought" vs "what won."
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:12px;
              margin-bottom:24px">
    <div style="position:relative;height:560px">
      <canvas id="detail-chart-winning"></canvas>
    </div>
  </div>

  <!-- Chart 3: final-scan distribution across ALL bins -->
  <div class="section-title" style="margin-top:24px">
    Final-scan distribution (all bins)
  </div>
  <div style="color:#94a3b8;font-size:12px;margin-bottom:8px">
    Our model's probability (blue) vs the market's (red) across every
    bin at the last scan.  Highlighted bars: BOUGHT (yellow ring) /
    WINNING (green ring).  Shows where the model and market diverged
    on the full bin range.
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:12px;
              margin-bottom:24px">
    <div style="position:relative;height:420px">
      <canvas id="detail-chart-distribution"></canvas>
    </div>
  </div>

</div>

<script>
// Mutable so the silent-refresh fetch can swap them in place
let SIGNALS = {sig_json};
let LIVE_ORDERS = {live_json};
let LIVE_POSITIONS = {positions_json};   // from Polymarket data API
let ANALYSIS = {analysis_json};          // Analysis tab data
const CITIES = {cities_json};

// Index live positions by asset (token_id) for O(1) lookup when computing
// LIVE-trade P&L.  Rebuilt on every silent-refresh via recomputeDerived().
let LIVE_POS_BY_TOKEN = {{}};
// Per-contract market-open state.  {{contract_id: {{ts, open}}}}, rebuilt
// every recompute from the freshest signal row's market_closed flag.
// Authoritative for "is this market still tradeable?"
let MARKET_OPEN_BY_CONTRACT = {{}};
const REFRESH_SEC = {refresh_sec_js};
const $ = id => document.getElementById(id);

// ===== Persistent UI state =====
// All toggles + filter values save to localStorage so they survive page
// reloads (manual F5, browser restart, etc.).  Date is NOT persisted —
// always defaults to today so users don't get stuck on yesterday's view.
const LS_KEY = "predictor_dashboard_state_v1";
function loadState() {{
  try {{ return JSON.parse(localStorage.getItem(LS_KEY) || "{{}}"); }}
  catch (e) {{ return {{}}; }}
}}
function saveState(updates) {{
  try {{
    const cur = loadState();
    localStorage.setItem(LS_KEY, JSON.stringify(Object.assign(cur, updates)));
  }} catch (e) {{ /* localStorage may be disabled */ }}
}}
const STATE = loadState();

// Active mode filter for the entire dashboard: 'paper' | 'live' | 'both'
// Resolution order:
//   1. STATE.mode (user's previously-saved choice via localStorage)
//   2. server-injected DEFAULT_MODE (= PREDICTOR_MODE from .env)
//   3. "paper" as ultimate fallback
const DEFAULT_MODE = "{default_mode}";
let MODE_FILTER = STATE.mode || DEFAULT_MODE || "paper";
// Right-side bottom table: 'signals' (all evaluations) or 'trades' (just BUYs
// with entry, current, P&L, live status).
let VIEW_MODE = STATE.view || "signals";

// The "date" filter is the master — KPIs, panels, signals table all
// respect it.  Defaults to today's UTC date but can be changed via the
// dropdown to inspect historical data.
let SELECTED_DATE = "";

// Recomputed on every refresh
let DATE_SIGNALS = [];      // signals matching SELECTED_DATE (scan side)
let DATE_LIVE_ORDERS = [];  // live orders placed on SELECTED_DATE
let LATEST_SCAN_TS = "";    // most-recent scan within SELECTED_DATE

function todayUtcStr() {{
  return new Date().toISOString().slice(0, 10);
}}

function recomputeDerived() {{
  if (!SELECTED_DATE) SELECTED_DATE = todayUtcStr();
  // === Three-signal position model (mirrors scheduled_predictor.py) ===
  //
  // LIVE_POS_BY_TOKEN — every token the wallet currently holds.
  //   NO value/dust filter.  Resolved-to-zero positions ARE kept here.
  //   This answers "did a fill happen?" not "is this active?"
  //
  // Whether a held position is rendered as LIVE vs RESOLVED is decided
  // PER-BIN by the signal row's `market_closed` flag (Gamma's
  // authoritative market resolution state at scan time), NOT by the
  // position's $-value.  See _resolveBuyState() below.
  LIVE_POS_BY_TOKEN = {{}};
  for (const p of (LIVE_POSITIONS || [])) {{
    const tok = p.asset || p.token_id || p.tokenId;
    if (!tok) continue;
    const size = parseFloat(p.size ?? 0);
    if (size <= 0) continue;
    LIVE_POS_BY_TOKEN[String(tok)] = p;
  }}
  // CHANGED: filter by event_date (the resolution day of the Polymarket
  // market), not by scanned_at_utc.  This is semantically what the user
  // means by "show me data for this day" — events resolving that day,
  // regardless of when our bot scanned them.  Also fixes the list-view
  // duplicate-rows issue: when two events for different dates were both
  // scanned today, the list previously showed bins from both.
  DATE_SIGNALS = SIGNALS.filter(s => s.event_date === SELECTED_DATE);
  // Live orders also have event_date, so filter the same way.  Fall
  // back to placed_at_utc match if event_date is missing on the row.
  DATE_LIVE_ORDERS = LIVE_ORDERS.filter(o =>
    (o.event_date && o.event_date === SELECTED_DATE)
    || (!o.event_date && o.placed_at_utc
         && o.placed_at_utc.slice(0, 10) === SELECTED_DATE));
  LATEST_SCAN_TS = DATE_SIGNALS.length
    ? DATE_SIGNALS.reduce((m, s) => s.scanned_at_utc > m ? s.scanned_at_utc : m,
                          DATE_SIGNALS[0].scanned_at_utc)
    : "";

  // MARKET_OPEN_BY_CONTRACT — Gamma's per-bin resolution state as of the
  // latest scan we have for it.  Authoritative source for "is this
  // market still tradeable?"  Read from the signal row's market_closed
  // column (written by scheduled_predictor every scan).
  MARKET_OPEN_BY_CONTRACT = {{}};
  for (const s of DATE_SIGNALS) {{
    if (!s.contract_id) continue;
    const prev = MARKET_OPEN_BY_CONTRACT[s.contract_id];
    if (!prev || s.scanned_at_utc > prev.ts) {{
      MARKET_OPEN_BY_CONTRACT[s.contract_id] =
        {{ ts: s.scanned_at_utc, open: !s.market_closed }};
    }}
  }}
}}
recomputeDerived();

// ===== Mode filter helpers =====
function isBuy(s) {{ return s.action === "PAPER_BUY" || s.action === "LIVE_BUY"; }}
function matchesMode(s) {{
  if (MODE_FILTER === "both") return true;
  if (MODE_FILTER === "live") return s.mode === "live";
  return s.mode === "paper";
}}
function matchesBuyMode(s) {{
  // For BUYs specifically: filter the buy action by current mode toggle
  if (MODE_FILTER === "both") return isBuy(s);
  if (MODE_FILTER === "live") return s.action === "LIVE_BUY";
  return s.action === "PAPER_BUY";
}}

// ===== Per-city aggregation =====
// Within a city, signals can belong to MULTIPLE events (Polymarket sometimes
// has more than one "highest temp" market per city per day).  We group bins
// by event_id and render each event as its own sub-table within the panel.
function buildCitySnapshot(cityMeta) {{
  const city = cityMeta.city;
  // Only signals where event_date matches the selected date — keeps
  // panels strictly scoped to the day the user picked (no future-event
  // bleed from older scans that evaluated tomorrow's markets).
  const todayInCity = DATE_SIGNALS.filter(s =>
    s.city === city && s.event_date === SELECTED_DATE);
  if (todayInCity.length === 0) {{
    return {{...cityMeta, hasData: false}};
  }}
  // Latest scan for this city (all events update together each scan)
  const latestTs = todayInCity.reduce((m, s) =>
    s.scanned_at_utc > m ? s.scanned_at_utc : m, todayInCity[0].scanned_at_utc);
  const latestRows = todayInCity.filter(s => s.scanned_at_utc === latestTs);

  // Map BUYs by contract_id for fast lookup (current mode only)
  const buyRows = todayInCity.filter(matchesBuyMode);
  const buyByContract = {{}};
  for (const b of buyRows) {{
    if (!buyByContract[b.contract_id] ||
        b.scanned_at_utc < buyByContract[b.contract_id].scanned_at_utc) {{
      buyByContract[b.contract_id] = b;
    }}
  }}

  // Group latest scan's rows by event_id
  const eventGroups = {{}};
  for (const r of latestRows) {{
    const eid = r.event_id || 'unknown';
    if (!eventGroups[eid]) {{
      eventGroups[eid] = {{
        event_id:   r.event_id,
        event_date: r.event_date,
        rows:       []
      }};
    }}
    eventGroups[eid].rows.push(r);
  }}

  // For each event group: figure out which bin has the highest our_p
  // (that's the ★ marker), then sort bins by TEMPERATURE for display.
  //
  // Sort by parsing the bin label directly — works regardless of whether
  // bin_range_low is populated, and gives unambiguous order:
  //   "≤69°F"   → -Infinity  (always first — colder than anything)
  //   "76-77°F" → 76         (numeric low end)
  //   "≥88°F"  → Infinity   (always last — hotter than anything)
  function binSortKey(label) {{
    if (!label) return 0;
    if (label.indexOf('≤') === 0 || label.indexOf('<') === 0) return -Infinity;
    if (label.indexOf('≥') === 0 || label.indexOf('>') === 0) return Infinity;
    const m = label.match(/-?\\d+(?:\\.\\d+)?/);
    return m ? parseFloat(m[0]) : 0;
  }}

  const events = Object.values(eventGroups).map(eg => {{
    // Find the top-P bin's contract_id BEFORE sorting (so the ★ marker
    // can be placed at its natural temperature position post-sort).
    const topPRow = eg.rows.reduce((best, r) =>
      r.our_prob > best.our_prob ? r : best, eg.rows[0]);
    const topPContract = topPRow.contract_id;
    // Sort by temperature: cold → hot, ≤ first, ≥ last
    eg.rows.sort((a, b) => binSortKey(a.bin_label) - binSortKey(b.bin_label));
    const bins = eg.rows.map(r => {{
      const buy = buyByContract[r.contract_id];
      let entry_price = null, stake_usd = null, pnl_usd = null;
      let pnl_source = null;     // 'api' (Polymarket) or 'formula' (paper)
      let active_buy = null;     // the BUY to display (null = no pill shown)
      let buy_state = "stale";    // "live" | "resolved" | "stale"
      if (buy) {{
        // Three-state resolution via the shared helper.  No more $-value
        // heuristics — the bot's `market_closed` flag drives "is this
        // still live?" and the wallet position list drives "did we
        // actually fill?"
        buy_state = _resolveBuyState(buy);
        const livePos = (buy.action === 'LIVE_BUY' && r.yes_token_id)
                          ? LIVE_POS_BY_TOKEN[String(r.yes_token_id)]
                          : null;

        if (buy_state === "live" || buy_state === "resolved") {{
          active_buy = buy;
          stake_usd = buy.recommended_stake_usd || 0;

          if (livePos) {{
            // LIVE_BUY with a wallet position (live OR resolved) → use
            // Polymarket's authoritative numbers (size × avgPrice, real
            // cashPnl).  For resolved positions, cashPnl reflects the
            // FINAL P&L since cur_price has settled to 1.0 or 0.0.
            const livSize = parseFloat(livePos.size ?? 0);
            const livAvg  = parseFloat(livePos.avgPrice ?? livePos.avg_price ?? 0);
            if (livSize > 0 && livAvg > 0) {{
              stake_usd   = livSize * livAvg;
              entry_price = livAvg;
            }} else {{
              entry_price = parseFloat(livePos.avgPrice ?? livePos.avg_price ?? buy.market_prob);
            }}
            const cashPnl = parseFloat(livePos.cashPnl ?? livePos.cash_pnl ?? NaN);
            if (!isNaN(cashPnl)) {{
              pnl_usd = cashPnl;
              pnl_source = 'api';
            }} else if (entry_price > 0) {{
              const curPx = parseFloat(livePos.curPrice ?? livePos.cur_price ?? r.market_prob);
              const size  = parseFloat(livePos.size ?? 0);
              pnl_usd = size * (curPx - entry_price);
              pnl_source = 'api';
            }}
          }} else {{
            // PAPER_BUY — use formula
            entry_price = buy.market_prob;
            if (entry_price > 0) {{
              pnl_usd = stake_usd * ((r.market_prob / entry_price) - 1);
              pnl_source = 'formula';
            }}
          }}
        }} else {{
          // "stale" — LIVE_BUY signal exists but no wallet position.
          // Order failed/cancelled/never filled.  Don't render as a
          // position; trades view shows the attempt for forensics.
          stake_usd = 0;
          entry_price = null;
          pnl_usd = null;
        }}
      }}
      return {{
        bin:          r.bin_label,
        our_prob:     r.our_prob,
        market_prob:  r.market_prob,
        edge:         r.edge,
        // action is null for stale LIVE buys (no pill displayed)
        action:       active_buy ? active_buy.action : null,
        buy_state:    buy_state,     // "live" | "resolved" | "stale"
        entry_price:  entry_price,
        stake_usd:    stake_usd,
        pnl_usd:      pnl_usd,
        pnl_source:   pnl_source,
        // ★ marks the bin with the highest our_p (the one we'd buy first)
        is_top_p:     r.contract_id === topPContract,
      }};
    }});
    return {{
      event_id:      eg.event_id,
      event_date:    eg.event_date,
      bins:          bins,
      // Count "live" only.  Resolved positions are shown in the panel
      // (so the user can see the final P&L) but they don't occupy a
      // slot or count toward "active BUYs."
      nBuys:         bins.filter(b => b.buy_state === "live").length,
      nLiveBuys:     bins.filter(b => b.action === "LIVE_BUY"
                                       && b.buy_state === "live").length,
      nResolved:     bins.filter(b => b.buy_state === "resolved").length,
      totalDeployed: bins.filter(b => b.buy_state === "live")
                          .reduce((s, b) => s + (b.stake_usd || 0), 0),
      totalPnl:      bins.reduce((s, b) => s + (b.pnl_usd || 0), 0),
    }};
  }});

  // Sort events: today first, then by date
  events.sort((a, b) => (a.event_date || '').localeCompare(b.event_date || ''));

  // City-level rollups across all events
  const anyRow = latestRows[0];
  return {{
    ...cityMeta,
    hasData:          true,
    events:           events,
    eventCount:       events.length,
    observedMax:      anyRow.observed_max_c,
    observedPeakHour: anyRow.observed_peak_hour,
    forecastHigh:     anyRow.forecast_high_c,
    forecastPeakHour: anyRow.forecast_peak_hour,
    windOctant:       anyRow.wind_octant,
    nBuys:            events.reduce((s, e) => s + e.nBuys, 0),
    nLiveBuys:        events.reduce((s, e) => s + e.nLiveBuys, 0),
    nResolved:        events.reduce((s, e) => s + (e.nResolved || 0), 0),
    totalDeployed:    events.reduce((s, e) => s + e.totalDeployed, 0),
    totalPnl:         events.reduce((s, e) => s + e.totalPnl, 0),
  }};
}}

// ===== KPI rendering =====
// Resolve a BUY signal row to one of three explicit states.  These are
// the SAME three signals the bot uses internally — keeping them
// separated here means the dashboard never has to use a $-value proxy
// to decide "is this real?"
//
//   "live"     → held + market still open.  Show LIVE pill, count in KPI.
//   "resolved" → held + market settled.  Show RESOLVED badge, exclude
//                  from "live BUYs" KPI but show in trades history with
//                  final P&L.
//   "stale"    → not held at all (failed/cancelled/never filled).
//                  Don't render as a position; the LIVE order log still
//                  has the attempt for forensics.
function _resolveBuyState(s) {{
  if (!s) return "stale";
  if (s.action === "PAPER_BUY") {{
    // Paper buys "fill" by writing the DB row.  Market still has to be
    // open for them to count as live (paper trading a closed market is
    // meaningless).  Use the latest known per-contract market_closed
    // flag; if we have no signal row for this contract yet, assume open.
    const mo = MARKET_OPEN_BY_CONTRACT[s.contract_id];
    return (!mo || mo.open) ? "live" : "resolved";
  }}
  if (s.action === "LIVE_BUY") {{
    const held = s.yes_token_id
              && LIVE_POS_BY_TOKEN[String(s.yes_token_id)] != null;
    if (!held) return "stale";
    const mo = MARKET_OPEN_BY_CONTRACT[s.contract_id];
    return (!mo || mo.open) ? "live" : "resolved";
  }}
  return "stale";
}}
function _isActiveBuy(s) {{
  // "Active" = countable in the live-BUYs KPI.  Resolved positions are
  // no longer countable (the market is settled, the bet is over).
  return _resolveBuyState(s) === "live";
}}

// Returns actual deployed $ for a single BUY row (API for live, signal for paper).
function _actualDeployedForBuy(s) {{
  if (s.action === "LIVE_BUY" && s.yes_token_id) {{
    const livePos = LIVE_POS_BY_TOKEN[String(s.yes_token_id)];
    if (livePos) {{
      const size = parseFloat(livePos.size ?? 0);
      const avg  = parseFloat(livePos.avgPrice ?? livePos.avg_price ?? 0);
      if (size > 0 && avg > 0) return size * avg;
    }}
    return 0;   // LIVE with no API match → not deployed
  }}
  return s.recommended_stake_usd || 0;
}}

// Reduce DATE_SIGNALS to one BUY row PER CONTRACT, filtered by buy state.
// A single position can produce many LIVE_BUY rows (original purchase
// + multiple topups), but the user has only one position per contract.
//
//   acceptStates: set of "live" | "resolved" | "stale" to include.
//     - KPI "live BUYs"     → {{"live"}}
//     - Trades view         → {{"live", "resolved"}}   (show final P&L)
//     - Forensics (future)  → {{"live", "resolved", "stale"}}
function _uniqueBuysByState(signals, acceptStates) {{
  const latestByContract = {{}};
  for (const s of signals) {{
    if (!matchesBuyMode(s)) continue;
    if (!acceptStates.has(_resolveBuyState(s))) continue;
    const k = s.contract_id;
    if (!latestByContract[k]
        || (s.scanned_at_utc || "") > (latestByContract[k].scanned_at_utc || "")) {{
      latestByContract[k] = s;
    }}
  }}
  return Object.values(latestByContract);
}}
function _uniqueActiveBuys(signals) {{
  return _uniqueBuysByState(signals, new Set(["live"]));
}}
function _uniqueHeldBuys(signals) {{
  return _uniqueBuysByState(signals, new Set(["live", "resolved"]));
}}

function renderKPIs() {{
  // Filter to CURRENTLY-ACTIVE buys, ONE PER CONTRACT.  The bot writes
  // a new LIVE_BUY signal row per topup, so a single position can have
  // 3-5 LIVE_BUY rows in the DB.  Counting rows here would massively
  // inflate "n BUYs" — we want to count distinct held contracts, which
  // is what the user sees on Polymarket.
  const buys = _uniqueActiveBuys(DATE_SIGNALS);
  const nBuys = buys.length;
  // Use the same active/inactive distinction for deployed $ — for active
  // LIVE buys, sum the API's actual size × avgPrice (matches Polymarket
  // "Traded").  For PAPER, use the recommended_stake_usd.
  const deployed = buys.reduce((s, b) => s + _actualDeployedForBuy(b), 0);

  // Compute total P&L across all buys in current mode
  let totalPnl = 0;
  for (const city of CITIES) {{
    const snap = buildCitySnapshot(city);
    totalPnl += snap.totalPnl || 0;
  }}

  const avgEdge = nBuys ? buys.reduce((s, b) => s + b.edge, 0) / nBuys : 0;
  const nSkips  = DATE_SIGNALS.filter(s => s.action === "SKIP" && matchesMode(s)).length;

  const pnlClass = totalPnl >= 0 ? "pos" : "neg";
  const pnlSign  = totalPnl >= 0 ? "+" : "";
  const buyClass = MODE_FILTER === "live" ? "live" : "buy";
  const buyLabel = MODE_FILTER === "live" ? "LIVE BUYs"
                : MODE_FILTER === "both" ? "Total BUYs"
                : "PAPER BUYs";

  $("kpis").innerHTML = `
    <div class="kpi ${{buyClass}}"><div class="label">${{buyLabel}}</div>
      <div class="val">${{nBuys}}</div>
      <div class="sub">$${{deployed.toFixed(2)}} deployed</div></div>
    <div class="kpi ${{pnlClass}}"><div class="label">Total P&L</div>
      <div class="val">${{pnlSign}}$${{totalPnl.toFixed(2)}}</div>
      <div class="sub">${{deployed > 0 ? (pnlSign + (100*totalPnl/deployed).toFixed(1) + '%') : '—'}} ROI</div></div>
    <div class="kpi"><div class="label">Avg edge on BUYs</div>
      <div class="val">${{nBuys ? (avgEdge >= 0 ? '+' : '') + (avgEdge*100).toFixed(1) + '%' : '—'}}</div></div>
    <div class="kpi"><div class="label">Skipped today</div>
      <div class="val">${{nSkips.toLocaleString()}}</div></div>
    <div class="kpi"><div class="label">Scanned today</div>
      <div class="val">${{DATE_SIGNALS.filter(matchesMode).length.toLocaleString()}}</div>
      <div class="sub">${{DATE_SIGNALS.length ? 'last: ' + LATEST_SCAN_TS.slice(11,16) + 'Z' : ''}}</div></div>
  `;
  $("hdr-counts").textContent =
    `${{nBuys}} BUYs · $${{deployed.toFixed(0)}} dep · ${{pnlSign}}$${{totalPnl.toFixed(2)}} P&L`;
}}

// ===== City panel rendering =====
function cToF(c) {{ return c * 9/5 + 32; }}

function renderBinRow(b) {{
  const edgeClass = b.edge >= 0 ? "edge pos" : "edge neg";
  const edgeStr = (b.edge >= 0 ? "+" : "") + (b.edge*100).toFixed(1) + "%";
  // Action pill: LIVE / PAPER for active positions, RESOLVED for held
  // positions whose market has settled, dash for idle bins.
  let action;
  if (b.buy_state === "resolved") {{
    action = `<span class="pill RESOLVED" title="Market has settled — final P&L">RESOLVED</span>`;
  }} else if (b.action) {{
    action = `<span class="pill ${{b.action}}">${{b.action === 'LIVE_BUY' ? 'LIVE' : 'PAPER'}}</span>`;
  }} else {{
    action = '<span class="pill idle">—</span>';
  }}
  const entry = b.entry_price !== null ? '$' + b.entry_price.toFixed(3) : '';
  const pnl = b.pnl_usd !== null
    ? `<span class="pnl ${{b.pnl_usd >= 0 ? 'pos' : 'neg'}}" title="${{b.pnl_source === 'api' ? 'from Polymarket API' : 'paper formula: stake × (current/entry − 1)'}}">${{b.pnl_usd >= 0 ? '+' : ''}}$${{b.pnl_usd.toFixed(2)}}${{b.pnl_source === 'api' ? '<sup style="font-size:8px;opacity:0.7">API</sup>' : ''}}</span>`
    : '';
  let rowCls = '';
  if (b.buy_state === "resolved") rowCls = 'bought resolved';
  else if (b.action === 'LIVE_BUY') rowCls = 'bought live';
  else if (b.action === 'PAPER_BUY') rowCls = 'bought';
  if (b.is_top_p) rowCls += ' top-p';
  return `<tr class="${{rowCls}}">
    <td class="bin-label ${{b.is_top_p ? 'top' : ''}}">${{b.is_top_p ? '★ ' : ''}}${{b.bin}}</td>
    <td class="num">${{(b.our_prob*100).toFixed(1)}}%</td>
    <td class="num">${{(b.market_prob*100).toFixed(1)}}%</td>
    <td class="num ${{edgeClass}}">${{edgeStr}}</td>
    <td class="action">${{action}}</td>
    <td class="num">${{entry}}</td>
    <td class="num">${{pnl}}</td>
  </tr>`;
}}

function renderEventSection(ev, showHeader) {{
  const binsHtml = ev.bins.map(renderBinRow).join('');
  const header = showHeader
    ? `<div class="event-header">
         <span class="event-date">Event ${{ev.event_date}}</span>
         <span class="event-stats">${{ev.bins.length}} bins
           ${{ev.nBuys ? '· ' + ev.nBuys + ' buy' + (ev.nBuys === 1 ? '' : 's') : ''}}
           ${{ev.totalDeployed > 0 ? '· $' + ev.totalDeployed.toFixed(2) + ' deployed' : ''}}
           ${{ev.totalPnl !== 0 ? '· P&L ' + (ev.totalPnl >= 0 ? '+' : '') + '$' + ev.totalPnl.toFixed(2) : ''}}
         </span>
       </div>`
    : '';
  return `${{header}}<table class="bins">
    <thead><tr>
      <th>Bin</th><th>Our P</th><th>Mkt P</th><th>Edge</th>
      <th>Action</th><th>Entry</th><th>P&L</th>
    </tr></thead>
    <tbody>${{binsHtml}}</tbody>
  </table>`;
}}

function renderCityPanel(snap) {{
  if (!snap.hasData) {{
    return `<div class="city-panel no-data">
      <div class="city-header">
        <div><div class="name">${{snap.city}}</div>
          <div class="station">${{snap.station}} · ${{snap.tz_label}}</div></div>
        <div class="now"><span style="color:#64748b">—</span></div>
      </div>
      <div class="city-empty">No scan data for today yet</div>
    </div>`;
  }}

  let cls = "city-panel";
  if (snap.nLiveBuys > 0) cls += " has-live";
  else if (snap.nBuys > 0) cls += " has-position";

  const obsClass = (snap.observedMax !== null && snap.forecastHigh !== null
                     && snap.observedMax >= snap.forecastHigh - 0.5) ? "hot" : "";

  // Header temperature: show both °C and °F
  let tempDisplay = '—';
  if (snap.observedMax !== null && snap.observedMax !== undefined) {{
    tempDisplay = `${{snap.observedMax.toFixed(1)}}°C
      <span style="color:#94a3b8;font-size:11px">/ ${{cToF(snap.observedMax).toFixed(1)}}°F</span>`;
  }}

  // Forecast high — also show both units
  let fcstDisplay = '—';
  if (snap.forecastHigh !== null && snap.forecastHigh !== undefined) {{
    fcstDisplay = `${{snap.forecastHigh.toFixed(1)}}°C /
      ${{cToF(snap.forecastHigh).toFixed(1)}}°F @ ${{snap.forecastPeakHour}}:00`;
  }}

  // Render each event as its own sub-table.  Hide event headers if only one
  // event (which is the common case — looks cleaner without them).
  const showEventHeaders = snap.events.length > 1;
  const eventSections = snap.events.map(ev =>
    renderEventSection(ev, showEventHeaders)).join('');

  return `<div class="${{cls}}">
    <div class="city-header">
      <div>
        <div class="name">${{snap.city}}</div>
        <div class="station">${{snap.station}} · ${{snap.tz_label}}${{snap.windOctant ? ' · wind ' + snap.windOctant : ''}}</div>
      </div>
      <div class="now">
        <div class="temp ${{obsClass}}">${{tempDisplay}}</div>
        <div>observed @ ${{snap.observedPeakHour}}:00 local</div>
      </div>
    </div>
    <div class="city-stats">
      <div>Forecast high: <span class="v">${{fcstDisplay}}</span></div>
      <div>Events: <span class="v">${{snap.eventCount}}</span></div>
      <div>Position(s): <span class="v">${{snap.nBuys || 0}}</span>
        ${{snap.nResolved > 0 ? '<span class="resolved-count" title="positions whose market has settled">+ ' + snap.nResolved + ' resolved</span>' : ''}}
        ${{snap.totalDeployed > 0 ? '· $' + snap.totalDeployed.toFixed(2) + ' deployed' : ''}}
        ${{snap.totalPnl !== 0 ? '· P&L <span class="' + (snap.totalPnl >= 0 ? 'pnl pos' : 'pnl neg') + '">' + (snap.totalPnl >= 0 ? '+' : '') + '$' + snap.totalPnl.toFixed(2) + '</span>' : ''}}
      </div>
    </div>
    ${{eventSections}}
  </div>`;
}}

function renderCityPanels() {{
  // City filter narrows which panels are shown.  Date filter (SELECTED_DATE)
  // is applied inside buildCitySnapshot.
  const cityFilter = $("f-city").value;
  const cities = cityFilter ? CITIES.filter(c => c.city === cityFilter) : CITIES;
  const html = cities.map(c => renderCityPanel(buildCitySnapshot(c))).join('');
  $("city-grid").innerHTML = html ||
    '<div class="empty">No matching cities — try changing the date or city filter</div>';
}}

// (renderLiveOrders removed — live order info now appears in the Trades
//  view's Status / Order ID / Error columns.)

// ===== Signals table =====
let SORT_KEY = "scanned_at_utc", SORT_DIR = -1;

function sigRow(s, addCityBreak) {{
  const eClass = s.edge >= 0 ? "pos" : "neg";
  const eStr = (s.edge >= 0 ? "+" : "") + (s.edge*100).toFixed(1) + "%";
  const stake = s.recommended_stake_usd ? "$" + s.recommended_stake_usd.toFixed(2) : "—";
  const gate = s.gate_blocked_by || "";
  let cls = s.action;
  if (addCityBreak) cls += " city-break";
  return `<tr class="${{cls}}">
    <td class="tstamp">${{s.scanned_at_utc ? s.scanned_at_utc.slice(0,16).replace('T',' ') : ''}}</td>
    <td><b>${{s.city}}</b><br><span style="color:#64748b;font-size:10px">${{s.settlement_station || ''}}</span></td>
    <td><b>${{s.bin_label || ''}}</b></td>
    <td class="num">${{(s.our_prob*100).toFixed(1)}}%</td>
    <td class="num">${{(s.market_prob*100).toFixed(1)}}%</td>
    <td class="num edge ${{eClass}}">${{eStr}}</td>
    <td class="num">$${{Math.round(s.liquidity_usd||0).toLocaleString()}}</td>
    <td class="num">${{stake}}</td>
    <td><span class="pill ${{s.action}}">${{s.action}}</span></td>
    <td style="color:#94a3b8;font-size:11px">${{gate}}</td>
  </tr>`;
}}

function renderSigTable() {{
  const city = $("f-city").value;
  const act  = $("f-action").value;
  const me   = parseFloat($("f-edge").value);
  const buys = $("f-buys").checked;
  const latest = $("f-latest").checked;
  let rows = DATE_SIGNALS.filter(s =>
       matchesMode(s)
    && (!latest || s.scanned_at_utc === LATEST_SCAN_TS)
    && (!city || s.city === city)
    && (!act  || s.action === act)
    && (isNaN(me) || s.edge >= me)
    && (!buys || isBuy(s))
  );
  rows.sort((a, b) => {{
    let av = a[SORT_KEY], bv = b[SORT_KEY];
    if (typeof av === "number") return SORT_DIR * (av - bv);
    return SORT_DIR * String(av || '').localeCompare(String(bv || ''));
  }});
  rows = rows.slice(0, 500);
  $("count").textContent = rows.length + " / " + DATE_SIGNALS.length;

  // Add city-break visual divider when city changes between rows
  let html = '';
  let lastCity = null;
  for (const r of rows) {{
    const breakNow = lastCity !== null && lastCity !== r.city;
    html += sigRow(r, breakNow);
    lastCity = r.city;
  }}
  $("sig-tbody").innerHTML = html ||
    '<tr><td colspan="10" class="empty">No signals match filters</td></tr>';
  document.querySelectorAll("th").forEach(th => {{
    th.classList.remove("sorted-asc","sorted-desc");
    if (th.dataset.key === SORT_KEY)
      th.classList.add(SORT_DIR > 0 ? "sorted-asc" : "sorted-desc");
  }});
}}

// ===== Date + city dropdown population (refreshed on every render) =====
function refreshDateDropdown() {{
  const dates = [...new Set(SIGNALS.map(s => s.event_date).filter(Boolean))]
    .sort().reverse();
  const sel = $("f-date");
  const today = todayUtcStr();
  // Preserve user's current selection if it's still in the new data
  const currentVal = SELECTED_DATE;
  const fallback = dates.includes(today) ? today : (dates[0] || today);
  sel.innerHTML = dates.map(d =>
    `<option value="${{d}}">${{d}}${{d === today ? ' (today)' : ''}}</option>`
  ).join('') || `<option value="${{today}}">${{today}} (today, no data yet)</option>`;
  const chosen = dates.includes(currentVal) ? currentVal : fallback;
  sel.value = chosen;
  SELECTED_DATE = chosen;
}}

function refreshCityDropdown() {{
  // Cities available on the selected date (preserve current selection)
  const citiesInData = [...new Set(DATE_SIGNALS.map(s => s.city).filter(Boolean))].sort();
  const sel = $("f-city");
  const currentVal = sel.value;
  sel.innerHTML = '<option value="">All</option>'
    + citiesInData.map(c => `<option>${{c}}</option>`).join('');
  if (citiesInData.includes(currentVal)) sel.value = currentVal;
}}

function updateSectionTitles() {{
  const dateLabel = SELECTED_DATE === todayUtcStr() ? '(today)' : `(${{SELECTED_DATE}})`;
  $("panels-title").textContent = `Per-city panels ${{dateLabel}}`;
  $("signals-title").textContent =
    (VIEW_MODE === 'trades' ? 'Trades ' : 'All signals ') + dateLabel;
}}

// Sort handlers
document.querySelectorAll("th").forEach(th => {{
  th.addEventListener("click", () => {{
    if (!th.dataset.key) return;
    if (SORT_KEY === th.dataset.key) SORT_DIR = -SORT_DIR;
    else {{ SORT_KEY = th.dataset.key; SORT_DIR = 1; }}
    renderSigTable();
  }});
}});
// ===== Trades view (paper + live BUYs with full lifecycle info) =====
let TRADES_SORT_KEY = "scanned_at_utc", TRADES_SORT_DIR = -1;

function buildTradesData() {{
  // HELD positions in the current date range, deduped to one per
  // contract.  Includes both live AND resolved positions so the user
  // can see final P&L on settled markets.  Stale BUY rows
  // (placed/cancelled/never filled) are excluded — they're forensics,
  // not trades.
  const buys = _uniqueHeldBuys(DATE_SIGNALS);
  // Latest known market price per contract (any scan in this date)
  const latestByContract = {{}};
  for (const s of DATE_SIGNALS) {{
    if (!latestByContract[s.contract_id]
        || s.scanned_at_utc > latestByContract[s.contract_id].scanned_at_utc) {{
      latestByContract[s.contract_id] = s;
    }}
  }}
  // Live order info per contract (most-recent placed order for that bin)
  const liveByContract = {{}};
  for (const o of DATE_LIVE_ORDERS) {{
    const k = o.contract_id;
    if (!liveByContract[k]
        || (o.placed_at_utc || "") > (liveByContract[k].placed_at_utc || "")) {{
      liveByContract[k] = o;
    }}
  }}
  return buys.map(b => {{
    const latest = latestByContract[b.contract_id] || b;
    const live   = b.action === "LIVE_BUY" ? liveByContract[b.contract_id] : null;
    // For LIVE: use Polymarket's actual avg fill price + actual deployed.
    // For PAPER: use signal-time market_prob and intended stake.
    const livePos = (b.action === "LIVE_BUY" && b.yes_token_id)
                    ? LIVE_POS_BY_TOKEN[String(b.yes_token_id)] : null;
    const entry  = livePos
                   ? parseFloat(livePos.avgPrice ?? livePos.avg_price ?? b.market_prob)
                   : b.market_prob;
    const cur    = latest.market_prob;
    const stake  = _actualDeployedForBuy(b);
    const shares = entry > 0 ? stake / entry : 0;
    const value  = shares * cur;
    const pnl    = value - stake;
    const pct    = entry > 0 ? (cur / entry - 1) : 0;
    return {{
      scanned_at_utc: b.scanned_at_utc,
      city:           b.city,
      station:        b.settlement_station,
      bin_label:      b.bin_label,
      action:         b.action,
      buy_state:      _resolveBuyState(b),
      entry_price:    entry,
      current_price:  cur,
      stake:          stake,
      shares:         shares,
      current_value:  value,
      pnl_usd:        pnl,
      pnl_pct:        pct,
      live_status:    live ? (live.status || "") : "",
      live_order_id:  live ? (live.order_id || "") : "",
      live_error:     live ? (live.error || "") : "",
    }};
  }});
}}

function tradeRow(t, addCityBreak) {{
  const modeCls = t.action;
  const modeLabel = t.action === "LIVE_BUY" ? "LIVE" : "PAPER";
  const pnlCls = t.pnl_usd >= 0 ? "pos" : "neg";
  const pnlSign = t.pnl_usd >= 0 ? "+" : "";
  let cls = modeCls + (addCityBreak ? " city-break" : "");
  if (t.buy_state === "resolved") cls += " resolved";
  // Show RESOLVED badge alongside the LIVE/PAPER label so the user can
  // tell at a glance which trades have settled vs. which are still moving.
  const stateBadge = t.buy_state === "resolved"
    ? ` <span class="pill RESOLVED" title="Market has settled">RESOLVED</span>`
    : "";
  return `<tr class="${{cls}}">
    <td class="tstamp">${{(t.scanned_at_utc || "").slice(0,16).replace("T"," ")}}</td>
    <td><b>${{t.city}}</b><br><span style="color:#64748b;font-size:10px">${{t.station || ""}}</span></td>
    <td><b>${{t.bin_label || ""}}</b></td>
    <td><span class="pill ${{modeCls}}">${{modeLabel}}</span>${{stateBadge}}</td>
    <td class="num">$${{t.entry_price.toFixed(3)}}</td>
    <td class="num">$${{t.current_price.toFixed(3)}}</td>
    <td class="num">$${{t.stake.toFixed(2)}}</td>
    <td class="num">${{t.shares.toFixed(2)}}</td>
    <td class="num">$${{t.current_value.toFixed(2)}}</td>
    <td class="num edge ${{pnlCls}}">${{pnlSign}}$${{t.pnl_usd.toFixed(2)}}</td>
    <td class="num edge ${{pnlCls}}">${{pnlSign}}${{(t.pnl_pct*100).toFixed(1)}}%</td>
    <td><span class="pill ${{(t.live_status || "").toLowerCase()}}">${{(t.live_status || "—").toUpperCase()}}</span></td>
    <td class="tstamp">${{(t.live_order_id || "").slice(0,14)}}</td>
  </tr>`;
}}

function renderTradesTable() {{
  const cityFilter = $("f-city").value;
  let trades = buildTradesData().filter(t =>
    matchesBuyMode({{action: t.action, mode: t.action === "LIVE_BUY" ? "live" : "paper"}})
    && (!cityFilter || t.city === cityFilter)
  );
  trades.sort((a, b) => {{
    let av = a[TRADES_SORT_KEY], bv = b[TRADES_SORT_KEY];
    if (typeof av === "number") return TRADES_SORT_DIR * (av - bv);
    return TRADES_SORT_DIR * String(av || "").localeCompare(String(bv || ""));
  }});
  trades = trades.slice(0, 500);
  $("count").textContent = trades.length + " trades";
  let html = "";
  let lastCity = null;
  for (const t of trades) {{
    const breakNow = lastCity !== null && lastCity !== t.city;
    html += tradeRow(t, breakNow);
    lastCity = t.city;
  }}
  $("trades-tbody").innerHTML = html ||
    '<tr><td colspan="13" class="empty">No trades match filters</td></tr>';
  // Update sort indicator on the trades table headers
  document.querySelectorAll("#trades-table th").forEach(th => {{
    th.classList.remove("sorted-asc","sorted-desc");
    if (th.dataset.key === TRADES_SORT_KEY)
      th.classList.add(TRADES_SORT_DIR > 0 ? "sorted-asc" : "sorted-desc");
  }});
}}

// Wire trades-table column sort handlers (same pattern as signals table)
document.querySelectorAll("#trades-table th").forEach(th => {{
  th.addEventListener("click", () => {{
    if (!th.dataset.key) return;
    if (TRADES_SORT_KEY === th.dataset.key) TRADES_SORT_DIR = -TRADES_SORT_DIR;
    else {{ TRADES_SORT_KEY = th.dataset.key; TRADES_SORT_DIR = 1; }}
    renderTradesTable();
  }});
}});

// ===== View toggle (Signals / Trades / Analysis) =====
function setView(v) {{
  VIEW_MODE = v;
  saveState({{view: v}});
  ["signals", "trades", "analysis"].forEach(x => {{
    const btn = $("view-" + x);
    if (btn) btn.classList.toggle("active", x === v);
  }});
  // Show the right view, hide the others
  $("sig-table").style.display       = (v === "signals"  ? "" : "none");
  $("trades-table").style.display    = (v === "trades"   ? "" : "none");
  const an = $("analysis-section");
  if (an) an.style.display           = (v === "analysis" ? "" : "none");
  // Exit the detail drilldown when switching primary views — otherwise
  // detail stays open over the new tab.  The hash gets cleared so
  // hashchange doesn't loop us back in.
  const ds = $("detail-section");
  if (ds && ds.style.display !== "none") {{
    ds.style.display = "none";
    _destroyDetailCharts();
    if (location.hash.startsWith("#detail/")) {{
      history.pushState("", document.title,
                          location.pathname + location.search);
    }}
  }}
  // Analysis tab hides the per-city panel + filters (they don't apply
  // to the historical-analysis view).
  const panelsTitle  = $("panels-title");
  const filters      = document.querySelector(".filters");
  const cityGrid     = $("city-grid");
  const signalsTitle = $("signals-title");
  const hideForAnalysis = (v === "analysis");
  if (panelsTitle)  panelsTitle.style.display  = hideForAnalysis ? "none" : "";
  if (filters)      filters.style.display      = hideForAnalysis ? "none" : "";
  if (cityGrid)     cityGrid.style.display     = hideForAnalysis ? "none" : "";
  if (signalsTitle) signalsTitle.style.display = hideForAnalysis ? "none" : "";
  renderAll();
}}
$("view-signals").addEventListener("click",  () => setView("signals"));
$("view-trades").addEventListener("click",   () => setView("trades"));
$("view-analysis").addEventListener("click", () => setView("analysis"));

// Belt-and-suspenders dispatch for the Analysis tab.  setView/setMode
// route through renderAll() which calls a chain of upstream functions
// (refreshDateDropdown, recomputeDerived, refreshCityDropdown,
// updateSectionTitles, renderKPIs, renderCityPanels) BEFORE reaching
// renderAnalysis.  If any of those throws, renderAnalysis is silently
// skipped.  We listen on the mode/view buttons directly and call
// renderAnalysis ourselves to make the Analysis tab independent of
// the rest of the dashboard's render path.
function _maybeRenderAnalysis() {{
  if (VIEW_MODE === "analysis") {{
    try {{ renderAnalysis(); }}
    catch (e) {{ console.error("renderAnalysis raised:", e); }}
  }}
}}
["mode-paper","mode-live","mode-both",
   "view-signals","view-trades","view-analysis"].forEach(id => {{
  const btn = $(id);
  if (btn) btn.addEventListener("click", () => {{
    // Defer so setMode/setView finish first (they fire on the same
    // click event before this handler).  setTimeout 0 puts us at the
    // end of the microtask queue.
    setTimeout(_maybeRenderAnalysis, 0);
  }});
}});

// In-tab date filter change handler.
const _anDateSel = $("an-f-date");
if (_anDateSel) {{
  _anDateSel.addEventListener("change", () => {{
    AN_DATE_FILTER = _anDateSel.value || "";
    _maybeRenderAnalysis();
  }});
}}

// ===== Analysis render =====
function fmtNum(v, places=2, sign=false) {{
  if (v == null || v === "" || isNaN(v)) return "--";
  const n = Number(v);
  const s = sign && n > 0 ? "+" : "";
  return s + n.toFixed(places);
}}
function fmtPct(v, places=1, sign=false) {{
  if (v == null || v === "" || isNaN(v)) return "--";
  const n = Number(v);
  const s = sign && n > 0 ? "+" : "";
  return s + n.toFixed(places) + "%";
}}
function fmtMoney(v, sign=false) {{
  if (v == null || v === "" || isNaN(v)) return "--";
  const n = Number(v);
  const sgn = sign && n > 0 ? "+" : (n < 0 ? "-" : "");
  return sgn + "$" + Math.abs(n).toFixed(2);
}}
function fmtDateShort(d) {{
  if (!d) return "--";
  const s = String(d);
  if (s.length >= 10 && s[4] === "-") {{
    return s.slice(5, 10) + " (" + s.slice(0, 4) + ")";
  }}
  return s;
}}
function pnlColor(v) {{
  if (v == null || isNaN(v)) return "";
  return Number(v) >= 0 ? "color:#22c55e" : "color:#ef4444";
}}
function gapColor(v) {{
  if (v == null || isNaN(v)) return "";
  const m = Math.abs(Number(v));
  if (m >= 0.10) return "color:#ef4444;font-weight:600";
  if (m >= 0.05) return "color:#f59e0b";
  return "color:#22c55e";
}}

// Temperature helpers — convert Celsius -> Fahrenheit and format with
// the right symbol so model context (always in C) lines up with bin
// labels (F for US, C for EU).
function cToF(c) {{
  if (c == null || isNaN(c)) return null;
  return Number(c) * 9 / 5 + 32;
}}
function fmtTempInUnit(temp_c, unit, places=1) {{
  if (temp_c == null || isNaN(temp_c)) return "--";
  const isF = (String(unit||"").toLowerCase() === "fahrenheit");
  const v = isF ? cToF(temp_c) : Number(temp_c);
  return v.toFixed(places) + "°" + (isF ? "F" : "C");
}}
function fmtBinLabel(lo, hi, unit) {{
  if (lo == null || hi == null) return "--";
  const isF = (String(unit||"").toLowerCase() === "fahrenheit");
  const sym = isF ? "°F" : "°C";
  return `${{Math.round(Number(lo))}}-${{Math.round(Number(hi))}}${{sym}}`;
}}
function fmtSigmaC(sig_c) {{
  if (sig_c == null || isNaN(sig_c)) return "--";
  return `±${{Number(sig_c).toFixed(2)}}°C`;
}}

// Map city -> IANA tz string, built once from the CITIES meta blob.
// Used by fmtCityLocalTime to render entry_time in the city's local
// time zone (Atlanta CT, NYC ET, etc.) so the operator can reason
// about "what was the bot doing at 2pm city local."
const CITY_TZ = (() => {{
  const m = {{}};
  for (const c of (CITIES || [])) {{
    if (c.city && c.tz_str) m[c.city] = c.tz_str;
  }}
  return m;
}})();

function fmtCityLocalTime(iso, city) {{
  if (!iso) return "--";
  const tz = CITY_TZ[city];
  try {{
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    const opts = {{
      year: "2-digit", month: "2-digit", day: "2-digit",
      hour: "2-digit", minute: "2-digit",
      hour12: false, timeZone: tz || undefined,
      timeZoneName: "short",
    }};
    // Returns like "06/13/26, 14:38 EDT"
    return new Intl.DateTimeFormat("en-US", opts).format(d);
  }} catch (e) {{
    return String(iso).slice(0, 19).replace("T", " ");
  }}
}}

// Compute headline KPIs from a filtered purchases set.  Mirrors the
// pre-deletion Python aggregation but runs client-side so it reacts
// to the Paper / Live / Both mode toggle.
function computePurchaseKpis(ps) {{
  let n_won = 0, n_lost = 0, n_pending = 0;
  let staked = 0, pnl_sum = 0;
  for (const r of ps) {{
    staked += Number(r.stake_usd) || 0;
    if (r.pnl != null) pnl_sum += Number(r.pnl) || 0;
    if (r.exit_px != null) {{
      if (Number(r.exit_px) >= 0.99) n_won++;
      else                            n_lost++;
    }} else if (r.win_lo != null && r.win_hi != null) {{
      const won = Number(r.bought_lo) === Number(r.win_lo)
                && Number(r.bought_hi) === Number(r.win_hi);
      won ? n_won++ : n_lost++;
    }} else {{
      n_pending++;
    }}
  }}
  const decided = n_won + n_lost;
  return {{
    n_total:      ps.length,
    n_won:        n_won,
    n_lost:       n_lost,
    n_pending:    n_pending,
    win_pct:      decided > 0 ? (100 * n_won / decided) : 0,
    staked_total: staked,
    pnl_total:    pnl_sum,
    roi_pct:      staked > 0 ? (100 * pnl_sum / staked) : 0,
  }};
}}

// Match the dashboard's mode-toggle semantics for a purchase row.
//   "paper" — only paper trades (is_paper=1)
//   "live"  — only live trades  (is_paper=0)
//   "both"  — everything
function matchesPurchaseMode(r) {{
  const isPaper = Number(r.is_paper) === 1;
  if (MODE_FILTER === "paper") return isPaper;
  if (MODE_FILTER === "live")  return !isPaper;
  return true;
}}

// In-tab date filter for the Analysis view.  Empty string = "All dates
// in the loaded window"; otherwise filter to the matching event_date.
let AN_DATE_FILTER = "";
function matchesPurchaseDate(r) {{
  if (!AN_DATE_FILTER) return true;
  return String(r.event_date || "") === AN_DATE_FILTER;
}}

// Populate the in-tab date dropdown with the distinct event_dates that
// appear in the purchases set, plus an "All dates" option at the top.
// Preserves the operator's selection across silent refreshes.
function refreshAnalysisDateDropdown() {{
  const sel = $("an-f-date");
  if (!sel) return;
  const dates = Array.from(new Set(
    (ANALYSIS.purchases || []).map(r => String(r.event_date || ""))
  )).filter(Boolean).sort().reverse();
  const prev = sel.value;
  let html = '<option value="">All dates (window)</option>';
  for (const d of dates) {{
    html += `<option value="${{d}}">${{fmtDateShort(d)}}</option>`;
  }}
  sel.innerHTML = html;
  // Restore previous selection if it's still present
  if (prev && dates.indexOf(prev) >= 0) {{
    sel.value = prev;
    AN_DATE_FILTER = prev;
  }} else {{
    AN_DATE_FILTER = "";
  }}
}}

function renderAnalysis() {{
  if (!ANALYSIS || typeof ANALYSIS !== "object") return;
  const lookback = ANALYSIS.lookback_days || 30;
  const lb1 = $("an-lookback-1"); if (lb1) lb1.textContent = lookback;

  // Refresh the in-tab date dropdown from whatever dates are present
  refreshAnalysisDateDropdown();

  // -- Filter purchases: mode AND date --
  const allPurchases = ANALYSIS.purchases || [];
  const ps = allPurchases
    .filter(matchesPurchaseMode)
    .filter(matchesPurchaseDate);

  // -- Headline KPI strip (computed from FILTERED set) --
  const k = computePurchaseKpis(ps);
  const modeLabel = MODE_FILTER === "both" ? "all"
                    : MODE_FILTER === "paper" ? "paper"
                    : "live";

  // Row count caption (above the table)
  const dateLabel = AN_DATE_FILTER ? ` on ${{fmtDateShort(AN_DATE_FILTER)}}` : "";
  const anCount = $("an-count");
  if (anCount) anCount.textContent =
    `${{ps.length}} purchase${{ps.length === 1 ? "" : "s"}} (${{modeLabel}})${{dateLabel}}`;
  $("an-headline-kpis").innerHTML = `
    <div class="kpi"><div class="label">Purchases (${{modeLabel}})</div>
      <div class="val">${{k.n_total||0}}</div></div>
    <div class="kpi"><div class="label">Won</div>
      <div class="val" style="color:#22c55e">${{k.n_won||0}}</div></div>
    <div class="kpi"><div class="label">Lost</div>
      <div class="val" style="color:#ef4444">${{k.n_lost||0}}</div></div>
    <div class="kpi"><div class="label">Pending</div>
      <div class="val" style="color:#94a3b8">${{k.n_pending||0}}</div></div>
    <div class="kpi"><div class="label">Win rate</div>
      <div class="val">${{fmtPct(k.win_pct, 1)}}</div></div>
    <div class="kpi"><div class="label">Staked</div>
      <div class="val">${{fmtMoney(k.staked_total)}}</div></div>
    <div class="kpi"><div class="label">P&amp;L</div>
      <div class="val" style="${{pnlColor(k.pnl_total)}}">${{fmtMoney(k.pnl_total, true)}}</div></div>
    <div class="kpi"><div class="label">ROI</div>
      <div class="val" style="${{pnlColor(k.roi_pct)}}">${{fmtPct(k.roi_pct, 2, true)}}</div></div>
  `;

  // -- Purchases & outcomes table (the headline) --
  let psHtml = "";
  for (const r of ps) {{
    const boughtBin = fmtBinLabel(r.bought_lo, r.bought_hi, r.bought_unit);
    const winBin    = (r.win_lo != null && r.win_hi != null)
                        ? fmtBinLabel(r.win_lo, r.win_hi,
                                        r.win_unit || r.bought_unit)
                        : '<span style="color:#94a3b8">--</span>';

    // Forecast high / mu shown in the same unit as the purchased bin
    const fcHigh = fmtTempInUnit(r.at_buy_fc_high_c, r.bought_unit, 1);
    const mu     = fmtTempInUnit(r.at_buy_mu_c, r.bought_unit, 1);
    const sigma  = fmtSigmaC(r.at_buy_sigma_c);

    // Result: prefer the position's exit_price (closed positions);
    // fall back to comparing bought vs winning bin.
    let result_html;
    if (r.exit_px != null) {{
      const won = Number(r.exit_px) >= 0.99;
      result_html = won
        ? '<span style="color:#22c55e;font-weight:600">WON</span>'
        : '<span style="color:#ef4444;font-weight:600">LOST</span>';
    }} else if (r.win_lo != null && r.win_hi != null) {{
      const won = (Number(r.bought_lo) === Number(r.win_lo)
                   && Number(r.bought_hi) === Number(r.win_hi));
      result_html = won
        ? '<span style="color:#22c55e;font-weight:600">WON</span>'
        : '<span style="color:#ef4444;font-weight:600">LOST</span>';
    }} else {{
      // Either still open OR no winner yet known
      const lbl = (r.status === 'open') ? 'OPEN' : 'PENDING';
      result_html = `<span style="color:#94a3b8">${{lbl}}</span>`;
    }}

    const pnlCell = (r.pnl != null)
      ? `<span style="${{pnlColor(r.pnl)}}">${{fmtMoney(r.pnl, true)}}</span>`
      : '<span style="color:#94a3b8">--</span>';

    const isPaper = Number(r.is_paper) === 1;
    const modeCell = isPaper
      ? '<span style="color:#3b82f6;font-weight:600">PAPER</span>'
      : '<span style="color:#ef4444;font-weight:600">LIVE</span>';

    const localTime = fmtCityLocalTime(r.entry_time, r.city);

    // City cell links to the detail drilldown for (city, event_date)
    const cityCell = r.city
      ? `<a href="#detail/${{encodeURIComponent(r.city)}}/${{encodeURIComponent(r.event_date)}}"
            style="color:#60a5fa;text-decoration:none;border-bottom:1px dotted #60a5fa">${{r.city}}</a>`
      : "";

    psHtml += `<tr>
      <td>${{fmtDateShort(r.event_date)}}</td>
      <td>${{cityCell}}</td>
      <td style="font-family:monospace;font-size:11px;color:#cbd5e1">${{localTime}}</td>
      <td>${{modeCell}}</td>
      <td><b>${{boughtBin}}</b></td>
      <td>${{r.side||""}}</td>
      <td>${{fmtMoney(r.stake_usd)}}</td>
      <td>${{fmtNum(r.entry_px, 3)}}</td>
      <td>${{fcHigh}}</td>
      <td>${{mu}}</td>
      <td style="color:#94a3b8">${{sigma}}</td>
      <td>${{fmtPct(Number(r.at_buy_our_p)*100, 1)}}</td>
      <td>${{fmtPct(Number(r.at_buy_mkt_p)*100, 1)}}</td>
      <td>${{winBin}}</td>
      <td>${{result_html}}</td>
      <td>${{pnlCell}}</td>
    </tr>`;
  }}
  $("an-purchases-tbody").innerHTML = psHtml ||
    `<tr><td colspan="16" class="empty">No ${{modeLabel}} purchases in window</td></tr>`;

  // -- Pipeline coverage KPIs --
  const pipe = ANALYSIS.pipeline_today
                 || {{signals:0, orders:0, positions:0}};
  $("an-pipeline-kpis").innerHTML = `
    <div class="kpi"><div class="lbl">LIVE_BUY signals today</div>
      <div class="val">${{pipe.signals}}</div></div>
    <div class="kpi"><div class="lbl">Orders placed today</div>
      <div class="val">${{pipe.orders}}</div></div>
    <div class="kpi"><div class="lbl">Live positions opened</div>
      <div class="val">${{pipe.positions}}</div></div>
  `;
  let alertHtml = "";
  if (pipe.signals > 0 && pipe.orders === 0) {{
    alertHtml = '<div style="background:#7f1d1d;color:#fee2e2;padding:10px;'
      + 'border-radius:6px;margin-top:8px">'
      + 'WARNING: LIVE_BUY signals exist but ZERO orders placed today. '
      + 'Check execute_signal logs for failures.</div>';
  }} else if (pipe.orders > 0 && pipe.positions === 0) {{
    alertHtml = '<div style="background:#78350f;color:#fef3c7;padding:10px;'
      + 'border-radius:6px;margin-top:8px">'
      + 'WARN: Orders placed but no positions opened.</div>';
  }}
  $("an-pipeline-alert").innerHTML = alertHtml;

  // -- Calibration --
  const cal = ANALYSIS.calibration_buckets || [];
  let calHtml = "";
  for (const r of cal) {{
    calHtml += `<tr>
      <td>${{r.conf_bucket||""}}</td>
      <td>${{r.n||0}}</td>
      <td>${{fmtNum(r.avg_model_p, 3)}}</td>
      <td>${{fmtNum(r.actual_win_rate, 3)}}</td>
      <td style="${{gapColor(r.calibration_gap)}}">${{fmtNum(r.calibration_gap, 3, true)}}</td>
    </tr>`;
  }}
  $("an-cal-tbody").innerHTML = calHtml ||
    '<tr><td colspan="5" class="empty">Need closed positions with a matching LIVE_BUY signal</td></tr>';

  // -- Stuck positions (compact warning only when present) --
  const stuck = ANALYSIS.stuck_positions || [];
  if (stuck.length === 0) {{
    $("an-stuck-summary").innerHTML = "";
    $("an-stuck-table").style.display = "none";
  }} else {{
    $("an-stuck-summary").innerHTML =
      `<div style="background:#78350f;color:#fef3c7;padding:10px;`
      + `border-radius:6px;margin-bottom:8px">`
      + `<b>${{stuck.length}} stuck position(s)</b> past event date, still open. `
      + `Investigate whether _settle_resolved_positions is firing OR `
      + `Gamma shows the market still open (delayed resolution).</div>`;
    let stuckHtml = "";
    for (const r of stuck) {{
      stuckHtml += `<tr>
        <td>${{r.id||""}}</td>
        <td>${{r.city||""}}</td>
        <td>${{fmtDateShort(r.date)}}</td>
        <td>${{r.side||""}}</td>
        <td>${{fmtMoney(r.stake_usd)}}</td>
        <td>${{fmtNum(r.entry_px, 3)}}</td>
        <td>${{r.status||""}}</td>
        <td>${{r.fill_status||""}}</td>
        <td>${{(r.entry_time||"").slice(0,19).replace("T"," ")}}</td>
        <td>${{fmtNum(r.days_past_event, 1)}}</td>
      </tr>`;
    }}
    $("an-stuck-tbody").innerHTML = stuckHtml;
    $("an-stuck-table").style.display = "";
  }}
}}

// ===========================================================================
// CITY-DETAIL DRILLDOWN
// ===========================================================================
// State: chart instances kept around so we can destroy/recreate on
// revisit (Chart.js leaks if you stack new charts onto the same canvas).
let DETAIL_CHARTS = {{bought: null, winning: null, distribution: null}};
let DETAIL_KEY = null;   // "<city>||<event_date>" of the currently-rendered detail

function _destroyDetailCharts() {{
  for (const k of Object.keys(DETAIL_CHARTS)) {{
    if (DETAIL_CHARTS[k]) {{ DETAIL_CHARTS[k].destroy(); DETAIL_CHARTS[k] = null; }}
  }}
}}

function _detailIsoToLocal(iso, city) {{
  // Same idea as fmtCityLocalTime but returns a compact HH:MM string
  // for chart axis labels — too noisy to render full date in every tick.
  if (!iso) return "";
  const tz = CITY_TZ[city];
  try {{
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return new Intl.DateTimeFormat("en-US", {{
      hour: "2-digit", minute: "2-digit",
      hour12: false, timeZone: tz || undefined,
    }}).format(d);
  }} catch (e) {{ return iso.slice(11, 16); }}
}}

function _purchaseForKey(city, date) {{
  // Find the first purchase row matching (city, event_date) — used to
  // build the summary card and choose bin labels.
  for (const r of (ANALYSIS.purchases || [])) {{
    if (r.city === city && String(r.event_date) === String(date)) return r;
  }}
  return null;
}}

function _findContractAtFinalScan(detail, lo, hi) {{
  for (const b of (detail.bin_distribution || [])) {{
    if (Number(b.range_low) === Number(lo)
        && Number(b.range_high) === Number(hi)) {{
      return b.contract_id;
    }}
  }}
  return null;
}}

function renderDetail(city, eventDate) {{
  const key = `${{city}}||${{eventDate}}`;
  const detail = (ANALYSIS.city_details || {{}})[key];
  const purchase = _purchaseForKey(city, eventDate);

  // Title + back button always work even when detail data is missing
  $("detail-title").textContent =
    `${{city}} — ${{fmtDateShort(eventDate)}}`;

  if (!detail) {{
    $("detail-summary").innerHTML =
      `<div class="kpi"><div class="label">Status</div>
       <div class="val" style="color:#ef4444">No detail data available</div></div>`;
    _destroyDetailCharts();
    return;
  }}
  DETAIL_KEY = key;

  // -- Summary card --
  const boughtBinLbl = purchase
    ? fmtBinLabel(purchase.bought_lo, purchase.bought_hi, purchase.bought_unit)
    : "--";
  const winBinLbl    = (purchase && purchase.win_lo != null)
    ? fmtBinLabel(purchase.win_lo, purchase.win_hi,
                    purchase.win_unit || purchase.bought_unit)
    : "--";

  // Determine result
  let resultLbl = "--", resultColor = "#94a3b8";
  if (purchase) {{
    if (purchase.exit_px != null) {{
      const won = Number(purchase.exit_px) >= 0.99;
      resultLbl   = won ? "WON" : "LOST";
      resultColor = won ? "#22c55e" : "#ef4444";
    }} else if (purchase.win_lo != null && purchase.win_hi != null) {{
      const won = Number(purchase.bought_lo) === Number(purchase.win_lo)
               && Number(purchase.bought_hi) === Number(purchase.win_hi);
      resultLbl   = won ? "WON" : "LOST";
      resultColor = won ? "#22c55e" : "#ef4444";
    }} else {{
      resultLbl = "PENDING";
    }}
  }}

  const pnlStr = purchase && purchase.pnl != null
    ? fmtMoney(purchase.pnl, true) : "--";
  const pnlStyle = (purchase && purchase.pnl != null) ? pnlColor(purchase.pnl) : "";
  const stakeStr = purchase ? fmtMoney(purchase.stake_usd) : "--";
  const entryStr = purchase ? fmtNum(purchase.entry_px, 3) : "--";
  const exitStr  = (purchase && purchase.exit_px != null)
                     ? fmtNum(purchase.exit_px, 3) : "--";
  const isPaperLbl = (purchase && Number(purchase.is_paper) === 1)
                       ? '<span style="color:#3b82f6">PAPER</span>'
                       : '<span style="color:#ef4444">LIVE</span>';

  $("detail-summary").innerHTML = `
    <div class="kpi"><div class="label">Mode</div>
      <div class="val">${{isPaperLbl}}</div></div>
    <div class="kpi"><div class="label">Bought bin</div>
      <div class="val">${{boughtBinLbl}}</div></div>
    <div class="kpi"><div class="label">Winning bin</div>
      <div class="val">${{winBinLbl}}</div></div>
    <div class="kpi"><div class="label">Result</div>
      <div class="val" style="color:${{resultColor}}">${{resultLbl}}</div></div>
    <div class="kpi"><div class="label">Stake</div>
      <div class="val">${{stakeStr}}</div></div>
    <div class="kpi"><div class="label">Entry $</div>
      <div class="val">${{entryStr}}</div></div>
    <div class="kpi"><div class="label">Exit $</div>
      <div class="val">${{exitStr}}</div></div>
    <div class="kpi"><div class="label">P&amp;L</div>
      <div class="val" style="${{pnlStyle}}">${{pnlStr}}</div></div>
  `;

  // Bin labels for the chart headers
  $("detail-bought-label").textContent = boughtBinLbl;
  $("detail-winning-label").textContent = winBinLbl;

  // -- Build chart data --
  _destroyDetailCharts();
  const ts = detail.timeseries || [];
  if (ts.length === 0) {{
    // No scan data for this event — show empty placeholders
    return;
  }}

  const unit = purchase ? purchase.bought_unit : null;
  const isF  = String(unit||"").toLowerCase() === "fahrenheit";
  const tempUnit = isF ? "°F" : "°C";
  const conv = isF ? cToF : (v) => v;

  const timeLabels = ts.map(p => _detailIsoToLocal(p.t, city));
  const tempArr    = ts.map(p => conv(p.observed_max_c));
  const muArr      = ts.map(p => conv(p.mu_c));
  const forecastC  = ts.length ? ts[ts.length-1].forecast_high_c : null;

  function _makeBinChart(canvasId, series, binLo, binHi, binLabel) {{
    const ctx = $(canvasId);
    if (!ctx) return null;
    // Align bin series to the shared timeseries by index — both are
    // ordered ASC by scanned_at_utc, but bin series can have fewer
    // points (skipped scans).  Build a map for O(1) lookup.
    const byT = {{}};
    for (const p of (series || [])) byT[p.t] = p;
    const ourArr = ts.map(p => byT[p.t] ? (Number(byT[p.t].our_prob) * 100) : null);
    const mktArr = ts.map(p => byT[p.t] ? (Number(byT[p.t].market_prob) * 100) : null);
    // Bin range is ALREADY stored in the bin's native unit (the same
    // unit we render the temperature in via conv).  Do NOT convert it
    // — that would F-to-F again, producing 168°F for a 76°F bin.
    const binLoDisp = (binLo != null) ? Number(binLo) : null;
    const binHiDisp = (binHi != null) ? Number(binHi) : null;
    const datasets = [
      {{
        label: `Temperature (observed max)`, data: tempArr,
        yAxisID: 'y-temp', borderColor: '#22c55e',
        backgroundColor: 'rgba(34, 197, 94, 0.1)', pointRadius: 2,
        borderWidth: 2, tension: 0.2,
      }},
      {{
        label: `Model μ`, data: muArr,
        yAxisID: 'y-temp', borderColor: '#a78bfa',
        borderDash: [3, 3], pointRadius: 1, borderWidth: 1.5,
        tension: 0.2,
      }},
      {{
        label: `Our P (${{binLabel}})`, data: ourArr,
        yAxisID: 'y-prob', borderColor: '#3b82f6',
        borderDash: [6, 4], pointRadius: 2, borderWidth: 2,
        tension: 0.2, spanGaps: true,
      }},
      {{
        label: `Market P (${{binLabel}})`, data: mktArr,
        yAxisID: 'y-prob', borderColor: '#ef4444', pointRadius: 2,
        borderWidth: 2, tension: 0.2, spanGaps: true,
      }},
    ];
    return new Chart(ctx, {{
      type: 'line',
      data: {{labels: timeLabels, datasets: datasets}},
      options: {{
        responsive: true, maintainAspectRatio: false,
        interaction: {{mode: 'index', intersect: false}},
        plugins: {{
          legend: {{labels: {{color: '#e2e8f0'}}}},
          tooltip: {{mode: 'index', intersect: false}},
          title: {{
            display: !!(forecastC || (binLoDisp != null)),
            text: `Forecast high: ${{forecastC != null ? conv(forecastC).toFixed(1) + tempUnit : '--'}}`
                  + (binLoDisp != null
                       ? `  ·  Bin: ${{binLoDisp}}-${{binHiDisp}}${{tempUnit}}`
                       : ''),
            color: '#94a3b8',
          }},
        }},
        scales: {{
          x: {{
            // Thin tick labels: 200+ scans/day would otherwise jam the
            // axis and eat all the chart height.  autoSkip + 12 limit
            // shows roughly one label per hour for a 12-hour window.
            ticks: {{
              color: '#94a3b8',
              autoSkip: true,
              maxTicksLimit: 12,
              maxRotation: 0,
              minRotation: 0,
            }},
            grid: {{color: '#1f2937'}},
          }},
          'y-temp': {{
            type: 'linear', position: 'left',
            title: {{display: true, text: `Temperature (${{tempUnit}})`, color: '#94a3b8'}},
            ticks: {{color: '#94a3b8'}}, grid: {{color: '#1f2937'}},
          }},
          'y-prob': {{
            type: 'linear', position: 'right', min: 0, max: 100,
            title: {{display: true, text: 'Probability (%)', color: '#94a3b8'}},
            ticks: {{color: '#94a3b8'}}, grid: {{display: false}},
          }},
        }},
      }},
    }});
  }}

  // -- Chart 1: bought bin --
  if (purchase) {{
    DETAIL_CHARTS.bought = _makeBinChart(
      "detail-chart-bought", detail.bought_series,
      purchase.bought_lo, purchase.bought_hi, boughtBinLbl
    );
  }}

  // -- Chart 2: winning bin --
  if (purchase && purchase.win_lo != null) {{
    DETAIL_CHARTS.winning = _makeBinChart(
      "detail-chart-winning", detail.winning_series,
      purchase.win_lo, purchase.win_hi, winBinLbl
    );
  }} else {{
    // No winner known yet — render empty placeholder so the title still
    // shows but the canvas stays blank.
  }}

  // -- Chart 3: final-scan distribution --
  const dist = detail.bin_distribution || [];
  if (dist.length > 0) {{
    const dctx = $("detail-chart-distribution");
    if (dctx) {{
      const lbls = dist.map(b => b.bin_label || "?");
      const ours = dist.map(b => Number(b.our_prob) * 100);
      const mkts = dist.map(b => Number(b.market_prob) * 100);
      // Per-bar border colors to highlight bought (yellow) / winning (green)
      const borders = dist.map(b =>
        b.is_winning ? '#22c55e'
        : b.is_bought ? '#f59e0b'
        : 'rgba(0,0,0,0)'
      );
      const borderWidths = dist.map(b => (b.is_bought || b.is_winning) ? 3 : 0);
      DETAIL_CHARTS.distribution = new Chart(dctx, {{
        type: 'bar',
        data: {{
          labels: lbls,
          datasets: [
            {{label: 'Our P (%)',    data: ours, backgroundColor: '#3b82f6',
              borderColor: borders, borderWidth: borderWidths}},
            {{label: 'Market P (%)', data: mkts, backgroundColor: '#ef4444',
              borderColor: borders, borderWidth: borderWidths}},
          ],
        }},
        options: {{
          responsive: true, maintainAspectRatio: false,
          plugins: {{
            legend: {{labels: {{color: '#e2e8f0'}}}},
            tooltip: {{mode: 'index', intersect: false}},
          }},
          scales: {{
            x: {{
              ticks: {{
                color: '#94a3b8',
                autoSkip: false,    // every bin label matters here
                maxRotation: 45, minRotation: 30,
              }},
              grid: {{color: '#1f2937'}},
            }},
            y: {{
              ticks: {{color: '#94a3b8'}}, grid: {{color: '#1f2937'}},
              title: {{display: true, text: 'Probability (%)', color: '#94a3b8'}},
              beginAtZero: true,
            }},
          }},
        }},
      }});
    }}
  }}
}}

// ===========================================================================
// Hash-based routing for the detail view
// ===========================================================================
// URL pattern: #detail/<city>/<event_date>
// Browser back button works for free — hashchange listener handles it.

function _parseDetailHash() {{
  const h = location.hash || "";
  const m = h.match(/^#detail\\/([^/]+)\\/([^/]+)$/);
  if (!m) return null;
  try {{
    return {{city: decodeURIComponent(m[1]), date: decodeURIComponent(m[2])}};
  }} catch (e) {{ return null; }}
}}

function showDetail(city, eventDate) {{
  // Hide every other section + show detail
  const sections = ["sig-table","trades-table","analysis-section"];
  for (const id of sections) {{
    const el = $(id);
    if (el) el.style.display = "none";
  }}
  for (const id of ["panels-title","filters","city-grid","signals-title"]) {{
    const el = (id === "filters")
      ? document.querySelector(".filters") : $(id);
    if (el) el.style.display = "none";
  }}
  // Hide the top-of-page KPI strip too — it doesn't apply to the
  // single-event drilldown.
  const kpis = $("kpis"); if (kpis) kpis.style.display = "none";
  const ds = $("detail-section");
  if (ds) ds.style.display = "block";
  renderDetail(city, eventDate);
}}

function hideDetail() {{
  const ds = $("detail-section");
  if (ds) ds.style.display = "none";
  _destroyDetailCharts();
  // Restore the top-of-page KPI strip
  const kpis = $("kpis"); if (kpis) kpis.style.display = "";
  // Re-apply the current view so the right section comes back
  setView(VIEW_MODE);
}}

function _maybeRouteDetail() {{
  const d = _parseDetailHash();
  if (d) showDetail(d.city, d.date);
  else   hideDetail();
}}

// Back-button & in-link navigation
window.addEventListener("hashchange", _maybeRouteDetail);
const _backBtn = $("detail-back");
if (_backBtn) _backBtn.addEventListener("click", () => {{
  // Clearing the hash fires hashchange → _maybeRouteDetail → hideDetail
  history.pushState("", document.title,
                      location.pathname + location.search);
  hideDetail();
}});

// Date dropdown drives the whole dashboard
$("f-date").addEventListener("change", () => {{
  SELECTED_DATE = $("f-date").value;
  recomputeDerived();
  renderAll();
}});
// City filter affects panels too, so it triggers renderAll (not just table)
$("f-city").addEventListener("change", () => renderAll());
// These only affect the signals table
["f-action","f-edge","f-buys","f-latest"].forEach(id =>
  $(id).addEventListener("input", renderSigTable));

// Mode toggle handlers
function setMode(m) {{
  MODE_FILTER = m;
  saveState({{mode: m}});
  ["paper","live","both"].forEach(x => {{
    const btn = $(`mode-${{x}}`);
    btn.classList.toggle("active", x === m);
    btn.className = btn.className.replace(/\\b(paper|live|both)\\b/g, '').trim();
    if (x === m) btn.classList.add("active", x);
  }});
  renderAll();
}}
$("mode-paper").addEventListener("click", () => setMode("paper"));
$("mode-live").addEventListener("click",  () => setMode("live"));
$("mode-both").addEventListener("click",  () => setMode("both"));

// ===== Restore persisted UI state on page load =====
// ALWAYS call setMode/setView to force button states to match the
// persisted MODE_FILTER/VIEW_MODE values.  Calling unconditionally
// (rather than only when != default) is bulletproof:
//   - if persisted = "paper", setMode("paper") is a no-op visually
//   - if persisted = "both", setMode("both") flips the visual state
//   - if HTML hard-coded the wrong default, this still fixes it
// Each call also triggers a renderAll(), so the data filters update.
function restorePersistedUI() {{
  setMode(MODE_FILTER);
  setView(VIEW_MODE);
  // Filter inputs — restore from STATE, but only those that exist in the
  // current data (selects may have different options than last session).
  const restoreInput = (id, key) => {{
    if (STATE[key] === undefined || STATE[key] === null) return;
    const el = $(id);
    if (!el) return;
    if (el.type === "checkbox") el.checked = !!STATE[key];
    else el.value = STATE[key];
  }};
  restoreInput("f-city",   "city");
  restoreInput("f-action", "action");
  restoreInput("f-edge",   "edge");
  restoreInput("f-buys",   "buys");
  restoreInput("f-latest", "latest");
}}

// Persist filter changes too
function saveFilterState() {{
  saveState({{
    city:   $("f-city").value,
    action: $("f-action").value,
    edge:   $("f-edge").value,
    buys:   $("f-buys").checked,
    latest: $("f-latest").checked,
  }});
}}
["f-city","f-action","f-edge","f-buys","f-latest"].forEach(id =>
  $(id).addEventListener("change", saveFilterState));

function renderAll() {{
  refreshDateDropdown();   // SELECTED_DATE may shift if today rolled over
  recomputeDerived();       // re-filter DATE_SIGNALS for new SELECTED_DATE
  refreshCityDropdown();    // city options depend on SELECTED_DATE
  updateSectionTitles();
  renderKPIs();
  renderCityPanels();
  if (VIEW_MODE === 'analysis') renderAnalysis();
  else if (VIEW_MODE === 'trades') renderTradesTable();
  else                              renderSigTable();
}}

// Initial render — must populate dropdowns BEFORE restoring filter
// values (otherwise the city dropdown is empty and the saved choice
// won't stick).
renderAll();
restorePersistedUI();
renderAll();

// If the URL already has a #detail/... hash (deep link / refresh
// from a detail page), route to it now that the initial render is done.
_maybeRouteDetail();

// ===== Silent refresh (no blank page on update) =====
// Re-fetch the same URL every REFRESH_SEC, extract the SIGNALS/LIVE_ORDERS
// blobs from the new HTML, swap them in place, and re-render.  Preserves:
//   * mode toggle state
//   * sort order
//   * all filter inputs
//   * scroll position
//   * dropdown open/closed state
function extractJsonBlob(text, varName) {{
  // Match: let VAR = [...]; or let VAR = {{...}};
  // Tolerant: accepts either array or object form.  We try array first
  // (most blobs), then object (used by ANALYSIS).
  const reArr = new RegExp('let\\\\s+' + varName + '\\\\s*=\\\\s*(\\\\[[\\\\s\\\\S]*?\\\\]);', 'm');
  let m = text.match(reArr);
  if (!m) {{
    const reObj = new RegExp('let\\\\s+' + varName + '\\\\s*=\\\\s*(\\\\{{[\\\\s\\\\S]*?\\\\}});', 'm');
    m = text.match(reObj);
  }}
  if (!m) return null;
  try {{ return JSON.parse(m[1]); }} catch (e) {{ return null; }}
}}

async function silentRefresh() {{
  try {{
    const resp = await fetch(window.location.href, {{cache: "no-store"}});
    if (!resp.ok) return;
    const text = await resp.text();
    const newSignals    = extractJsonBlob(text, "SIGNALS");
    const newLiveOrders = extractJsonBlob(text, "LIVE_ORDERS");
    const newLivePos    = extractJsonBlob(text, "LIVE_POSITIONS");
    const newAnalysis   = extractJsonBlob(text, "ANALYSIS");
    if (newSignals === null && newLiveOrders === null
        && newLivePos === null && newAnalysis === null) return;
    if (newSignals !== null)    SIGNALS = newSignals;
    if (newLiveOrders !== null) LIVE_ORDERS = newLiveOrders;
    if (newLivePos !== null)    LIVE_POSITIONS = newLivePos;
    if (newAnalysis !== null)   ANALYSIS = newAnalysis;
    recomputeDerived();
    renderAll();
    // Re-assert button states after render — defensive against any
    // accidental DOM reset during the silent refresh cycle.
    setMode(MODE_FILTER);
    setView(VIEW_MODE);
    // Update generated-at timestamp in header if present
    const tsMatch = text.match(/generated ([0-9-]+ [0-9:]+ UTC)/);
    if (tsMatch) {{
      document.querySelectorAll("header .meta").forEach(el => {{
        const html = el.innerHTML.replace(/generated [0-9-]+ [0-9:]+ UTC/, "generated " + tsMatch[1]);
        if (html !== el.innerHTML) el.innerHTML = html;
      }});
    }}
  }} catch (e) {{
    console.error("silent refresh failed:", e);
  }}
}}

if (REFRESH_SEC > 0) {{
  setInterval(silentRefresh, REFRESH_SEC * 1000);
}}
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# HTTP server + watch loop
# ---------------------------------------------------------------------------

def serve(path: str, port: int, regenerate_fn=None, watch_sec: int | None = None) -> None:
    serve_dir = os.path.dirname(os.path.abspath(path)) or "."
    fname = os.path.basename(path)
    os.chdir(serve_dir)

    class Reusable(socketserver.TCPServer):
        allow_reuse_address = True

    print()
    print("=" * 72)
    print(f"  Serving {path} on 0.0.0.0:{port}")
    print(f"  Tailscale URL:  http://<vps-tailscale-ip>:{port}/{fname}")
    if regenerate_fn and watch_sec:
        print(f"  Watch:          regenerating HTML every {watch_sec}s")
    print("=" * 72)

    if regenerate_fn and watch_sec:
        import threading, time as _time
        def _watcher():
            while True:
                _time.sleep(watch_sec)
                try:
                    regenerate_fn()
                except Exception as e:
                    log.warning(f"regenerate failed: {e}")
        threading.Thread(target=_watcher, daemon=True).start()

    with Reusable(("0.0.0.0", port), http.server.SimpleHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    default_port  = int(os.getenv("DASHBOARD_PORT", "8082"))
    default_watch = int(os.getenv("DASHBOARD_WATCH_SEC", "30"))
    default_days  = int(os.getenv("DASHBOARD_DAYS", "1"))

    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--days", type=int, default=default_days)
    p.add_argument("--db", default=DB_PATH)
    p.add_argument("--html", default=os.path.join(_BOT_DIR, "data",
                                                    "predictor_dashboard.html"))
    p.add_argument("--serve", type=int, default=default_port, nargs="?")
    p.add_argument("--no-serve", action="store_true")
    p.add_argument("--watch", type=int, default=default_watch)
    args = p.parse_args()
    if args.no_serve:
        args.serve = None

    # Analysis tab lookback (env-tunable; default 30 days).  Bounded
    # at the SQL layer for cost — the joins on closed positions can
    # otherwise scan the full history every regen.
    _analysis_days = int(os.getenv("DASHBOARD_ANALYSIS_LOOKBACK_DAYS", "30"))

    def regenerate() -> str:
        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        signals = load_signals(args.db, since)
        live_orders = load_live_orders(args.db, since)
        live_positions = fetch_live_positions()    # Polymarket data API
        analysis = load_analysis_data(args.db, lookback_days=_analysis_days)
        _pu = analysis.get("purchases", [])
        _live = sum(1 for r in _pu if int(r.get("is_paper", 0)) == 0)
        _paper = len(_pu) - _live
        log.info(f"regenerate: {len(signals)} signals + "
                  f"{len(live_orders)} live orders + "
                  f"{len(live_positions)} live positions + "
                  f"analysis(purchases={len(_pu)} = "
                  f"{_live} live + {_paper} paper)")
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        html = build_dashboard(signals, live_orders, generated_at,
                                auto_refresh_sec=args.watch or None,
                                live_positions=live_positions,
                                analysis=analysis)
        os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(html)
        log.info(f"wrote {os.path.getsize(args.html)/1024:.0f} KB to {args.html}")
        return args.html

    regenerate()

    if args.serve:
        serve(args.html, args.serve,
              regenerate_fn=regenerate if args.watch else None,
              watch_sec=args.watch or None)
    return 0


if __name__ == "__main__":
    sys.exit(main())