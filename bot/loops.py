"""
loops.py — Phase 2 scheduled loops.

Two new jobs that sit alongside the existing discovery/trading/monitor cycle:

  forecast_pull_run()   — every 2 hours at :05
                          For each active event within MAX_FORECAST_DAYS,
                          pull ECMWF + GFS ensembles (via weather.py helpers),
                          append raw rows to forecast_runs + forecast_hourly,
                          and update temp_events current-state columns.

  live_observation_run() — every 20 minutes (:00, :20, :40)
                          For each active event, call Visual Crossing
                          fetch_intraday(), append one row to live_observations,
                          and update temp_events current-state columns.

Both jobs are append-only and non-destructive — they never mutate historical
rows.  Trading logic is NOT touched here; these loops just record state.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

from config import (
    MAX_FORECAST_DAYS,
    VC_DIAG_BIN_WIDTH_C,
    VC_DIAG_DISAGREE_LARGE_C,
    VC_DIAG_FUTURE_DAY_HORIZON,
    VC_DIAG_RMSE_WINDOW_HOURS,
    VC_DIAG_WARN_DIRECTIONAL_C,
)
from db import (
    insert_forecast_run,
    insert_forecast_hourly_bulk,
    get_latest_forecast_run,
    get_latest_forecast_hourly,
    get_previous_forecast_run,
    insert_live_observation,
    get_latest_observation,
    get_recent_observations,
    update_event_current_state,
    bump_vc_usage,
    insert_vc_forecast_diagnostic,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ahead(target_date: str) -> int | None:
    try:
        return (date.fromisoformat(target_date) - date.today()).days
    except (ValueError, TypeError):
        return None


def _event_iter(events: list[dict]) -> Iterable[dict]:
    """Yield events with valid date + lat/lon within MAX_FORECAST_DAYS."""
    for ev in events or []:
        d = ev.get("date")
        lat, lon = ev.get("lat"), ev.get("lon")
        if not d or lat is None or lon is None:
            continue
        da = _days_ahead(d)
        if da is None or da < 0 or da > MAX_FORECAST_DAYS:
            continue
        yield ev


# ---------------------------------------------------------------------------
# Forecast pull (2-hour cadence)
# ---------------------------------------------------------------------------

def _fetch_and_write_source(
    event: dict, source: str, pulled_at: str,
) -> tuple[int, float | None, float | None]:
    """
    Fetch μ/σ for one (event, source) and write forecast_runs row.
    Returns (run_id, mu_c, sigma_c).  run_id = 0 on failure.
    """
    # Import inside the function to keep module import cheap and avoid cycles.
    from weather import (
        _get_ecmwf_ensemble_distribution,
        _get_gfs_ensemble_distribution,
    )

    lat, lon = event["lat"], event["lon"]
    date_str = event["date"]

    if source == "ecmwf":
        dist = _get_ecmwf_ensemble_distribution(lat, lon, date_str)
    elif source == "gfs":
        dist = _get_gfs_ensemble_distribution(lat, lon, date_str)
    else:
        logger.error(f"Unknown forecast source: {source}")
        return (0, None, None)

    if dist is None:
        logger.debug(f"No {source} distribution for {event.get('city')} {date_str}")
        return (0, None, None)

    mu_c    = dist.get("mu_c")
    sigma_c = dist.get("sigma_c")

    run_id = insert_forecast_run(
        event_id        = event.get("event_id") or "",
        city            = event.get("city"),
        date            = date_str,
        lat             = lat,
        lon             = lon,
        source          = source,
        pulled_at       = pulled_at,
        forecast_mu_c   = mu_c,
        forecast_sigma_c= sigma_c,
        forecast_high_c = mu_c,   # daily-max ensemble → mu already represents the high
        days_ahead      = _days_ahead(date_str),
    )
    return (run_id, mu_c, sigma_c)


def _fetch_vc_hourly_path(
    lat: float, lon: float, target_date: str
) -> list[dict]:
    """
    Pull hourly temperature / humidity / cloud / wind / precip for a future
    date via Open-Meteo's forecast API (not VC — VC is observations-only
    in v1).  Returns a list of hour dicts ready for insert_forecast_hourly_bulk.

    For events inside 3 days, returns ~72 hourly rows.
    For events >3 days out, returns a single is_target_day=1 daily-summary row.
    """
    import httpx

    days_out = _days_ahead(target_date) or 0

    # Inside 3 days: dense hourly path
    if days_out <= 3:
        try:
            r = httpx.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude":   lat,
                    "longitude":  lon,
                    "hourly": ",".join([
                        "temperature_2m", "relative_humidity_2m",
                        "cloud_cover", "wind_speed_10m",
                        "precipitation", "precipitation_probability",
                    ]),
                    "forecast_days": min(max(days_out + 1, 1), 7),
                    "timezone":   "UTC",
                },
                timeout=20,
            )
            r.raise_for_status()
            h = r.json().get("hourly", {}) or {}
        except Exception as e:
            logger.debug(f"hourly path fetch failed for ({lat},{lon}) {target_date}: {e}")
            return []

        times = h.get("time", []) or []
        rows: list[dict] = []
        # Only keep the next 72 hours from now, regardless of target date.
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
        horizon = now_utc + timedelta(hours=72)
        for i, t in enumerate(times):
            try:
                ts = datetime.fromisoformat(t)
            except ValueError:
                continue
            if ts < now_utc or ts > horizon:
                continue
            rows.append({
                "hour_ts_utc":   ts.isoformat() + "Z",
                "is_target_day": 0,
                "temp_c":        _at(h, "temperature_2m",            i),
                "humidity":      _at(h, "relative_humidity_2m",      i),
                "cloudcover":    _at(h, "cloud_cover",               i),
                "windspeed_kph": _at(h, "wind_speed_10m",            i),
                "precip_mm":     _at(h, "precipitation",             i),
                "precip_prob":   _at(h, "precipitation_probability", i),
            })
        return rows

    # Beyond 3 days: single daily-summary row for the target date.
    try:
        r = httpx.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "daily":      "temperature_2m_max",
                "start_date": target_date,
                "end_date":   target_date,
                "timezone":   "UTC",
            },
            timeout=15,
        )
        r.raise_for_status()
        d = r.json().get("daily", {}) or {}
    except Exception as e:
        logger.debug(f"daily summary fetch failed for {target_date}: {e}")
        return []

    tmax_list = d.get("temperature_2m_max") or []
    if not tmax_list:
        return []
    return [{
        "hour_ts_utc":   f"{target_date}T12:00:00Z",
        "is_target_day": 1,
        "temp_c":        tmax_list[0],
    }]


def _at(d: dict, key: str, i: int):
    arr = d.get(key) or []
    return arr[i] if i < len(arr) else None


def forecast_pull_run(events: list[dict] | None = None) -> dict:
    """
    Pull ECMWF + GFS forecasts for every active event and persist to
    forecast_runs / forecast_hourly.  Also updates temp_events current-state
    columns (forecast_delta_mu_c, forecast_delta_sigma_c, forecast_agreement_c,
    latest_forecast_ts).

    Args:
        events: optional pre-fetched event list.  If None, pulls fresh from
                Polymarket via search_temp_high_events().
    """
    logger.info("=== FORECAST PULL RUN START ===")
    pulled_at = _utc_now_iso()

    if events is None:
        from polymarket import search_temp_high_events
        from config import MIN_LIQUIDITY_USD
        events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)

    counts = {"events": 0, "runs": 0, "hourly_rows": 0, "skipped": 0}

    for ev in _event_iter(events):
        counts["events"] += 1
        event_id = ev.get("event_id") or ""
        city     = ev.get("city")
        date_str = ev.get("date")

        # ECMWF + GFS forecast_runs rows
        run_ids: dict[str, int] = {}
        mus: dict[str, float | None] = {}
        sigmas: dict[str, float | None] = {}
        for source in ("ecmwf", "gfs"):
            # Capture prior μ/σ before inserting the new row so we can compute delta.
            prev = get_latest_forecast_run(event_id, source)
            rid, mu, sigma = _fetch_and_write_source(ev, source, pulled_at)
            run_ids[source] = rid
            mus[source]     = mu
            sigmas[source]  = sigma
            if rid:
                counts["runs"] += 1
            # Store "prev" as the one we just replaced as latest — we held prev above.
            ev.setdefault("_prev_runs", {})[source] = prev

        # Hourly path (written under the ECMWF run if present, else GFS, else
        # skip).  We tag it under whichever source we picked so downstream
        # queries can JOIN via run_id.
        carrier_run = run_ids.get("ecmwf") or run_ids.get("gfs") or 0
        if carrier_run:
            try:
                hourly = _fetch_vc_hourly_path(ev["lat"], ev["lon"], date_str)
                if hourly:
                    counts["hourly_rows"] += insert_forecast_hourly_bulk(
                        carrier_run, hourly
                    )
            except Exception as e:
                logger.debug(f"hourly insert failed for {city} {date_str}: {e}")

        # Deltas vs prior pull, per source
        deltas_mu: list[float] = []
        deltas_sigma: list[float] = []
        for source in ("ecmwf", "gfs"):
            prev = ev.get("_prev_runs", {}).get(source)
            if not prev:
                continue
            if mus[source] is not None and prev.get("forecast_mu_c") is not None:
                deltas_mu.append(mus[source] - prev["forecast_mu_c"])
            if sigmas[source] is not None and prev.get("forecast_sigma_c") is not None:
                deltas_sigma.append(sigmas[source] - prev["forecast_sigma_c"])

        delta_mu    = (sum(deltas_mu) / len(deltas_mu))       if deltas_mu else None
        delta_sigma = (sum(deltas_sigma) / len(deltas_sigma)) if deltas_sigma else None

        # Agreement = |ecmwf_mu - gfs_mu|
        agreement = None
        if mus["ecmwf"] is not None and mus["gfs"] is not None:
            agreement = abs(mus["ecmwf"] - mus["gfs"])

        try:
            update_event_current_state(
                event_id,
                latest_forecast_ts     = pulled_at,
                forecast_delta_mu_c    = delta_mu,
                forecast_delta_sigma_c = delta_sigma,
                forecast_agreement_c   = agreement,
            )
        except Exception as e:
            logger.debug(f"event-state update failed for {event_id}: {e}")

    logger.info(
        f"Forecast pull: events={counts['events']} runs={counts['runs']} "
        f"hourly_rows={counts['hourly_rows']}"
    )
    logger.info("=== FORECAST PULL RUN END ===")
    return counts


# ---------------------------------------------------------------------------
# Live observation (20-minute cadence)
# ---------------------------------------------------------------------------

def _interp_expected_temp(run_id: int, observed_at_utc: str | None) -> float | None:
    """Find the forecast_hourly row nearest `observed_at_utc` and return its temp."""
    if not run_id or not observed_at_utc:
        return None
    try:
        obs_dt = datetime.fromisoformat(observed_at_utc.replace("Z", "+00:00"))
    except Exception:
        return None
    from db import get_latest_forecast_hourly
    rows = get_latest_forecast_hourly(run_id)
    if not rows:
        return None
    best = None
    best_gap = None
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["hour_ts_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        gap = abs((ts - obs_dt).total_seconds())
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = r
    if best is None or best_gap is None or best_gap > 3 * 3600:
        return None   # no hourly row within 3 hours
    return best.get("temp_c")


def _temp_change(event_id: str, current_temp_c: float | None, hours: int) -> float | None:
    if current_temp_c is None:
        return None
    since = (datetime.now(timezone.utc) - timedelta(hours=hours + 1)).isoformat()
    rows = get_recent_observations(event_id, since)
    if not rows:
        return None
    target = datetime.now(timezone.utc) - timedelta(hours=hours)
    # Find obs closest to `target` time (but before now)
    best = None
    best_gap = None
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["pulled_at_utc"].replace("Z", "+00:00"))
        except Exception:
            continue
        gap = abs((ts - target).total_seconds())
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = r
    if best is None or best.get("current_temp_c") is None:
        return None
    return current_temp_c - best["current_temp_c"]


def _build_vc_diagnostic_row(
    *,
    event_id: str,
    city: str | None,
    target_date: str,
    pulled_at_utc: str,
    kind: str,
    vc_payload: dict,
    event_state: dict | None,
) -> dict | None:
    """Compute disagreement metrics + flags from a VC payload and write a
    vc_forecast_diagnostics row.  Returns the inserted row dict (or None if
    VC didn't provide a usable day-max estimate)."""
    import json
    import math

    vc_day_max = vc_payload.get("day_tempmax_estimate_c")
    if vc_day_max is None:
        return None

    vc_fcst_remaining = vc_payload.get("forecast_remaining_max_c")
    vc_day_source = vc_payload.get("day_vc_source")

    blended_mu    = (event_state or {}).get("forecast_mu_c")
    blended_sigma = (event_state or {}).get("forecast_sigma_c")
    adjusted_mu   = (event_state or {}).get("adjusted_mu_c")
    adjusted_sigma= (event_state or {}).get("adjusted_sigma_c")
    cur_temp      = vc_payload.get("current_temp_c")
    obs_max       = vc_payload.get("observed_max_so_far_c")

    def _sub(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    d_blended = _sub(vc_day_max, blended_mu)
    d_adjusted = _sub(vc_day_max, adjusted_mu)
    d_obs     = _sub(vc_day_max, obs_max)

    abs_blended  = abs(d_blended)  if d_blended  is not None else None
    abs_adjusted = abs(d_adjusted) if d_adjusted is not None else None

    # Bins-apart uses disagreement vs blended (the model's production number)
    if abs_blended is not None and VC_DIAG_BIN_WIDTH_C > 0:
        bins_apart = int(round(abs_blended / VC_DIAG_BIN_WIDTH_C))
    else:
        bins_apart = None

    # Flags
    flag_large   = int(abs_blended is not None and abs_blended >= VC_DIAG_DISAGREE_LARGE_C)
    flag_hotter  = int(d_blended   is not None and d_blended   >=  VC_DIAG_WARN_DIRECTIONAL_C)
    flag_colder  = int(d_blended   is not None and d_blended   <= -VC_DIAG_WARN_DIRECTIONAL_C)

    # Hourly-path RMSE: compare next-6h VC forecast_hours to our stored
    # forecast_hourly for the latest ECMWF run.  Same-day only; NULL for
    # future_day rows (caller passes empty vc.forecast_hours then).
    rmse_c = None
    n_rmse = None
    if kind == "same_day":
        vc_future_hours = [h for h in (vc_payload.get("forecast_hours") or [])
                           if h.get("source") == "fcst" and h.get("temp_c") is not None]
        if vc_future_hours:
            latest_ecmwf = get_latest_forecast_run(event_id, "ecmwf")
            our_hours_map: dict[str, float] = {}
            if latest_ecmwf:
                hrs = get_latest_forecast_hourly(latest_ecmwf["id"]) or []
                for h in hrs:
                    ts = h.get("hour_ts_utc")
                    t = h.get("temp_c")
                    if ts and t is not None:
                        # normalize to "YYYY-MM-DDTHH" prefix for hour-match
                        our_hours_map[ts[:13]] = float(t)
            if our_hours_map:
                diffs: list[float] = []
                for h in vc_future_hours[:VC_DIAG_RMSE_WINDOW_HOURS]:
                    vc_dt = h.get("datetime")  # "YYYY-MM-DD HH:MM:SS" (local per VC)
                    if not vc_dt:
                        continue
                    # VC returns local-time datetime; match only by hour in UTC
                    # is hard, so match by datetime_epoch → ISO hour.
                    epoch = h.get("datetime_epoch")
                    if epoch is None:
                        continue
                    from datetime import datetime, timezone as _tz
                    utc_hr = datetime.fromtimestamp(int(epoch), _tz.utc).strftime("%Y-%m-%dT%H")
                    our_t = our_hours_map.get(utc_hr)
                    if our_t is None:
                        continue
                    diffs.append(float(h["temp_c"]) - our_t)
                if diffs:
                    n_rmse = len(diffs)
                    rmse_c = math.sqrt(sum(d * d for d in diffs) / n_rmse)

    # Compact JSON of VC hourly forecast path (future hours only) for later
    # re-analysis; skip for future_day where the payload format differs.
    vc_hourly_json = None
    try:
        future_hours = [
            {"dt": h.get("datetime"), "epoch": h.get("datetime_epoch"),
             "temp": h.get("temp_c"), "src": h.get("source")}
            for h in (vc_payload.get("forecast_hours") or [])
        ]
        if future_hours:
            vc_hourly_json = json.dumps(future_hours, default=str,
                                        separators=(",", ":"))
    except Exception:
        vc_hourly_json = None

    return {
        "event_id": event_id, "city": city,
        "target_date": target_date, "pulled_at_utc": pulled_at_utc, "kind": kind,
        "vc_projected_day_max_c":      vc_day_max,
        "vc_forecast_remaining_max_c": vc_fcst_remaining,
        "vc_day_vc_source":            vc_day_source,
        "blended_mu_c":                blended_mu,
        "blended_sigma_c":             blended_sigma,
        "adjusted_mu_c":               adjusted_mu,
        "adjusted_sigma_c":            adjusted_sigma,
        "current_temp_c":              cur_temp,
        "observed_max_so_far_c":       obs_max,
        "vc_vs_blended_mu_c":          d_blended,
        "vc_vs_adjusted_mu_c":         d_adjusted,
        "vc_vs_observed_max_c":        d_obs,
        "abs_vc_vs_blended":           abs_blended,
        "abs_vc_vs_adjusted":          abs_adjusted,
        "vc_bins_apart":               bins_apart,
        "vc_hourly_path_rmse_c":       rmse_c,
        "vc_hourly_path_n":            n_rmse,
        "flag_vc_disagreement_large":  flag_large,
        "flag_vc_warns_hotter":        flag_hotter,
        "flag_vc_warns_colder":        flag_colder,
        "vc_hourly_forecast_json":     vc_hourly_json,
    }


def _get_event_state(event_id: str) -> dict | None:
    """Return the most recent temp_events row with forecast/adjusted μ/σ."""
    if not event_id:
        return None
    try:
        import sqlite3
        from config import DB_PATH
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT forecast_mu_c, forecast_sigma_c, adjusted_mu_c, "
                "adjusted_sigma_c FROM temp_events WHERE event_id = ? "
                "ORDER BY scan_timestamp DESC LIMIT 1",
                (event_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except Exception:
        return None


def live_observation_run(events: list[dict] | None = None) -> dict:
    """
    Poll Visual Crossing for every active event, insert a live_observations
    row, and refresh temp_events current-state columns.
    """
    logger.info("=== LIVE OBSERVATION RUN START ===")
    pulled_at = _utc_now_iso()

    if events is None:
        from polymarket import search_temp_high_events
        from config import MIN_LIQUIDITY_USD
        events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)

    try:
        from visualcrossing import fetch_intraday
    except Exception as e:
        logger.error(f"Cannot import visualcrossing: {e}")
        return {"events": 0, "observations": 0, "errors": 1}

    counts = {"events": 0, "observations": 0, "errors": 0}

    for ev in _event_iter(events):
        counts["events"] += 1
        event_id = ev.get("event_id") or ""
        city     = ev.get("city")
        lat, lon = ev["lat"], ev["lon"]
        date_str = ev["date"]

        try:
            j = fetch_intraday(lat, lon)
        except Exception as e:
            logger.warning(f"VC intraday failed for {city}: {e}")
            counts["errors"] += 1
            continue

        bump_vc_usage(j.get("query_cost"))

        stations_json = json.dumps(j.get("current_stations") or [])
        obs_id = insert_live_observation(
            event_id              = event_id,
            city                  = city,
            date                  = date_str,
            lat                   = lat,
            lon                   = lon,
            pulled_at_utc         = pulled_at,
            observed_at_utc       = j.get("current_time"),
            observed_at_local     = j.get("current_time"),
            vc_source             = j.get("current_vc_source"),
            current_temp_c        = j.get("current_temp_c"),
            humidity              = j.get("current_humidity"),
            cloudcover            = j.get("current_cloudcover"),
            windspeed_kph         = j.get("current_windspeed_kph"),
            precip_mm             = j.get("current_precip_mm"),
            conditions            = j.get("current_conditions"),
            observed_max_so_far_c = j.get("observed_max_so_far_c"),
            stations              = stations_json,
            query_cost            = j.get("query_cost"),
        )
        counts["observations"] += 1

        # 1h / 3h temp change — depend on prior observations, not this one.
        # _temp_change looks up rows in the window that are strictly earlier,
        # so this must be called BEFORE writing the row above? No — the
        # new row is timestamped `pulled_at`; finding an obs ~1h ago will
        # correctly pick a prior row.  OK.
        current_temp = j.get("current_temp_c")

        # Expected-temp-now from latest ECMWF forecast_hourly.
        latest_ecmwf = get_latest_forecast_run(event_id, "ecmwf")
        expected_now = None
        if latest_ecmwf and current_temp is not None:
            expected_now = _interp_expected_temp(
                latest_ecmwf["id"], j.get("current_time") or pulled_at
            )

        actual_minus_expected = None
        if current_temp is not None and expected_now is not None:
            actual_minus_expected = current_temp - expected_now

        try:
            update_event_current_state(
                event_id,
                timezone                = j.get("timezone"),
                latest_observation_ts   = pulled_at,
                current_temp_c          = current_temp,
                observed_max_so_far_c   = j.get("observed_max_so_far_c"),
                expected_temp_now_c     = expected_now,
                actual_minus_expected_c = actual_minus_expected,
            )
        except Exception as e:
            logger.debug(f"event-state update failed for {event_id}: {e}")

        # --- VC forecast diagnostic (Phase 2c, shadow-only) ---
        try:
            diag = _build_vc_diagnostic_row(
                event_id      = event_id,
                city          = city,
                target_date   = date_str,
                pulled_at_utc = pulled_at,
                kind          = "same_day",
                vc_payload    = j,
                event_state   = _get_event_state(event_id),
            )
            if diag:
                insert_vc_forecast_diagnostic(diag)
                if diag.get("flag_vc_disagreement_large") or \
                   diag.get("flag_vc_warns_hotter") or \
                   diag.get("flag_vc_warns_colder"):
                    tags = []
                    if diag.get("flag_vc_disagreement_large"): tags.append("LARGE")
                    if diag.get("flag_vc_warns_hotter"): tags.append("hotter")
                    if diag.get("flag_vc_warns_colder"): tags.append("colder")
                    logger.info(
                        f"VCDiag: {city} {date_str} | vc_day_max="
                        f"{diag['vc_projected_day_max_c']:.2f}°C "
                        f"blended_mu={diag.get('blended_mu_c')} "
                        f"delta={diag.get('vc_vs_blended_mu_c'):+.2f}°C "
                        f"bins={diag.get('vc_bins_apart')} "
                        f"flags={','.join(tags)}"
                    )
        except Exception as e:
            logger.debug(f"VC diagnostic write failed for {event_id}: {e}")

    logger.info(
        f"Live observations: events={counts['events']} "
        f"observations={counts['observations']} errors={counts['errors']}"
    )
    logger.info("=== LIVE OBSERVATION RUN END ===")
    return counts


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------

def retention_run(older_than_days: int = 90) -> dict:
    """Nightly retention: purge forecast_hourly + live_observations +
    vc_forecast_diagnostics older than N days.  Summary tables are kept
    indefinitely."""
    from db import (
        purge_forecast_hourly, purge_live_observations,
        purge_vc_forecast_diagnostics,
    )
    logger.info("=== RETENTION RUN START ===")
    fh = purge_forecast_hourly(older_than_days)
    lo = purge_live_observations(older_than_days)
    vd = purge_vc_forecast_diagnostics(older_than_days)
    logger.info(f"Retention: purged forecast_hourly={fh} live_observations={lo} "
                f"vc_forecast_diagnostics={vd} (older than {older_than_days}d)")
    logger.info("=== RETENTION RUN END ===")
    return {
        "forecast_hourly_deleted":         fh,
        "live_observations_deleted":       lo,
        "vc_forecast_diagnostics_deleted": vd,
    }


# ---------------------------------------------------------------------------
# Future-day VC diagnostic loop (Phase 2c)
# ---------------------------------------------------------------------------

def vc_future_diagnostic_run(events: list[dict] | None = None) -> dict:
    """Pull VC forecast diagnostics for events that are 1..VC_DIAG_FUTURE_DAY_HORIZON
    days out.  Shadow-only — writes vc_forecast_diagnostics rows tagged
    kind='future_day'.  Runs every 2 hours alongside the forecast pull."""
    if VC_DIAG_FUTURE_DAY_HORIZON <= 0:
        logger.debug("VC future-day diagnostic disabled (horizon=0)")
        return {"events": 0, "diagnostics": 0, "skipped": 0, "errors": 0}

    logger.info("=== VC FUTURE-DAY DIAGNOSTIC RUN START ===")
    pulled_at = _utc_now_iso()

    if events is None:
        from polymarket import search_temp_high_events
        from config import MIN_LIQUIDITY_USD
        events = search_temp_high_events(min_liquidity=MIN_LIQUIDITY_USD)

    try:
        from visualcrossing import fetch_future_day
    except Exception as e:
        logger.error(f"Cannot import fetch_future_day: {e}")
        return {"events": 0, "diagnostics": 0, "errors": 1}

    counts = {"events": 0, "diagnostics": 0, "skipped": 0, "errors": 0}

    for ev in _event_iter(events):
        d = ev.get("date")
        da = _days_ahead(d)
        if da is None or da < 1 or da > VC_DIAG_FUTURE_DAY_HORIZON:
            counts["skipped"] += 1
            continue
        counts["events"] += 1
        event_id = ev.get("event_id") or ""
        city = ev.get("city")
        lat, lon = ev["lat"], ev["lon"]
        try:
            from datetime import date as _date
            j = fetch_future_day(lat, lon, _date.fromisoformat(d))
        except Exception as e:
            logger.warning(f"VC future_day failed for {city} {d}: {e}")
            counts["errors"] += 1
            continue

        bump_vc_usage(j.get("query_cost"))

        try:
            diag = _build_vc_diagnostic_row(
                event_id      = event_id,
                city          = city,
                target_date   = d,
                pulled_at_utc = pulled_at,
                kind          = "future_day",
                vc_payload    = j,
                event_state   = _get_event_state(event_id),
            )
            if diag:
                insert_vc_forecast_diagnostic(diag)
                counts["diagnostics"] += 1
                if diag.get("flag_vc_disagreement_large"):
                    logger.info(
                        f"VCDiag[D+{da}]: {city} {d} | "
                        f"vc_day_max={diag['vc_projected_day_max_c']:.2f}°C "
                        f"blended_mu={diag.get('blended_mu_c')} "
                        f"delta={diag.get('vc_vs_blended_mu_c'):+.2f}°C LARGE"
                    )
        except Exception as e:
            logger.debug(f"future_day diag insert failed for {event_id}: {e}")

    logger.info(
        f"VC future-day diagnostics: events={counts['events']} "
        f"diagnostics={counts['diagnostics']} "
        f"skipped={counts['skipped']} errors={counts['errors']}"
    )
    logger.info("=== VC FUTURE-DAY DIAGNOSTIC RUN END ===")
    return counts
