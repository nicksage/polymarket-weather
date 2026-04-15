"""
solar.py — Sunrise/sunset utilities for live-adjustment σ shrinkage.

Thin wrapper around `astral` so the rest of the codebase can ask simple
questions like "what fraction of daylight remains at this UTC time for this
lat/lon on this date?" without dealing with timezone bookkeeping.

Cached per (round(lat,1), round(lon,1), date) because the values are
deterministic and exact precision doesn't matter at the degree-latitude
level our σ shrinkage consumes.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional

from astral import LocationInfo
from astral.sun import sun


@lru_cache(maxsize=4096)
def _sun_for_date(lat_key: float, lon_key: float, d: date) -> dict | None:
    """Raw astral result in UTC for a given date; None on failure/polar."""
    try:
        loc = LocationInfo(latitude=lat_key, longitude=lon_key)
        return sun(loc.observer, date=d, tzinfo=timezone.utc)
    except Exception:
        return None


def _active_daylight_window(
    lat: float, lon: float, at_utc: datetime,
) -> tuple[datetime, datetime] | None:
    """Return (sunrise_utc, sunset_utc) for the daylight window that
    contains `at_utc`, or the NEXT window if we're currently at night.

    Astral's `sun(date=D, tz=UTC)` returns sunrise/sunset *as they occur on
    UTC date D*, which for western-hemisphere locations can give a sunset
    earlier than the sunrise if the local day straddles the UTC date
    boundary.  We probe D-1, D, D+1 and select a pair where sunrise < sunset
    and `at_utc` sits within (or nearest future) the sunrise → sunset span.
    """
    d = at_utc.date()
    lat_k, lon_k = round(lat, 1), round(lon, 1)

    # Collect all sunrises and sunsets across a 3-day window; astral binds
    # both to the same UTC date, so for western-hemisphere locations the
    # "sunset" on UTC date D is actually that calendar-UTC day's evening
    # which may correspond to the local evening of D-1.  Pair each sunrise
    # with the next sunset > sunrise to reconstruct valid daylight windows.
    sunrises: list[datetime] = []
    sunsets:  list[datetime] = []
    for offset in (-1, 0, 1, 2):
        s = _sun_for_date(lat_k, lon_k, d + timedelta(days=offset))
        if not s:
            continue
        sunrises.append(s["sunrise"])
        sunsets.append(s["sunset"])
    if not sunrises or not sunsets:
        return None

    sunrises.sort()
    sunsets.sort()

    candidates: list[tuple[datetime, datetime]] = []
    for rise in sunrises:
        later = [s for s in sunsets if s > rise]
        if later:
            candidates.append((rise, min(later)))
    if not candidates:
        return None

    # Choose: (a) window containing at_utc; else (b) the earliest window with sunset > at_utc
    containing = [w for w in candidates if w[0] <= at_utc <= w[1]]
    if containing:
        return containing[0]
    future = [w for w in candidates if w[1] > at_utc]
    if future:
        return min(future, key=lambda w: w[0])
    # All behind us — return the latest
    return max(candidates, key=lambda w: w[1])


def sun_events(lat: float, lon: float, at_utc: datetime) -> tuple[datetime, datetime] | None:
    """Public helper: the sunrise/sunset pair for the daylight window
    containing (or next after) the given UTC instant."""
    return _active_daylight_window(lat, lon, at_utc)


def remaining_daylight_fraction(
    lat: float, lon: float, at_utc: datetime,
) -> Optional[float]:
    """Return the fraction of total daylight remaining at `at_utc`.

    Returns:
        1.0  if `at_utc` is at or before sunrise (full day ahead)
        0.0  if `at_utc` is at or after sunset of the active window
        frac in [0,1] during daylight
        None if solar events cannot be computed for this location/date
    """
    window = _active_daylight_window(lat, lon, at_utc)
    if window is None:
        return None
    sunrise, sunset = window
    total = (sunset - sunrise).total_seconds()
    if total <= 0:
        return None
    if at_utc <= sunrise:
        return 1.0
    if at_utc >= sunset:
        return 0.0
    remaining = (sunset - at_utc).total_seconds()
    return max(0.0, min(1.0, remaining / total))
