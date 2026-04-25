"""
solar.py — Lightweight clear-sky solar geometry / irradiance helpers.

Used by the v2.0 feature builder to compute:
  * solar elevation angle at a given (lat, lon, datetime_utc)
  * approximate clear-sky GHI (W/m²) using a simple cos(zenith) model
  * integrated clear-sky GHI from `decision_dt_local` to a peak hour

These are deliberately simple — no aerosol, no Linke turbidity, no diffuse
component split.  The intent is to give the ML model a usable "potential
sunlight" signal at zero dependency cost.  When the application needs
publication-grade radiation estimates, pull in `pvlib` instead.

References:
  Iqbal, M. (1983) "An Introduction to Solar Radiation" — chapters 3-4
  for the geometry; chapter 6 for the simple cosine clear-sky model used
  here (Bird-style at zero atmospheric mass).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

# Solar constant W/m² at the top of the atmosphere
_SOLAR_CONSTANT = 1361.0


def _day_of_year(dt: datetime) -> int:
    return dt.timetuple().tm_yday


def _equation_of_time_minutes(doy: int) -> float:
    """Spencer (1971) approximation for the equation of time, minutes."""
    b = 2.0 * math.pi * (doy - 1) / 365.0
    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(b)
        - 0.032077 * math.sin(b)
        - 0.014615 * math.cos(2 * b)
        - 0.040849 * math.sin(2 * b)
    )


def _declination_rad(doy: int) -> float:
    """Solar declination in radians, simple Cooper formula."""
    return math.radians(23.45) * math.sin(math.radians(360.0 * (284 + doy) / 365.0))


def solar_elevation_deg(lat: float, lon: float, dt_utc: datetime) -> float:
    """Solar elevation above the horizon (degrees) at a UTC moment.
    Negative when the sun is below the horizon."""
    doy = _day_of_year(dt_utc)
    decl = _declination_rad(doy)
    eot = _equation_of_time_minutes(doy)

    # Local solar time (hours), accounting for longitude and equation of time.
    utc_hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    solar_time = utc_hours + lon / 15.0 + eot / 60.0
    hour_angle = math.radians(15.0 * (solar_time - 12.0))

    lat_rad = math.radians(lat)
    sin_alpha = (
        math.sin(lat_rad) * math.sin(decl)
        + math.cos(lat_rad) * math.cos(decl) * math.cos(hour_angle)
    )
    sin_alpha = max(-1.0, min(1.0, sin_alpha))
    return math.degrees(math.asin(sin_alpha))


def clear_sky_ghi_wm2(lat: float, lon: float, dt_utc: datetime) -> float:
    """Approximate clear-sky GHI in W/m² at (lat, lon, dt_utc).
    Returns 0 when the sun is below the horizon.  Uses a 0.7-air-mass
    transmission heuristic to be slightly more realistic than pure
    cos(zenith) — good enough for ML input, not for solar engineering."""
    elev_deg = solar_elevation_deg(lat, lon, dt_utc)
    if elev_deg <= 0:
        return 0.0
    cos_zenith = math.sin(math.radians(elev_deg))
    # Crude atmospheric attenuation: ~0.7 transmission at zenith=0,
    # decreasing toward the horizon via cos(zenith) exponent ~0.678.
    air_mass = 1.0 / max(cos_zenith, 0.05)
    transmission = 0.7 ** (air_mass ** 0.678)
    return _SOLAR_CONSTANT * cos_zenith * transmission


def clear_sky_ghi_remaining_kwh(
    lat: float,
    lon: float,
    decision_dt_utc: datetime,
    end_local_hour: int = 16,
    tz_offset_hours: float = 0.0,
    step_minutes: int = 30,
) -> float:
    """Integrated clear-sky GHI in kWh/m² from `decision_dt_utc` to the
    target day's `end_local_hour` local time.  The integration is a
    simple trapezoidal sum over `step_minutes`-spaced samples.

    `tz_offset_hours` is the local timezone offset (e.g., -6 for CDT).
    Caller supplies it to define when "end_local_hour" is in UTC terms.

    Returns 0 if the decision is already past end_local_hour or if the
    sun is set for the rest of the window.
    """
    # Convert end_local_hour to UTC for this date
    decision_local = decision_dt_utc + timedelta(hours=tz_offset_hours)
    end_local = decision_local.replace(
        hour=end_local_hour, minute=0, second=0, microsecond=0
    )
    end_utc = end_local - timedelta(hours=tz_offset_hours)
    if end_utc <= decision_dt_utc:
        return 0.0

    total_wh = 0.0
    t = decision_dt_utc
    step = timedelta(minutes=step_minutes)
    prev_w = clear_sky_ghi_wm2(lat, lon, t)
    while t < end_utc:
        t_next = min(t + step, end_utc)
        next_w = clear_sky_ghi_wm2(lat, lon, t_next)
        # trapezoid: avg W * hours
        dt_hours = (t_next - t).total_seconds() / 3600.0
        total_wh += 0.5 * (prev_w + next_w) * dt_hours
        prev_w = next_w
        t = t_next
    return total_wh / 1000.0   # Wh -> kWh


def noon_solar_elevation_deg(lat: float, lon: float, dt_utc: datetime) -> float:
    """Sun elevation at solar noon for the given date — useful as a 'how
    high will the sun get today' geometry feature."""
    doy = _day_of_year(dt_utc)
    decl = math.degrees(_declination_rad(doy))
    return 90.0 - abs(lat - decl)
