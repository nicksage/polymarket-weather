"""
city_features.py — Static per-city feature loader.

Reads bot/ml/data/city_static_features.json once on first import; provides
get_city_features(city) which returns a stable dict the feature builder
folds into every training row for that city.  See the JSON file's _README
field for the source and definitions.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_DATA_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "city_static_features.json"

# Default zeros for cities we don't recognize — feature_builder will apply
# these silently and HistGradientBoosting handles "all zero" rows fine
# (the model just doesn't split on them).
_FALLBACK = {
    "elevation_m": 0.0,
    "koppen_zone": "?",
    "koppen_main": "?",
    "hemisphere": 1,
    "coastal":    0,
}


@lru_cache(maxsize=1)
def _load_all() -> dict:
    if not _DATA_PATH.exists():
        logger.warning(f"city_static_features.json missing at {_DATA_PATH}")
        return {}
    try:
        with open(_DATA_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"failed to load {_DATA_PATH}: {e}")
        return {}
    return data.get("cities", {})


def get_city_features(city: str | None) -> dict:
    """Return the static feature dict for a city, or _FALLBACK if missing."""
    if not city:
        return dict(_FALLBACK)
    data = _load_all()
    if city in data:
        return data[city]
    # Try case-insensitive match
    lower = city.lower()
    for k, v in data.items():
        if k.lower() == lower:
            return v
    logger.debug(f"no static features for city={city!r}; using fallback")
    return dict(_FALLBACK)


def koppen_one_hot(koppen_main: str | None) -> tuple[float, float, float, float]:
    """Encode the main Köppen letter (A/B/C/D) as one-hot over (A, B, C, D).
    Unknown / Polar (E) maps to all zeros — model treats these as a 5th
    implicit category."""
    main = (koppen_main or "?").upper()
    return (
        1.0 if main == "A" else 0.0,
        1.0 if main == "B" else 0.0,
        1.0 if main == "C" else 0.0,
        1.0 if main == "D" else 0.0,
    )
