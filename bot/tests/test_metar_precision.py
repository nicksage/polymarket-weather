"""
test_metar_precision.py — Lock in the METAR T-group parser + the
conservative-bound semantics for body-only readings.

Regression context: Atlanta 2026-06-12.  KATL synoptic METARs at :52
past the hour reported `T0328...` (32.8°C tenths-precision) while
body-only 5-minute cycles reported "33" (whole-°C body value, which
NWS API serves as 33.0°C with no tenths precision).  The bot took
max() across both and recorded observed_max = 33.0°C, while the actual
peak was 32.8°C.  Wunderground (which uses T-group / DSM precision)
correctly displayed 32°C as the day's high.

The fix:
   - Parse T-group when raw_message contains one → tenths precision
   - For body-only readings, apply -0.5°C conservative lower bound
     (body "33" means precise was in [32.5, 33.5); we claim 32.5)

Run:
    cd bot
    python -m pytest tests/test_metar_precision.py -v
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOT_DIR = os.path.dirname(_HERE)
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)

from scripts.intraday_predictor import (   # type: ignore
    parse_metar_t_group,
    precise_temp_from_cycle,
    METAR_BODY_CONSERVATIVE_OFFSET_C,
)


# ============================================================
# parse_metar_t_group
# ============================================================

def test_t_group_positive_temp_and_dewpoint():
    """Atlanta 18:52 UTC actual METAR snippet."""
    metar = ("KATL 121852Z 28008G17KT 10SM FEW045TCU SCT100 BKN250 "
             "33/21 A3001 RMK AO2 SLP150 TCU ALQDS T03280206")
    temp, dewpoint = parse_metar_t_group(metar)
    assert abs(temp - 32.8) < 1e-9, f"expected 32.8°C, got {temp}"
    assert abs(dewpoint - 20.6) < 1e-9, f"expected 20.6°C, got {dewpoint}"


def test_t_group_negative_temperature():
    """Winter case: temp below freezing."""
    metar = "KORD 122353Z 27015KT 10SM CLR M02/M08 A3010 RMK AO2 SLP188 T10171083"
    temp, dewpoint = parse_metar_t_group(metar)
    assert abs(temp - (-1.7)) < 1e-9, f"expected -1.7°C, got {temp}"
    assert abs(dewpoint - (-8.3)) < 1e-9, f"expected -8.3°C, got {dewpoint}"


def test_t_group_positive_temp_negative_dewpoint():
    """Dry conditions: positive temp, negative dewpoint."""
    metar = "KDEN 121723Z 28015KT 10SM CLR 33/M05 A3005 RMK AO2 SLP132 T03281050"
    temp, dewpoint = parse_metar_t_group(metar)
    assert abs(temp - 32.8) < 1e-9
    assert abs(dewpoint - (-5.0)) < 1e-9


def test_t_group_absent_returns_none():
    """Body-only METAR with no T-group in remarks."""
    metar = "KATL 121752Z 28009KT 10SM CLR 33/22 A3005"
    temp, dewpoint = parse_metar_t_group(metar)
    assert temp is None
    assert dewpoint is None


def test_t_group_empty_message():
    """Defensive: empty/None inputs."""
    assert parse_metar_t_group(None) == (None, None)
    assert parse_metar_t_group("") == (None, None)


def test_t_group_does_not_false_match_other_t_codes():
    """METARs contain other T-prefixed codes (TCU = towering cumulus).
    The regex must require the digit pattern, not just the literal T."""
    metar = "KATL 121852Z 28008G17KT 10SM TCU SCT100 33/21 A3001"
    # No T-group with proper digit pattern → returns None
    temp, dewpoint = parse_metar_t_group(metar)
    assert temp is None
    assert dewpoint is None


# ============================================================
# precise_temp_from_cycle — the wrapper that picks tenths vs. whole
# ============================================================

def test_precise_temp_prefers_t_group_when_present():
    """When raw_message has a T-group, use it regardless of API body value."""
    api_value = 33.0   # NWS API serves the rounded body
    raw = "KATL 121852Z 33/21 A3001 RMK AO2 T03280206"
    precise, precision = precise_temp_from_cycle(api_value, raw)
    assert abs(precise - 32.8) < 1e-9
    assert precision == "tenths"


def test_precise_temp_body_only_applies_conservative_offset():
    """No T-group → body value with -0.5°C conservative bound.

    Body of '33' means precise temp was in [32.5, 33.5).  Conservative
    interpretation (truthful lower bound): 32.5.
    """
    api_value = 33.0
    raw = "KATL 121752Z 28009KT 10SM CLR 33/22 A3005"   # no T-group
    precise, precision = precise_temp_from_cycle(api_value, raw)
    expected = 33.0 + METAR_BODY_CONSERVATIVE_OFFSET_C   # 32.5
    assert abs(precise - expected) < 1e-9, (
        f"body-only reading should apply conservative offset "
        f"({METAR_BODY_CONSERVATIVE_OFFSET_C}); got {precise}, "
        f"expected {expected}"
    )
    assert precision == "whole"


def test_precise_temp_missing_data_returns_none():
    api_value = None
    raw = None
    precise, precision = precise_temp_from_cycle(api_value, raw)
    assert precise is None
    assert precision == "missing"


def test_precise_temp_missing_body_with_t_group_uses_t_group():
    """Edge case: API didn't parse temp but T-group is in remarks."""
    api_value = None
    raw = "KATL 121852Z RMK AO2 T03280206"
    precise, precision = precise_temp_from_cycle(api_value, raw)
    assert abs(precise - 32.8) < 1e-9
    assert precision == "tenths"


# ============================================================
# The Atlanta 2026-06-12 scenario, end to end
# ============================================================

def test_atlanta_20260612_scenario_produces_correct_observed_max():
    """Replay of Atlanta 2026-06-12's actual METAR sequence.

    The bot's observed_max should NOT ratchet to 33.0°C from the
    5-minute body-only cycles — it should cap at 32.8°C from the
    T-group precision in the :52 synoptics, with body-only cycles
    contributing conservative-bounded values like 32.5°C.

    This is the exact bug that put the bot's 92-93°F Atlanta position
    on the wrong side of the 91.5°F bin floor.
    """
    cycles = [
        # 5-min body-only cycles, scattered through the afternoon
        ("KATL 121725Z 28009KT 10SM SCT042TCU BKN250 33/22 A3005", 33.0),
        ("KATL 121730Z 28010KT 10SM SCT042TCU BKN250 32/22 A3005", 32.0),
        ("KATL 121815Z 28010KT 10SM FEW045TCU BKN250 33/21 A3005", 33.0),
        ("KATL 121825Z 28009KT 10SM FEW045TCU BKN250 33/21 A3005", 33.0),
        ("KATL 121845Z 28008KT 10SM FEW045TCU BKN250 33/21 A3001", 33.0),
        # Synoptic at 18:52 with T-group
        ("KATL 121852Z 28008G17KT 10SM FEW045TCU SCT100 BKN250 "
         "33/21 A3001 RMK AO2 SLP150 TCU ALQDS T03280206", 33.0),
    ]
    precise_values = []
    for raw_msg, api_t_c in cycles:
        precise, _ = precise_temp_from_cycle(api_t_c, raw_msg)
        precise_values.append(precise)

    # The maximum of all precise readings is what observed_max would be.
    observed_max = max(precise_values)

    # T-group says 32.8, body-only cycles contribute 32.5 (33.0 - 0.5).
    # max(32.5, 32.5, 32.5, 32.5, 32.5, 32.8) = 32.8
    assert abs(observed_max - 32.8) < 1e-9, (
        f"Atlanta 2026-06-12 replay: observed_max should be 32.8°C "
        f"(from T-group), not the body-only rounded-up value.  "
        f"Got {observed_max}"
    )

    # And critically: 32.8°C = 91.04°F, BELOW the 91.5°F floor of the
    # 92-93°F bin.  That means the bot would correctly recognize this
    # market belongs in 90-91°F, not 92-93°F.
    obs_max_f = observed_max * 9 / 5 + 32
    BIN_FLOOR_92_93 = 91.5
    assert obs_max_f < BIN_FLOOR_92_93, (
        f"observed_max ({obs_max_f:.2f}°F) should fall below the "
        f"92-93°F bin floor (91.5°F).  Got >= floor."
    )