"""Tests for the deterministic English-to-strategy parser (`nlt.translate.rules`).

No LLM is involved anywhere in this file, and none of these tests requires an
API key. The highest-priority coverage here is the crossing-vs-state split --
see the module docstring in `rules.py` for why getting that wrong is the worst
mistake this component can make.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nlt.spec.models import StrategySpec
from nlt.translate.rules import CONDITION_PATTERNS, parse

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "strategies.json").read_text())


def _spec_of(text: str, **kw) -> StrategySpec:
    result = parse(text, **kw)
    assert result.spec is not None, f"expected a spec for {text!r}, got questions: {result.questions}"
    return result.spec


# --------------------------------------------------------------- crossing vs state

CROSSES_BELOW_PHRASES = [
    "rsi cracks 30",
    "rsi crosses below 30",
    "rsi falls below 30",
    "rsi breaks below 30",
    "rsi dips under 30",
    "rsi drops below 30",
    "rsi goes below 30",
    "RSI Cracks 30",
]

STATE_BELOW_PHRASES = [
    "rsi is below 30",
    "rsi is under 30",
    "while rsi is below 30",
    "rsi as long as it's under 30",
    "rsi below 30",
    "rsi under 30",
    "rsi < 30",
]

CROSSES_ABOVE_PHRASES = [
    "rsi crosses above 70",
    "rsi breaks above 70",
    "rsi goes above 70",
    "rsi rises above 70",
    "rsi climbs above 70",
]

STATE_ABOVE_PHRASES = [
    "rsi is above 70",
    "rsi is over 70",
    "rsi above 70",
    "rsi over 70",
    "rsi > 70",
]


@pytest.mark.parametrize("phrase", CROSSES_BELOW_PHRASES)
def test_crossing_below_phrasings(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when {phrase}, stop 1%, target 2%")
    assert spec.entry.kind == "compare"
    assert spec.entry.op == "crosses_below"
    assert spec.entry.right.value == 30.0


@pytest.mark.parametrize("phrase", STATE_BELOW_PHRASES)
def test_state_below_phrasings(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when {phrase}, stop 1%, target 2%")
    assert spec.entry.kind == "compare"
    assert spec.entry.op == "lt"
    assert spec.entry.right.value == 30.0


@pytest.mark.parametrize("phrase", CROSSES_ABOVE_PHRASES)
def test_crossing_above_phrasings(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when {phrase}, stop 1%, target 2%")
    assert spec.entry.op == "crosses_above"
    assert spec.entry.right.value == 70.0


@pytest.mark.parametrize("phrase", STATE_ABOVE_PHRASES)
def test_state_above_phrasings(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when {phrase}, stop 1%, target 2%")
    assert spec.entry.op == "gt"
    assert spec.entry.right.value == 70.0


# --------------------------------------------------------------------- instrument


@pytest.mark.parametrize("phrase", ["nifty", "NIFTY", "nifty 50"])
def test_instrument_nifty(phrase: str) -> None:
    spec = _spec_of(f"buy {phrase} when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY"


@pytest.mark.parametrize("phrase", ["banknifty", "bank nifty", "BANKNIFTY"])
def test_instrument_banknifty(phrase: str) -> None:
    spec = _spec_of(f"buy {phrase} when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "BANKNIFTY"


def test_instrument_defaults_to_nifty() -> None:
    spec = _spec_of("buy when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY"


# --------------------------------------------------------------------- direction


@pytest.mark.parametrize("phrase", ["buy", "long", "go long"])
def test_direction_long(phrase: str) -> None:
    spec = _spec_of(f"{phrase} nifty when rsi cracks 30, stop 1%, target 2%")
    assert spec.direction == "long"


@pytest.mark.parametrize("phrase", ["sell short", "short sell", "short", "go short"])
def test_direction_short(phrase: str) -> None:
    spec = _spec_of(f"{phrase} nifty when rsi crosses above 70, stop 1%, target 2%")
    assert spec.direction == "short"


def test_direction_defaults_to_long() -> None:
    spec = _spec_of("nifty when rsi cracks 30, stop 1%, target 2%")
    assert spec.direction == "long"


# --------------------------------------------------------------- indicator conditions


def test_moving_average_crossover() -> None:
    spec = _spec_of("buy nifty when 20 MA crosses above 50 MA, stop 1%, target 2%")
    ids = {i.id for i in spec.indicators}
    assert {"sma20", "sma50"} <= ids
    assert spec.entry.op == "crosses_above"


def test_moving_average_crossover_worded() -> None:
    spec = _spec_of(
        "buy nifty when 20 day moving average crosses the 50 day, stop 1%, target 2%"
    )
    ids = {i.id for i in spec.indicators}
    assert {"sma20", "sma50"} <= ids


def test_ema_above() -> None:
    spec = _spec_of("buy nifty when 20 EMA above 200 EMA, stop 1%, target 2%")
    ids = {i.id for i in spec.indicators}
    assert {"ema20", "ema200"} <= ids
    assert spec.entry.op == "gt"


def test_golden_cross() -> None:
    spec = _spec_of("buy nifty on golden cross, stop 1%, target 2%")
    ids = {i.id for i in spec.indicators}
    assert {"sma50", "sma200"} <= ids
    assert spec.entry.op == "crosses_above"


def test_death_cross_forces_short() -> None:
    spec = _spec_of("nifty on death cross, stop 1%, target 2%")
    assert spec.direction == "short"
    assert spec.entry.op == "crosses_below"


def test_price_above_moving_average() -> None:
    spec = _spec_of("buy nifty when price above the 200 day moving average, stop 1%, target 2%")
    assert spec.entry.op == "gt"
    assert spec.entry.left.name == "close"


def test_close_below_ema() -> None:
    spec = _spec_of("sell nifty when close below the 50 EMA, stop 1%, target 2%")
    assert spec.entry.op == "lt"


def test_macd_crosses_above_signal() -> None:
    spec = _spec_of("buy nifty when MACD crosses above signal, stop 1%, target 2%")
    assert spec.entry.op == "crosses_above"
    assert spec.entry.left.output == "macd"
    assert spec.entry.right.output == "signal"


def test_macd_turns_positive() -> None:
    spec = _spec_of("buy nifty when MACD turns positive, stop 1%, target 2%")
    assert spec.entry.op == "crosses_above"
    assert spec.entry.right.value == 0


def test_adx_above() -> None:
    spec = _spec_of("buy nifty when ADX above 25, stop 1%, target 2%")
    assert spec.entry.op == "gt"
    assert spec.entry.right.value == 25.0


def test_trend_is_strong_notes_assumption() -> None:
    result = parse("buy nifty when the trend is strong, stop 1%, target 2%")
    assert result.spec is not None
    assert any("adx" in n.lower() for n in result.notes)


def test_supertrend_bullish() -> None:
    spec = _spec_of("buy nifty when supertrend turns green, stop 1%, target 2%")
    assert spec.entry.op == "crosses_above"
    assert spec.entry.left.output == "direction"


def test_bollinger_touch_lower_band() -> None:
    spec = _spec_of("buy nifty when price touches the lower bollinger band, stop 1%, target 2%")
    assert spec.entry.right.output == "lower"


def test_bollinger_close_above_upper() -> None:
    spec = _spec_of("buy nifty when it closes above the upper band, stop 1%, target 2%")
    assert spec.entry.right.output == "upper"


def test_bollinger_squeeze_notes_assumption() -> None:
    result = parse("buy nifty on a bollinger squeeze, stop 1%, target 2%")
    assert result.spec is not None
    assert any("squeeze" in n.lower() for n in result.notes)


def test_stochastic_level() -> None:
    spec = _spec_of("buy nifty when stochastic below 20, stop 1%, target 2%")
    assert spec.entry.op == "lt"


def test_cci_level() -> None:
    spec = _spec_of("buy nifty when cci below -100, stop 1%, target 2%")
    assert spec.entry.op == "lt"
    assert spec.entry.right.value == -100.0


def test_williams_r_level() -> None:
    spec = _spec_of("buy nifty when williams %r below -80, stop 1%, target 2%")
    assert spec.entry.op == "lt"


def test_prior_day_high_breakout() -> None:
    spec = _spec_of("buy nifty when it breaks yesterday's high, stop 1%, target 2%")
    assert spec.entry.op == "crosses_above"
    ind = spec.indicators[0]
    assert ind.type == "prior_period"
    assert ind.params["period"] == "day"


def test_prior_week_low() -> None:
    spec = _spec_of("sell nifty when below last week's low, stop 1%, target 2%")
    ind = spec.indicators[0]
    assert ind.params["period"] == "week"


def test_breakout_high() -> None:
    spec = _spec_of("buy nifty when it breaks out to a 50 day high, stop 1%, target 2%")
    ind = spec.indicators[0]
    assert ind.type == "rolling_extremes"
    assert ind.params["length"] == 50


@pytest.mark.parametrize(
    "phrase,pattern_id",
    [
        ("bullish engulfing", "engulfing_bullish"),
        ("bearish engulfing", "engulfing_bearish"),
        ("hammer", "hammer"),
        ("hanging man", "hanging_man"),
        ("doji", "doji"),
        ("morning star", "morning_star"),
        ("evening star", "evening_star"),
        ("shooting star", "shooting_star"),
        ("bullish harami", "harami_bullish"),
        ("spinning top", "spinning_top"),
    ],
)
def test_candlestick_patterns(phrase: str, pattern_id: str) -> None:
    spec = _spec_of(f"buy nifty on a {phrase} candle, stop 1%, target 2%")
    assert spec.indicators[0].type == pattern_id
    assert spec.entry.kind == "is_true"


def test_percent_move_falls() -> None:
    spec = _spec_of("buy nifty when it falls 1%, stop 1%, target 2%")
    assert spec.entry.kind == "percent_change"
    assert spec.entry.value == -1.0


def test_percent_move_drops_over_bars() -> None:
    spec = _spec_of("buy nifty when it drops 2% in 3 days, stop 1%, target 2%")
    assert spec.entry.lookback == 3
    assert spec.entry.value == -2.0


# -------------------------------------------------------------------------- exits


def test_target_and_stop_combined() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30, exit at +2% or -1%")
    assert spec.exit.target_pct == 2.0
    assert spec.exit.stop_pct == 1.0


@pytest.mark.parametrize(
    "phrase",
    ["exit at +2%", "take profit 2%", "target 2%", "sell at 2% gain", "book profit at 2%"],
)
def test_target_phrasings(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when rsi cracks 30, {phrase}, stop 1%")
    assert spec.exit.target_pct == 2.0


@pytest.mark.parametrize("phrase", ["stop loss 1%", "SL 1%", "stop at 1%", "with a 1% stop"])
def test_stop_phrasings(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when rsi cracks 30, target 2%, {phrase}")
    assert spec.exit.stop_pct == 1.0


def test_atr_stop() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30, stop loss 1.5 times ATR, target 2%")
    assert spec.exit.stop_atr_mult == 1.5
    assert spec.exit.atr_id == "atr14"
    assert any(i.type == "atr" for i in spec.indicators)


@pytest.mark.parametrize("phrase", ["trailing stop 1%", "trail by 1%"])
def test_trailing_stop(phrase: str) -> None:
    spec = _spec_of(f"buy nifty when rsi cracks 30, {phrase}, target 2%")
    assert spec.exit.trailing_stop_pct == 1.0


def test_exit_condition() -> None:
    spec = _spec_of(
        "buy nifty when 20 MA crosses above 50 MA, exit when the 20 MA crosses below the 50 MA, stop 1%"
    )
    assert spec.exit.condition is not None
    assert spec.exit.condition.op == "crosses_below"


def test_max_bars_held() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30, hold for max 5 days, stop 1%, target 2%")
    assert spec.exit.max_bars_held == 5


def test_square_off_time() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30, square off at 15:15, stop 1%, target 2%")
    assert spec.schedule.square_off.hour == 15
    assert spec.schedule.square_off.minute == 15


# -------------------------------------------------------------------------- sizing


def test_lots_sizing() -> None:
    spec = _spec_of("buy 2 lots of nifty when rsi cracks 30, stop 1%, target 2%")
    assert spec.sizing.mode == "fixed_lots"
    assert spec.sizing.lots == 2


def test_risk_based_sizing() -> None:
    spec = _spec_of("buy nifty risk 1% per trade when rsi cracks 30, stop 1%, target 2%")
    assert spec.sizing.mode == "risk_based"
    assert spec.sizing.risk_pct == 1.0


def test_default_sizing_is_one_lot() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30, stop 1%, target 2%")
    assert spec.sizing.lots == 1


# ----------------------------------------------------------------------- combining


def test_and_combination() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30 and adx above 25, stop 1%, target 2%")
    assert spec.entry.kind == "all"
    assert len(spec.entry.conditions) == 2


def test_or_combination() -> None:
    spec = _spec_of(
        "buy nifty when rsi cracks 30 or macd turns positive, stop 1%, target 2%"
    )
    assert spec.entry.kind == "any"
    assert len(spec.entry.conditions) == 2


def test_three_condition_and() -> None:
    spec = _spec_of(
        "buy nifty when rsi cracks 30 and adx above 25 and macd turns positive, stop 1%, target 2%"
    )
    assert spec.entry.kind == "all"
    assert len(spec.entry.conditions) == 3


# ------------------------------------------------------------------------- sloppy


def test_sloppy_lowercase_no_punctuation() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30 exit at 2% sl 1%")
    assert spec.entry.op == "crosses_below"
    assert spec.exit.target_pct == 2.0
    assert spec.exit.stop_pct == 1.0


# ---------------------------------------------------------------------- refusals


def test_refuses_gibberish() -> None:
    result = parse("asdkjaslkdj random gibberish text here")
    assert result.spec is None
    assert result.questions


def test_refuses_unavailable_indicator() -> None:
    result = parse("buy nifty when the Elliott Wave count hits 5, stop 1%, target 2%")
    assert result.spec is None
    assert result.questions
    assert any("elliott wave" in q.text.lower() for q in result.questions)


def test_no_exit_at_all_asks() -> None:
    result = parse("buy nifty when rsi cracks 30")
    assert result.spec is None
    assert any(q.field.startswith("exit") for q in result.questions)


def test_no_stop_loss_never_defaults_one_in() -> None:
    result = parse("buy nifty when rsi cracks 30, target 2%")
    assert result.spec is None
    assert any(q.field == "exit.stop_pct" for q in result.questions)


# ---------------------------------------------------------------------- unparsed


def test_unparsed_is_populated_for_unrecognised_text() -> None:
    result = parse("buy nifty when the moon is full, stop 1%, target 2%")
    assert result.unparsed


def test_clean_input_leaves_nothing_unparsed() -> None:
    result = parse("Buy NIFTY when RSI cracks 30, exit at +2% or -1%")
    assert result.unparsed == []


# ------------------------------------------------------------------------ answers


def test_answers_round_trip_fills_missing_stop_loss() -> None:
    missing = parse("buy nifty when rsi cracks 30, target 2%")
    assert missing.spec is None
    assert any(q.field == "exit.stop_pct" for q in missing.questions)

    filled = parse("buy nifty when rsi cracks 30, target 2%", answers={"exit.stop_pct": "1"})
    assert filled.spec is not None
    assert filled.spec.exit.stop_pct == 1.0


# -------------------------------------------------------------------- determinism


def test_deterministic_ids() -> None:
    a = _spec_of("buy nifty when rsi cracks 30, stop 1%, target 2%")
    b = _spec_of("buy nifty when rsi cracks 30, stop 1%, target 2%")
    assert [i.id for i in a.indicators] == [i.id for i in b.indicators]


def test_deterministic_output() -> None:
    text = "Short BANKNIFTY when the 20 MA crosses below the 50 MA and ADX is above 25, stop loss 1.5 times ATR, take profit 3%"
    a = parse(text)
    b = parse(text)
    assert a.spec.model_dump_json() == b.spec.model_dump_json()


# ------------------------------------------------------------------------ fixtures


@pytest.mark.parametrize("case", FIXTURES, ids=[c["text"][:40] for c in FIXTURES])
def test_fixtures_validate(case: dict) -> None:
    """Every fixture spec, if produced, must pass StrategySpec validation on its own
    (a property already guaranteed by `parse`, since it constructs via the model)
    and the parser must never silently mis-handle what the fixture says it should do.
    """
    result = parse(case["text"])
    if case["expected"] == "spec":
        assert result.spec is not None, f"expected a spec: {case['note']}"
        # Constructing StrategySpec already validated it; re-validate defensively.
        StrategySpec.model_validate(result.spec.model_dump())
    else:
        assert result.spec is None, f"expected a refusal/questions: {case['note']}"
        assert result.questions, f"a refusal with no questions leaves the user with nothing: {case['note']}"


def test_every_fixture_spec_passes_validation() -> None:
    """Property test: every spec produced across all fixtures is a valid StrategySpec."""
    for case in FIXTURES:
        result = parse(case["text"])
        if result.spec is not None:
            StrategySpec.model_validate(result.spec.model_dump())


# ---------------------------------------------------------------------------
# Regression: pattern-table ordering
#
# Two rules matched "the previous day's high" and the crossing one listed
# "closes above" while sitting earlier in the table, so it shadowed the state
# rule completely -- "closes above yesterday's high" was read as a crossing and
# the dedicated rule for it was unreachable.
#
# Nothing else catches this: both rules produce a valid spec, so validation,
# the readback and every existing test stayed green while the meaning was wrong.
# ---------------------------------------------------------------------------

def _entry_op(text: str):
    result = parse(f"buy nifty when {text}, target 2%, stop loss 1%")
    assert result.spec is not None, f"failed to parse: {text!r} ({result.unparsed})"
    return result.spec.entry.op


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("price closes above the previous day's high", "gt"),
        ("closes above yesterday's high", "gt"),
        ("breaks yesterday's high", "crosses_above"),
        ("crosses above the previous day's high", "crosses_above"),
    ],
)
def test_closes_above_is_a_state_breaks_is_a_crossing(phrase, expected):
    assert _entry_op(phrase) == expected


def test_no_pattern_is_shadowed_by_an_earlier_one():
    """Every rule's own example must be matched by that rule, not an earlier one.

    A rule whose example is claimed by something above it is dead code, and its
    intended meaning silently never fires.
    """
    shadowed = []
    for i, rule in enumerate(CONDITION_PATTERNS):
        for earlier in CONDITION_PATTERNS[:i]:
            if earlier.regex.search(rule.example.lower()):
                shadowed.append(f"{rule.name!r} (example {rule.example!r}) shadowed by {earlier.name!r}")
                break
    assert not shadowed, "unreachable pattern rules:\n  " + "\n  ".join(shadowed)


@pytest.mark.parametrize(
    "verb, expected",
    [
        ("goes under", "crosses_below"), ("dips below", "crosses_below"),
        ("slips below", "crosses_below"), ("drops under", "crosses_below"),
        ("crosses under", "crosses_below"), ("falls under", "crosses_below"),
        ("goes over", "crosses_above"), ("crosses over", "crosses_above"),
        ("moves above", "crosses_above"), ("pops above", "crosses_above"),
    ],
)
def test_widened_crossing_verbs(verb, expected):
    assert _entry_op(f"rsi {verb} 30") == expected


@pytest.mark.parametrize("subject", ["price", "the price", "close", "nifty", "it", ""])
def test_leading_subject_does_not_break_parsing(subject):
    """"price closes above ..." must not leave "price" as an unparsed fragment."""
    result = parse(f"buy nifty when {subject} closes above yesterday's high, target 2%, stop 1%")
    assert result.spec is not None, result.unparsed
    assert not result.unparsed
