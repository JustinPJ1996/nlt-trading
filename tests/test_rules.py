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
from nlt.translate.readback import describe
from nlt.translate.rules import CONDITION_PATTERNS, _known_tickers, parse

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "strategies.json").read_text())


def _spec_of(text: str, **kw) -> StrategySpec:
    result = parse(text, **kw)
    assert result.spec is not None, (
        f"expected a spec for {text!r}, got questions: {result.questions}"
    )
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


@pytest.mark.parametrize("phrase", ["nifty", "NIFTY"])
def test_instrument_nifty(phrase: str) -> None:
    spec = _spec_of(f"buy {phrase} when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY"
    assert spec.instrument.trade_as == "index"


@pytest.mark.parametrize("phrase", ["banknifty", "bank nifty", "BANKNIFTY"])
def test_instrument_banknifty(phrase: str) -> None:
    spec = _spec_of(f"buy {phrase} when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "BANKNIFTY"
    assert spec.instrument.trade_as == "index"


def test_instrument_defaults_to_nifty() -> None:
    spec = _spec_of("buy when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY"
    assert spec.instrument.trade_as == "index"


# ------------------------------------------------------- index vs universe


# "NIFTY" bare -- with no stock count attached at all -- is unambiguously the
# index, in every phrasing a strategy is likely to open with.
INDEX_PHRASINGS = [
    "buy nifty when rsi cracks 30",
    "Buy NIFTY when RSI cracks 30",
    "buy nifty when rsi < 30",
    "short nifty when rsi crosses above 70",
    "sell nifty when rsi crosses above 70",
    "i want to buy nifty when rsi cracks 30",
]


@pytest.mark.parametrize("text", INDEX_PHRASINGS)
def test_bare_nifty_is_always_the_index(text: str) -> None:
    spec = _spec_of(f"{text}, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY"
    assert spec.instrument.trade_as == "index"


# "stocks"/"shares" (or "stocks in") next to a Nifty count is the unambiguous
# signal that the *basket* is meant, in every spelling users actually typed.
UNIVERSE_PHRASINGS = [
    ("buy nifty 50 stocks when rsi cracks 30", "NIFTY 50"),
    ("buy NIFTY50 stocks when rsi cracks 30", "NIFTY 50"),
    ("buy nifty50 stocks when rsi cracks 30", "NIFTY 50"),
    ("buy stocks in nifty 500 when rsi cracks 30", "NIFTY 500"),
    ("buy nifty 100 stocks when rsi cracks 30", "NIFTY 100"),
    ("buy nifty 100 stocks above the 200 dma when rsi cracks 30", "NIFTY 100"),
]


@pytest.mark.parametrize("text,canonical", UNIVERSE_PHRASINGS)
def test_universe_spellings_resolve_to_canonical_symbol(text: str, canonical: str) -> None:
    spec = _spec_of(f"{text}, stop 1%, target 2%")
    assert spec.instrument.symbol == canonical
    assert spec.instrument.trade_as == "stock"
    assert spec.instrument.is_universe


@pytest.mark.parametrize(
    "text",
    [
        "buy nifty 100 when rsi cracks 30",
        "buy nifty500 when rsi cracks 30",
        "buy when nifty 100 rsi cracks 30",
    ],
)
def test_nifty_100_and_500_are_always_the_universe_even_without_the_word_stocks(text: str) -> None:
    """NIFTY 100/500 name no tradeable index in this platform (only NIFTY and
    BANKNIFTY do), so mentioning them is never ambiguous -- unlike NIFTY 50."""
    spec = _spec_of(f"{text}, stop 1%, target 2%")
    assert spec.instrument.trade_as == "stock"
    assert spec.instrument.is_universe


def test_nifty_50_with_no_qualifier_is_a_genuine_ambiguity() -> None:
    """NIFTY 50 is both the literal name of the index and the name of the
    50-stock basket -- an honest refusal beats guessing either way."""
    result = parse("buy nifty 50 when rsi cracks 30, stop 1%, target 2%")
    assert result.spec is None
    assert result.questions
    assert result.questions[0].field == "instrument.symbol"


def test_nifty_50_stocks_resolves_the_ambiguity() -> None:
    spec = _spec_of("buy nifty 50 stocks when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY 50"
    assert spec.instrument.trade_as == "stock"


def test_instantly_is_read_as_filler_not_a_feature() -> None:
    spec = _spec_of("Instantly buy NIFTY 50 stocks when RSI cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY 50"
    assert spec.instrument.trade_as == "stock"


# ------------------------------------------------------------------- stocks


@pytest.mark.parametrize("ticker", ["TCS", "RELIANCE", "ITC", "INFY"])
def test_known_tickers_are_recognised(ticker: str) -> None:
    spec = _spec_of(f"buy {ticker} when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == ticker
    assert spec.instrument.trade_as == "stock"
    assert not spec.instrument.is_universe


@pytest.mark.parametrize("ticker", ["tcs", "reliance", "itc", "infy"])
def test_known_tickers_are_recognised_lowercase(ticker: str) -> None:
    spec = _spec_of(f"buy {ticker} when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.symbol == ticker.upper()


def test_unknown_word_after_buy_is_not_invented_as_a_ticker() -> None:
    """'dips' is not an NSE symbol -- it must land in unparsed, not become a
    stock symbol just because it sits where a ticker would."""
    result = parse("buy dips when rsi cracks 30, stop 1%, target 2%")
    assert result.spec is None
    assert any("dips" in u for u in result.unparsed)


@pytest.mark.parametrize("word", ["buy", "when", "the", "stop", "target"])
def test_ordinary_english_words_are_never_mistaken_for_tickers(word: str) -> None:
    assert word.upper() not in _known_tickers()


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
    spec = _spec_of("buy nifty when 20 day moving average crosses the 50 day, stop 1%, target 2%")
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
    spec = _spec_of("buy nifty when rsi cracks 30 or macd turns positive, stop 1%, target 2%")
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
        assert result.questions, (
            f"a refusal with no questions leaves the user with nothing: {case['note']}"
        )


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
                shadowed.append(
                    f"{rule.name!r} (example {rule.example!r}) shadowed by {earlier.name!r}"
                )
                break
    assert not shadowed, "unreachable pattern rules:\n  " + "\n  ".join(shadowed)


@pytest.mark.parametrize(
    "verb, expected",
    [
        ("goes under", "crosses_below"),
        ("dips below", "crosses_below"),
        ("slips below", "crosses_below"),
        ("drops under", "crosses_below"),
        ("crosses under", "crosses_below"),
        ("falls under", "crosses_below"),
        ("goes over", "crosses_above"),
        ("crosses over", "crosses_above"),
        ("moves above", "crosses_above"),
        ("pops above", "crosses_above"),
    ],
)
def test_widened_crossing_verbs(verb, expected):
    assert _entry_op(f"rsi {verb} 30") == expected


@pytest.mark.parametrize("subject", ["price", "the price", "close", "nifty", "it", ""])
def test_leading_subject_does_not_break_parsing(subject):
    """ "price closes above ..." must not leave "price" as an unparsed fragment."""
    result = parse(f"buy nifty when {subject} closes above yesterday's high, target 2%, stop 1%")
    assert result.spec is not None, result.unparsed
    assert not result.unparsed


# --------------------------------------------------------------- timeframe


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("use 5-minute candles", "5m"),
        ("use 5 minute candles", "5m"),
        ("on the 15 minute chart", "15m"),
        ("1 hour candles", "1h"),
        ("use 1-hour candles", "1h"),
        ("daily", "1d"),
        ("1 day timeframe", "1d"),
        ("30 minute bars", "30m"),
        ("3 minute timeframe", "3m"),
        ("1 minute candles", "1m"),
    ],
)
def test_supported_timeframes_are_recognised(phrase: str, expected: str) -> None:
    spec = _spec_of(f"buy nifty when rsi cracks 30, stop 1%, target 2%, {phrase}")
    assert spec.instrument.timeframe == expected


def test_no_timeframe_word_defaults_to_daily() -> None:
    spec = _spec_of("buy nifty when rsi cracks 30, stop 1%, target 2%")
    assert spec.instrument.timeframe == "1d"


def test_bare_intraday_does_not_invent_a_bar_size() -> None:
    """'intraday' alone tells us nothing about *which* sub-day bar size is
    meant, so it must not silently become 5m/15m/etc, and it must not surface
    as an unparsed leftover either -- it is a recognised word, just one that
    correctly carries no timeframe information on its own."""
    result = parse("buy nifty when rsi cracks 30, stop 1%, target 2%, intraday")
    assert result.spec is not None, result.questions
    assert result.spec.instrument.timeframe == "1d"
    assert not result.unparsed


@pytest.mark.parametrize(
    "phrase",
    [
        "use 2 minute candles",
        "7 minute chart",
        "4 hour candles",
        "3 day timeframe",
    ],
)
def test_unsupported_timeframes_are_refused_not_rounded(phrase: str) -> None:
    result = parse(f"buy nifty when rsi cracks 30, stop 1%, target 2%, {phrase}")
    assert result.spec is None
    assert result.questions
    assert result.questions[0].field == "instrument.timeframe"


def test_use_5_minute_candles_on_a_stock_strategy() -> None:
    spec = _spec_of("buy itc when rsi drops below 30, stop 1%, target 2%. use 5-minute candles.")
    assert spec.instrument.symbol == "ITC"
    assert spec.instrument.timeframe == "5m"


# ------------------------------------------------------- stocks: crossing vs state


@pytest.mark.parametrize(
    "text",
    [
        "buy tcs when rsi cracks 30, stop 1%, target 2%",
        "buy tcs when rsi drops below 30, stop 1%, target 2%",
    ],
)
def test_crossing_still_holds_for_a_single_stock(text: str) -> None:
    spec = _spec_of(text)
    assert spec.instrument.symbol == "TCS"
    assert spec.entry.op == "crosses_below"


def test_state_still_holds_for_a_stock_universe() -> None:
    spec = _spec_of("buy nifty 100 stocks when rsi is below 30, stop 1%, target 2%")
    assert spec.instrument.symbol == "NIFTY 100"
    assert spec.entry.op == "lt"


# ------------------------------------------------------------- determinism (stocks)


def test_deterministic_output_for_a_stock_strategy() -> None:
    text = "Buy TCS when RSI drops below 30, target 5%, stop 2%"
    a = parse(text)
    b = parse(text)
    assert a.spec is not None
    assert a.spec.model_dump_json() == b.spec.model_dump_json()


def test_deterministic_output_for_a_universe_strategy() -> None:
    text = "Buy Nifty 100 stocks above the 200 DMA when RSI drops below 40. 2% stop loss, 6% take profit"
    a = parse(text)
    b = parse(text)
    assert a.spec is not None
    assert a.spec.model_dump_json() == b.spec.model_dump_json()


# ------------------------------------------------------- the 70 user strategies


USER_FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "user_strategies.json").read_text()
)
_NEVER_A_STRATEGY = {"question", "robo", "fundamental", "event"}

# Pinned so a future change that silently drops coverage (or, far worse,
# starts turning a "question"/"robo"/"fundamental"/"event" sentence into a
# tradeable spec) fails loudly instead of drifting unnoticed.
#
# Grew from 1 to 3 when the parser learned: spelled-out percents and "at RSI
# N" level phrasing (#13, a single stock with an explicit stop and target),
# and inline-period indicators plus indicator-vs-indicator comparisons (#59,
# a NIFTY 100 universe with an EMA crossover and an explicit stop and
# target). Every other previously-refused sentence in the 70 stays refused,
# most now for a cleaner, more specific reason (VWAP needs an intraday
# timeframe and cannot run on a stock basket at all yet; a still-missing
# stop loss) instead of a garbled "I didn't understand" pile-up.
_EXPECTED_SPEC_COUNT = 3


def test_all_70_user_strategies_do_not_crash() -> None:
    for case in USER_FIXTURES:
        parse(case["text"])  # must not raise


def test_no_wrong_parses_among_the_70_user_strategies() -> None:
    """The single most important assertion in this file: a sentence classified
    as a question, a robo-advisory request, a fundamentals lookup or a
    corporate-action event must never come back as a tradeable spec. Parsing
    "Which sectors are strongest right now?" into a strategy is the worst
    failure this parser can make.
    """
    wrong = []
    for case in USER_FIXTURES:
        result = parse(case["text"])
        if result.spec is not None and case["klass"] in _NEVER_A_STRATEGY:
            wrong.append((case["n"], case["klass"], case["text"]))
    assert not wrong, f"produced a spec for out-of-scope sentences: {wrong}"


def test_pinned_spec_count_across_the_70_user_strategies() -> None:
    produced = [case["n"] for case in USER_FIXTURES if parse(case["text"]).spec is not None]
    assert len(produced) == _EXPECTED_SPEC_COUNT, (
        f"expected {_EXPECTED_SPEC_COUNT} of the 70 user strategies to produce a spec, "
        f"got {len(produced)}: {produced}. If this grew, update _EXPECTED_SPEC_COUNT "
        "deliberately; if it shrank, something regressed."
    )


def test_every_produced_spec_among_the_70_validates() -> None:
    for case in USER_FIXTURES:
        result = parse(case["text"])
        if result.spec is not None:
            StrategySpec.model_validate(result.spec.model_dump())


# ============================================================================
# Six gaps closed: anaphora, value-first exits, inline-period indicators,
# indicator-vs-indicator comparisons, spelled-out numbers/"at RSI N", and VWAP.
# ============================================================================

# ------------------------------------------------------------------ anaphora


@pytest.mark.parametrize(
    "text",
    [
        "buy nifty when rsi drops below 30, sell when it crosses above 70, stop 1%, target 2%",
        "buy tcs when rsi cracks 30 and sell when it goes above 70, stop 1%, target 2%",
    ],
)
def test_anaphoric_it_resolves_to_the_entry_indicator(text: str) -> None:
    spec = _spec_of(text)
    assert spec.entry.op == "crosses_below"
    assert spec.exit.condition is not None
    assert spec.exit.condition.op == "crosses_above"
    assert spec.exit.condition.right.value == 70.0
    # "it" must resolve to the *same* indicator id as the entry's RSI, not a
    # second, independently-declared one.
    assert spec.exit.condition.left.name == spec.entry.left.name
    assert len([i for i in spec.indicators if i.type == "rsi"]) == 1


def test_anaphoric_it_with_inline_period_carries_the_same_length() -> None:
    spec = _spec_of(
        "buy nifty when rsi14 drops below 30, sell when it crosses above 70, stop 1%, target 2%"
    )
    assert spec.entry.left.name == "rsi14"
    assert spec.exit.condition.left.name == "rsi14"


def test_anaphoric_it_is_ambiguous_with_two_entry_indicators() -> None:
    """The entry names both RSI and ADX -- 'it' cannot mean both, so this must
    raise a Question, never silently pick one."""
    result = parse(
        "buy nifty when rsi drops below 30 and adx above 25, sell when it crosses 70, "
        "stop 1%, target 2%"
    )
    assert result.spec is None
    assert result.questions
    assert result.questions[0].field == "exit.condition"
    assert "rsi" not in result.questions[0].text.lower().split("mean either the ")[0]


def test_it_with_no_recognised_antecedent_falls_through_to_unparsed() -> None:
    """No indicator at all in the entry -- 'it' has nothing to resolve to, so
    this must fail safe (unparsed), not crash."""
    result = parse(
        "buy nifty when price touches the lower bollinger band, sell when it crosses 70, "
        "stop 1%, target 2%"
    )
    assert result.spec is None


# ------------------------------------------------------------- value-first exits


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("sell at 1% profit", 1.0),
        ("Sell at 2% profit", 2.0),
        ("book 3% profit", 3.0),
        ("price reaches 1% profit", 1.0),
    ],
)
def test_value_first_target_phrasings(phrase: str, expected: float) -> None:
    spec = _spec_of(f"buy nifty when rsi cracks 30, {phrase}, stop loss 1%")
    assert spec.exit.target_pct == expected


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("stop loss at 0.8%", 0.8),
        ("stop loss 0.8%", 0.8),
    ],
)
def test_value_first_and_at_stop_phrasings(phrase: str, expected: float) -> None:
    spec = _spec_of(f"buy nifty when rsi cracks 30, target 2%, {phrase}")
    assert spec.exit.stop_pct == expected


# ------------------------------------------------------------- inline-period


@pytest.mark.parametrize(
    "phrase,ind_id,length",
    [
        ("rsi7 < 28", "rsi7", 7),
        ("rsi14 < 30", "rsi14", 14),
        ("adx14 > 18", "adx14", 14),
        ("ema8 crosses above ema21", "ema8", 8),
    ],
)
def test_inline_period_indicators_are_recognised(phrase: str, ind_id: str, length: int) -> None:
    spec = _spec_of(f"buy nifty when {phrase}, stop 1%, target 2%")
    ids = {i.id: i for i in spec.indicators}
    assert ind_id in ids
    assert ids[ind_id].params["length"] == length


@pytest.mark.parametrize("text", ["rsi0", "rsi9999"])
def test_inline_period_zero_or_absurd_is_refused_not_clamped(text: str) -> None:
    result = parse(f"buy nifty when {text} < 30, stop 1%, target 2%")
    assert result.spec is None
    assert result.questions


def test_inline_period_ema9999_is_refused_not_clamped() -> None:
    result = parse("buy nifty when ema9999 crosses above ema21, stop 1%, target 2%")
    assert result.spec is None
    assert result.questions


# --------------------------------------------------------- indicator-vs-indicator


@pytest.mark.parametrize(
    "phrase",
    ["ema8 crosses above ema21", "close crosses above vwap"],
)
def test_indicator_vs_indicator_crossing(phrase: str) -> None:
    spec = _spec_of(f"buy tcs when {phrase}, stop 1%, target 2%, use 5-minute candles")
    assert spec.entry.kind == "compare"
    assert spec.entry.op == "crosses_above"
    assert isinstance(spec.entry.left, type(spec.entry.right))  # both Ref


@pytest.mark.parametrize(
    "phrase,left_id,right_id",
    [
        ("ema8 > ema21", "ema8", "ema21"),
        ("close > vwap", "close", "vwap"),
    ],
)
def test_indicator_vs_indicator_state(phrase: str, left_id: str, right_id: str) -> None:
    spec = _spec_of(f"buy tcs when {phrase}, stop 1%, target 2%, use 5-minute candles")
    assert spec.entry.op == "gt"
    assert spec.entry.left.name == left_id
    assert spec.entry.right.name == right_id


# ------------------------------------------------------- spelled-out numbers


def test_spelled_out_percent_stop_and_target() -> None:
    spec = _spec_of(
        "buy reliance when rsi cracks 30 with 5 percent stop loss and 10 percent target"
    )
    assert spec.exit.stop_pct == 5.0
    assert spec.exit.target_pct == 10.0


def test_at_rsi_level_is_a_crossing_direction_dependent() -> None:
    long_spec = _spec_of("buy reliance at rsi 30 with 5 percent stop loss and 10 percent target")
    assert long_spec.entry.op == "crosses_below"
    assert long_spec.entry.right.value == 30.0

    short_spec = _spec_of(
        "sell short reliance at rsi 70 with 5 percent stop loss and 10 percent target"
    )
    assert short_spec.entry.op == "crosses_above"
    assert short_spec.entry.right.value == 70.0


def test_rsi_between_becomes_an_all_of_two_bounds() -> None:
    spec = _spec_of("buy nifty when rsi is between 40-60, stop 1%, target 2%")
    assert spec.entry.kind == "all"
    ops = sorted((c.op, c.right.value) for c in spec.entry.conditions)
    assert ops == [("gte", 40.0), ("lte", 60.0)]


# -------------------------------------------------------------------- vwap


def test_vwap_on_daily_bars_refuses_naming_the_timeframe() -> None:
    result = parse("buy tcs when close > vwap, stop 1%, target 2%")
    assert result.spec is None
    assert result.questions
    assert result.questions[0].field == "instrument.timeframe"
    assert "intraday" in result.questions[0].text.lower()


def test_vwap_on_intraday_timeframe_parses() -> None:
    spec = _spec_of("buy tcs when close > vwap, stop 1%, target 2%, use 15-minute candles")
    assert any(i.type == "vwap" for i in spec.indicators)
    assert spec.instrument.timeframe == "15m"


def test_vwap_on_a_stock_universe_refuses_regardless_of_timeframe() -> None:
    """Baskets are daily-only in this platform today (see nlt/engine/basket.py),
    so VWAP on a universe can never run, not even on an intraday timeframe."""
    result = parse(
        "buy nifty 100 stocks when close > vwap, stop 1%, target 2%, use 5-minute candles"
    )
    assert result.spec is None
    assert result.questions
    assert "basket" in result.questions[0].text.lower()


# ---------------------------------------------------- crossing vs state, stocks


@pytest.mark.parametrize(
    "text,expected_op",
    [
        ("rsi14 cracks 30", "crosses_below"),
        ("rsi14 < 30", "lt"),
        ("rsi7 crosses above 70", "crosses_above"),
        ("rsi7 > 70", "gt"),
    ],
)
def test_inline_period_still_honours_crossing_vs_state(text: str, expected_op: str) -> None:
    spec = _spec_of(f"buy tcs when {text}, stop 1%, target 2%")
    assert spec.entry.op == expected_op


# --------------------------------------------------------------- backtest window


@pytest.mark.parametrize(
    ("phrase", "last", "unit", "year"),
    [
        ("backtest this for the last 3 months", 3, "months", None),
        ("backtest the last 2 years", 2, "years", None),
        ("backtest the past 1 year", 1, "years", None),
        ("backtest over the last 6 months", 6, "months", None),
        ("backtest for 2024", None, None, 2024),
        ("full year 2024", None, None, 2024),
    ],
)
def test_a_test_period_is_read_and_kept_off_the_spec(phrase, last, unit, year) -> None:
    """Tradl's prompts all end "Backtest the last N years"; it used to block them."""
    result = parse(f"buy itc when rsi cracks 30, stop 1%, target 2%. {phrase}.")
    assert result.spec is not None, result.questions
    assert result.unparsed == []
    assert (result.period.last, result.period.unit, result.period.year) == (last, unit, year)
    without = parse("buy itc when rsi cracks 30, stop 1%, target 2%.").spec
    keep = {"name", "description"}
    assert result.spec.model_dump(exclude=keep) == without.model_dump(exclude=keep)


def test_a_vague_test_period_sets_no_dates_but_is_named() -> None:
    result = parse("buy itc when rsi cracks 30, stop 1%, target 2%, for last year")
    assert result.spec is not None
    assert result.period is None
    assert any("for last year" in n for n in result.notes)


def test_a_lookback_inside_a_rule_is_not_a_test_period() -> None:
    """ "The highest high of the last 20 days" is part of the entry, not the dates."""
    result = parse("buy itc when close is above the highest high of the last 20 days, stop 1%")
    assert result.period is None


# ------------------------------------------------------- bar size and overnight

_BASE = "buy RELIANCE when rsi crosses below 30 on {}, target 2%, stop loss 1%"


@pytest.mark.parametrize(
    ("words", "timeframe"),
    [
        ("hourly bars", "1h"),
        ("hourly candles", "1h"),
        ("hourly chart", "1h"),
        ("60 minute candles", "1h"),
        ("1 hour candles", "1h"),
        ("15-minute bars", "15m"),
        ("daily bars", "1d"),
    ],
)
def test_bar_size_wording(words, timeframe) -> None:
    result = parse(_BASE.format(words))
    assert result.spec is not None, result.questions
    assert result.spec.instrument.timeframe == timeframe


def test_an_hour_long_average_is_not_a_bar_size() -> None:
    result = parse("buy RELIANCE when close is above 20-hour ema, target 2%, stop loss 1%")
    assert result.spec is None or result.spec.instrument.timeframe == "1d"


def test_hold_overnight_turns_off_the_square_off_and_says_so() -> None:
    result = parse(_BASE.format("15 minute candles") + ", hold overnight")
    assert result.spec is not None, result.questions
    assert result.spec.schedule.intraday is False
    text = describe(result.spec)
    assert "Hold overnight" in text
    assert "Square off" not in text


def test_without_hold_overnight_minute_bars_still_square_off() -> None:
    result = parse(_BASE.format("15 minute candles"))
    assert result.spec.schedule.intraday is True
    assert "Square off at 15:15" in describe(result.spec)


def test_hold_overnight_and_square_off_together_is_a_question() -> None:
    result = parse(_BASE.format("15 minute candles") + ", hold overnight, square off at 15:15")
    assert result.spec is None
    assert [q.field for q in result.questions] == ["schedule.intraday"]


# --------------------------------------------------------------- determinism


def test_deterministic_output_for_indicator_vs_indicator() -> None:
    text = "Buy NIFTY 100 stocks when ema8 crosses above ema21 and adx14 > 20, sell at 3% profit, stop loss 1.5%"
    a = parse(text)
    b = parse(text)
    assert a.spec is not None
    assert a.spec.model_dump_json() == b.spec.model_dump_json()


def test_deterministic_output_for_anaphoric_exit() -> None:
    text = "buy nifty when rsi drops below 30, sell when it crosses above 70, stop 1%, target 2%"
    a = parse(text)
    b = parse(text)
    assert a.spec is not None
    assert a.spec.model_dump_json() == b.spec.model_dump_json()


# ------------------------------------------------------------ tap-to-answer

_NO_STOP = "buy nifty when rsi cracks 30, target 2%"
_IT = "buy nifty when rsi crosses below 30 and adx > 25, sell when it crosses above 70, stop 1%"
_BOTH = _BASE.format("15 minute candles") + ", hold overnight, square off at 15:15"


def test_the_stop_question_offers_buttons_and_takes_typing() -> None:
    q = parse(_NO_STOP).questions[0]
    assert q.field == "exit.stop_pct"
    assert [c.value for c in q.choices] == ["0.5", "1", "2"]
    assert q.answerable


@pytest.mark.parametrize("value", ["0.5", "1", "2"])
def test_every_stop_button_is_understood(value) -> None:
    result = parse(_NO_STOP, answers={"exit.stop_pct": value})
    assert result.spec.exit.stop_pct == float(value)


def test_a_target_can_be_picked_alongside_the_stop() -> None:
    questions = parse("buy nifty when rsi cracks 30").questions
    target = next(q for q in questions if q.field == "exit.rule")
    assert [c.value for c in target.choices] == ["2", "3", "5"]
    result = parse("buy nifty when rsi cracks 30", answers={"exit.stop_pct": "1", "exit.rule": "3"})
    assert (result.spec.exit.stop_pct, result.spec.exit.target_pct) == (1.0, 3.0)


def test_a_target_alone_never_stands_in_for_the_stop() -> None:
    result = parse("buy nifty when rsi cracks 30", answers={"exit.rule": "3"})
    assert result.spec is None
    assert "exit.stop_pct" in [q.field for q in result.questions]


def test_overnight_or_square_off_is_answered_by_a_button() -> None:
    q = parse(_BOTH).questions[0]
    assert [c.value for c in q.choices] == ["overnight", "square_off"]
    assert not q.answerable
    held = parse(_BOTH, answers={"schedule.intraday": "overnight"}).spec
    assert held.schedule.intraday is False
    closed = parse(_BOTH, answers={"schedule.intraday": "square_off"}).spec
    assert closed.schedule.intraday is True
    assert "Square off at 15:15" in describe(closed)


def test_an_unknown_answer_to_a_button_question_asks_again() -> None:
    result = parse(_BOTH, answers={"schedule.intraday": "yes"})
    assert result.spec is None
    assert [q.field for q in result.questions] == ["schedule.intraday"]


def test_which_indicator_it_means_is_a_button_per_indicator() -> None:
    q = parse(_IT).questions[0]
    assert q.field == "exit.condition"
    assert [(c.label, c.value) for c in q.choices] == [("RSI", "rsi"), ("ADX", "adx")]
    picked = parse(_IT, answers={"exit.condition": "adx"}).spec
    assert picked.exit.condition.left.name.startswith("adx")
    # CCI is a family we could read, but the sentence never named it.
    assert parse(_IT, answers={"exit.condition": "cci"}).spec is None


def test_a_question_only_a_new_description_can_answer_says_so() -> None:
    q = parse("buy nifty, stop 1%").questions[0]
    assert q.field == "entry.condition"
    assert not q.answerable and q.choices == []


# ------------------------------------------- "50-day" said with a hyphen
#
# 2026-10-16: "50-day SMA" was not read at all, though our own refusal (a daily
# average on a daily chart) tells users to type "close above the 50-day SMA".
# And on intraday candles "50 day SMA" was silently read as 50 *candles*.


@pytest.mark.parametrize("words", ["50-day sma", "50 day sma", "50-day SMA", "50 sma"])
def test_a_50_day_average_reads_on_daily_candles(words: str) -> None:
    result = parse(f"buy nifty when close is above the {words}, stop 1%")
    assert result.spec is not None, result.questions
    assert result.spec.indicators[0].params == {"length": 50}


def test_the_phrase_our_refusal_suggests_actually_reads() -> None:
    """`_intraday_only_refusal` tells a user with a daily average on a daily
    chart to say "close above the 50-day SMA" instead; that must then read."""
    assert parse("buy nifty when close above the 50-day SMA, stop 1%").spec is not None


def test_the_daily_close_above_a_50_day_average_on_daily_candles_is_that_average() -> None:
    spec = parse("buy nifty when the daily close is above the 50-day sma, stop 1%").spec
    assert spec is not None
    assert spec.instrument.timeframe == "1d"
    assert [(i.type, i.params) for i in spec.indicators] == [("sma", {"length": 50})]


@pytest.mark.parametrize(
    "condition",
    [
        "close is above the 50-day sma",
        "close is above the 50 day sma",
        "close is above the 50 dma",
        "20 ema crosses above 50 day ema",
        "20-day ema crosses above 50 ema",
    ],
)
def test_an_average_said_in_days_on_intraday_candles_is_asked_not_read_as_candles(
    condition: str,
) -> None:
    result = parse(f"buy nifty on 15 minute candles when {condition}, stop 1%")
    assert result.spec is None
    assert "is not the same as" in result.questions[0].text


def test_an_average_in_candles_on_intraday_candles_still_reads() -> None:
    result = parse("buy nifty on 15 minute candles when close is above the 50 sma, stop 1%")
    assert result.spec is not None
