"""Tests for `nlt.translate.readback.describe`.

No LLM here either -- the readback is generated mechanically from the spec, so
these tests check that every field a spec can carry actually shows up in the
text a non-technical user would read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from nlt.spec.models import (
    All,
    Any_,
    Compare,
    Const,
    ExitRules,
    IndicatorSpec,
    Instrument,
    IsTrue,
    Not,
    PercentChange,
    Ref,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)
from nlt.translate.readback import describe, format_inr
from nlt.translate.rules import parse

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "strategies.json").read_text())


def _minimal_spec(entry, indicators=None, **exit_kwargs) -> StrategySpec:
    return StrategySpec(
        name="Test Strategy",
        description="test",
        indicators=indicators or [],
        entry=entry,
        exit=ExitRules(stop_pct=1.0, **exit_kwargs),
    )


# ------------------------------------------------------------------- node coverage


def test_compare_renders() -> None:
    spec = _minimal_spec(
        Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
    )
    text = describe(spec)
    assert "RSI(14)" in text
    assert "crosses below" in text
    assert "30" in text


def test_is_true_renders() -> None:
    spec = _minimal_spec(
        IsTrue(ref=Ref(name="hammer")),
        indicators=[IndicatorSpec(id="hammer", type="hammer", params={})],
    )
    assert "Hammer" in describe(spec)


def test_percent_change_renders() -> None:
    spec = _minimal_spec(PercentChange(ref=Ref(name="close"), lookback=3, op="lte", value=-1.0))
    text = describe(spec)
    assert "falls 1% or more" in text
    assert "3 days" in text or "3 bars" in text


@pytest.mark.parametrize(
    ("op", "value", "words"),
    [
        ("lte", -2.0, "falls 2% or more"),
        ("lt", -2.0, "falls more than 2%"),
        ("gte", 2.0, "rises 2% or more"),
        ("gt", 2.0, "rises more than 2%"),
    ],
)
def test_percent_change_says_whether_the_boundary_counts(op, value, words) -> None:
    """A fall of exactly 2% triggers "lte -2" but not "lt -2"; the readback must say which."""
    spec = _minimal_spec(PercentChange(ref=Ref(name="close"), lookback=1, op=op, value=value))
    assert words in describe(spec)


def test_all_renders() -> None:
    spec = _minimal_spec(
        All(
            conditions=[
                Compare(op="gt", left=Ref(name="close"), right=Const(value=100)),
                Compare(op="lt", left=Ref(name="close"), right=Const(value=200)),
            ]
        )
    )
    text = describe(spec)
    assert " and " in text


def test_any_renders() -> None:
    spec = _minimal_spec(
        Any_(
            conditions=[
                Compare(op="gt", left=Ref(name="close"), right=Const(value=100)),
                Compare(op="lt", left=Ref(name="close"), right=Const(value=50)),
            ]
        )
    )
    assert " or " in describe(spec)


def test_not_renders() -> None:
    spec = _minimal_spec(
        Not(condition=Compare(op="gt", left=Ref(name="close"), right=Const(value=100)))
    )
    assert "NOT" in describe(spec)


def test_bars_ago_renders() -> None:
    spec = _minimal_spec(
        Compare(op="gt", left=Ref(name="close", bars_ago=2), right=Const(value=100))
    )
    assert "2 days ago" in describe(spec)


def test_deeply_nested_condition_renders_without_raising() -> None:
    nested = All(
        conditions=[
            Any_(
                conditions=[
                    Compare(op="gt", left=Ref(name="close"), right=Const(value=100)),
                    Not(condition=Compare(op="lt", left=Ref(name="close"), right=Const(value=50))),
                ]
            ),
            Compare(op="crosses_above", left=Ref(name="close"), right=Const(value=200)),
        ]
    )
    spec = _minimal_spec(nested)
    text = describe(spec)
    assert " and " in text
    assert " or " in text
    assert "NOT" in text


@pytest.mark.parametrize(
    "cond",
    [
        Compare(op="lt", left=Ref(name="close"), right=Const(value=1)),
        Compare(op="lte", left=Ref(name="close"), right=Const(value=1)),
        Compare(op="gt", left=Ref(name="close"), right=Const(value=1)),
        Compare(op="gte", left=Ref(name="close"), right=Const(value=1)),
        Compare(op="eq", left=Ref(name="close"), right=Const(value=1)),
        Compare(op="crosses_above", left=Ref(name="close"), right=Const(value=1)),
        Compare(op="crosses_below", left=Ref(name="close"), right=Const(value=1)),
        IsTrue(ref=Ref(name="doji")),
        PercentChange(ref=Ref(name="close"), op="gte", value=1.0),
        All(conditions=[Compare(op="gt", left=Ref(name="close"), right=Const(value=1))]),
        Any_(conditions=[Compare(op="gt", left=Ref(name="close"), right=Const(value=1))]),
        Not(condition=Compare(op="gt", left=Ref(name="close"), right=Const(value=1))),
    ],
)
def test_every_condition_node_type_renders_without_raising(cond) -> None:
    indicators = [IndicatorSpec(id="doji", type="doji", params={})] if _uses_doji(cond) else []
    spec = _minimal_spec(cond, indicators=indicators)
    text = describe(spec)
    assert isinstance(text, str) and text


def _uses_doji(cond) -> bool:
    return isinstance(cond, IsTrue) and cond.ref.name == "doji"


# ---------------------------------------------------------------------- rupee format


@pytest.mark.parametrize(
    "value,expected",
    [
        (2000, "Rs 2,000"),
        (100000, "Rs 1,00,000"),
        (5000, "Rs 5,000"),
        (1234567, "Rs 12,34,567"),
        (500, "Rs 500"),
    ],
)
def test_indian_digit_grouping(value: float, expected: str) -> None:
    assert format_inr(value) == expected


# ------------------------------------------------------------------- completeness


def test_completeness_every_optional_field_appears() -> None:
    """Build a spec exercising every optional field; every value must be visible.

    This is what stops a future field from being invisible to the user -- if
    someone adds a field to StrategySpec and forgets to render it, this test
    should start failing the moment the field is actually used.
    """
    spec = StrategySpec(
        name="Full Coverage Strategy",
        description="exercise every field",
        instrument=Instrument(
            symbol="BANKNIFTY",
            timeframe="1d",
            trade_as="option",
            option_type="CE",
            strike_offset=1,
            expiry="monthly",
        ),
        indicators=[
            IndicatorSpec(id="rsi14", type="rsi", params={"length": 14}),
            IndicatorSpec(id="adx14", type="adx", params={"length": 14}),
            IndicatorSpec(id="atr14", type="atr", params={"length": 14}),
        ],
        direction="long",
        entry=All(
            conditions=[
                Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30)),
                Compare(op="gt", left=Ref(name="adx14", output="adx"), right=Const(value=25)),
            ]
        ),
        exit=ExitRules(
            target_pct=2.0,
            stop_pct=1.0,
            trailing_stop_pct=0.5,
            stop_atr_mult=1.5,
            atr_id="atr14",
            condition=Compare(op="crosses_above", left=Ref(name="rsi14"), right=Const(value=70)),
            max_bars_held=5,
        ),
        sizing=Sizing(mode="fixed_lots", lots=3),
        risk=RiskLimits(
            max_daily_loss=2500.0,
            max_loss_per_trade=6000.0,
            max_concurrent_positions=2,
            max_lots=4,
        ),
        schedule=Schedule(intraday=True),
    )

    text = describe(spec)

    # instrument
    assert "BANKNIFTY" in text
    assert "call" in text.lower()
    assert "monthly" in text.lower()
    assert "1 strike" in text.lower() or "strike" in text.lower()
    # entry
    assert "RSI(14)" in text
    assert "ADX(14)" in text
    # exit
    assert "+2%" in text
    assert "-1%" in text
    assert "0.5%" in text
    assert "1.5x" in text
    assert "ATR(14)" in text
    assert "5 days" in text or "5 bars" in text
    assert "70" in text  # exit condition threshold
    # sizing
    assert "3 lots" in text
    # risk -- the rupee loss limits are deliberately absent: nothing enforces
    # them yet (see test_the_readback_never_promises_a_loss_limit_nothing_enforces)
    assert format_inr(2500.0) not in text
    assert format_inr(6000.0) not in text
    assert "2 positions" in text
    assert "4 lots" in text


# ------------------------------------------------------------------------ round-trip

ROUND_TRIP_CASES = [
    ("Buy NIFTY when RSI cracks 30, exit at +2% or -1%", ["NIFTY", "RSI", "30", "2%", "1%"]),
    (
        "buy on golden cross, stop 1%, target 2%",
        ["Simple Moving Average", "50", "200", "crosses above"],
    ),
    (
        "Buy NIFTY when MACD crosses above signal, stop loss 1%, target 2%",
        ["MACD", "signal"],
    ),
    (
        "Buy NIFTY when stochastic is below 20, stop 1%, target 2%",
        ["Stochastic", "20"],
    ),
    (
        # "2 lots" of an index is a quantity of 2: a lot is an options contract
        # term, and saying "2 lots" about something that has no lots would be
        # the readback repeating the user's words rather than reporting what
        # the engine will actually do.
        "Buy 2 lots of NIFTY when RSI cracks 30, stop 1%, target 2%",
        ["Quantity 2", "RSI"],
    ),
]


@pytest.mark.parametrize("text,expected_terms", ROUND_TRIP_CASES)
def test_round_trip_contains_key_concepts(text: str, expected_terms: list[str]) -> None:
    result = parse(text)
    assert result.spec is not None, f"expected a spec for {text!r}: {result.questions}"
    rendered = describe(result.spec)
    for term in expected_terms:
        assert term.lower() in rendered.lower(), f"{term!r} missing from readback:\n{rendered}"


# ------------------------------------------------------------- stocks & universes


def test_completeness_single_stock_instrument() -> None:
    spec = StrategySpec(
        name="Single Stock",
        description="test",
        instrument=Instrument(symbol="TCS", trade_as="stock", timeframe="15m"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        exit=ExitRules(stop_pct=1.0, target_pct=2.0),
    )
    text = describe(spec)
    assert "TCS" in text
    assert "NSE" in text
    assert "15-minute" in text


def test_completeness_universe_instrument_states_symbol_and_count() -> None:
    """The whole safety point of a universe readback: a user must not be able
    to miss that "NIFTY 100" means a hundred separate instruments, not one."""
    spec = StrategySpec(
        name="Universe Strategy",
        description="test",
        instrument=Instrument(symbol="NIFTY 100", trade_as="stock"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=40)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        exit=ExitRules(stop_pct=2.0, target_pct=6.0),
    )
    text = describe(spec)
    assert "NIFTY 100" in text
    assert "100" in text
    assert "all" in text.lower()


def test_completeness_universe_instrument_shows_actual_membership_count() -> None:
    """NIFTY 500's bundled snapshot actually has 501 constituents, not 500 --
    the readback must say what is really going to be backtested, not what the
    index's name implies."""
    spec = StrategySpec(
        name="Universe Strategy",
        description="test",
        instrument=Instrument(symbol="NIFTY 500", trade_as="stock"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=40)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        exit=ExitRules(stop_pct=2.0, target_pct=6.0),
    )
    text = describe(spec)
    assert "501" in text


def test_index_instrument_unchanged_by_the_stock_work() -> None:
    spec = StrategySpec(
        name="Index Strategy",
        description="test",
        instrument=Instrument(symbol="NIFTY", trade_as="index"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        exit=ExitRules(stop_pct=1.0, target_pct=2.0),
    )
    lines = describe(spec).splitlines()
    assert "  NIFTY index" in lines
    assert "  Daily candles" in lines


def test_the_bar_size_is_its_own_line_even_when_daily() -> None:
    """Its own line so it can be labelled: daily is usually assumed, not said."""
    spec = StrategySpec(
        name="Single Stock",
        description="test",
        instrument=Instrument(symbol="TCS", trade_as="stock", timeframe="1d"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        exit=ExitRules(stop_pct=1.0, target_pct=2.0),
    )
    lines = describe(spec).splitlines()
    assert lines[3:5] == ["  TCS (NSE)", "  Daily candles"]


STOCK_ROUND_TRIP_CASES = [
    (
        "Buy Nifty 100 stocks above the 200 DMA when RSI drops below 40. 2% stop loss, 6% take profit",
        ["NIFTY 100", "100 stocks", "RSI(14)", "crosses below", "40", "+6%", "-2%"],
    ),
    (
        "Buy TCS when RSI drops below 30, target 5%, stop 2%",
        ["TCS", "NSE", "RSI(14)", "crosses below", "30", "+5%", "-2%"],
    ),
    (
        "Buy NIFTY when RSI cracks 30, target 2%, stop 1%",
        ["NIFTY index", "Daily candles", "RSI(14)", "crosses below", "30", "+2%", "-1%"],
    ),
    (
        "Buy ITC when RSI drops below 30, stop 1%, target 2%. use 5-minute candles.",
        ["ITC", "NSE", "5-minute"],
    ),
]


@pytest.mark.parametrize("text,expected_terms", STOCK_ROUND_TRIP_CASES)
def test_stock_and_universe_round_trip_contains_key_concepts(
    text: str, expected_terms: list[str]
) -> None:
    result = parse(text)
    assert result.spec is not None, f"expected a spec for {text!r}: {result.questions}"
    rendered = describe(result.spec)
    for term in expected_terms:
        assert term.lower() in rendered.lower(), f"{term!r} missing from readback:\n{rendered}"


# ============================================================================
# Readback for the six new gaps -- these must read in plain English a
# beginner could check against what they actually typed.
# ============================================================================


def test_inline_period_indicator_readback() -> None:
    spec = _minimal_spec(
        Compare(op="lt", left=Ref(name="rsi7"), right=Const(value=28)),
        indicators=[IndicatorSpec(id="rsi7", type="rsi", params={"length": 7})],
    )
    assert "RSI(7)" in describe(spec)


def test_indicator_vs_indicator_readback() -> None:
    spec = _minimal_spec(
        Compare(op="crosses_above", left=Ref(name="ema8"), right=Ref(name="ema21")),
        indicators=[
            IndicatorSpec(id="ema8", type="ema", params={"length": 8}),
            IndicatorSpec(id="ema21", type="ema", params={"length": 21}),
        ],
    )
    text = describe(spec)
    assert "crosses above" in text
    assert "8" in text and "21" in text
    assert "Exponential Moving Average" in text


def test_close_vs_vwap_readback() -> None:
    spec = StrategySpec(
        name="Test Strategy",
        description="test",
        instrument=Instrument(symbol="TCS", trade_as="stock", timeframe="15m"),
        indicators=[IndicatorSpec(id="vwap", type="vwap", params={})],
        entry=Compare(op="gt", left=Ref(name="close"), right=Ref(name="vwap")),
        exit=ExitRules(stop_pct=1.0),
    )
    text = describe(spec)
    assert "the price" in text
    assert "VWAP" in text
    assert "is above" in text


def test_rsi_between_readback_shows_both_bounds() -> None:
    spec = _minimal_spec(
        All(
            conditions=[
                Compare(op="gte", left=Ref(name="rsi14"), right=Const(value=40)),
                Compare(op="lte", left=Ref(name="rsi14"), right=Const(value=60)),
            ]
        ),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
    )
    text = describe(spec)
    assert "40" in text and "60" in text
    assert " and " in text


def test_anaphoric_exit_condition_reads_as_the_same_indicator() -> None:
    """The exit condition, once 'it' has been resolved during parsing, renders
    exactly like any other indicator comparison -- nothing anaphora-specific
    is left for the readback to handle, which is itself part of the safety
    story: the user reads the same plain sentence either way."""
    result = parse(
        "buy nifty when rsi drops below 30, sell when it crosses above 70, stop 1%, target 2%"
    )
    assert result.spec is not None
    text = describe(result.spec)
    assert "Exit when RSI(14) crosses above 70" in text


ROUND_TRIP_NEW_GAP_CASES = [
    (
        "Buy RELIANCE at RSI 30 with 5 percent stop loss and 10 percent target",
        ["RELIANCE", "RSI(14)", "crosses below", "30", "+10%", "-5%"],
    ),
    (
        "Instantly buy NIFTY 100 stocks when ema8 crosses above ema21 and adx14 > 20. "
        "Sell at 3% profit. Stop loss 1.5%.",
        [
            "NIFTY 100",
            "8-day Exponential Moving Average",
            "crosses above",
            "ADX(14)",
            "+3%",
            "-1.5%",
        ],
    ),
]


@pytest.mark.parametrize("text,expected_terms", ROUND_TRIP_NEW_GAP_CASES)
def test_new_gap_round_trip_contains_key_concepts(text: str, expected_terms: list[str]) -> None:
    result = parse(text)
    assert result.spec is not None, f"expected a spec for {text!r}: {result.questions}"
    rendered = describe(result.spec)
    for term in expected_terms:
        assert term.lower() in rendered.lower(), f"{term!r} missing from readback:\n{rendered}"


# ---------------------------------------------------------------------------
# The readback must not promise behaviour the engine will not perform
#
# It claimed "Square off at 15:15 if still open" on daily strategies, where the
# engine skips every time-of-day rule -- a daily bar's timestamp carries no
# intraday clock, and the engine says so in its own warnings. A readback line
# that is merely plausible is worse than no line: its entire value is that a
# non-technical reader can trust it literally.
# ---------------------------------------------------------------------------


def test_daily_strategy_does_not_promise_a_square_off():
    from nlt.translate.rules import parse

    result = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%")
    assert result.spec is not None
    assert result.spec.instrument.timeframe == "1d"

    text = describe(result.spec)
    assert "Square off" not in text, (
        "the readback promises a square-off on a daily strategy, which the engine explicitly skips"
    )


def test_intraday_strategy_does_promise_a_square_off():
    """The mirror: where it genuinely happens, it must be stated."""
    from nlt.translate.rules import parse

    result = parse("buy nifty when rsi cracks 30 on 15 minute chart, target 2%, stop 1%")
    assert result.spec is not None
    assert result.spec.instrument.timeframe == "15m"
    assert "Square off at 15:15" in describe(result.spec)


# ---------------------------------------------------------------------------
# Quantity, not lots
#
# A lot is an options contract term. Saying "1 lot of TCS" is not something
# anyone says, and a reader who owns shares would reasonably ask how many shares
# a lot is. Quantity is correct everywhere and is what a contract note and a
# broker's order window both show.
#
# The wording change surfaced a real defect behind it: max_lots defaulted to 2,
# and for a stock one lot is one share, so every stock position was capped at
# two shares whatever sizing was asked for.
# ---------------------------------------------------------------------------


def test_stock_readback_says_quantity_not_lots():
    from nlt.translate.rules import parse

    spec = parse("Buy TCS when RSI drops below 30, target 5%, stop 2%").spec
    assert spec is not None
    rendered = describe(spec)
    assert "lot" not in rendered.lower(), f"a stock readback must not mention lots:\n{rendered}"


def test_stock_has_no_share_count_cap():
    """A fixed count cannot bound risk across a Rs 50 and a Rs 20,000 share."""
    from nlt.translate.rules import parse

    spec = parse("Buy TCS when RSI drops below 30, target 5%, stop 2%").spec
    assert spec is not None
    assert spec.risk.max_lots is None
    assert "quantity above" not in describe(spec).lower(), (
        "the readback states a cap the engine does not enforce"
    )


def test_stock_sizing_defaults_to_risking_capital_not_one_share():
    """ "1 lot" of a stock is one share -- about Rs 640 on a Rs 1,00,000 account,
    which tests nothing. Risk-based sizing works at any share price."""
    from nlt.translate.rules import parse

    spec = parse("Buy TCS when RSI drops below 30, target 5%, stop 2%").spec
    assert spec is not None
    assert spec.sizing.mode == "risk_based"
    assert spec.sizing.risk_pct == 1.0
    assert "Risk 1% of capital per trade" in describe(spec)


def test_an_explicit_lot_count_is_still_honoured():
    """Saying "2 lots" must still mean two lots, not be overridden by a default."""
    from nlt.translate.rules import parse

    spec = parse("Buy 2 lots of NIFTY when RSI cracks 30, stop 1%, target 2%").spec
    assert spec is not None
    assert spec.sizing.mode == "fixed_lots"
    assert spec.sizing.lots == 2


def test_risk_limits_default_to_no_quantity_cap():
    """The spec's own default matters, not just the parser's override.

    The parser sets this explicitly for every strategy it produces, which masks
    the default entirely -- restoring the old two-share cap left the whole suite
    green. Anything building a spec directly (a saved strategy, a future API, a
    test) would still inherit it, so the default is asserted at its source.
    """
    from nlt.spec.models import RiskLimits

    assert RiskLimits().max_lots is None, (
        "a cap counted in lots means nothing for a stock, where one lot is one "
        "share -- the default must not silently bound every position"
    )


# ------------------------------------------- you said / we worked out / default

from nlt.translate.readback import DEFAULT, SAID, WORKED_OUT, describe_lines  # noqa: E402


def _labels(text: str, answers: dict | None = None) -> dict[str, str]:
    result = parse(text, answers=answers)
    assert result.spec is not None, result.questions
    return cast(
        "dict[str, str]",
        {line: label for _h, line, label in describe_lines(result.spec, result.sources, answers)},
    )


def _body_lines(spec) -> list[str]:
    rows = describe(spec).splitlines()
    return [r.strip() for r in rows[1:] if r.startswith("  ")]


@pytest.mark.parametrize(
    "text",
    [
        "buy nifty when rsi cracks 30, target 2%, stop loss 1%",
        "buy crude oil futures when rsi cracks 30 on 15 minute candles, target 2%, stop 1%",
        "buy nifty 100 stocks when ema8 crosses above ema21, target 3%, stop 1.5%",
        "buy RELIANCE when rsi crosses below 30 on 15 minute candles, stop loss 1%, hold overnight",
        "short banknifty on death cross, target 2%, stop 1%, 2 lots",
    ],
)
def test_labelled_lines_are_exactly_the_readback(text) -> None:
    """The labels sit beside the readback; they must never change a word of it."""
    result = parse(text)
    assert result.spec is not None
    labelled = [line for _h, line, _l in describe_lines(result.spec, result.sources)]
    assert labelled == _body_lines(result.spec)


def test_no_record_never_reads_as_you_said() -> None:
    """The safety rule: "You said" is a claim; without a record it is never made."""
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    labels = {label for _h, _l, label in describe_lines(spec, {})}
    assert SAID not in labels
    assert {label for _h, _l, label in describe_lines(spec, None)} == {None}


def test_what_the_user_typed_is_you_said_and_the_rest_is_not() -> None:
    labels = _labels("buy nifty when rsi cracks 30, target 2%, stop loss 1%")
    assert labels["NIFTY index"] == SAID
    assert labels["Take profit at +2%"] == SAID
    assert labels["Stop loss at -1%"] == SAID
    assert labels["Daily candles"] == DEFAULT
    assert labels["Risk 1% of capital per trade"] == DEFAULT
    assert labels["At most 1 position open at a time"] == DEFAULT


def test_an_assumed_instrument_is_default() -> None:
    assert _labels("buy when rsi cracks 30, target 2%, stop 1%")["NIFTY index"] == DEFAULT


def test_a_number_we_filled_in_is_worked_out() -> None:
    """They typed "rsi"; the 14 is ours."""
    assert _labels("buy tcs when rsi cracks 30, stop 1%")["When RSI(14) crosses below 30"] == (
        WORKED_OUT
    )
    assert _labels("buy tcs when rsi(14) cracks 30, stop 1%")["When RSI(14) crosses below 30"] == (
        SAID
    )


def test_an_answer_to_our_question_is_you_said() -> None:
    labels = _labels("buy tcs when rsi(14) cracks 30, target 2%", {"exit.stop_pct": "1.5%"})
    assert labels["Stop loss at -1.5%"] == SAID


def test_stated_size_bar_and_overnight_are_you_said() -> None:
    labels = _labels(
        "buy RELIANCE when rsi(14) crosses below 30 on hourly candles, stop loss 1%, "
        "risk 2% per trade, hold overnight"
    )
    assert labels["Hourly candles"] == SAID
    assert labels["Risk 2% of capital per trade"] == SAID
    assert labels["Hold overnight: no square-off before the close"] == SAID


def test_a_default_square_off_is_default() -> None:
    labels = _labels("buy RELIANCE when rsi(14) crosses below 30 on 15 minute candles, stop 1%")
    assert labels["Square off at 15:15 if still open"] == DEFAULT


def test_futures_contract_facts_are_worked_out() -> None:
    labels = _labels("buy crude oil futures when rsi(14) cracks 30 on 15 minute candles, stop 1%")
    assert labels["Crude Oil futures (MCX)"] == SAID
    contract = next(line for line in labels if line.startswith("Nearest-month contract"))
    assert labels[contract] == WORKED_OUT
    assert labels["Square off at 23:15 if still open"] == WORKED_OUT
    assert labels["Close at the end of the contract's expiry day if still open"] == WORKED_OUT


def test_the_readback_never_promises_a_loss_limit_nothing_enforces() -> None:
    """No engine applies the daily or per-trade rupee loss limits, so the readback
    must not state them (Justin's call, 2026-10-12)."""
    spec = parse("buy nifty when rsi cracks 30, target 2%, stop loss 1%").spec
    assert spec is not None
    text = describe(spec)
    assert "Stop trading for the day" not in text
    assert "Never risk more than" not in text
    assert format_inr(spec.risk.max_daily_loss) not in text
    assert format_inr(spec.risk.max_loss_per_trade) not in text


# ------------------------------------------------- no code names on screen
#
# 2026-10-16: "MACD(12, 26, 9) macd crosses above MACD(12, 26, 9) signal line"
# and "the price is above Previous Period Levels high" were what users read.


@pytest.mark.parametrize(
    ("sentence", "line"),
    [
        (
            "buy nifty when macd crosses above signal, stop 1%",
            "When MACD(12, 26, 9) line crosses above its signal line",
        ),
        (
            "buy nifty when close above previous day high, stop 1%",
            "When the price is above the previous day's high",
        ),
        (
            "buy nifty when supertrend turns green, stop 1%",
            "When Supertrend(10, 3.0) turns to an uptrend",
        ),
        (
            "sell nifty when supertrend turns red, stop 1%",
            "When Supertrend(10, 3.0) turns to a downtrend",
        ),
    ],
)
def test_indicator_outputs_are_read_out_in_words(sentence: str, line: str) -> None:
    spec = parse(sentence).spec
    assert spec is not None
    assert f"  {line}" in describe(spec).splitlines()


def test_no_multi_output_indicator_prints_a_raw_output_name() -> None:
    """Every output of every multi-output indicator, worded on its own: none
    may print the underscore names the code uses."""
    from nlt.indicators.registry import REGISTRY
    from nlt.translate.readback import _ref_desc

    for type_, d in REGISTRY.items():
        for output in d.outputs if len(d.outputs) > 1 else ():
            words = _ref_desc(
                Ref(name="x", output=output), {"x": IndicatorSpec(id="x", type=type_)}, "15m"
            )
            assert "_" not in words, (type_, output, words)
            # Output names that are code, not English, must never be printed.
            if output in (
                "macd",
                "sar",
                "supertrend",
                "tc",
                "bc",
                "r1",
                "r2",
                "r3",
                "s1",
                "s2",
                "s3",
            ):
                assert not words.lower().endswith(f" {output}"), (type_, output, words)


def test_its_is_only_used_for_two_outputs_of_one_indicator() -> None:
    """Two different MACDs must each be named: "its" would hide which is which."""
    inds = [
        IndicatorSpec(id="m1", type="macd"),
        IndicatorSpec(id="m2", type="macd", params={"fast": 5, "slow": 35, "signal": 5}),
    ]
    cond = Compare(
        op="crosses_above",
        left=Ref(name="m1", output="macd"),
        right=Ref(name="m2", output="signal"),
    )
    from nlt.translate.readback import _describe_condition

    text = _describe_condition(cond, {i.id: i for i in inds}, "1d")
    assert "its" not in text
    assert "MACD(5, 35, 5) signal line" in text


def test_a_supertrend_compared_with_anything_but_zero_is_not_called_a_trend_turn() -> None:
    from nlt.translate.readback import _describe_condition

    inds = {"st": IndicatorSpec(id="st", type="supertrend")}
    cond = Compare(
        op="crosses_above", left=Ref(name="st", output="direction"), right=Const(value=0.5)
    )
    assert "uptrend" not in _describe_condition(cond, inds, "1d")
