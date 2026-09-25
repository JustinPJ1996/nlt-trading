"""Tests for `nlt.translate.readback.describe`.

No LLM here either -- the readback is generated mechanically from the spec, so
these tests check that every field a spec can carry actually shows up in the
text a non-technical user would read.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    spec = _minimal_spec(
        PercentChange(ref=Ref(name="close"), lookback=3, op="lte", value=-1.0)
    )
    text = describe(spec)
    assert "falls more than 1%" in text
    assert "3 days" in text or "3 bars" in text


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
    spec = _minimal_spec(Not(condition=Compare(op="gt", left=Ref(name="close"), right=Const(value=100))))
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
    # risk
    assert format_inr(2500.0) in text
    assert format_inr(6000.0) in text
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
        "Buy 2 lots of NIFTY when RSI cracks 30, stop 1%, target 2%",
        ["2 lots", "RSI"],
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
    text = describe(spec)
    assert "NIFTY, on daily bars" in text


def test_daily_timeframe_produces_no_redundant_candle_suffix_for_a_stock() -> None:
    spec = StrategySpec(
        name="Single Stock",
        description="test",
        instrument=Instrument(symbol="TCS", trade_as="stock", timeframe="1d"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30)),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        exit=ExitRules(stop_pct=1.0, target_pct=2.0),
    )
    text = describe(spec)
    assert "TCS (NSE)" in text
    assert "candles" not in text.lower()


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
        ["NIFTY, on daily bars", "RSI(14)", "crosses below", "30", "+2%", "-1%"],
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
        ["NIFTY 100", "8-day Exponential Moving Average", "crosses above", "ADX(14)", "+3%", "-1.5%"],
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
        "the readback promises a square-off on a daily strategy, which the "
        "engine explicitly skips"
    )


def test_intraday_strategy_does_promise_a_square_off():
    """The mirror: where it genuinely happens, it must be stated."""
    from nlt.translate.rules import parse

    result = parse("buy nifty when rsi cracks 30 on 15 minute chart, target 2%, stop 1%")
    assert result.spec is not None
    assert result.spec.instrument.timeframe == "15m"
    assert "Square off at 15:15" in describe(result.spec)
