"""Boxes for the numbers inside the buy and sell rules (2026-10-16).

"Fine-tune the numbers" can change an oscillator's level ("RSI below 30") and
each indicator's length. What matters: the strategy tested is the one the
boxes describe, the readback says so (it is regenerated, never patched), only
numbers already in the rule can change, and a value the indicator cannot take
is refused rather than clipped.
"""

from __future__ import annotations

import pytest

from app import logic

RSI = "buy nifty when rsi cracks 30, target 2%, stop loss 1%"
TWO_SMA = "buy nifty when 20 sma crosses above 50 sma, stop loss 1%"
BOTH = (
    "buy nifty when rsi crosses below 30 and close is above 200 sma, stop loss 1%, "
    "sell when rsi crosses above 70"
)


def _read(text: str):
    t = logic.translate_text(text)
    assert t.spec is not None, t.questions
    return t


def _boxes(text: str) -> dict[str, dict]:
    return {b["key"]: b for b in logic.rule_boxes(_read(text).spec)}


def test_an_rsi_rule_offers_its_level_and_length() -> None:
    boxes = _boxes(RSI)
    assert boxes["level:entry.right"]["value"] == 30
    assert boxes["level:entry.right"]["group"] == "When to buy"
    assert boxes["length:rsi14"]["value"] == 14


def test_changing_the_level_changes_what_is_tested_and_the_readback() -> None:
    t = _read(RSI)
    # Start from a line the user did not word themselves: a changed box makes it theirs.
    before = {**t.sources, "entry": "worked_out"}
    spec, sources, typed = logic.adjust_spec(t.spec, {"level:entry.right": "25"}, before)
    assert spec.entry.right.value == 25
    assert "When RSI(14) crosses below 25" in logic.describe_spec(spec)
    assert "crosses below 30" not in logic.describe_spec(spec)
    assert sources["entry"] == "said"
    assert typed == {"form.level:entry.right": "25"}


def test_changing_a_length_changes_the_indicator_and_the_readback() -> None:
    t = _read(RSI)
    spec, _s, _t = logic.adjust_spec(t.spec, {"length:rsi14": "7"}, t.sources)
    (rsi,) = [i for i in spec.indicators if i.type == "rsi"]
    assert rsi.params["length"] == 7
    assert "RSI(7)" in logic.describe_spec(spec)


def test_buy_and_sell_levels_are_separate_boxes() -> None:
    t = _read(BOTH)
    spec, _s, _t = logic.adjust_spec(
        t.spec, {"level:entry.0.right": "35", "level:exit.condition.right": "65"}, t.sources
    )
    text = logic.describe_spec(spec)
    assert "RSI(14) crosses below 35" in text
    assert "Exit when RSI(14) crosses above 65" in text


def test_two_averages_get_boxes_named_by_position() -> None:
    boxes = _boxes(TWO_SMA)
    assert boxes["length:sma20"]["label"].endswith("first in the rule")
    assert boxes["length:sma50"]["label"].endswith("second in the rule")
    t = _read(TWO_SMA)
    spec, _s, _t = logic.adjust_spec(t.spec, {"length:sma20": "10"}, t.sources)
    assert "10-day Simple Moving Average crosses above 50-day" in logic.describe_spec(spec)


@pytest.mark.parametrize(
    ("text", "edits", "says"),
    [
        (RSI, {"level:entry.right": "130"}, "between 0 and 100"),
        (RSI, {"level:entry.right": "0"}, "between 0 and 100"),
        (RSI, {"level:entry.right": "thirty"}, "must be a number"),
        (RSI, {"length:rsi14": "7.5"}, "whole number"),
        (RSI, {"length:rsi14": "1"}, "whole number from 2 to 500"),
        (RSI, {"length:sma200": "50"}, "does not belong to this strategy"),
        (RSI, {"level:entry.left": "50"}, "does not belong to this strategy"),
        (TWO_SMA, {"length:sma20": "50"}, "could never trigger"),
    ],
)
def test_a_value_the_rule_cannot_take_is_refused(text, edits, says) -> None:
    t = _read(text)
    with pytest.raises(logic.AdjustError, match=says):
        logic.adjust_spec(t.spec, edits, t.sources)


def test_a_rule_with_no_level_or_length_offers_no_boxes() -> None:
    # MACD's crossing has no level, and its three settings are not one length.
    assert _boxes("buy nifty when macd crosses above signal, stop loss 1%") == {}


def test_a_spec_back_from_the_browser_reads_back_in_the_same_words() -> None:
    """JSON from the browser turns 2.0 into 2; the readback must not change."""
    import json

    from nlt.spec.models import StrategySpec

    spec = _read("buy nifty when supertrend(7, 2) turns green, stop loss 1%").spec
    raw = json.loads(json.dumps(spec.model_dump(mode="json")).replace("2.0", "2"))
    back = StrategySpec.model_validate(raw)
    assert logic.describe_spec(back) == logic.describe_spec(spec)
    t = _read("buy nifty when supertrend(7, 2) turns green, stop loss 1%")
    edited, _s, _t = logic.adjust_spec(back, {"length:supertrend7x2": "12"}, t.sources)
    assert "Supertrend(12, 2.0)" in logic.describe_spec(edited)
