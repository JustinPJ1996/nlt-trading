"""The "Change the numbers before testing" boxes for sold and multi-leg options."""

from __future__ import annotations

import pytest

from app import api, logic
from nlt.spec.models import StrategySpec
from nlt.translate.readback import describe_lines

STRADDLE = (
    "Sell the NIFTY weekly ATM straddle at 09:20 with a 25% stop on each leg. "
    "Square off at 15:15. Backtest the last 1 year."
)
CONDOR = (
    "At 09:30, sell a NIFTY weekly iron condor with short strikes 200 points out and "
    "wings 100 points beyond them. Exit at 50% of the credit received or at 15:15. "
    "Backtest the last 1 year."
)
STOCK = "buy nifty when rsi cracks 30, stop loss 1%, target 2%"


def _read(text):
    tr = logic.translate_text(text)
    assert tr.spec is not None
    return tr.spec, tr.sources


def _lines(spec, sources, typed):
    return {line: label for _h, line, label in describe_lines(spec, sources, typed)}


def _line(lines, start):
    (match,) = [k for k in lines if k.startswith(start)]
    return match, lines[match]


def _with_combined_stop(spec, pct):
    return spec.model_copy(update={"exit": spec.exit.model_copy(update={"combined_stop_pct": pct})})


def test_straddle_leg_stop_changes_every_sold_leg_and_says_you_said() -> None:
    spec, sources = _read(STRADDLE)
    new, new_sources, typed = logic.adjust_spec(spec, {"leg_stop_pct": "30"}, sources)
    assert [leg.stop_pct for leg in new.instrument.legs] == [30, 30]
    text, label = _line(_lines(new, new_sources, typed), "Stop loss on each sold option")
    assert "by 30% of its price" in text
    assert label == "said"


def test_a_leg_stop_typed_in_the_box_is_said_even_if_the_original_was_worked_out() -> None:
    spec, sources = _read(STRADDLE)
    sources = {**sources, "exit.stop": "worked_out"}
    new, new_sources, typed = logic.adjust_spec(spec, {"leg_stop_pct": "30"}, sources)
    assert _line(_lines(new, new_sources, typed), "Stop loss on each")[1] == "said"


def test_leg_stop_cannot_be_removed_or_be_nonsense() -> None:
    spec, sources = _read(STRADDLE)
    for bad in ["", "0", "-5", "abc", "1001"]:
        with pytest.raises(logic.AdjustError):
            logic.adjust_spec(spec, {"leg_stop_pct": bad}, sources)
    new, _s, _t = logic.adjust_spec(spec, {"leg_stop_pct": "1000"}, sources)
    assert new.instrument.legs[0].stop_pct == 1000


def test_straddle_take_profit_adds_a_combined_target() -> None:
    spec, sources = _read(STRADDLE)
    new, new_sources, typed = logic.adjust_spec(spec, {"combined_target_pct": "40"}, sources)
    assert new.exit.combined_target_pct == 40
    text, label = _line(_lines(new, new_sources, typed), "Take profit when")
    assert "reaches 40% of the credit" in text and label == "said"


def test_condor_take_profit_changes_and_relabels_only_that_line() -> None:
    spec, sources = _read(CONDOR)
    assert spec.exit.combined_target_pct == 50
    # As first read, the sentence's own take profit is labelled "You said".
    assert _line(_lines(spec, sources, {}), "Take profit when")[1] == "said"
    sources = {**sources, "exit.combined_target": "worked_out"}
    new, new_sources, typed = logic.adjust_spec(spec, {"combined_target_pct": "40"}, sources)
    assert new.exit.combined_target_pct == 40
    text, label = _line(_lines(new, new_sources, typed), "Take profit when")
    assert "reaches 40% of the credit" in text and label == "said"
    assert new_sources is not None
    assert new_sources["entry"] == sources["entry"]


def test_a_leg_stop_edit_does_not_relabel_the_combined_lines() -> None:
    spec, sources = _read(STRADDLE)
    spec = _with_combined_stop(spec, 60)
    sources = {**sources, "exit.combined_stop": "worked_out"}
    _new, new_sources, _t = logic.adjust_spec(spec, {"leg_stop_pct": "30"}, sources)
    assert new_sources is not None
    assert new_sources["exit.combined_stop"] == "worked_out"
    assert new_sources["exit.stop"] == "said"


def test_take_profit_can_be_emptied_when_the_strategy_still_validates() -> None:
    spec, sources = _read(STRADDLE)  # its exits are the per-leg stops
    withtp, sources, _t = logic.adjust_spec(spec, {"combined_target_pct": "40"}, sources)
    new, new_sources, _t = logic.adjust_spec(withtp, {"combined_target_pct": ""}, sources)
    assert new.exit.combined_target_pct is None
    assert new_sources is not None
    assert "exit.combined_target" not in new_sources


def test_emptying_the_only_exit_is_refused_with_the_validators_words() -> None:
    spec, sources = _read(CONDOR)  # the take profit is its only exit
    with pytest.raises(logic.AdjustError) as err:
        logic.adjust_spec(spec, {"combined_target_pct": ""}, sources)
    message = str(err.value)
    assert "no exit rule" in message
    assert "Value error" not in message


def test_take_profit_out_of_range_is_refused_plainly() -> None:
    spec, sources = _read(CONDOR)
    for bad in ["0", "150", "abc"]:
        with pytest.raises(logic.AdjustError):
            logic.adjust_spec(spec, {"combined_target_pct": bad}, sources)
    with pytest.raises(logic.AdjustError, match="at most 100%"):
        logic.adjust_spec(spec, {"combined_target_pct": "150"}, sources)


def test_combined_stop_box_only_when_set_and_never_removed() -> None:
    spec, sources = _read(STRADDLE)
    new, _s, typed = logic.adjust_spec(spec, {"combined_stop_pct": "70"}, sources)
    assert new.exit.combined_stop_pct is None and "form.combined_stop_pct" not in typed
    withstop = _with_combined_stop(spec, 60)
    moved, moved_sources, _t = logic.adjust_spec(withstop, {"combined_stop_pct": "70"}, sources)
    assert moved.exit.combined_stop_pct == 70
    assert moved_sources is not None
    assert moved_sources["exit.combined_stop"] == "said"
    with pytest.raises(logic.AdjustError, match="cannot be empty"):
        logic.adjust_spec(withstop, {"combined_stop_pct": ""}, sources)


def test_stock_exit_edits_are_ignored_on_a_legs_strategy() -> None:
    spec, sources = _read(STRADDLE)
    new, _s, typed = logic.adjust_spec(spec, {"target_pct": "3", "stop_pct": "2"}, sources)
    assert new.exit.target_pct is None and new.exit.stop_pct is None and not typed


def test_candles_with_an_entry_time_offer_only_sizes_that_end_at_it() -> None:
    spec, _ = _read(STRADDLE)  # 09:20 is 5 minutes after 09:15
    assert logic.allowed_timeframes(spec) == ["1m", "5m"]
    condor, _ = _read(CONDOR)  # 09:30 is 15 minutes after
    assert logic.allowed_timeframes(condor) == ["1m", "3m", "5m", "15m"]


def test_a_candle_that_breaks_the_entry_time_is_refused_plainly() -> None:
    spec, sources = _read(STRADDLE)
    for bad in ["15m", "30m", "1h", "1d", "3m"]:
        with pytest.raises(logic.AdjustError, match="do not end at that time"):
            logic.adjust_spec(spec, {"timeframe": bad}, sources)
    new, new_sources, typed = logic.adjust_spec(spec, {"timeframe": "5m"}, sources)
    assert new.instrument.timeframe == "5m"
    assert _line(_lines(new, new_sources, typed), "5-minute candles")[1] == "said"


def test_an_entry_time_nested_in_a_condition_still_limits_the_candles() -> None:
    spec, _ = _read(STRADDLE)
    nested = StrategySpec.model_validate(
        {
            **spec.model_dump(mode="json"),
            "entry": {"kind": "all", "conditions": [spec.entry.model_dump(mode="json")]},
        }
    )
    assert logic.allowed_timeframes(nested) == ["1m", "5m"]


def test_the_payload_carries_what_the_boxes_need() -> None:
    spec, sources = _read(STRADDLE)
    payload = api._spec_payload(spec, sources)
    assert payload["allowed_timeframes"] == ["1m", "5m"]
    assert payload["spec"]["instrument"]["legs"][0]["side"] == "sell"
    assert payload["spec"]["instrument"]["legs"][0]["stop_pct"] == 25
    assert "combined_target_pct" in payload["spec"]["exit"]


def test_the_stock_path_is_unchanged() -> None:
    spec, sources = _read(STOCK)
    assert logic.allowed_timeframes(spec) is None
    assert api._spec_payload(spec, sources)["allowed_timeframes"] is None
    new, new_sources, _t = logic.adjust_spec(spec, {"stop_pct": "1.5", "target_pct": "3"}, sources)
    assert (new.exit.stop_pct, new.exit.target_pct) == (1.5, 3.0)
    assert new_sources is not None
    assert new_sources["exit.stop"] == "said"
    with pytest.raises(logic.AdjustError, match="at most 50%"):
        logic.adjust_spec(spec, {"stop_pct": "60"}, sources)
    new, _s, typed = logic.adjust_spec(spec, {"combined_target_pct": "40"}, sources)
    assert new.exit.combined_target_pct is None and not typed
