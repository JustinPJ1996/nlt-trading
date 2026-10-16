"""The recipe rules for selling and multi-leg option positions (Phase 2, batch 2).

Justin, 2026-10-14: a sold option with nothing protecting it must have a stop
loss; a position whose every sold option is covered by a bought one needs none
(the hedge is the stop).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from pydantic import ValidationError

from nlt.engine.conditions import evaluate
from nlt.spec.models import AtTime, OptionLeg, StrategySpec, is_defined_risk


def _spec(legs: list[dict], exit_: dict | None = None) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "x",
            "description": "x",
            "instrument": {
                "symbol": "NIFTY",
                "trade_as": "option",
                "timeframe": "1m",
                "legs": legs,
            },
            "entry": {"kind": "at_time", "time": "09:20"},
            "exit": exit_ or {"exits_on_legs": True},
        }
    )


def _leg(side, kind, **kw) -> dict:
    return {"side": side, "option_type": kind, **kw}


def test_straddle_with_a_stop_on_each_leg_is_valid() -> None:
    spec = _spec([_leg("sell", "CE", stop_pct=25), _leg("sell", "PE", stop_pct=25)])
    assert spec.instrument.is_multi_leg


def test_straddle_without_a_stop_is_refused() -> None:
    with pytest.raises(ValidationError, match="needs a stop loss"):
        _spec([_leg("sell", "CE"), _leg("sell", "PE")], {"combined_target_pct": 50})


def test_one_unstopped_sold_leg_is_enough_to_refuse() -> None:
    with pytest.raises(ValidationError, match="needs a stop loss"):
        _spec([_leg("sell", "CE", stop_pct=25), _leg("sell", "PE")])


def test_a_combined_stop_covers_every_sold_leg() -> None:
    _spec([_leg("sell", "CE"), _leg("sell", "PE")], {"combined_stop_pct": 30})


def test_iron_condor_needs_no_stop() -> None:
    legs = [
        _leg("sell", "CE", strike_points=200),
        _leg("buy", "CE", strike_points=300),
        _leg("sell", "PE", strike_points=-200),
        _leg("buy", "PE", strike_points=-300),
    ]
    _spec(legs, {"combined_target_pct": 50})


def test_a_bought_put_nearer_the_money_still_caps_the_sold_put() -> None:
    """A bear put spread (buy the -200 put, sell the -400): the most it can lose
    is what was paid. Justin, 2026-10-16: covered at any strike; until then
    this was asked for a stop it did not need."""
    legs = [_leg("sell", "PE", strike_points=-400), _leg("buy", "PE", strike_points=-200)]
    _spec(legs, {"combined_target_pct": 50})
    assert is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_a_call_cannot_hedge_a_put() -> None:
    legs = [_leg("sell", "PE", strike_points=-200), _leg("buy", "CE", strike_points=400)]
    assert not is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_a_ratio_spread_is_not_covered() -> None:
    legs = [_leg("sell", "PE", strike_offset=1, lots=2), _leg("buy", "PE", strike_offset=3)]
    assert not is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_mixed_strike_units_cover_too() -> None:
    """Which strike is further out no longer matters, so mixed units do not either."""
    legs = [_leg("sell", "PE", strike_offset=1), _leg("buy", "PE", strike_points=-500)]
    assert is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_a_hedge_in_another_expiry_is_not_covered() -> None:
    legs = [
        _leg("sell", "PE", strike_offset=1),
        _leg("buy", "PE", strike_offset=3, expiry="monthly"),
    ]
    assert not is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_offset_hedges_count_out_of_the_money_for_each_type() -> None:
    legs = [
        _leg("sell", "CE", strike_offset=2),
        _leg("buy", "CE", strike_offset=4),
        _leg("sell", "PE", strike_offset=2),
        _leg("buy", "PE", strike_offset=4),
    ]
    assert is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_only_bought_legs_is_defined_risk() -> None:
    assert is_defined_risk([OptionLeg(side="buy", option_type="CE")])


def test_stock_style_stop_is_refused_on_legs() -> None:
    with pytest.raises(ValidationError, match="per leg"):
        _spec([_leg("sell", "CE", stop_pct=25)], {"stop_pct": 10})


def test_reentry_needs_a_stop_on_that_leg() -> None:
    legs = [
        _leg("sell", "CE", reentry="same_strike_at_cost", reentry_max=1),
        _leg("buy", "CE", strike_offset=3),
    ]
    with pytest.raises(ValidationError, match="re-enter"):
        _spec(legs, {"combined_target_pct": 50})


def test_reentry_kind_and_count_go_together() -> None:
    with pytest.raises(ValidationError, match="re-entry"):
        OptionLeg(side="sell", option_type="CE", stop_pct=25, reentry_max=1)


def test_combined_exits_refused_without_legs() -> None:
    with pytest.raises(ValidationError, match="multi-leg"):
        StrategySpec.model_validate(
            {
                "name": "x",
                "description": "x",
                "entry": {"kind": "at_time", "time": "09:20"},
                "exit": {"combined_target_pct": 50},
            }
        )


def test_entry_time_outside_session_is_refused() -> None:
    with pytest.raises(ValidationError, match="outside"):
        AtTime(time=dt.time(9, 15))


def _bars(minutes: int, day: str = "2025-03-12") -> pd.DataFrame:
    idx = pd.date_range(f"{day} 09:15", f"{day} 15:29", freq=f"{minutes}min", tz="Asia/Kolkata")
    return pd.DataFrame({"close": 1.0}, index=idx)


def test_at_time_fires_on_the_candle_that_closes_at_that_time() -> None:
    fired = evaluate(AtTime(time=dt.time(9, 20)), _bars(1))
    assert list(fired[fired].index.strftime("%H:%M")) == ["09:19"]
    fired = evaluate(AtTime(time=dt.time(9, 20)), _bars(5))
    assert list(fired[fired].index.strftime("%H:%M")) == ["09:15"]


def test_at_time_refuses_a_time_no_candle_closes_at() -> None:
    with pytest.raises(ValueError, match="closes at 09:20"):
        evaluate(AtTime(time=dt.time(9, 20)), _bars(15))


def test_at_time_refuses_daily_candles() -> None:
    idx = pd.date_range("2025-03-10", periods=5, freq="D", tz="Asia/Kolkata")
    with pytest.raises(ValueError, match="intraday"):
        evaluate(AtTime(time=dt.time(9, 20)), pd.DataFrame({"close": 1.0}, index=idx))


def test_a_bull_call_spread_is_covered() -> None:
    """Buy the call 1 strike out, sell the one 3 strikes out: a debit spread."""
    legs = [_leg("sell", "CE", strike_offset=3), _leg("buy", "CE", strike_offset=1)]
    assert is_defined_risk([OptionLeg(**leg) for leg in legs])


def test_fewer_bought_lots_than_sold_is_not_covered_at_any_strike() -> None:
    legs = [_leg("sell", "CE", strike_offset=3, lots=2), _leg("buy", "CE", strike_offset=1)]
    assert not is_defined_risk([OptionLeg(**leg) for leg in legs])
