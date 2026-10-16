"""A strike the user names ("buy the 21,600 call"): paper trading only.

Justin, 2026-10-16: a backtest refuses it (over past months the index was far
from it), and paper trading uses exactly that strike or none -- never the
nearest listed one.
"""

from __future__ import annotations

import datetime as dt

import pytest

from app import logic
from nlt.paper import runner
from nlt.spec.models import ExitRules, Instrument, OptionLeg, Schedule, StrategySpec
from tests import test_options_engine as oe


def _spec(strike: int):
    base = oe._spec()
    return base.model_copy(
        update={"instrument": base.instrument.model_copy(update={"strike": strike})}
    )


def _run(spec, bars, source):
    from nlt.engine.options_backtest import run_option_backtest
    from nlt.options.execution import OptionExecution

    i = spec.instrument
    ex = OptionExecution(source, "NIFTY", i.timeframe, i.option_type, 0, i.expiry, strike=i.strike)
    return run_option_backtest(spec, bars, ex, capital=oe.CAPITAL)


def test_the_named_strike_is_traded_not_the_nearest() -> None:
    """The index is at 21,510 at the signal (ATM 21,500); the 21,600 is named."""
    res = _run(_spec(21600), oe._index_bars([oe.D1], oe.SIGNAL), oe._chain([oe.D1], {}))
    (t,) = res.trades[:1]
    assert t.symbol == oe._symbol(21600, "CE")
    assert "names the 21,600 strike" in t.contract_why


def test_a_named_strike_that_is_not_listed_is_no_trade() -> None:
    res = _run(_spec(21700), oe._index_bars([oe.D1], oe.SIGNAL), oe._chain([oe.D1], {}))
    assert res.trades == []
    (d,) = [d for d in res.decisions if d.kind == "signal_skipped"]
    assert d.reason_code == "strike_not_listed"
    assert "another strike is never used instead" in d.detail["message"]


def test_paper_trading_passes_the_named_strike_to_the_engine() -> None:
    import pandas as pd

    bars = oe._index_bars([oe.D1], oe.SIGNAL)
    res = runner._run_option_engine(
        _spec(21600),
        bars,
        capital=oe.CAPITAL,
        charge_fn=None,
        trade_from=None,
        now=pd.Timestamp(dt.datetime.combine(oe.D2, dt.time(9, 0)), tz=oe.IST),
        blocked_windows=[],
        slippage=oe.Slippage(),
        source=oe._chain([oe.D1], {}),
        feed=None,
    )
    assert res.trades[0].symbol == oe._symbol(21600, "CE")


def test_a_named_strike_is_not_backtested_and_says_why() -> None:
    result = logic.backtest_spec(_spec(21600))
    assert not result.ok
    assert "A fixed strike like 21,600 can't be backtested" in result.error
    assert "Paper trade it instead" in result.error
    assert logic.paper_problem(_spec(21600)) is None


def test_a_leg_with_a_named_strike_is_not_backtested_either() -> None:
    spec = StrategySpec(
        name="t",
        description="t",
        instrument=Instrument(
            symbol="NIFTY",
            timeframe="5m",
            trade_as="option",
            legs=[OptionLeg(side="sell", option_type="CE", strike=25000, stop_pct=30)],
        ),
        entry=oe.Compare(op="gt", left=oe.Ref(name="close"), right=oe.Const(value=1.0)),
        exit=ExitRules(exits_on_legs=True),
        schedule=Schedule(intraday=True),
    )
    assert "25,000" in logic.backtest_problem(spec)


@pytest.mark.parametrize(
    "kw", [{"strike": 25000, "strike_offset": 1}, {"strike": 25000, "strike_points": 100}]
)
def test_a_leg_has_one_strike_rule(kw) -> None:
    with pytest.raises(ValueError):
        OptionLeg(side="sell", option_type="CE", **kw)


def test_a_leg_with_a_named_strike_trades_exactly_that_strike() -> None:
    from tests import test_legs_engine as le

    legs = [le.sell("CE", strike=21700, stop_pct=25), le.sell("PE", stop_pct=25)]
    (t,) = le._run(
        le._spec(legs), le._source(), days=(le.D1,), index=le._index(days=(le.D1,))
    ).trades
    assert {f.symbol for f in t.legs} == {le.sym(21700, "CE"), le.sym(21500, "PE")}
