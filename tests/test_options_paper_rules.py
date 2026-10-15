"""The engine rules that only paper trading uses (Phase 2 batch 3).

* Expiry (Justin, 2026-10-15): in paper, an option still open on its expiry
  day is held to the F&O close (15:40) and valued at its last traded price;
  a backtest closes it at the square-off time instead.
* A paper pass sees the day so far, never its end: the newest candle is not
  the session's last, and running out of candles is not a reason to give up
  a re-entry.

Small synthetic chains from the engine tests; no real data, no network.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from test_legs_engine import CE as LEG_CE
from test_legs_engine import OPEN_PRICES, PE_FLAT, _legs, _with, sell
from test_legs_engine import PE as LEG_PE
from test_legs_engine import _index as _legs_index
from test_legs_engine import _run as _run_legs
from test_legs_engine import _source as _legs_source
from test_legs_engine import _spec as _legs_spec
from test_options_engine import ATM_CE, D1, D2, D3, SIGNAL, _chain, _index_bars, _run, _spec, _ts

from nlt.options.data import FrameSource

CLOSE = dt.time(15, 40)


def _plus(source: FrameSource, rows: list[dict]) -> FrameSource:
    """`source` with extra 1-minute rows (copied from an existing row of the
    same contract, with new time and prices)."""
    df = source._df
    extra = []
    for r in rows:
        like = df[(df["tradingsymbol"] == r["tradingsymbol"]) & (df["timeframe"] == "1m")].iloc[0]
        extra.append({**like.to_dict(), "volume": 10, **r})
    return FrameSource(pd.concat([df, pd.DataFrame(extra)], ignore_index=True))


def _minute(sym, ts, price, volume=10):
    return {
        "tradingsymbol": sym,
        "timestamp": ts,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": volume,
    }


# ------------------------------------------------------------- bought option


def _held_to_expiry(**kw):
    source = _plus(
        _chain([D1, D2, D3], {ATM_CE: {_ts(D3, 15, 15): (130, 140, 129, 139)}}),
        [
            _minute(ATM_CE, _ts(D3, 15, 31), 50.0),
            _minute(ATM_CE, _ts(D3, 15, 39), 12.0),  # above the 90% stop (10)
        ],
    )
    spec = _spec(stop=90.0, target=100.0, intraday=False)
    return _run(spec, _index_bars([D1, D2, D3], SIGNAL), source, **kw).trades[0]


def test_paper_holds_an_expiring_option_to_the_close_at_its_last_traded_price():
    t = _held_to_expiry(fno_close=CLOSE)
    assert t.exit_reason == "expiry"
    assert t.exit_price == pytest.approx(12.0)  # 15:39's trade, not 15:15's open
    assert not t.exit_untraded


def test_the_backtest_still_closes_an_expiring_option_at_1515():
    t = _held_to_expiry()
    assert (t.exit_reason, t.exit_time, t.exit_price) == ("expiry", _ts(D3, 15, 15), 139.0)


def test_paper_still_watches_the_stop_on_expiry_day_before_the_close():
    source = _chain(
        [D1, D2, D3],
        {ATM_CE: {_ts(D3, 11, 0): (60, 61, 5, 6)}},
    )
    spec = _spec(stop=50.0, target=100.0, intraday=False)
    t = _run(spec, _index_bars([D1, D2, D3], SIGNAL), source, fno_close=CLOSE).trades[0]
    assert (t.exit_reason, t.exit_time) == ("stop", _ts(D3, 11, 0))


def test_paper_keeps_the_expiring_position_open_until_the_day_is_over():
    """Mid-afternoon on expiry day, the newest candle is not the day's last:
    the position must still be open, not valued at a price from 15:00."""
    bars = _index_bars([D1, D2, D3], SIGNAL)
    bars = bars[bars.index <= _ts(D3, 15, 0)]
    source = _chain([D1, D2, D3], {})
    spec = _spec(stop=90.0, target=100.0, intraday=False)
    res = _run(
        spec,
        bars,
        source,
        fno_close=CLOSE,
        drop_partial_last_bar=False,
        final_session_in_progress=True,
    )
    assert [t.exit_reason for t in res.trades] == ["end_of_data"]


def test_an_intraday_position_is_not_squared_off_on_every_mid_day_pass():
    bars = _index_bars([D1], SIGNAL)
    bars = bars[bars.index <= _ts(D1, 12, 0)]
    source = _chain([D1], {})
    res = _run(
        _spec(stop=90.0, target=100.0),
        bars,
        source,
        drop_partial_last_bar=False,
        final_session_in_progress=True,
    )
    assert [t.exit_reason for t in res.trades] == ["end_of_data"]


# --------------------------------------------------------------- legs engine


def _straddle_to_expiry(**kw):
    src = _plus(
        _legs_source(OPEN_PRICES, PE_FLAT),
        [_minute(LEG_CE, _ts(D3, 15, 39), 3.0), _minute(LEG_PE, _ts(D3, 15, 39), 4.0)],
    )
    return _run_legs(_legs_spec(intraday=False), src, **kw).trades[0]


def test_paper_holds_an_expiring_straddle_to_the_close_at_its_last_traded_prices():
    t = _straddle_to_expiry(fno_close=CLOSE)
    (ce,), (pe,) = _legs(t, LEG_CE), _legs(t, LEG_PE)
    assert (ce.exit_reason, ce.exit_price) == ("expiry", 3.0)
    assert (pe.exit_reason, pe.exit_price) == ("expiry", 4.0)
    # Valued after the index's last candle (15:25), not at the 15:15 square-off.
    assert ce.exit_time == pe.exit_time == _ts(D3, 15, 35)
    assert t.exit_reason == "expiry"


def test_the_backtest_still_closes_an_expiring_straddle_at_1515():
    t = _straddle_to_expiry()
    (ce,) = _legs(t, LEG_CE)
    assert (ce.exit_reason, ce.exit_time) == ("expiry", _ts(D3, 15, 15))


def test_a_leg_waiting_to_re_enter_keeps_the_position_open_at_the_newest_candle():
    """CE stopped on 10:00, the newest candle of a paper pass. It may re-enter
    on a later candle, so the position is still open -- not reported closed
    with reason "stop", then reopened on the next pass."""
    legs = [sell("CE", stop_pct=25, reentry="same_strike_at_cost", reentry_max=1)]
    src = _legs_source(_with(OPEN_PRICES, {LEG_CE: {_ts(D1, 10, 0): (110.0, 130.0, 100.0, 100.0)}}))
    index = _legs_index(days=(D1,))
    index = index[index.index <= _ts(D1, 10, 0)]
    res = _run_legs(
        _legs_spec(legs),
        src,
        index=index,
        days=(D1,),
        drop_partial_last_bar=False,
        final_session_in_progress=True,
    )
    (t,) = res.trades
    assert t.exit_reason == "end_of_data"
    (stopped,) = _legs(t, LEG_CE)
    assert stopped.exit_reason == "stop"


# ------------------------------------------- 15:30 to 15:40, after the index


def _overnight_call(extra_minutes, *, stop=30.0, **kw):
    source = _plus(_chain([D1, D2, D3], {}), extra_minutes)
    spec = _spec(stop=stop, target=100.0, intraday=False)
    return _run(spec, _index_bars([D1, D2, D3], SIGNAL), source, **kw)


def test_paper_watches_the_options_stop_after_the_index_has_closed():
    """Bought at 100 on D1, stop 70. At 15:33 on D2 -- after the index's last
    candle, but while options still trade -- it trades at 60."""
    res = _overnight_call([_minute(ATM_CE, _ts(D2, 15, 33), 60.0)], fno_close=CLOSE)
    t = res.trades[0]
    assert (t.exit_reason, t.exit_time, t.exit_price) == ("stop", _ts(D2, 15, 30), 60.0)


def test_the_backtest_never_looks_after_the_index_close():
    res = _overnight_call([_minute(ATM_CE, _ts(D2, 15, 33), 60.0)])
    assert res.trades[0].exit_reason == "expiry"


def test_an_exit_rule_met_on_the_last_index_candle_fills_after_the_close():
    """The exit rule is true on D2's last index candle (15:15): the option is
    sold at the next option price, 15:30's open -- not the next morning."""
    from nlt.spec.models import Compare, Const, Ref

    bars = _index_bars([D1, D2, D3], SIGNAL)
    bars.loc[_ts(D2, 15, 15), ["open", "high", "low", "close"]] = 21000.0
    source = _plus(_chain([D1, D2, D3], {}), [_minute(ATM_CE, _ts(D2, 15, 32), 88.0)])
    spec = _spec(stop=50.0, target=100.0, intraday=False)
    spec = spec.model_copy(
        update={
            "exit": spec.exit.model_copy(
                update={
                    "condition": Compare(
                        op="lt", left=Ref(name="close"), right=Const(value=21100.0)
                    )
                }
            )
        }
    )
    paper = _run(spec, bars, source, fno_close=CLOSE).trades[0]
    assert (paper.exit_reason, paper.exit_time, paper.exit_price) == (
        "condition",
        _ts(D2, 15, 30),
        88.0,
    )
    backtest = _run(spec, bars, source).trades[0]
    assert (backtest.exit_reason, backtest.exit_time) == ("condition", _ts(D3, 9, 15))


def test_paper_watches_a_sold_legs_stop_after_the_index_has_closed():
    """Straddle sold on D1 (CE at 102, stop 127), held overnight. At 15:37 on
    D1 the call trades at 130: bought back at 130, the put stays open."""
    src = _plus(_legs_source(OPEN_PRICES, PE_FLAT), [_minute(LEG_CE, _ts(D1, 15, 37), 130.0)])
    t = _run_legs(_legs_spec(intraday=False), src, fno_close=CLOSE).trades[0]
    (ce,), (pe,) = _legs(t, LEG_CE), _legs(t, LEG_PE)
    assert (ce.exit_reason, ce.exit_time, ce.exit_price) == ("stop", _ts(D1, 15, 35), 130.0)
    assert pe.exit_reason == "expiry"
    backtest = _run_legs(_legs_spec(intraday=False), src).trades[0]
    assert _legs(backtest, LEG_CE)[0].exit_reason == "expiry"


def test_a_combined_stop_on_the_last_index_candle_fills_after_the_close():
    """Combined stop met on D1's last index candle (15:25): both options are
    bought back at the next option candle's open (15:30), not the next morning."""
    legs = [sell("CE", stop_pct=100), sell("PE", stop_pct=100)]  # far away
    src = _plus(
        _legs_source(
            _with(OPEN_PRICES, {LEG_CE: {_ts(D1, 15, 25): (150.0, 200.0, 150.0, 190.0)}}),
            PE_FLAT,
        ),
        [_minute(LEG_CE, _ts(D1, 15, 32), 170.0), _minute(LEG_PE, _ts(D1, 15, 32), 81.0)],
    )
    spec = _legs_spec(legs, combined_stop=50, intraday=False)
    t = _run_legs(spec, src, fno_close=CLOSE).trades[0]
    (ce,), (pe,) = _legs(t, LEG_CE), _legs(t, LEG_PE)
    assert (ce.exit_reason, ce.exit_time, ce.exit_price) == (
        "combined_stop",
        _ts(D1, 15, 30),
        170.0,
    )
    assert (pe.exit_reason, pe.exit_price) == ("combined_stop", 81.0)
    backtest = _run_legs(spec, src).trades[0]
    assert _legs(backtest, LEG_CE)[0].exit_time == _ts(D2, 9, 15)
