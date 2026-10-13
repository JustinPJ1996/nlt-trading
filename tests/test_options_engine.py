"""Tests for `nlt/engine/options_backtest.py`: buying one call or put.

Every price here is synthetic and stated outright, so each rule can be
checked by hand. The chain: NIFTY, 15-minute candles, strikes 21,400 /
21,500 / 21,600, a weekly expiring Thursday 4 Jan 2024 and a monthly on
25 Jan. The index sits at 21,490 (nearest strike 21,500) except on the one
candle where the entry rule fires.

One 1-minute row is written at the start of each 15-minute bucket, so the
resampled 15-minute bar's open/high/low/close are that row's -- unless the
row has volume 0, in which case the bucket did not trade at all.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.costs.slippage import Slippage
from nlt.engine.options_backtest import run_option_backtest
from nlt.options.data import FrameSource
from nlt.options.execution import OptionExecution
from nlt.spec.models import (
    Compare,
    Const,
    ExitRules,
    Instrument,
    Ref,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)

IST = "Asia/Kolkata"
D1, D2, D3 = dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)
WEEKLY, MONTHLY = D3, dt.date(2024, 1, 25)
STRIKES = (21400, 21500, 21600)
CAPITAL = 2_500_000.0


def _ts(day: dt.date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hh, mm)), tz=IST)


def _slots(day: dt.date) -> list[pd.Timestamp]:
    start = _ts(day, 9, 15)
    return [start + pd.Timedelta(minutes=15 * k) for k in range(25)]  # 09:15..15:15


def _symbol(strike: int, kind: str, expiry: dt.date = WEEKLY) -> str:
    return f"NIFTY{expiry:%y%m%d}{strike}{kind}"


def _index_bars(days, signal_at: pd.Timestamp | None, signal_close: float = 21510.0):
    rows = []
    for day in days:
        for ts in _slots(day):
            c = signal_close if ts == signal_at else 21490.0
            rows.append({"timestamp": ts, "open": c, "high": c, "low": c, "close": c, "volume": 0})
    return pd.DataFrame(rows).set_index("timestamp")


def _chain(days, prices: dict[str, dict[pd.Timestamp, tuple]], default=(100.0, 100.0, 100.0, 100.0)):
    """Raw rows for FrameSource.

    `prices[symbol][ts] = (open, high, low, close[, volume])` overrides the
    default flat 100 for that option at that 15-minute slot; volume 0 means
    it did not trade there.
    """
    rows = []
    for day in days:
        for expiry in (WEEKLY, MONTHLY):
            if day > expiry:
                continue
            for strike in STRIKES:
                for kind in ("CE", "PE"):
                    sym = _symbol(strike, kind, expiry)
                    base = dict(
                        underlying="NIFTY",
                        tradingsymbol=sym,
                        instrument_type=kind,
                        expiry=expiry,
                        strike=float(strike),
                        best_buy_price=0.0,
                        best_sell_price=0.0,
                        oi=0,
                        bad_ohlc=False,
                    )
                    rows.append(
                        {**base, "timeframe": "1d", "timestamp": _ts(day, 9, 15), "open": 1.0,
                         "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
                    )
                    for ts in _slots(day):
                        p = prices.get(sym, {}).get(ts, default)
                        o, h, low, c = p[:4]
                        vol = p[4] if len(p) > 4 else 10
                        rows.append(
                            {**base, "timeframe": "1m", "timestamp": ts, "open": o, "high": h,
                             "low": low, "close": c, "volume": vol}
                        )
    return FrameSource(pd.DataFrame(rows))


def _spec(*, stop=30.0, target=60.0, intraday=True, timeframe="15m", option_type="CE"):
    return StrategySpec(
        name="t",
        description="t",
        instrument=Instrument(
            symbol="NIFTY",
            timeframe=timeframe,
            trade_as="option",
            option_type=option_type,
            strike_offset=0,
            expiry="nearest_weekly",
        ),
        entry=Compare(op="gt", left=Ref(name="close"), right=Const(value=21500.0)),
        exit=ExitRules(stop_pct=stop, target_pct=target),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_lots=2, max_position_pct=10.0),
        schedule=Schedule(intraday=intraday),
    )


def _run(spec, bars, source, **kw):
    ex = OptionExecution(
        source,
        "NIFTY",
        spec.instrument.timeframe,
        spec.instrument.option_type,
        spec.instrument.strike_offset,
        spec.instrument.expiry,
    )
    return run_option_backtest(spec, bars, ex, capital=CAPITAL, **kw)


ATM_CE = _symbol(21500, "CE")
SIGNAL = _ts(D1, 10, 0)  # closes 10:15
FILL = _ts(D1, 10, 15)


def test_fills_next_candles_option_open_and_measures_levels_from_the_premium():
    """Signal on the 10:00 candle; the option closed at 100 on it (the basis).
    Fill at the 10:15 candle's option open, 104. Stop 30% of 100 below the
    fill = 74; target 60% of 100 above = 164. Not the index's 21,510."""
    source = _chain([D1], {ATM_CE: {SIGNAL: (98, 101, 97, 100), FILL: (104, 106, 103, 105)}})
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    assert len(res.trades) == 1
    t = res.trades[0]
    assert t.symbol == ATM_CE
    assert t.entry_time == FILL
    assert t.entry_price == 104.0
    assert t.stop_level == pytest.approx(74.0)
    assert t.target_level == pytest.approx(164.0)
    assert t.quantity == 65  # today's NIFTY lot
    assert "21,500" in t.contract_why and "21,510" in t.contract_why


def test_option_that_did_not_trade_at_the_fill_skips_the_signal_never_another_strike():
    untraded = (104, 104, 104, 104, 0)
    source = _chain([D1], {ATM_CE: {FILL: untraded}})
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    assert res.trades == []
    skipped = [d for d in res.decisions if d.kind == "signal_skipped"]
    assert [d.reason_code for d in skipped] == ["option_not_traded"]
    assert "21500 CE did not trade" in skipped[0].detail["message"].replace(",", "")
    assert any("did not trade in the candle" in w for w in res.warnings)


def test_stop_fills_at_its_level_and_a_gap_through_it_fills_at_the_open():
    later = _ts(D1, 11, 0)
    source = _chain([D1], {ATM_CE: {FILL: (100, 100, 100, 100), later: (90, 92, 60, 65)}})
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    assert res.trades[0].exit_reason == "stop"
    assert res.trades[0].exit_price == pytest.approx(70.0)

    source = _chain([D1], {ATM_CE: {FILL: (100, 100, 100, 100), later: (55, 58, 50, 52)}})
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    assert res.trades[0].exit_price == pytest.approx(55.0)


def test_exits_wait_for_a_candle_where_the_option_actually_traded():
    """The 11:00 candle 'shows' a low of 10 but did not trade (volume 0): no
    exit there. The next traded candle (11:15) opens at 65, below the 70
    stop, so the stop fills at 65 at 11:15."""
    source = _chain(
        [D1],
        {
            ATM_CE: {
                FILL: (100, 100, 100, 100),
                _ts(D1, 11, 0): (10, 10, 10, 10, 0),
                _ts(D1, 11, 15): (65, 66, 60, 61),
            }
        },
    )
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    t = res.trades[0]
    assert t.exit_time == _ts(D1, 11, 15)
    assert t.exit_price == pytest.approx(65.0)


def test_stop_is_assumed_first_when_one_candle_hits_both():
    source = _chain([D1], {ATM_CE: {FILL: (100, 100, 100, 100), _ts(D1, 11, 0): (100, 200, 50, 100)}})
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    t = res.trades[0]
    assert t.exit_reason == "stop" and t.both_hit


def test_square_off_fills_at_the_1515_candles_open_not_its_close():
    sq = _ts(D1, 15, 15)
    source = _chain([D1], {ATM_CE: {FILL: (100, 100, 100, 100), sq: (110, 125, 108, 120)}})
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    t = res.trades[0]
    assert t.exit_reason == "square_off"
    assert t.exit_time == sq
    assert t.exit_price == pytest.approx(110.0)
    assert not t.exit_untraded


def test_square_off_on_an_untraded_candle_uses_the_last_traded_price_and_flags_it():
    sq = _ts(D1, 15, 15)
    source = _chain(
        [D1],
        {ATM_CE: {FILL: (100, 100, 100, 100), _ts(D1, 15, 0): (100, 112, 100, 111), sq: (5, 5, 5, 5, 0)}},
    )
    res = _run(_spec(), _index_bars([D1], SIGNAL), source)
    t = res.trades[0]
    assert t.exit_reason == "square_off"
    assert t.exit_price == pytest.approx(111.0)
    assert t.exit_untraded
    assert any("last traded price" in w for w in res.warnings)


def test_overnight_position_is_closed_at_1515_on_expiry_day_never_held_past_it():
    sq = _ts(D3, 15, 15)
    source = _chain([D1, D2, D3], {ATM_CE: {sq: (130, 140, 129, 139)}}, default=(100, 101, 99, 100))
    res = _run(_spec(stop=90.0, target=90.0, intraday=False), _index_bars([D1, D2, D3], SIGNAL), source)
    t = res.trades[0]
    assert t.exit_reason == "expiry"
    assert t.exit_time == sq
    assert t.exit_price == pytest.approx(130.0)
    assert t.exit_time.date() <= WEEKLY


def test_signal_that_would_fill_after_the_expiry_day_close_is_skipped():
    """On expiry day a 15:00 signal would fill at 15:15 -- the contract's last
    close time -- so it is skipped. (Overnight spec, so the 15:00 no-entry
    cut-off does not apply and the expiry rule is what is tested.)"""
    signal = _ts(D3, 15, 0)
    source = _chain([D3], {})
    res = _run(_spec(intraday=False), _index_bars([D3], signal), source)
    assert res.trades == []
    assert [d.reason_code for d in res.decisions if d.kind == "signal_skipped"] == ["expiry"]


def test_put_and_slippage_and_charges_reach_the_trade():
    atm_pe = _symbol(21500, "PE")
    source = _chain([D1], {atm_pe: {FILL: (100, 100, 100, 100)}})
    calls = []

    def charge_fn(price, qty, side):
        calls.append((round(price, 4), qty, side))
        return 10.0

    res = _run(
        _spec(option_type="PE"),
        _index_bars([D1], SIGNAL),
        source,
        slippage=Slippage("rs", 1.0),
        charge_fn=charge_fn,
    )
    t = res.trades[0]
    assert t.symbol == atm_pe
    assert t.entry_price == pytest.approx(101.0)
    assert t.raw_entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(99.0)  # square-off at 100 open, minus Rs 1
    assert t.charges == pytest.approx(20.0)
    assert calls[0] == (101.0, 65, "buy") and calls[1] == (99.0, 65, "sell")
    assert t.net_pnl == pytest.approx((99.0 - 101.0) * 65 - 20.0)


def test_levels_never_depend_on_candles_after_the_signal():
    """Changing every option price after the fill must not move the stop or
    target: they come from the signal candle only."""
    a = _chain([D1], {ATM_CE: {SIGNAL: (98, 101, 97, 100)}}, default=(100, 101, 99, 100))
    b = _chain([D1], {ATM_CE: {SIGNAL: (98, 101, 97, 100)}}, default=(100, 300, 99, 250))
    ra = _run(_spec(), _index_bars([D1], SIGNAL), a)
    rb = _run(_spec(), _index_bars([D1], SIGNAL), b)
    assert ra.trades[0].stop_level == rb.trades[0].stop_level
    assert ra.trades[0].target_level == pytest.approx(rb.trades[0].target_level)


def test_a_sold_option_is_refused():
    spec = _spec()
    sold = spec.model_copy(update={"direction": "short"})
    with pytest.raises(ValueError):
        _run(sold, _index_bars([D1], SIGNAL), _chain([D1], {}))


def test_a_hold_for_n_candles_exit_also_waits_for_a_traded_candle():
    """'Exit after 2 candles' would fall on 10:30, where the option did not
    trade; it fills on the next traded candle (10:45) at a real price."""
    spec = _spec(stop=90.0, target=90.0)
    spec = spec.model_copy(update={"exit": ExitRules(stop_pct=90.0, max_bars_held=2)})
    source = _chain(
        [D1],
        {ATM_CE: {FILL: (100, 100, 100, 100), _ts(D1, 10, 30): (1, 1, 1, 1, 0), _ts(D1, 10, 45): (97, 99, 96, 98)}},
    )
    res = _run(spec, _index_bars([D1], SIGNAL), source)
    t = res.trades[0]
    assert t.exit_reason == "max_bars"
    assert t.exit_time == _ts(D1, 10, 45)
    assert t.exit_price == pytest.approx(98.0)
