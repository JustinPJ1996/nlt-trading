"""Tests for `nlt/engine/legs_backtest.py`: sold options and multi-leg positions.

Every price is synthetic and stated outright so each rule can be checked by
hand. The chain: NIFTY, 5-minute candles, strikes 21,000 to 21,900 every 100,
a weekly expiring Thursday 4 Jan 2024 and a monthly on 25 Jan. The index sits
at 21,490 (nearest strike 21,500) unless a test moves it. Every option trades
at a flat price (100 unless a test says otherwise) on every candle.

One 1-minute row is written at the start of each 5-minute bucket, so the
resampled 5-minute bar's open/high/low/close are that row's -- unless the row
has volume 0, in which case the bucket did not trade at all.

Margin is a fake (`FixedMargin`): Rs `per_unit` for every unit SOLD, nothing
for bought units -- the real SPAN module is tested on its own.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

import pandas as pd
import pytest

from nlt.costs.charges import NseOptionsCharges, charge_fn
from nlt.costs.slippage import Slippage
from nlt.engine.legs_backtest import run_legs_backtest
from nlt.options.contracts import resolve_points
from nlt.options.data import FrameSource
from nlt.options.execution import OptionExecution
from nlt.spec.models import (
    All,
    AtTime,
    ExitRules,
    Instrument,
    IvPercentile,
    OnExpiryDay,
    OptionLeg,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)

IST = "Asia/Kolkata"
D1, D2, D3 = dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)
DAYS = (D1, D2, D3)
WEEKLY, MONTHLY = D3, dt.date(2024, 1, 25)
STRIKES = tuple(range(21000, 22000, 100))
CAPITAL = 2_500_000.0
LOT = 65


def _ts(day: dt.date, hh: int, mm: int) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hh, mm)), tz=IST)


def _slots(day: dt.date) -> list[pd.Timestamp]:
    start = _ts(day, 9, 15)
    return [start + pd.Timedelta(minutes=5 * k) for k in range(75)]  # 09:15..15:25


def sym(strike: int, kind: str, expiry: dt.date = WEEKLY) -> str:
    return f"NIFTY{expiry:%y%m%d}{strike}{kind}"


CE, PE = sym(21500, "CE"), sym(21500, "PE")


def _index(moves: dict[pd.Timestamp, float] | None = None, days=DAYS) -> pd.DataFrame:
    moves = moves or {}
    rows = []
    for day in days:
        for ts in _slots(day):
            c = moves.get(ts, 21490.0)
            rows.append({"timestamp": ts, "open": c, "high": c, "low": c, "close": c, "volume": 0})
    return pd.DataFrame(rows).set_index("timestamp")


def _source(prices=None, defaults=None, days=DAYS, weekly=WEEKLY) -> FrameSource:
    """`prices[symbol][ts] = (open, high, low, close[, volume])`; volume 0 = no trade.

    `defaults[symbol]` replaces the flat 100 for that option on every other candle.
    """
    prices = prices or {}
    defaults = defaults or {}
    rows = []
    for day in days:
        for expiry in (weekly, MONTHLY):
            if day > expiry:
                continue
            for strike in STRIKES:
                for kind in ("CE", "PE"):
                    s = sym(strike, kind, expiry)
                    base = dict(
                        underlying="NIFTY",
                        tradingsymbol=s,
                        instrument_type=kind,
                        expiry=expiry,
                        strike=float(strike),
                        best_buy_price=0.0,
                        best_sell_price=0.0,
                        oi=0,
                        bad_ohlc=False,
                    )
                    rows.append(
                        {
                            **base,
                            "timeframe": "1d",
                            "timestamp": _ts(day, 9, 15),
                            "open": 1.0,
                            "high": 1.0,
                            "low": 1.0,
                            "close": 1.0,
                            "volume": 1,
                        }
                    )
                    flat = defaults.get(s, (100.0, 100.0, 100.0, 100.0))
                    for ts in _slots(day):
                        p = prices.get(s, {}).get(ts, flat)
                        rows.append(
                            {
                                **base,
                                "timeframe": "1m",
                                "timestamp": ts,
                                "open": p[0],
                                "high": p[1],
                                "low": p[2],
                                "close": p[3],
                                "volume": p[4] if len(p) > 4 else 10,
                            }
                        )
    return FrameSource(pd.DataFrame(rows))


@dataclass(frozen=True)
class Breakdown:
    total: float
    source: str = "test margin"
    note: str | None = None
    span: float = 0.0
    exposure: float = 0.0
    expiry_day_extra: float = 0.0


@dataclass
class FixedMargin:
    per_unit: float = 1000.0
    calls: list = field(default_factory=list)

    def __call__(self, underlying, day, legs, prev_close):
        self.calls.append((underlying, day, list(legs), prev_close))
        sold = sum(-leg.quantity for leg in legs if leg.quantity < 0)
        return Breakdown(total=sold * self.per_unit)


def sell(kind, **kw):
    return OptionLeg(side="sell", option_type=kind, **kw)


def buy(kind, **kw):
    return OptionLeg(side="buy", option_type=kind, **kw)


STRADDLE = [sell("CE", stop_pct=25), sell("PE", stop_pct=25)]


def _spec(
    legs=None,
    *,
    entry=None,
    combined_target=None,
    combined_stop=None,
    intraday=True,
    lots=1,
    max_pct=None,
    max_concurrent=1,
    no_entry_after=dt.time(15, 0),
):
    legs = STRADDLE if legs is None else legs
    return StrategySpec(
        name="t",
        description="t",
        instrument=Instrument(symbol="NIFTY", timeframe="5m", trade_as="option", legs=legs),
        entry=entry or AtTime(time=dt.time(9, 20)),
        exit=ExitRules(
            combined_target_pct=combined_target,
            combined_stop_pct=combined_stop,
            exits_on_legs=True,
        ),
        sizing=Sizing(mode="fixed_lots", lots=lots),
        risk=RiskLimits(max_position_pct=max_pct, max_concurrent_positions=max_concurrent),
        schedule=Schedule(intraday=intraday, no_entry_after=no_entry_after),
    )


def _run(spec, source, *, index=None, margin=None, days=DAYS, **kw):
    ex = OptionExecution(source, "NIFTY", "5m")
    kw.setdefault("capital", CAPITAL)
    return run_legs_backtest(
        spec,
        index if index is not None else _index(days=days),
        ex,
        margin or FixedMargin(),
        **kw,
    )


def _legs(trade, symbol):
    return [f for f in trade.legs if f.symbol == symbol]


# Basis = each option's close on the 09:15 candle (the signal); the fill is the
# 09:20 candle's open. CE: basis 100, fill 102. PE: basis 80, fill 79.
OPEN_PRICES = {
    CE: {_ts(D1, 9, 20): (102.0, 102.0, 102.0, 102.0)},
    PE: {_ts(D1, 9, 15): (80.0, 80.0, 80.0, 80.0), _ts(D1, 9, 20): (79.0, 79.0, 79.0, 79.0)},
}
PE_FLAT = {PE: (80.0, 80.0, 80.0, 80.0)}


def _with(base, extra):
    out = {k: dict(v) for k, v in base.items()}
    for k, v in extra.items():
        out.setdefault(k, {}).update(v)
    return out


# ------------------------------------------------------------------ entries


def test_straddle_fills_both_legs_at_0920_open_with_each_stop_above_its_own_fill():
    """Sold CE: fill 102, stop 102 + 25% x 100 = 127 (ABOVE). Sold PE: fill 79,
    stop 79 + 25% x 80 = 99. Nothing touches either, so both close at the 15:15
    candle's open. One Trade for the position, two legs inside it."""
    r = _run(_spec(), _source(OPEN_PRICES, PE_FLAT), days=(D1,))
    assert len(r.trades) == 1
    t = r.trades[0]
    (ce,), (pe,) = _legs(t, CE), _legs(t, PE)
    assert ce.side == pe.side == "sell"
    assert ce.entry_time == pe.entry_time == _ts(D1, 9, 20)
    assert (ce.entry_price, ce.basis, ce.stop_level) == (102.0, 100.0, 127.0)
    assert (pe.entry_price, pe.basis, pe.stop_level) == (79.0, 80.0, 99.0)
    assert ce.exit_reason == pe.exit_reason == "square_off"
    assert ce.exit_time == _ts(D1, 15, 15)
    assert ce.quantity == pe.quantity == LOT
    # Position: a net credit of 102 + 79 = 181 a unit, bought back at 100 + 80.
    assert t.direction == "short"
    assert t.entry_price == pytest.approx(181.0)
    assert t.exit_price == pytest.approx(180.0)
    assert t.quantity == LOT
    assert t.gross_pnl == pytest.approx((102 - 100) * LOT + (79 - 80) * LOT)


def test_one_leg_that_did_not_trade_at_the_fill_skips_the_whole_position():
    src = _source(_with(OPEN_PRICES, {PE: {_ts(D1, 9, 20): (79.0, 79.0, 79.0, 79.0, 0)}}), PE_FLAT)
    r = _run(_spec(), src, days=(D1,))
    assert r.trades == []
    skipped = [d for d in r.decisions if d.reason_code == "leg_not_traded"]
    assert len(skipped) == 1
    assert "21,500 put did not trade at 09:20" in skipped[0].detail["message"]


# ------------------------------------------------------------- per-leg exits


def test_sold_leg_stops_above_and_the_other_leg_stays_open():
    """CE's 10:00 candle reaches 127 exactly: stopped there. PE untouched, so it
    stays open to the square-off -- a stop on one leg is not a stop on all."""
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (110.0, 127.0, 108.0, 120.0)}}), PE_FLAT)
    t = _run(_spec(), src, days=(D1,)).trades[0]
    (ce,), (pe,) = _legs(t, CE), _legs(t, PE)
    assert (ce.exit_reason, ce.exit_price, ce.exit_time) == ("stop", 127.0, _ts(D1, 10, 0))
    assert pe.exit_reason == "square_off"
    assert pe.exit_time == _ts(D1, 15, 15)
    assert ce.gross_pnl == pytest.approx((102 - 127) * LOT)


def test_a_candle_below_the_stop_does_not_stop_a_sold_leg():
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (110.0, 126.9, 60.0, 120.0)}}), PE_FLAT)
    (ce,) = _legs(_run(_spec(), src, days=(D1,)).trades[0], CE)
    assert ce.exit_reason == "square_off"


def test_gap_past_a_sold_legs_stop_fills_at_the_open():
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (140.0, 150.0, 135.0, 145.0)}}), PE_FLAT)
    (ce,) = _legs(_run(_spec(), src, days=(D1,)).trades[0], CE)
    assert (ce.exit_reason, ce.exit_price) == ("stop", 140.0)


def test_stop_is_assumed_first_when_one_candle_hits_stop_and_target():
    """CE target 50% of 100 below the fill: 52. One candle spans 50..130."""
    legs = [sell("CE", stop_pct=25, target_pct=50), sell("PE", stop_pct=25)]
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (100.0, 130.0, 50.0, 90.0)}}), PE_FLAT)
    t = _run(_spec(legs), src, days=(D1,)).trades[0]
    (ce,) = _legs(t, CE)
    assert (ce.exit_reason, ce.exit_price, ce.both_hit) == ("stop", 127.0, True)
    assert ce.target_level == 52.0
    assert t.both_hit


def test_sold_legs_target_is_below_its_fill():
    legs = [sell("CE", stop_pct=25, target_pct=50), sell("PE", stop_pct=25)]
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (60.0, 60.0, 51.0, 55.0)}}), PE_FLAT)
    (ce,) = _legs(_run(_spec(legs), src, days=(D1,)).trades[0], CE)
    assert (ce.exit_reason, ce.exit_price) == ("target", 52.0)


def test_a_legs_stop_waits_for_a_candle_in_which_it_traded():
    """10:00 did not trade (its row's 200 is a stale repeat, not a trade); the
    next real trade, at 10:05, opened at 130 -- past 127, so it fills at 130."""
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {
                    _ts(D1, 10, 0): (200.0, 200.0, 200.0, 200.0, 0),
                    _ts(D1, 10, 5): (130.0, 131.0, 129.0, 130.0),
                }
            },
        ),
        PE_FLAT,
    )
    (ce,) = _legs(_run(_spec(), src, days=(D1,)).trades[0], CE)
    assert (ce.exit_time, ce.exit_price) == (_ts(D1, 10, 5), 130.0)


# ------------------------------------------------------------ combined exits

# Credit at the signal: CE 100 + PE 80 = 180 a unit (x 65).


def test_combined_target_is_judged_on_closes_and_fills_every_leg_at_the_next_open():
    """50% of 180 = 90 a unit. At the 11:00 close CE 40, PE 40: profit
    (102 - 40) + (79 - 40) = 101 >= 90. Every leg closes at 11:05's open."""
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {
                    _ts(D1, 11, 0): (60.0, 60.0, 40.0, 40.0),
                    _ts(D1, 11, 5): (41.0, 44.0, 38.0, 43.0),
                },
                PE: {
                    _ts(D1, 11, 0): (60.0, 60.0, 40.0, 40.0),
                    _ts(D1, 11, 5): (39.0, 42.0, 36.0, 37.0),
                },
            },
        ),
        PE_FLAT,
    )
    spec = _spec([sell("CE"), sell("PE")], combined_target=50, combined_stop=100)
    t = _run(spec, src, days=(D1,)).trades[0]
    (ce,), (pe,) = _legs(t, CE), _legs(t, PE)
    assert ce.exit_reason == pe.exit_reason == t.exit_reason == "combined_target"
    assert ce.exit_time == pe.exit_time == _ts(D1, 11, 5)
    assert (ce.exit_price, pe.exit_price) == (41.0, 39.0)


def test_combined_target_ignores_lows_that_only_touched_it_inside_the_candle():
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {_ts(D1, 11, 0): (100.0, 100.0, 10.0, 100.0)},
                PE: {_ts(D1, 11, 0): (80.0, 80.0, 10.0, 80.0)},
            },
        ),
        PE_FLAT,
    )
    spec = _spec([sell("CE"), sell("PE")], combined_target=50, combined_stop=100)
    t = _run(spec, src, days=(D1,)).trades[0]
    assert t.exit_reason == "square_off"


def test_combined_stop_assumes_every_leg_was_at_its_worst_together():
    """30% of the credit at the SIGNAL (180) = 54 a unit -- not of the fills' 181,
    which would be 54.30. On 10:00 CE's high is 140 (loss 38) and PE's high
    95.20 (loss 16.20): 54.20 together, though each closed calm. The stop is
    decided and every leg buys back at 10:05's open."""
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {_ts(D1, 10, 0): (100.0, 140.0, 100.0, 100.0), _ts(D1, 10, 5): (103.0,) * 4},
                PE: {_ts(D1, 10, 0): (80.0, 95.2, 80.0, 80.0), _ts(D1, 10, 5): (81.0,) * 4},
            },
        ),
        PE_FLAT,
    )
    spec = _spec([sell("CE"), sell("PE")], combined_target=50, combined_stop=30)
    t = _run(spec, src, days=(D1,)).trades[0]
    (ce,), (pe,) = _legs(t, CE), _legs(t, PE)
    assert ce.exit_reason == pe.exit_reason == "combined_stop"
    assert (ce.exit_time, ce.exit_price, pe.exit_price) == (_ts(D1, 10, 5), 103.0, 81.0)


def test_combined_stop_just_short_of_its_level_does_not_fire():
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {_ts(D1, 10, 0): (100.0, 140.0, 100.0, 100.0)},
                PE: {_ts(D1, 10, 0): (80.0, 94.0, 80.0, 80.0)},
            },
        ),
        PE_FLAT,
    )
    spec = _spec([sell("CE"), sell("PE")], combined_target=50, combined_stop=30)
    assert _run(spec, src, days=(D1,)).trades[0].exit_reason == "square_off"


def test_combined_stop_wins_when_one_candle_meets_both():
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {_ts(D1, 10, 0): (100.0, 200.0, 10.0, 10.0)},
                PE: {_ts(D1, 10, 0): (80.0, 80.0, 10.0, 10.0)},
            },
        ),
        PE_FLAT,
    )
    spec = _spec([sell("CE"), sell("PE")], combined_target=50, combined_stop=30)
    t = _run(spec, src, days=(D1,)).trades[0]
    assert t.exit_reason == "combined_stop"
    assert t.both_hit


def test_combined_exit_on_a_leg_that_did_not_trade_uses_its_last_price_and_flags_it():
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {
                    _ts(D1, 10, 0): (100.0, 140.0, 100.0, 100.0),
                    _ts(D1, 10, 5): (999.0, 999.0, 999.0, 999.0, 0),
                },
                PE: {_ts(D1, 10, 0): (80.0, 96.0, 80.0, 80.0)},
            },
        ),
        PE_FLAT,
    )
    spec = _spec([sell("CE"), sell("PE")], combined_target=50, combined_stop=30)
    t = _run(spec, src, days=(D1,)).trades[0]
    (ce,) = _legs(t, CE)
    assert (ce.exit_price, ce.exit_untraded) == (100.0, True)
    assert t.exit_untraded


# ------------------------------------------------------- strikes from points


def _listing():
    return _source(days=(D1,)).listing("NIFTY", D1)


def test_points_from_spot_takes_the_nearest_listed_strike_and_says_why():
    r = resolve_points(_listing(), "NIFTY", _ts(D1, 9, 25), 21490.0, "PE", -200, "nearest_weekly")
    assert r.contract.strike == 21300
    assert "21,490.00" in r.why and "21,290.00" in r.why and "21,300" in r.why


def test_points_from_spot_breaks_a_tie_to_the_lower_strike():
    r = resolve_points(_listing(), "NIFTY", _ts(D1, 9, 25), 21400.0, "CE", 50, "nearest_weekly")
    assert r.contract.strike == 21400
    assert "tie" in r.why


def test_points_beyond_the_listed_strikes_are_refused_never_clamped():
    """-700 from 21,490 is 20,790; the lowest listed is 21,000, 210 away --
    more than one 100-point gap."""
    r = resolve_points(_listing(), "NIFTY", _ts(D1, 9, 25), 21490.0, "PE", -700, "nearest_weekly")
    assert r.contract is None and r.reason_code == "points_out_of_range"
    assert "20,790.00" in r.why and "21,000" in r.why
    # One gap away is still accepted: -580 -> 20,910, 90 from 21,000.
    ok = resolve_points(_listing(), "NIFTY", _ts(D1, 9, 25), 21490.0, "PE", -580, "nearest_weekly")
    assert ok.contract.strike == 21000


def test_bull_put_spread_by_points_trades_both_legs_and_needs_no_stop():
    spec = _spec([sell("PE", strike_points=-200), buy("PE", strike_points=-400)])
    t = _run(spec, _source(), days=(D1,)).trades[0]
    assert {(f.symbol, f.side) for f in t.legs} == {
        (sym(21300, "PE"), "sell"),
        (sym(21100, "PE"), "buy"),
    }


def test_a_leg_whose_strike_is_out_of_range_skips_the_signal():
    spec = _spec([sell("PE", strike_points=-700, stop_pct=25)])
    r = _run(spec, _source(), days=(D1,))
    assert r.trades == []
    assert [d.reason_code for d in r.decisions if d.kind == "signal_skipped"] == [
        "points_out_of_range"
    ]


# ----------------------------------------------------------------- re-entry


def test_same_strike_reentry_resells_when_a_later_close_is_back_at_cost_and_stops_at_max():
    """CE stopped at 127 on 10:00. 10:00's own close (100) is not "later". The
    10:30 close is 101 -- at or below the first fill, 102 -- so it re-sells at
    10:35's open, 100, stop 100 + 25% x 101 = 125.25. Stopped again on 11:00;
    the one re-entry is used, so 11:30 coming back does nothing."""
    legs = [
        sell("CE", stop_pct=25, reentry="same_strike_at_cost", reentry_max=1),
        sell("PE", stop_pct=25),
    ]
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {
                    _ts(D1, 10, 0): (110.0, 130.0, 100.0, 100.0),
                    _ts(D1, 10, 5): (110.0, 110.0, 110.0, 110.0),
                    _ts(D1, 10, 10): (110.0, 110.0, 110.0, 110.0),
                    _ts(D1, 10, 15): (110.0, 110.0, 110.0, 110.0),
                    _ts(D1, 10, 20): (110.0, 110.0, 110.0, 110.0),
                    _ts(D1, 10, 25): (110.0, 110.0, 110.0, 110.0),
                    _ts(D1, 10, 30): (101.0, 101.0, 101.0, 101.0),
                    _ts(D1, 10, 35): (100.0, 100.0, 100.0, 100.0),
                    _ts(D1, 11, 0): (120.0, 126.0, 120.0, 125.0),
                    _ts(D1, 11, 30): (100.0, 100.0, 100.0, 100.0),
                    _ts(D1, 11, 35): (100.0, 100.0, 100.0, 100.0),
                }
            },
        ),
        {**PE_FLAT, CE: (110.0, 110.0, 110.0, 110.0)},
    )
    t = _run(_spec(legs), src, days=(D1,)).trades[0]
    first, second = sorted(_legs(t, CE), key=lambda f: f.entry_time)
    assert (first.exit_reason, first.exit_time, first.reentry) == ("stop", _ts(D1, 10, 0), 0)
    assert (second.entry_time, second.entry_price, second.basis) == (_ts(D1, 10, 35), 100.0, 101.0)
    assert second.stop_level == pytest.approx(125.25)
    assert (second.exit_reason, second.exit_price, second.reentry) == ("stop", 125.25, 1)
    assert len(_legs(t, CE)) == 2


def test_same_strike_reentry_does_not_use_the_stop_candles_own_close():
    legs = [
        sell("CE", stop_pct=25, reentry="same_strike_at_cost", reentry_max=1),
        sell("PE", stop_pct=25),
    ]
    src = _source(
        _with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (110.0, 130.0, 100.0, 100.0)}}),
        {**PE_FLAT, CE: (110.0, 110.0, 110.0, 110.0)},
    )
    t = _run(_spec(legs), src, days=(D1,)).trades[0]
    assert len(_legs(t, CE)) == 1


def test_fresh_strike_reentry_picks_the_strike_again_from_the_index_close():
    """CE stopped on 10:00, when the index closed at 21,610: the 21,600 call is
    sold at 10:05's open (95), basis its 10:00 close (90)."""
    legs = [
        sell("CE", stop_pct=25, reentry="fresh_strike_now", reentry_max=1),
        sell("PE", stop_pct=25),
    ]
    ce600 = sym(21600, "CE")
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {_ts(D1, 10, 0): (110.0, 130.0, 110.0, 120.0)},
                ce600: {_ts(D1, 10, 0): (90.0,) * 4, _ts(D1, 10, 5): (95.0,) * 4},
            },
        ),
        {**PE_FLAT, CE: (110.0,) * 4},
    )
    index = _index({_ts(D1, 10, 0): 21610.0}, days=(D1,))
    t = _run(_spec(legs), src, index=index, days=(D1,)).trades[0]
    (again,) = _legs(t, ce600)
    assert (again.entry_time, again.entry_price, again.basis) == (_ts(D1, 10, 5), 95.0, 90.0)
    assert again.stop_level == pytest.approx(95 + 22.5)
    assert again.reentry == 1


def test_no_reentry_at_or_after_the_no_entry_time():
    legs = [
        sell("CE", stop_pct=25, reentry="fresh_strike_now", reentry_max=2),
        sell("PE", stop_pct=25),
    ]
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 15, 0): (110.0, 130.0, 110.0, 120.0)}}), PE_FLAT)
    t = _run(_spec(legs), src, days=(D1,)).trades[0]
    assert len(_legs(t, CE)) == 1
    assert {f.symbol for f in t.legs} == {CE, PE}


# ------------------------------------------------------------------- sizing


def test_sets_are_capped_so_margin_stays_within_20_percent():
    """One set sells 130 units x Rs 1,000 = Rs 1,30,000 of margin. 20% of
    Rs 25 lakh is Rs 5,00,000, so 3 of the 5 sets asked for."""
    margin = FixedMargin()
    t = _run(
        _spec(lots=5, max_pct=20), _source(OPEN_PRICES, PE_FLAT), margin=margin, days=(D1,)
    ).trades[0]
    assert t.quantity == 3 * LOT
    assert {f.quantity for f in t.legs} == {3 * LOT}
    assert margin.calls[-1][2][0].quantity == -3 * LOT


def test_selling_defaults_to_a_20_percent_cap_when_the_spec_sets_none():
    t = _run(_spec(lots=5), _source(OPEN_PRICES, PE_FLAT), days=(D1,)).trades[0]
    assert t.quantity == 3 * LOT


def test_zero_sets_skips_with_the_numbers_in_the_reason():
    r = _run(
        _spec(max_pct=20), _source(OPEN_PRICES, PE_FLAT), margin=FixedMargin(5000.0), days=(D1,)
    )
    assert r.trades == []
    (d,) = [d for d in r.decisions if d.reason_code == "margin"]
    assert d.detail["message"] == (
        "One set needs Rs 6,50,000 margin; 20% of the account is Rs 5,00,000, so the "
        "position was skipped."
    )


def test_bought_premium_is_added_on_top_of_margin():
    """`MarginBreakdown.total` is read as EXCLUDING bought premium. Bull put
    spread: margin 65 x 1,000 = 65,000; the bought put's premium 100 x 65 =
    6,500 on top: 71,500. 20% of 3,50,000 is 70,000 -- one set does not fit."""
    spec = _spec([sell("PE", strike_points=-200), buy("PE", strike_points=-400)], max_pct=20)
    r = _run(spec, _source(), capital=350_000, days=(D1,))
    assert r.trades == []
    (d,) = [d for d in r.decisions if d.reason_code == "margin"]
    assert "Rs 71,500" in d.detail["message"]
    ok = _run(spec, _source(), capital=360_000, days=(D1,))
    assert len(ok.trades) == 1


def test_a_second_position_must_fit_in_the_money_left_spare():
    """Overnight, two at once allowed, 60% cap. Margin 1,200 a sold unit: one
    set 1,56,000 against a cap of 1,80,000 on Rs 3 lakh. The D2 signal finds
    only 3,00,000 - 1,56,000 = 1,44,000 spare, and is skipped saying so; D3's too."""
    spec = _spec(intraday=False, max_pct=60, max_concurrent=2)
    r = _run(spec, _source(OPEN_PRICES, PE_FLAT), capital=300_000, margin=FixedMargin(1200.0))
    assert len(r.trades) == 1
    skipped = [d for d in r.decisions if d.reason_code == "margin"]
    assert [d.time.date() for d in skipped] == [D2, D3]
    assert all("the spare money was Rs 1,44,000" in d.detail["message"] for d in skipped)


# ------------------------------------------------------------------ charges


def test_a_sold_leg_is_charged_as_a_sell_on_entry_and_a_buy_on_exit():
    calls = []

    def spy(price, qty, side):
        calls.append((price, qty, side))
        return 0.0

    _run(_spec([sell("CE", stop_pct=25)]), _source(OPEN_PRICES, PE_FLAT), charge_fn=spy, days=(D1,))
    assert calls == [(102.0, LOT, "sell"), (100.0, LOT, "buy")]


def test_stt_falls_on_the_short_legs_opening_sale():
    cf = charge_fn(NseOptionsCharges())
    t = _run(
        _spec([sell("CE", stop_pct=25)]), _source(OPEN_PRICES, PE_FLAT), charge_fn=cf, days=(D1,)
    ).trades[0]
    model = NseOptionsCharges()
    expected = model.charges(102.0, LOT, "sell").total + model.charges(100.0, LOT, "buy").total
    assert t.charges == pytest.approx(expected)
    assert model.charges(102.0, LOT, "sell").stt == pytest.approx(round(0.0015 * 102 * LOT, 2))


def test_slippage_lowers_a_sale_and_raises_a_buy_back():
    t = _run(
        _spec([sell("CE", stop_pct=25)]),
        _source(OPEN_PRICES, PE_FLAT),
        slippage=Slippage("rs", 1.0),
        days=(D1,),
    ).trades[0]
    (ce,) = t.legs
    assert (ce.raw_entry_price, ce.entry_price) == (102.0, 101.0)
    assert (ce.raw_exit_price, ce.exit_price) == (100.0, 101.0)


# ------------------------------------------------------ square-off and expiry


def test_square_off_at_the_end_of_the_1515_minute_and_last_traded_price_when_untraded():
    """The call: the 15:15 minute's close, 93 (the price at 15:15:59). The put
    did not trade in that minute: its last trade before it, 77, flagged."""
    src = _source(
        _with(
            OPEN_PRICES,
            {
                CE: {_ts(D1, 15, 15): (90.0, 95.0, 85.0, 93.0)},
                PE: {
                    _ts(D1, 15, 10): (70.0, 70.0, 70.0, 77.0),
                    _ts(D1, 15, 15): (60.0, 60.0, 60.0, 60.0, 0),
                },
            },
        ),
        PE_FLAT,
    )
    t = _run(_spec(), src, days=(D1,)).trades[0]
    (ce,), (pe,) = _legs(t, CE), _legs(t, PE)
    assert (ce.exit_price, ce.exit_untraded) == (93.0, False)
    assert (pe.exit_price, pe.exit_untraded) == (77.0, True)
    assert t.exit_untraded


def test_overnight_position_closes_every_leg_on_expiry_day_never_past_it():
    """Entered D1 (and D2 refused: one position at a time); expiry Thursday D3
    closes both legs at the 15:15 open."""
    src = _source(
        _with(
            OPEN_PRICES, {CE: {_ts(D3, 15, 15): (55.0,) * 4}, PE: {_ts(D3, 15, 15): (45.0,) * 4}}
        ),
        PE_FLAT,
    )
    t = _run(_spec(intraday=False), src).trades[0]
    assert {f.exit_reason for f in t.legs} == {"expiry"}
    assert {f.exit_time for f in t.legs} == {_ts(D3, 15, 15)}
    assert sorted(f.exit_price for f in t.legs) == [45.0, 55.0]


def test_a_calendar_closes_all_legs_on_the_near_legs_expiry():
    legs = [sell("CE", stop_pct=25), buy("CE", expiry="monthly")]
    t = _run(_spec(legs, intraday=False), _source(OPEN_PRICES, PE_FLAT)).trades[0]
    assert {f.exit_time for f in t.legs} == {_ts(D3, 15, 15)}
    assert {f.exit_reason for f in t.legs} == {"expiry"}


def test_no_entry_whose_fill_would_be_at_the_expiry_close():
    spec = _spec(
        entry=All(conditions=[AtTime(time=dt.time(15, 15)), OnExpiryDay()]), intraday=False
    )
    r = _run(spec, _source())
    assert r.trades == []
    assert [d.reason_code for d in r.decisions if d.kind == "signal_skipped"] == ["expiry"]


# -------------------------------------------- expiry-day and IV conditions


def test_expiry_day_condition_trades_only_on_the_weekly_expiry():
    spec = _spec(entry=All(conditions=[AtTime(time=dt.time(9, 20)), OnExpiryDay()]))
    r = _run(spec, _source())
    assert [t.entry_time.date() for t in r.trades] == [D3]


def test_iv_percentile_condition_uses_the_supplied_series_and_states_its_warm_up():
    def iv(underlying, index_bars, lookback):
        assert (underlying, lookback) == ("NIFTY", 252)
        by_day = {D1: float("nan"), D2: 80.0, D3: 50.0}
        return pd.Series([by_day[d] for d in index_bars.index.date], index=index_bars.index)

    spec = _spec(
        entry=All(conditions=[AtTime(time=dt.time(9, 20)), IvPercentile(op="gt", value=70)])
    )
    r = _run(spec, _source(), iv_fn=iv)
    assert [t.entry_time.date() for t in r.trades] == [D2]
    assert any("first known on 03 Jan 2024" in w for w in r.warnings)


# ------------------------------------------------------------- the account


def test_a_loss_bigger_than_the_account_is_reported_not_hidden():
    """A 1000% stop on a sold call that gaps to 5,000 loses ~Rs 3.2 lakh on a
    Rs 2 lakh account. Nothing asserts -- that loss is possible -- but it is said."""
    legs = [sell("CE", stop_pct=1000)]
    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (5000.0,) * 4}}), PE_FLAT)
    r = _run(_spec(legs, max_pct=100), src, capital=200_000, days=(D1,))
    assert r.trades[0].net_pnl < -200_000
    assert any(w.startswith("LOSS LARGER THAN THE ACCOUNT") for w in r.warnings)


def test_margin_source_and_entry_only_rule_reach_the_warnings():
    r = _run(_spec(), _source(OPEN_PRICES, PE_FLAT), days=(D1,))
    (w,) = [w for w in r.warnings if w.startswith("MARGIN AT ENTRY")]
    assert "test margin" in w and "checked at entry only" in w


# ----------------------------------------------- what the report says about it


def test_report_states_the_selling_rules_and_weak_spots():
    from nlt.costs.slippage import ZERO
    from nlt.report import method

    spec = _spec(
        [sell("CE", stop_pct=25, reentry="same_strike_at_cost", reentry_max=1), sell("PE")],
        combined_stop=30,
    )
    said = " ".join(
        method.assumptions(
            spec,
            capital=CAPITAL,
            charge_model=NseOptionsCharges(),
            first=D1,
            last=D3,
            slippage=ZERO,
        )
    )
    assert "must trade in the candle" in said
    assert "A sold option's stop is above" in said
    assert "at its worst at the same moment" in said
    assert "sold again once its price comes back" in said
    assert "within 20% of the account" in said
    assert "only a reference point" in said
    spots, notes = method.weak_spots(spec, ["MARGIN AT ENTRY: x", "other"], slippage=ZERO)
    joined = " ".join(spots)
    assert "blow straight past the stop" in joined
    assert "20 Nov 2024" in joined
    assert "MARGIN AT ENTRY: x" in spots and notes == ["other"]


def test_trade_table_lists_every_leg_of_a_position():
    from app.api import _trade_row

    src = _source(_with(OPEN_PRICES, {CE: {_ts(D1, 10, 0): (110.0, 127.0, 108.0, 120.0)}}), PE_FLAT)
    spec = _spec()
    row = _trade_row(_run(spec, src, days=(D1,)).trades[0], spec, "5m")
    assert row["direction"] == "Net credit"
    assert row["symbol"] == "NIFTY 21,500 CE + NIFTY 21,500 PE"
    assert [(leg["side"], leg["label"], leg["reason"]) for leg in row["legs"]] == [
        ("Sold", "NIFTY 21,500 CE", "stop loss hit"),
        ("Sold", "NIFTY 21,500 PE", "end-of-day square-off"),
    ]
    # Option prices to the paisa, on the position and on every leg.
    shown = [row["entry_price"], row["exit_price"]] + [
        leg[k] for leg in row["legs"] for k in ("entry_price", "exit_price", "stop")
    ]
    assert all(re.search(r"Rs [\d,]+\.\d\d", text) for text in shown), shown


# --------------------------------------------- known answer, on the real data

_REAL_1M = (
    __import__("pathlib").Path(__file__).resolve().parents[1] / "data/fno/1m/underlying=NIFTY"
)


@pytest.mark.skipif(not _REAL_1M.exists(), reason="the F&O 1-minute data is not on this machine")
def test_known_answer_0920_short_straddle_on_2_jan_2025():
    """Checked by hand against the raw 1-minute rows (2026-10-14):

    NIFTY closed at 23,788.85 on the 09:19 minute (the candle ending 09:20); the
    nearest listed strike of the 2 Jan 2025 weekly (expiring that day) is 23,800.
    23800 CE: last trade 09:19 closed 111.10 (basis); 09:20 opened 107.65 (sold).
    Stop 107.65 + 25% x 111.10 = 135.425. Highs stay below it until 09:41's
    127.45; 09:42 OPENS at 136.55, past the stop, so it is bought back at 136.55.
    23800 PE: basis 116.50, sold 118.60, stop 147.725; its highest trade all day
    was 140.00, so it is bought back at the 15:15 minute's close, 0.15 (the
    price at 15:15:59; it opened at 0.20).
    Charges (Zerodha's options schedule): CE 37.05 sell + 27.60 buy; PE 38.40
    sell + 23.60 buy = 126.65. Gross (107.65 - 136.55) x 65 + (118.60 - 0.15)
    x 65 = 5,820.75; net 5,694.10. Margin is a fixed Rs 1,000 a sold unit here.
    """
    from app.logic import _backtest_legs_spec
    from nlt.translate.rules import TranslationResult

    spec = StrategySpec(
        name="09:20 straddle",
        description="t",
        instrument=Instrument(symbol="NIFTY", timeframe="1m", trade_as="option", legs=STRADDLE),
        entry=AtTime(time=dt.time(9, 20)),
        exit=ExitRules(exits_on_legs=True),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_position_pct=20),
        schedule=Schedule(intraday=True),
    )
    r = _backtest_legs_spec(
        spec,
        capital=2_500_000,
        cost_model_label="Index options (NIFTY/BANKNIFTY weekly)",
        start=dt.date(2025, 1, 2),
        end=dt.date(2025, 1, 2),
        translation=TranslationResult(spec=spec),
        slippage=Slippage(),
        margin_fn=FixedMargin(),
    )
    assert r.error is None
    (t,) = r.backtest.trades
    ce = next(f for f in t.legs if f.symbol == "NIFTY2510223800CE")
    pe = next(f for f in t.legs if f.symbol == "NIFTY2510223800PE")
    assert t.signal_time == pd.Timestamp("2025-01-02 09:19", tz=IST)
    assert ce.entry_time == pe.entry_time == pd.Timestamp("2025-01-02 09:20", tz=IST)
    assert (ce.basis, ce.entry_price) == (pytest.approx(111.10), pytest.approx(107.65))
    assert ce.stop_level == pytest.approx(135.425)
    assert ce.exit_time == pd.Timestamp("2025-01-02 09:42", tz=IST)
    assert (ce.exit_reason, ce.exit_price) == ("stop", pytest.approx(136.55))
    assert (pe.basis, pe.entry_price) == (pytest.approx(116.50), pytest.approx(118.60))
    assert pe.stop_level == pytest.approx(147.725)
    assert pe.exit_time == pd.Timestamp("2025-01-02 15:15", tz=IST)
    assert (pe.exit_reason, pe.exit_price) == ("square_off", pytest.approx(0.15))
    assert ce.quantity == pe.quantity == 65
    assert ce.charges == pytest.approx(37.05 + 27.60)
    assert pe.charges == pytest.approx(38.40 + 23.60)
    assert t.gross_pnl == pytest.approx(5820.75)
    assert t.net_pnl == pytest.approx(5694.10)


@pytest.mark.skipif(not _REAL_1M.exists(), reason="the F&O 1-minute data is not on this machine")
def test_known_answer_0920_bought_straddle_on_2_jan_2025():
    """The same day and strikes, BOUGHT, with a 25% stop on each option.
    Checked by hand against the raw 1-minute rows (2026-10-16):

    23800 CE: basis 111.10 (09:19 close), bought at 107.65 (09:20 open). Stop
    107.65 - 25% x 111.10 = 79.875; its lowest trade after 09:20 was 90.40, so
    it is sold at the 15:15 minute's close, 390.85.
    23800 PE: basis 116.50, bought at 118.60, stop 118.60 - 29.125 = 89.475.
    09:37 opened 103.10 and traded down to 86.95: stopped at 89.475 (no gap).
    Gross (390.85 - 107.65) x 65 + (89.475 - 118.60) x 65 = 16,514.875.
    Premium paid 14,706.25, under 10% of Rs 25 lakh: one set.
    """
    from app.logic import _backtest_legs_spec
    from nlt.translate.rules import TranslationResult

    legs = [buy("CE", stop_pct=25), buy("PE", stop_pct=25)]
    spec = StrategySpec(
        name="09:20 bought straddle",
        description="t",
        instrument=Instrument(symbol="NIFTY", timeframe="1m", trade_as="option", legs=legs),
        entry=AtTime(time=dt.time(9, 20)),
        exit=ExitRules(exits_on_legs=True),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_position_pct=10),
        schedule=Schedule(intraday=True),
    )
    margin = FixedMargin()
    r = _backtest_legs_spec(
        spec,
        capital=2_500_000,
        cost_model_label="Index options (NIFTY/BANKNIFTY weekly)",
        start=dt.date(2025, 1, 2),
        end=dt.date(2025, 1, 2),
        translation=TranslationResult(spec=spec),
        slippage=Slippage(),
        margin_fn=margin,
    )
    assert r.error is None
    assert margin.calls == []  # nothing sold, no exchange margin
    (t,) = r.backtest.trades
    ce = next(f for f in t.legs if f.symbol == "NIFTY2510223800CE")
    pe = next(f for f in t.legs if f.symbol == "NIFTY2510223800PE")
    assert ce.side == pe.side == "buy"
    assert (ce.basis, ce.entry_price) == (pytest.approx(111.10), pytest.approx(107.65))
    assert ce.stop_level == pytest.approx(79.875)
    assert (ce.exit_reason, ce.exit_price) == ("square_off", pytest.approx(390.85))
    assert ce.exit_time == pd.Timestamp("2025-01-02 15:15", tz=IST)
    assert (pe.basis, pe.entry_price) == (pytest.approx(116.50), pytest.approx(118.60))
    assert pe.stop_level == pytest.approx(89.475)
    assert (pe.exit_reason, pe.exit_price) == ("stop", pytest.approx(89.475))
    assert pe.exit_time == pd.Timestamp("2025-01-02 09:37", tz=IST)
    assert ce.quantity == pe.quantity == 65
    assert t.gross_pnl == pytest.approx(16514.875)
    assert t.net_pnl == pytest.approx(t.gross_pnl - ce.charges - pe.charges)


@pytest.mark.skipif(not _REAL_1M.exists(), reason="the F&O 1-minute data is not on this machine")
def test_known_answer_0920_bull_call_spread_on_2_jan_2025():
    """A debit spread with no stop -- covered since 2026-10-16. By hand from the
    raw 1-minute rows: index 23,788.85 at 09:19, so ATM 23800 and +200 points
    -> 24000. Bought 23800 CE at 107.65 (09:20 open), sold 24000 CE at 35.00
    (09:20 open; basis 35.15). Both squared off at the 15:15 close: 390.85 and
    190.15. Gross (390.85 - 107.65) x 65 + (35.00 - 190.15) x 65 = 8,323.25.
    """
    from app.logic import _backtest_legs_spec
    from nlt.translate.rules import TranslationResult

    legs = [buy("CE"), sell("CE", strike_points=200)]
    spec = StrategySpec(
        name="bull call spread",
        description="t",
        instrument=Instrument(symbol="NIFTY", timeframe="1m", trade_as="option", legs=legs),
        entry=AtTime(time=dt.time(9, 20)),
        exit=ExitRules(exits_on_legs=True),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_position_pct=20),
        schedule=Schedule(intraday=True),
    )
    r = _backtest_legs_spec(
        spec,
        capital=2_500_000,
        cost_model_label="Index options (NIFTY/BANKNIFTY weekly)",
        start=dt.date(2025, 1, 2),
        end=dt.date(2025, 1, 2),
        translation=TranslationResult(spec=spec),
        slippage=Slippage(),
        margin_fn=FixedMargin(),
    )
    assert r.error is None
    (t,) = r.backtest.trades
    bought = next(f for f in t.legs if f.symbol == "NIFTY2510223800CE")
    sold = next(f for f in t.legs if f.symbol == "NIFTY2510224000CE")
    assert (bought.side, bought.entry_price, bought.exit_price) == (
        "buy",
        107.65,
        pytest.approx(390.85),
    )
    assert (sold.side, sold.basis, sold.entry_price) == ("sell", pytest.approx(35.15), 35.00)
    assert (sold.exit_reason, sold.exit_price) == ("square_off", pytest.approx(190.15))
    assert sold.stop_level is None
    assert t.gross_pnl == pytest.approx(8323.25)


def test_backtest_spec_sends_a_spec_with_legs_to_the_legs_engine(monkeypatch):
    import app.logic as logic

    seen = []
    monkeypatch.setattr(
        logic, "_backtest_legs_spec", lambda spec, **kw: seen.append(spec) or "legs"
    )
    monkeypatch.setattr(logic, "_backtest_option_spec", lambda spec, **kw: "single")
    assert logic.backtest_spec(_spec()) == "legs"
    assert logic.paper_problem(_spec()) is None  # options paper trade since batch 3


def test_charges_note_says_the_strategy_sells_options():
    from app.logic import charge_model_for_spec

    _model, note = charge_model_for_spec(_spec(), "Index futures")
    assert "buys and sells index options" in note


# ------------------------------------------ an expiry listed on a holiday

# The dataset lists NIFTY's 30 Mar 2023 and 29 Jun 2023 expiries at their
# nominal dates, which were market holidays: the contracts really expired the
# session before. Here the weekly is listed for Friday 5 Jan (no session that
# day; the next is Monday 8 Jan), so it really expires Thursday 4 Jan.
HOLIDAY = dt.date(2024, 1, 5)
D8 = dt.date(2024, 1, 8)
H_DAYS = (D1, D2, D3, D8)
H_CE, H_PE = sym(21500, "CE", HOLIDAY), sym(21500, "PE", HOLIDAY)


def _holiday_source(prices=None):
    return _source(prices, {H_PE: (80.0,) * 4}, days=H_DAYS, weekly=HOLIDAY)


def test_holiday_listed_expiry_closes_on_the_session_before():
    src = _holiday_source(
        {H_CE: {_ts(D3, 15, 15): (55.0,) * 4}, H_PE: {_ts(D3, 15, 15): (45.0,) * 4}}
    )
    t = _run(_spec(intraday=False), src, days=H_DAYS).trades[0]
    assert {f.symbol for f in t.legs} == {H_CE, H_PE}
    assert {(f.exit_reason, f.exit_time) for f in t.legs} == {("expiry", _ts(D3, 15, 15))}


def test_holiday_listed_expiry_day_is_the_session_before_for_conditions_and_entries():
    spec = _spec(entry=All(conditions=[AtTime(time=dt.time(9, 20)), OnExpiryDay()]))
    r = _run(spec, _holiday_source(), days=H_DAYS)
    assert [t.entry_time.date() for t in r.trades] == [D3]
    late = _spec(entry=AtTime(time=dt.time(15, 15)), intraday=False)
    r = _run(late, _source(days=(D3, D8), weekly=HOLIDAY), days=(D3, D8))
    skipped = [(d.time.date(), d.reason_code) for d in r.decisions if d.kind == "signal_skipped"]
    assert (D3, "expiry") in skipped


def test_batch_1_bought_option_also_closes_on_the_real_last_day():
    """The same bug existed in `run_option_backtest`: a bought call held
    overnight into a holiday-listed expiry was never closed on its real last
    day, and the engine then failed holding it past expiry."""
    from nlt.engine.options_backtest import run_option_backtest
    from nlt.spec.models import Compare, Const, Ref

    spec = StrategySpec(
        name="t",
        description="t",
        instrument=Instrument(
            symbol="NIFTY",
            timeframe="5m",
            trade_as="option",
            option_type="CE",
            expiry="nearest_weekly",
        ),
        entry=Compare(op="gt", left=Ref(name="close"), right=Const(value=21500.0)),
        exit=ExitRules(stop_pct=90.0),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_lots=1, max_position_pct=10.0),
        schedule=Schedule(intraday=False),
    )
    src = _holiday_source({H_CE: {_ts(D3, 15, 15): (55.0,) * 4}})
    index = _index({_ts(D1, 10, 0): 21510.0}, days=H_DAYS)
    ex = OptionExecution(src, "NIFTY", "5m", "CE", 0, "nearest_weekly")
    (t,) = run_option_backtest(spec, index, ex, capital=CAPITAL).trades
    assert (t.symbol, t.exit_reason, t.exit_time, t.exit_price) == (
        H_CE,
        "expiry",
        _ts(D3, 15, 15),
        55.0,
    )


# ------------------------------------------------ when no margin figure exists


def test_a_day_with_no_margin_figure_is_skipped_never_guessed():
    from nlt.engine.legs_backtest import MarginNotAvailable

    def no_file(underlying, day, legs, prev_close):
        raise MarginNotAvailable("We have NIFTY margin files from 1 Dec 2022 only.")

    r = _run(_spec(), _source(OPEN_PRICES, PE_FLAT), margin=no_file, days=(D1,))
    assert r.trades == []
    (d,) = [d for d in r.decisions if d.reason_code == "margin_unavailable"]
    assert "rather than sized on a guess" in d.detail["message"]
    assert any("1 Dec 2022 only" in w for w in r.warnings)


_REAL_SPAN = __import__("pathlib").Path(__file__).resolve().parents[1] / "data/span/NSE"


@pytest.mark.skipif(
    not (_REAL_1M.exists() and _REAL_SPAN.exists()),
    reason="the F&O 1-minute data or the exchange margin files are not on this machine",
)
def test_known_answer_straddle_with_the_real_exchange_margin():
    """The same 2 Jan 2025 trade through the app's real SPAN margin: one set
    needs Rs 2,75,317.38 (SPAN 1,51,854.30 + exposure 1,23,463.08, of which
    61,731.54 is the expiry-day extra) -- within 20% of Rs 25 lakh, so one set,
    the same fills. A day before the margin files is refused, not guessed."""
    from app.logic import _backtest_legs_spec, span_margin_fn
    from nlt.engine.legs_backtest import MarginNotAvailable, MarginQuery
    from nlt.translate.rules import TranslationResult

    fn = span_margin_fn()
    expiry = dt.date(2025, 1, 2)
    b = fn(
        "NIFTY",
        expiry,
        [MarginQuery("CE", 23800.0, expiry, -65), MarginQuery("PE", 23800.0, expiry, -65)],
        23742.90,
    )
    assert b.total == pytest.approx(275317.38)
    with pytest.raises(MarginNotAvailable):
        fn("NIFTY", dt.date(2019, 1, 1), [MarginQuery("CE", 18000.0, dt.date(2019, 1, 3), -65)], 1)

    spec = StrategySpec(
        name="09:20 straddle",
        description="t",
        instrument=Instrument(symbol="NIFTY", timeframe="1m", trade_as="option", legs=STRADDLE),
        entry=AtTime(time=dt.time(9, 20)),
        exit=ExitRules(exits_on_legs=True),
        sizing=Sizing(mode="fixed_lots", lots=1),
        risk=RiskLimits(max_position_pct=20),
        schedule=Schedule(intraday=True),
    )
    r = _backtest_legs_spec(
        spec,
        capital=2_500_000,
        cost_model_label="Index options (NIFTY/BANKNIFTY weekly)",
        start=expiry,
        end=expiry,
        translation=TranslationResult(spec=spec),
        slippage=Slippage(),
    )
    assert r.error is None
    (t,) = r.backtest.trades
    assert t.quantity == 65
    assert t.net_pnl == pytest.approx(5694.10)
    assert any("NSE's margin file for 2 Jan 2025" in w for w in r.weak_spots)


def test_margin_sources_are_listed_first_to_last_opened_not_alphabetically() -> None:
    from nlt.engine.legs_backtest import _count_warnings, _Counts

    counts = _Counts()
    for source in (
        "NSE's file for 31 Dec 2024",
        "NSE's file for 2 Jan 2025",
        "NSE's file for 1 Apr 2025",
    ):
        counts.margin_sources[source] = 1
    text = " ".join(_count_warnings(counts, dt.time(15, 15), _spec(), 2_500_000))
    assert "(NSE's file for 31 Dec 2024 ... NSE's file for 1 Apr 2025)" in text


def test_option_prices_show_paise_never_round_to_zero():
    from app.api import _price_inr

    assert _price_inr(0.2) == "Rs 0.20"
    assert _price_inr(135.425) in ("Rs 135.42", "Rs 135.43")
    assert _price_inr(1999.999) == "Rs 2,000.00"
    assert _price_inr(-12345.5) == "-Rs 12,345.50"
