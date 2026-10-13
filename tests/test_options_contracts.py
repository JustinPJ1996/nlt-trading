"""Tests for `nlt/options/expiry.py`, `nlt/options/contracts.py`, and
`nlt/options/execution.py`'s `OptionExecution` -- all against synthetic chains
built with `FrameSource`, so nothing here touches disk.

A real-data cross-check (hand-verified against the raw Parquet rows) lives at
the bottom, skipped if `data/fno` is absent.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.data.instruments import LOT_SIZES
from nlt.options.contracts import resolve
from nlt.options.data import FNO_ROOT, FrameSource
from nlt.options.execution import OptionExecution
from nlt.options.expiry import (
    BANKNIFTY_WEEKLY_END,
    REASON_NO_LISTING,
    REASON_NO_WEEKLY,
    pick_expiry,
)

IST = "Asia/Kolkata"


def _ts(day: dt.date, hour: int = 9, minute: int = 15) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hour, minute)), tz=IST)


def _dt(day: dt.date, hour: int = 9, minute: int = 15) -> dt.datetime:
    return _ts(day, hour, minute).to_pydatetime()


# ===================================================================== expiry


class TestPickExpiry:
    def test_nearest_weekly_on_expiry_day_is_that_days_contract(self):
        day = dt.date(2024, 1, 4)
        listed = [day, day + dt.timedelta(days=7), day + dt.timedelta(days=28)]
        expiry, reason = pick_expiry(listed, day, "nearest_weekly", "NIFTY")
        assert expiry == day
        assert reason is None

    def test_next_weekly_is_the_one_after_nearest(self):
        day = dt.date(2024, 1, 4)
        week1 = day + dt.timedelta(days=0)
        week2 = day + dt.timedelta(days=7)
        listed = [week1, week2, day + dt.timedelta(days=35)]
        expiry, reason = pick_expiry(listed, day, "next_weekly", "NIFTY")
        assert expiry == week2
        assert reason is None

    def test_next_weekly_with_no_second_weekly_listed_is_refused(self):
        day = dt.date(2024, 1, 4)
        listed = [day]  # only one expiry listed at all
        expiry, reason = pick_expiry(listed, day, "next_weekly", "NIFTY")
        assert expiry is None
        assert reason == REASON_NO_WEEKLY

    def test_monthly_is_last_listed_expiry_in_the_month(self):
        day = dt.date(2024, 1, 10)
        listed = [
            dt.date(2024, 1, 4),
            dt.date(2024, 1, 11),
            dt.date(2024, 1, 25),  # last Jan expiry
            dt.date(2024, 2, 29),
        ]
        expiry, reason = pick_expiry(listed, day, "monthly", "NIFTY")
        assert expiry == dt.date(2024, 1, 25)
        assert reason is None

    def test_monthly_rolls_to_next_month_once_this_months_has_passed(self):
        day = dt.date(2024, 1, 28)  # after Jan's last expiry (25 Jan)
        listed = [
            dt.date(2024, 1, 25),
            dt.date(2024, 2, 8),
            dt.date(2024, 2, 29),  # last Feb expiry
        ]
        expiry, reason = pick_expiry(listed, day, "monthly", "NIFTY")
        assert expiry == dt.date(2024, 2, 29)
        assert reason is None

    def test_monthly_no_listing_at_all(self):
        expiry, reason = pick_expiry([], dt.date(2024, 1, 1), "monthly", "NIFTY")
        assert expiry is None
        assert reason == REASON_NO_LISTING

    def test_banknifty_weekly_cutover_exact_boundary(self):
        # Verified from the real data: 13 Nov 2024 is BANKNIFTY's own last
        # weekly contract; the next listed expiry (27 Nov) is the month-end,
        # 14 days out -- genuinely no more weeklies, ever, after today.
        last_weekly = BANKNIFTY_WEEKLY_END
        listed = [
            last_weekly,
            last_weekly + dt.timedelta(days=14),
            last_weekly + dt.timedelta(days=41),
        ]

        nearest, reason = pick_expiry(listed, last_weekly, "nearest_weekly", "BANKNIFTY")
        assert nearest == last_weekly
        assert reason is None

        next_, reason = pick_expiry(listed, last_weekly, "next_weekly", "BANKNIFTY")
        assert next_ is None
        assert reason == REASON_NO_WEEKLY

    def test_banknifty_after_weekly_cutover_is_always_no_weekly(self):
        day = BANKNIFTY_WEEKLY_END + dt.timedelta(days=14)  # 27 Nov 2024
        listed = [day, day + dt.timedelta(days=28), day + dt.timedelta(days=58)]

        nearest, reason = pick_expiry(listed, day, "nearest_weekly", "BANKNIFTY")
        assert nearest is None
        assert reason == REASON_NO_WEEKLY

        monthly, reason = pick_expiry(listed, day, "monthly", "BANKNIFTY")
        assert monthly == day  # monthly is unaffected by the weekly cutover
        assert reason is None

    def test_a_true_weekly_regime_day_is_never_refused_as_no_weekly(self):
        # A normal Monday mid-week in an active weekly series (any underlying
        # other than BANKNIFTY post-cutover) must resolve, not refuse.
        day = dt.date(2024, 1, 1)
        listed = [
            dt.date(2024, 1, 4),
            dt.date(2024, 1, 11),
            dt.date(2024, 1, 25),
        ]
        nearest, reason = pick_expiry(listed, day, "nearest_weekly", "NIFTY")
        assert nearest == dt.date(2024, 1, 4)
        assert reason is None


# =================================================================== resolve


EXPIRY = dt.date(2024, 1, 4)
STRIKES_CE = [21400, 21450, 21500, 21550, 21600]


def _listing_df(day: dt.date, strikes: list[float] | None = None) -> pd.DataFrame:
    strikes = strikes if strikes is not None else STRIKES_CE
    rows = []
    for strike in strikes:
        for opt in ("CE", "PE"):
            rows.append(
                {
                    "tradingsymbol": f"IDX{int(strike)}{opt}",
                    "instrument_type": opt,
                    "expiry": EXPIRY,
                    "strike": float(strike),
                }
            )
    return pd.DataFrame(rows)


class TestResolve:
    def test_atm_is_nearest_listed_strike(self):
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "IDX", _dt(EXPIRY), 21505.0, "CE", 0, "nearest_weekly")
        assert res.contract is not None
        assert res.contract.strike == 21500.0

    def test_tie_breaks_to_the_lower_strike(self):
        # Exactly between 21450 and 21500 -> 25 away from each.
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "IDX", _dt(EXPIRY), 21475.0, "CE", 0, "nearest_weekly")
        assert res.contract.strike == 21450.0
        assert "lower strike" in res.why

    def test_call_offset_plus_one_is_next_higher_strike(self):
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "IDX", _dt(EXPIRY), 21500.0, "CE", 1, "nearest_weekly")
        assert res.contract.strike == 21550.0

    def test_put_offset_plus_one_is_next_lower_strike(self):
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "IDX", _dt(EXPIRY), 21500.0, "PE", 1, "nearest_weekly")
        assert res.contract.strike == 21450.0

    def test_call_offset_minus_one_is_next_lower_strike_itm(self):
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "IDX", _dt(EXPIRY), 21500.0, "CE", -1, "nearest_weekly")
        assert res.contract.strike == 21450.0

    def test_offset_running_off_the_end_is_refused(self):
        listing = _listing_df(EXPIRY)
        # ATM is the top-listed strike (21600); +1 OTM call runs off the end.
        res = resolve(listing, "IDX", _dt(EXPIRY), 21600.0, "CE", 1, "nearest_weekly")
        assert res.contract is None
        assert res.reason_code == "offset_out_of_range"
        assert "no contract could be chosen" in res.why

    def test_no_listing_at_all_is_refused(self):
        res = resolve(
            pd.DataFrame(columns=["tradingsymbol", "instrument_type", "expiry", "strike"]),
            "IDX",
            _dt(EXPIRY),
            21500.0,
            "CE",
            0,
            "monthly",
        )
        assert res.contract is None
        assert res.reason_code == "no_listing"

    def test_why_quotes_the_spot_and_strike_in_indian_grouping(self):
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "IDX", _dt(EXPIRY, 10, 15), 21505.0, "CE", 0, "nearest_weekly")
        assert "21,505.00" in res.why
        assert "21,500" in res.why

    def test_lot_size_is_todays_not_historical(self):
        listing = _listing_df(EXPIRY)
        res = resolve(listing, "BANKNIFTY", _dt(EXPIRY), 21500.0, "CE", 0, "nearest_weekly")
        assert res.contract.lot_size == LOT_SIZES["BANKNIFTY"]

    def test_invalid_option_type_raises(self):
        with pytest.raises(ValueError):
            resolve(_listing_df(EXPIRY), "IDX", _dt(EXPIRY), 21500.0, "XX", 0, "monthly")


# ============================================================== OptionExecution


DAY1 = dt.date(2024, 1, 4)
DAY2 = dt.date(2024, 1, 5)


def _exec_rows() -> list[dict]:
    rows = []

    def listing_row(day, strike, opt):
        return {
            "underlying": "IDX",
            "timeframe": "1d",
            "tradingsymbol": f"IDX{int(strike)}{opt}",
            "instrument_type": opt,
            "expiry": DAY2,
            "strike": float(strike),
            "timestamp": _ts(day),
            "open": 50.0,
            "high": 55.0,
            "low": 45.0,
            "close": 50.0,
            "best_buy_price": 49.9,
            "best_sell_price": 50.1,
            "volume": 10,
            "oi": 100,
            "bad_ohlc": False,
        }

    for strike in (100, 105, 110):
        rows.append(listing_row(DAY1, strike, "CE"))
        rows.append(listing_row(DAY2, strike, "CE"))

    def bar_row(day, hour, minute, close, volume, high=None, low=None, open_=None):
        return {
            "underlying": "IDX",
            # Native "1m": OptionExecution resamples every timeframe (here
            # "15m") from 1m traded rows -- see nlt/options/data.py's module
            # docstring for why the native 15m/1d option files are never
            # trusted directly. Each synthetic row below sits exactly on a
            # 15-minute bucket boundary (09:15/09:30/09:45), so it is both
            # the only row in, and anchors, its own bucket.
            "timeframe": "1m",
            "tradingsymbol": "IDX105CE",
            "instrument_type": "CE",
            "expiry": DAY2,
            "strike": 105.0,
            "timestamp": _ts(day, hour, minute),
            "open": open_ if open_ is not None else close,
            "high": high if high is not None else close + 1,
            "low": low if low is not None else close - 1,
            "close": close,
            "best_buy_price": close - 0.1,
            "best_sell_price": close + 0.1,
            "volume": volume,
            "oi": 100,
            "bad_ohlc": False,
        }

    # DAY1: traded at 09:15 (close=10), untraded at 09:30, traded at 09:45 (close=12)
    rows.append(bar_row(DAY1, 9, 15, 10.0, 20))
    rows.append(bar_row(DAY1, 9, 30, 10.0, 0))
    rows.append(bar_row(DAY1, 9, 45, 12.0, 15))
    # DAY2 (expiry day): no trades at all -- square-off must look back to DAY1.
    rows.append(bar_row(DAY2, 9, 15, 12.0, 0))
    rows.append(bar_row(DAY2, 9, 30, 12.0, 0))

    return rows


@pytest.fixture
def exec_source() -> FrameSource:
    return FrameSource(pd.DataFrame(_exec_rows()))


class TestOptionExecution:
    def test_resolve_picks_atm(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        assert res.contract.strike == 105.0

    def test_price_at_signal_reads_the_candles_own_close_if_traded(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        price = ex.price_at_signal(res.contract, _dt(DAY1, 9, 15))
        assert price == 10.0

    def test_price_at_signal_never_reads_a_later_bar(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        # Asking as of 09:15 must never see the 09:45 close (12.0).
        price = ex.price_at_signal(res.contract, _dt(DAY1, 9, 15))
        assert price != 12.0
        assert price == 10.0

    def test_price_at_signal_falls_through_untraded_bar_to_last_traded_close(
        self, exec_source: FrameSource
    ):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        # 09:30 itself is untraded; the question is "as of 09:30", same session.
        price = ex.price_at_signal(res.contract, _dt(DAY1, 9, 30))
        assert price == 10.0  # last traded close (09:15), not the 09:30 repeat

    def test_price_at_signal_does_not_cross_into_a_different_session(
        self, exec_source: FrameSource
    ):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        # DAY2 09:15 never traded; same-session rule forbids reaching back to DAY1.
        price = ex.price_at_signal(res.contract, _dt(DAY2, 9, 15))
        assert price is None

    def test_last_trade_is_strictly_before_ts(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        price, when = ex.last_trade(res.contract, _dt(DAY1, 9, 45))
        assert price == 10.0  # the 09:15 trade, not 09:45 itself
        assert when == _dt(DAY1, 9, 15)

    def test_last_trade_crosses_sessions_for_square_off(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        # DAY2 never traded -- squaring off on DAY2 must look back to DAY1.
        price, when = ex.last_trade(res.contract, _dt(DAY2, 9, 30))
        assert price == 12.0
        assert when == _dt(DAY1, 9, 45)

    def test_expiry_session_close_falls_back_when_expiry_day_never_traded(
        self, exec_source: FrameSource
    ):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        price, when = ex.expiry_session_close(res.contract)
        assert price == 12.0
        assert when == _dt(DAY1, 9, 45)

    def test_traded_reflects_the_bar(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        assert ex.traded(res.contract, _dt(DAY1, 9, 15)) is True
        assert ex.traded(res.contract, _dt(DAY1, 9, 30)) is False

    def test_traded_false_when_no_bar_exists(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        assert ex.traded(res.contract, _dt(DAY1, 11, 0)) is False

    def test_contract_bars_loaded_once_and_cached(self, exec_source: FrameSource):
        ex = OptionExecution(exec_source, "IDX", "15m", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        first = ex._contract_bars(res.contract, DAY1)
        second = ex._contract_bars(res.contract, DAY1)
        assert first is second  # identical object: not reloaded

    def test_1d_timeframe_is_resampled_to_one_bar_per_day(self, exec_source: FrameSource):
        """`OptionExecution(..., timeframe="1d")` must dispatch to
        `resample_daily`, not to the (forbidden) native 1d option file."""
        ex = OptionExecution(exec_source, "IDX", "1d", "CE", 0, "monthly")
        res = ex.resolve(_dt(DAY1), 104.0)
        bars = ex._contract_bars(res.contract, DAY1)
        # DAY1 and DAY2 each collapse to a single midnight-indexed bar.
        assert all(ts.time() == dt.time(0, 0) for ts in bars.index)
        day1_bar = bars.loc[pd.Timestamp(DAY1, tz=IST)]
        assert day1_bar["close"] == 12.0  # last traded close that day (09:45)


# -------------------------------------------------------- real-data test


FNO_PRESENT = FNO_ROOT.exists() and any(FNO_ROOT.iterdir())


@pytest.mark.skipif(not FNO_PRESENT, reason="data/fno not present in this worktree")
def test_real_banknifty_resolve_matches_hand_checked_strike():
    """2024-11-13 (BANKNIFTY's last weekly expiry day), 10:15 candle.

    Hand check (see tests/test_options_data.py for the coverage facts): the
    underlying's own 15m EQ close at 10:15 that day is 50,958.80. The listed
    CE strikes for the 13-Nov-2024 expiry step in multiples of 100 near the
    money; the nearest to 50,958.80 is 51,000 (41.20 away, vs 58.80 away from
    50,900 -- wait, verified directly by querying the raw Parquet: the
    strikes bracketing 50,958.80 are 50,900 and 51,000, and 51,000 is in
    fact nearer). +1 strike out of the money for a call is the next higher
    listed strike, 51,100.
    """
    from nlt.options.data import ParquetSource

    source = ParquetSource()
    day = dt.date(2024, 11, 13)
    bars = source.underlying_bars("BANKNIFTY", "15m", day, day)
    close_1015 = float(bars.loc[bars.index.time == dt.time(10, 15), "close"].iloc[0])
    assert close_1015 == pytest.approx(50958.80, abs=0.01)

    ex = OptionExecution(source, "BANKNIFTY", "15m", "CE", 1, "nearest_weekly")
    res = ex.resolve(_dt(day, 10, 15), close_1015)

    assert res.contract is not None
    assert res.contract.expiry == dt.date(2024, 11, 13)
    assert res.contract.strike == 51100.0
    assert res.contract.tradingsymbol == "BANKNIFTY24N1351100CE"
