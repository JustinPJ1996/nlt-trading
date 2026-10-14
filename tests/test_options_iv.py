"""Tests for `nlt/options/iv.py`: implied volatility from the ATM options' own
prices, its no-look-ahead percentile, and the feature column the engine reads.

Synthetic chains go through `FrameSource`, so these run with no data on disk.
The known-answer test uses figures copied from the raw 1-minute rows; a twin
checks the real dataset gives the same row when it is present.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from nlt.engine.conditions import iv_percentile_column
from nlt.options import data as option_data
from nlt.options import iv
from nlt.options.data import FrameSource, ParquetSource
from nlt.options.iv import (
    RATE,
    bs_price,
    daily_atm_iv,
    implied_vol,
    iv_feature,
    iv_percentile,
)

IST = "Asia/Kolkata"


# ------------------------------------------------------------- Black-Scholes


def test_black_scholes_matches_the_textbook_example():
    # Hull, Options Futures and Other Derivatives, example 15.6: S=42, K=40,
    # r=10%, T=0.5, vol=20% -> call 4.76, put 0.81.
    assert bs_price(42, 40, 0.5, 0.2, "CE", rate=0.10) == pytest.approx(4.76, abs=0.005)
    assert bs_price(42, 40, 0.5, 0.2, "PE", rate=0.10) == pytest.approx(0.81, abs=0.005)


@pytest.mark.parametrize("option_type", ["CE", "PE"])
@pytest.mark.parametrize(
    "spot, strike, days, vol",
    [
        (22000, 22000, 7, 0.12),
        (22000, 22050, 1.2, 0.35),
        (48000, 47500, 30, 0.18),
        (80000, 80100, 0.3, 0.6),
        (22000, 21000, 14, 0.15),  # deep in or out of the money
        (22000, 22000, 0.5, 1.5),  # a wild half-day: still solved, not capped at 100%
    ],
)
def test_price_to_iv_to_price_round_trips(option_type, spot, strike, days, vol):
    price = bs_price(spot, strike, days / 365, vol, option_type)
    solved = implied_vol(price, spot, strike, days, option_type)
    assert solved == pytest.approx(vol, abs=1e-5)
    assert bs_price(spot, strike, days / 365, solved, option_type) == pytest.approx(price, abs=1e-4)


def test_a_price_below_what_the_option_is_worth_at_zero_volatility_has_no_iv():
    # A call 200 in the money, 7 days out: worth at least 200 + a little rate.
    floor = 22200 - 22000 * math.exp(-RATE * 7 / 365)
    assert implied_vol(floor - 0.5, 22200, 22000, 7, "CE") is None
    assert implied_vol(floor, 22200, 22000, 7, "CE") is None
    assert implied_vol(floor + 5, 22200, 22000, 7, "CE") is not None
    # And a put 200 in the money.
    put_floor = 22000 * math.exp(-RATE * 7 / 365) - 21800
    assert implied_vol(put_floor - 0.5, 21800, 22000, 7, "PE") is None
    assert implied_vol(put_floor + 5, 21800, 22000, 7, "PE") is not None


@pytest.mark.parametrize(
    "price, days", [(0, 7), (-1, 7), (100, 0), (100, -1), (float("nan"), 7), (None, 7)]
)
def test_nonsense_inputs_have_no_iv(price, days):
    assert implied_vol(price, 22000, 22000, days, "CE") is None


def test_a_price_more_than_any_sane_volatility_explains_has_no_iv():
    # An ATM weekly at 15,000 on a 22,000 index would need a vol far above 500%.
    assert implied_vol(15000, 22000, 22000, 7, "CE") is None


# ---------------------------------------------------------------- percentile


def _sessions(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-01", periods=n, tz=IST, name="date")


def test_percentile_is_the_share_of_the_window_below_the_previous_day():
    values = [0.10, 0.30, 0.20, 0.40, 0.25, 0.99]
    daily = pd.Series(values, index=_sessions(6))
    out = iv_percentile(daily, 4)
    # Day 5's value comes from days 1-4 (0.10, 0.30, 0.20, 0.40) and day 4's
    # IV 0.40: three of the four are below it.
    assert out.iloc[4] == pytest.approx(75.0)
    # Day 6: window days 2-5 (0.30, 0.20, 0.40, 0.25), day 5's IV 0.25: one below.
    assert out.iloc[5] == pytest.approx(25.0)


def test_percentile_never_uses_the_day_itself():
    """A strategy deciding at 09:20 on D cannot know D's close. Changing D's
    IV must not change D's percentile -- only the next day's."""
    rng = np.random.default_rng(37)
    daily = pd.Series(rng.uniform(0.1, 0.3, 40), index=_sessions(40))
    before = iv_percentile(daily, 20)
    for spike in (0.0001, 9.0):
        moved = daily.copy()
        moved.iloc[30] = spike
        after = iv_percentile(moved, 20)
        assert after.iloc[30] == before.iloc[30]
        pd.testing.assert_series_equal(after.iloc[:31], before.iloc[:31])
    # ... and day 31 does see it.
    assert iv_percentile(moved, 20).iloc[31] != before.iloc[31]


def test_percentile_is_nan_until_a_full_window_exists():
    daily = pd.Series(np.linspace(0.1, 0.2, 30), index=_sessions(30))
    out = iv_percentile(daily, 10)
    # Session 11 (position 10) is the first whose previous day has 10 IVs up to it.
    assert out.iloc[:10].isna().all()
    assert out.iloc[10:].notna().all()
    # A rising series: the previous day is always the highest of its window.
    assert out.iloc[10] == pytest.approx(90.0)


def test_percentile_is_nan_when_the_previous_day_has_no_iv_and_skips_gaps():
    values = [0.1, 0.2, 0.3, np.nan, 0.15, 0.25]
    daily = pd.Series(values, index=_sessions(6))
    out = iv_percentile(daily, 3)
    assert math.isnan(out.iloc[4])  # previous day (position 3) has no IV
    # Position 5 uses day 4's IV 0.15 against the last three IVs: 0.2, 0.3, 0.15.
    assert out.iloc[5] == pytest.approx(0.0)
    # Position 3 uses day 2 (0.3) against 0.1, 0.2, 0.3.
    assert out.iloc[3] == pytest.approx(200 / 3)


# ------------------------------------------------------ a synthetic options chain


def _ts(day: dt.date, hour: int, minute: int) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hour, minute)), tz=IST)


def _row(timeframe, symbol, kind, expiry, strike, stamp, close, volume) -> dict:
    return {
        "underlying": "IDX",
        "timeframe": timeframe,
        "tradingsymbol": symbol,
        "instrument_type": kind,
        "expiry": expiry,
        "strike": strike,
        "timestamp": stamp,
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "best_buy_price": close - 0.05,
        "best_sell_price": close + 0.05,
        "volume": volume,
        "oi": 0.0,
        "bad_ohlc": False,
    }


def _symbol(expiry: dt.date, strike: float, kind: str) -> str:
    return f"IDX{expiry:%y%m%d}{int(strike)}{kind}"


class Chain:
    """Builds FrameSource rows: index minutes, a listing, option minutes."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def index(self, day: dt.date, closes: dict[tuple[int, int], float]) -> Chain:
        for (h, m), close in closes.items():
            self.rows.append(_row("1m", "IDX", "EQ", day, 0.0, _ts(day, h, m), close, 0))
        return self

    def list(self, day: dt.date, expiry: dt.date, strikes: list[float]) -> Chain:
        for strike in strikes:
            for kind in ("CE", "PE"):
                symbol = _symbol(expiry, strike, kind)
                stamp = _ts(day, 9, 15)
                self.rows.append(_row("1d", symbol, kind, expiry, strike, stamp, 1.0, 1))
        return self

    def option(
        self,
        day: dt.date,
        expiry: dt.date,
        strike: float,
        kind: str,
        prints: dict[tuple[int, int], tuple[float, float]],
    ) -> Chain:
        symbol = _symbol(expiry, strike, kind)
        for (h, m), (close, volume) in prints.items():
            self.rows.append(
                _row("1m", symbol, kind, expiry, strike, _ts(day, h, m), close, volume)
            )
        return self

    def source(self) -> FrameSource:
        return FrameSource(pd.DataFrame(self.rows))


DAY = dt.date(2024, 6, 5)  # a Wednesday
EXPIRY = dt.date(2024, 6, 6)
NEXT_EXPIRY = dt.date(2024, 6, 13)


def _years(day: dt.date, h: int, m: int, expiry: dt.date) -> float:
    """From the close of the candle starting h:m to 15:30 on expiry day."""
    start = dt.datetime.combine(day, dt.time(h, m)) + dt.timedelta(minutes=1)
    return (dt.datetime.combine(expiry, dt.time(15, 30)) - start).total_seconds() / (365 * 86400)


def test_the_day_iv_is_the_average_of_the_atm_call_and_put():
    spot, strike = 22010.0, 22000.0
    t = _years(DAY, 15, 29, EXPIRY)
    call = round(bs_price(spot, strike, t, 0.12, "CE"), 2)
    put = round(bs_price(spot, strike, t, 0.18, "PE"), 2)
    src = (
        Chain()
        .index(DAY, {(15, 28): 21990.0, (15, 29): spot})
        .list(DAY, EXPIRY, [21950, 22000, 22050])
        .option(DAY, EXPIRY, strike, "CE", {(15, 29): (call, 100)})
        .option(DAY, EXPIRY, strike, "PE", {(15, 29): (put, 100)})
        .source()
    )
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert row["strike"] == strike
    assert row["expiry"] == EXPIRY
    assert row["legs"] == iv.LEGS_BOTH
    assert row["call_iv"] == pytest.approx(0.12, abs=2e-3)
    assert row["put_iv"] == pytest.approx(0.18, abs=2e-3)
    assert row["iv"] == pytest.approx((row["call_iv"] + row["put_iv"]) / 2)
    assert row["iv"] == pytest.approx(0.15, abs=2e-3)


def test_one_side_without_a_trade_uses_the_other_and_says_so():
    spot, strike = 22010.0, 22000.0
    t = _years(DAY, 15, 29, EXPIRY)
    call = round(bs_price(spot, strike, t, 0.12, "CE"), 2)
    src = (
        Chain()
        .index(DAY, {(15, 29): spot})
        .list(DAY, EXPIRY, [22000])
        .option(DAY, EXPIRY, strike, "CE", {(15, 29): (call, 100)})
        .option(DAY, EXPIRY, strike, "PE", {(15, 29): (90.0, 0)})  # quoted, never traded
        .source()
    )
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert row["legs"] == iv.LEGS_CALL
    assert row["iv"] == pytest.approx(0.12, abs=2e-3)
    assert "put did not trade" in row["reason"]


def test_only_the_put_traded():
    spot, strike = 22010.0, 22000.0
    t = _years(DAY, 15, 29, EXPIRY)
    put = round(bs_price(spot, strike, t, 0.16, "PE"), 2)
    src = (
        Chain()
        .index(DAY, {(15, 29): spot})
        .list(DAY, EXPIRY, [22000])
        .option(DAY, EXPIRY, strike, "PE", {(15, 29): (put, 100)})
        .source()
    )
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert row["legs"] == iv.LEGS_PUT
    assert row["iv"] == pytest.approx(0.16, abs=2e-3)
    assert "call did not trade" in row["reason"]


def test_a_day_where_neither_side_traded_has_no_iv():
    src = (
        Chain()
        .index(DAY, {(15, 29): 22010.0})
        .list(DAY, EXPIRY, [22000])
        .option(DAY, EXPIRY, 22000, "CE", {(15, 29): (50.0, 0)})
        .source()
    )
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert math.isnan(row["iv"])
    assert row["legs"] == iv.LEGS_NONE
    assert "call did not trade" in row["reason"]
    assert "put did not trade" in row["reason"]


def test_a_forward_filled_minute_is_never_used_as_a_price():
    """The vendor repeats a price on minutes with no trade (volume 0). The IV
    must come from the last real trade, and the index level at that minute."""
    spot_then, strike = 22000.0, 22000.0
    t = _years(DAY, 15, 20, EXPIRY)
    call = round(bs_price(spot_then, strike, t, 0.14, "CE"), 2)
    put = round(bs_price(spot_then, strike, t, 0.14, "PE"), 2)
    index = {(15, m): 22000.0 + 0.3 * (m - 20) for m in range(20, 30)}
    filled = {(15, m): (call + 40.0, 0) for m in range(21, 30)}  # volume 0: not trades
    src = (
        Chain()
        .index(DAY, index)
        .list(DAY, EXPIRY, [21950, 22000, 22050])
        .option(DAY, EXPIRY, strike, "CE", {(15, 20): (call, 50), **filled})
        .option(DAY, EXPIRY, strike, "PE", {(15, 20): (put, 50)})
        .source()
    )
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert row["call_price"] == call
    assert row["call_time"] == _ts(DAY, 15, 21)  # the 15:20 candle's close
    assert row["call_spot"] == spot_then
    assert row["call_iv"] == pytest.approx(0.14, abs=2e-3)
    assert row["put_price"] == put
    assert row["strike"] == strike
    assert row["spot_close"] == pytest.approx(22002.7)


def test_a_trade_after_the_session_close_is_not_used():
    spot, strike = 22000.0, 22000.0
    t = _years(DAY, 15, 29, EXPIRY)
    call = round(bs_price(spot, strike, t, 0.13, "CE"), 2)
    put = round(bs_price(spot, strike, t, 0.13, "PE"), 2)
    src = (
        Chain()
        .index(DAY, {(15, 29): spot})
        .list(DAY, EXPIRY, [22000])
        .option(DAY, EXPIRY, strike, "CE", {(15, 29): (call, 10), (15, 45): (call * 3, 10)})
        .option(DAY, EXPIRY, strike, "PE", {(15, 29): (put, 10)})
        .source()
    )
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert row["call_price"] == call


def test_on_expiry_day_the_next_expiry_is_used():
    spot, strike = 22000.0, 22000.0
    t = _years(EXPIRY, 15, 29, NEXT_EXPIRY)
    call = round(bs_price(spot, strike, t, 0.13, "CE"), 2)
    put = round(bs_price(spot, strike, t, 0.13, "PE"), 2)
    src = (
        Chain()
        .index(EXPIRY, {(15, 29): spot})
        .list(EXPIRY, EXPIRY, [22000])
        .list(EXPIRY, NEXT_EXPIRY, [22000])
        # Expiring this afternoon: a 0-day price that would give a silly IV.
        .option(EXPIRY, EXPIRY, strike, "CE", {(15, 29): (0.5, 1000)})
        .option(EXPIRY, EXPIRY, strike, "PE", {(15, 29): (0.6, 1000)})
        .option(EXPIRY, NEXT_EXPIRY, strike, "CE", {(15, 29): (call, 100)})
        .option(EXPIRY, NEXT_EXPIRY, strike, "PE", {(15, 29): (put, 100)})
        .source()
    )
    row = daily_atm_iv("IDX", EXPIRY, EXPIRY, src).iloc[0]
    assert row["expiry"] == NEXT_EXPIRY
    assert row["iv"] == pytest.approx(0.13, abs=2e-3)


def test_the_day_before_expiry_uses_the_nearest_expiry():
    spot = 22000.0
    src = (
        Chain()
        .index(DAY, {(15, 29): spot})
        .list(DAY, EXPIRY, [22000])
        .list(DAY, NEXT_EXPIRY, [22000])
        .source()
    )
    assert daily_atm_iv("IDX", DAY, DAY, src).iloc[0]["expiry"] == EXPIRY


def test_an_expiry_moved_earlier_by_a_holiday_counts_as_expiring_that_day():
    """NIFTY's 30 Mar 2023 expiry is listed at its nominal date, but 30 Mar
    was a holiday and it really expired on 29 Mar. On 29 Mar it is a 0-day
    option and must be skipped like any other expiry-day contract."""
    wed, thu, fri = dt.date(2023, 3, 29), dt.date(2023, 3, 30), dt.date(2023, 3, 31)
    nxt = dt.date(2023, 4, 6)
    spot, strike = 17100.0, 17100.0
    t = _years(wed, 15, 29, nxt)
    call = round(bs_price(spot, strike, t, 0.13, "CE"), 2)
    put = round(bs_price(spot, strike, t, 0.13, "PE"), 2)
    src = (
        Chain()
        .index(wed, {(15, 29): spot})
        .index(fri, {(15, 29): spot})  # no session on the 30th: a holiday
        .list(wed, thu, [17100])
        .list(wed, nxt, [17100])
        .option(wed, thu, strike, "CE", {(15, 29): (100.0, 1000)})
        .option(wed, thu, strike, "PE", {(15, 29): (0.05, 1000)})
        .option(wed, nxt, strike, "CE", {(15, 29): (call, 100)})
        .option(wed, nxt, strike, "PE", {(15, 29): (put, 100)})
        .source()
    )
    row = daily_atm_iv("IDX", wed, wed, src).iloc[0]
    assert row["expiry"] == nxt
    assert row["iv"] == pytest.approx(0.13, abs=2e-3)


def test_time_to_expiry_runs_to_the_real_last_day_of_a_moved_expiry():
    tue, wed, fri = dt.date(2023, 3, 28), dt.date(2023, 3, 29), dt.date(2023, 3, 31)
    listed = dt.date(2023, 3, 30)  # a holiday; really expires on the 29th
    spot, strike = 17100.0, 17100.0
    t = _years(tue, 15, 29, wed)  # one day, not two
    call = round(bs_price(spot, strike, t, 0.13, "CE"), 2)
    put = round(bs_price(spot, strike, t, 0.13, "PE"), 2)
    src = (
        Chain()
        .index(tue, {(15, 29): spot})
        .index(wed, {(15, 29): spot})
        .index(fri, {(15, 29): spot})
        .list(tue, listed, [17100])
        .option(tue, listed, strike, "CE", {(15, 29): (call, 100)})
        .option(tue, listed, strike, "PE", {(15, 29): (put, 100)})
        .source()
    )
    row = daily_atm_iv("IDX", tue, tue, src).iloc[0]
    assert row["expiry"] == listed
    assert row["expires"] == wed
    assert row["iv"] == pytest.approx(0.13, abs=2e-3)


def test_a_tie_between_two_strikes_goes_to_the_lower():
    src = (
        Chain()
        .index(DAY, {(15, 29): 22025.0})
        .list(DAY, EXPIRY, [21950, 22000, 22050, 22100])
        .source()
    )
    assert daily_atm_iv("IDX", DAY, DAY, src).iloc[0]["strike"] == 22000


# ------------------------------------------------------------- the feature


def test_the_feature_gives_every_candle_of_a_day_that_days_percentile():
    sessions = _sessions(6)
    daily = pd.Series([0.10, 0.30, 0.20, 0.40, 0.25, 0.99], index=sessions)
    expected = iv_percentile(daily, 4)
    candles = []
    for day in sessions[3:]:
        for minute in (15, 30, 45):
            candles.append(day.tz_localize(None) + pd.Timedelta(hours=9, minutes=minute))
    candles.append(pd.Timestamp("2024-01-10 09:15"))  # a day with no IV row
    bars = pd.DataFrame(
        {"close": 1.0}, index=pd.DatetimeIndex(candles).tz_localize(IST).sort_values()
    )
    out = iv_feature("IDX", bars, 4, daily_iv=daily)
    assert out.name == iv_percentile_column(4)
    assert out.index.equals(bars.index)
    for stamp, value in out.items():
        day = stamp.normalize()
        if day in expected.index:
            want = expected.loc[day]
            assert (math.isnan(value) and math.isnan(want)) or value == want
        else:
            assert math.isnan(value)
    assert out.loc["2024-01-05"].tolist() == [75.0] * 3
    assert out.loc["2024-01-08"].tolist() == [25.0] * 3


def test_the_feature_accepts_the_daily_frame_as_well_as_the_series():
    sessions = _sessions(6)
    frame = pd.DataFrame({"iv": [0.10, 0.30, 0.20, 0.40, 0.25, 0.99]}, index=sessions)
    bars = pd.DataFrame({"close": 1.0}, index=sessions)
    assert iv_feature("IDX", bars, 4, daily_iv=frame).iloc[5] == 25.0


# ------------------------------------------------------------------ the cache


def test_the_cache_is_read_for_the_real_source_and_ignored_when_out_of_date(tmp_path, monkeypatch):
    monkeypatch.setattr(option_data, "FNO_ROOT", tmp_path)
    sessions = _sessions(3)
    frame = pd.DataFrame({c: [None] * 3 for c in iv.IV_COLUMNS}, index=sessions).assign(
        iv=[0.1, 0.2, 0.3], legs=iv.LEGS_BOTH, expiry=pd.Timestamp("2024-01-04")
    )
    path = iv.write_cache("IDX", frame)
    assert path == tmp_path / "iv" / "underlying=IDX.parquet"

    got = daily_atm_iv("IDX", sessions[1].date(), sessions[2].date(), ParquetSource())
    assert got["iv"].tolist() == [0.2, 0.3]

    # Written by an older method: not trusted. Nothing else is on disk here,
    # so working it out afresh finds no sessions at all.
    monkeypatch.setattr(iv, "METHOD_VERSION", iv.METHOD_VERSION + 1)
    assert daily_atm_iv("IDX", sessions[0].date(), sessions[2].date(), ParquetSource()).empty


def test_an_in_memory_source_never_reads_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(option_data, "FNO_ROOT", tmp_path)
    stamp = pd.DatetimeIndex([pd.Timestamp(DAY, tz=IST)], name="date")
    stale = pd.DataFrame({c: [None] for c in iv.IV_COLUMNS}, index=stamp).assign(
        iv=[0.99], legs=iv.LEGS_BOTH, expiry=pd.Timestamp(EXPIRY)
    )
    iv.write_cache("IDX", stale)
    src = Chain().index(DAY, {(15, 29): 22000.0}).list(DAY, EXPIRY, [22000]).source()
    row = daily_atm_iv("IDX", DAY, DAY, src).iloc[0]
    assert math.isnan(row["iv"])


# --------------------------------------------- known answer from the raw data

# NIFTY, Friday 7 June 2024, copied from data/fno/1m (prices already in
# rupees, timestamps IST). The index closed the 15:29 candle at 23,270.05;
# listed strikes are every 50 points, so the ATM strike is 23,250 and the
# nearest expiry after the day is Thursday 13 June 2024.
#
#   NIFTY2461323250CE 15:29  close 242.15  volume 95,500
#   NIFTY2461323250PE 15:29  close 175.00  volume 83,625
#
# Worked by hand: from 15:30 on 7 June to 15:30 on 13 June is exactly 6 days,
# T = 6/365. Black-Scholes at r = 6.5% gives call IV 18.416% and put IV
# 16.536%; their average is 17.476%. (Checked against the raw Parquet rows
# read directly, with a separate solver built on statistics.NormalDist.)
KNOWN_DAY = dt.date(2024, 6, 7)
KNOWN_EXPIRY = dt.date(2024, 6, 13)
KNOWN_CALL_IV = 0.18416
KNOWN_PUT_IV = 0.16536


def test_known_answer_nifty_7_june_2024():
    src = (
        Chain()
        .index(KNOWN_DAY, {(15, 28): 23271.55, (15, 29): 23270.05})
        .list(KNOWN_DAY, KNOWN_EXPIRY, [23200, 23250, 23300])
        .option(KNOWN_DAY, KNOWN_EXPIRY, 23250, "CE", {(15, 29): (242.15, 95500)})
        .option(KNOWN_DAY, KNOWN_EXPIRY, 23250, "PE", {(15, 29): (175.00, 83625)})
        .source()
    )
    row = daily_atm_iv("IDX", KNOWN_DAY, KNOWN_DAY, src).iloc[0]
    assert row["strike"] == 23250
    assert row["call_iv"] == pytest.approx(KNOWN_CALL_IV, abs=5e-5)
    assert row["put_iv"] == pytest.approx(KNOWN_PUT_IV, abs=5e-5)
    assert row["iv"] == pytest.approx((KNOWN_CALL_IV + KNOWN_PUT_IV) / 2, abs=5e-5)


ONE_M_PRESENT = (option_data.FNO_ROOT / "1m").exists()


@pytest.mark.skipif(not ONE_M_PRESENT, reason="1m data not present in this worktree")
def test_known_answer_nifty_7_june_2024_from_the_real_data():
    row = daily_atm_iv("NIFTY", KNOWN_DAY, KNOWN_DAY, ParquetSource(), use_cache=False).iloc[0]
    assert row["expiry"] == KNOWN_EXPIRY
    assert row["strike"] == 23250
    assert row["spot_close"] == pytest.approx(23270.05)
    assert row["call_price"] == pytest.approx(242.15)
    assert row["put_price"] == pytest.approx(175.00)
    assert row["call_iv"] == pytest.approx(KNOWN_CALL_IV, abs=5e-5)
    assert row["put_iv"] == pytest.approx(KNOWN_PUT_IV, abs=5e-5)
