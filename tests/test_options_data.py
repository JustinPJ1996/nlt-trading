"""Tests for `nlt/options/data.py`: `FrameSource` (synthetic, no disk) and the
real Parquet dataset (skipped if `data/fno` is absent).

Synthetic rows are built to look exactly like one row of the Parquet dataset
`nlt/data/fno_import.py` writes, plus a `timeframe` column (since one real
Parquet directory per timeframe becomes one `FrameSource` DataFrame here).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from nlt.options.data import (
    FNO_ROOT,
    FrameSource,
    contract_bars,
    coverage,
    listing,
    traded_mask,
)
from nlt.options.execution import resample_daily, resample_traded

IST = "Asia/Kolkata"


def _ts(day: dt.date, hour: int, minute: int) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(hour, minute)), tz=IST)


def _row(
    *,
    underlying: str,
    timeframe: str,
    tradingsymbol: str,
    instrument_type: str,
    expiry: dt.date,
    strike: float,
    timestamp: pd.Timestamp,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 0.0,
    bbp: float | None = None,
    bsp: float | None = None,
    oi: float = 0.0,
    bad_ohlc: bool = False,
) -> dict:
    return {
        "underlying": underlying,
        "timeframe": timeframe,
        "tradingsymbol": tradingsymbol,
        "instrument_type": instrument_type,
        "expiry": expiry,
        "strike": strike,
        "timestamp": timestamp,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "best_buy_price": bbp if bbp is not None else close - 0.05,
        "best_sell_price": bsp if bsp is not None else close + 0.05,
        "volume": volume,
        "oi": oi,
        "bad_ohlc": bad_ohlc,
    }


DAY1 = dt.date(2024, 1, 4)  # a Thursday
DAY2 = dt.date(2024, 1, 5)
EXPIRY = dt.date(2024, 1, 4)


def _synthetic_rows() -> list[dict]:
    rows = []

    # EQ (index) rows, 1d and 15m, two days.
    for day, close in [(DAY1, 21500.0), (DAY2, 21550.0)]:
        rows.append(
            _row(
                underlying="IDX",
                timeframe="1d",
                tradingsymbol="IDX",
                instrument_type="EQ",
                expiry=EXPIRY,
                strike=0.0,
                timestamp=_ts(day, 9, 15),
                open_=close - 10,
                high=close + 20,
                low=close - 20,
                close=close,
                volume=0,
            )
        )
        for i, minute in enumerate((15, 30, 45)):
            rows.append(
                _row(
                    underlying="IDX",
                    timeframe="15m",
                    tradingsymbol="IDX",
                    instrument_type="EQ",
                    expiry=EXPIRY,
                    strike=0.0,
                    timestamp=_ts(day, 9, minute),
                    open_=close + i,
                    high=close + i + 5,
                    low=close + i - 5,
                    close=close + i,
                    volume=0,
                )
            )

    # Listing: 1d CE/PE rows for DAY1 and DAY2 at several strikes, one expiry.
    # A strike listed only "tomorrow" (DAY2) must never appear in DAY1's listing.
    for strike in (21400, 21450, 21500, 21550, 21600):
        rows.append(
            _row(
                underlying="IDX",
                timeframe="1d",
                tradingsymbol=f"IDX{int(strike)}CE",
                instrument_type="CE",
                expiry=EXPIRY,
                strike=float(strike),
                timestamp=_ts(DAY1, 9, 15),
                open_=50,
                high=55,
                low=45,
                close=50,
                volume=10,
            )
        )
        rows.append(
            _row(
                underlying="IDX",
                timeframe="1d",
                tradingsymbol=f"IDX{int(strike)}PE",
                instrument_type="PE",
                expiry=EXPIRY,
                strike=float(strike),
                timestamp=_ts(DAY1, 9, 15),
                open_=50,
                high=55,
                low=45,
                close=50,
                volume=10,
            )
        )
    # A strike that only exists the next day.
    rows.append(
        _row(
            underlying="IDX",
            timeframe="1d",
            tradingsymbol="IDX21650CE",
            instrument_type="CE",
            expiry=EXPIRY,
            strike=21650.0,
            timestamp=_ts(DAY2, 9, 15),
            open_=20,
            high=22,
            low=18,
            close=20,
            volume=5,
        )
    )

    # 1m bars with a forward-filled (volume 0) row carrying a fake high, to
    # prove the resampler never lets it leak into an aggregated bucket.
    rows.append(
        _row(
            underlying="IDX",
            timeframe="1m",
            tradingsymbol="IDX21500CE",
            instrument_type="CE",
            expiry=EXPIRY,
            strike=21500.0,
            timestamp=_ts(DAY1, 9, 15),
            open_=100,
            high=102,
            low=98,
            close=100,
            volume=10,
        )
    )
    rows.append(
        _row(
            underlying="IDX",
            timeframe="1m",
            tradingsymbol="IDX21500CE",
            instrument_type="CE",
            expiry=EXPIRY,
            strike=21500.0,
            timestamp=_ts(DAY1, 9, 16),
            open_=100,
            high=9999,  # fake forward-filled high -- must never win
            low=100,
            close=100,
            volume=0,
        )
    )
    rows.append(
        _row(
            underlying="IDX",
            timeframe="1m",
            tradingsymbol="IDX21500CE",
            instrument_type="CE",
            expiry=EXPIRY,
            strike=21500.0,
            timestamp=_ts(DAY1, 9, 17),
            open_=101,
            high=103,
            low=99,
            close=102,
            volume=15,
        )
    )
    # A whole untraded bucket: minutes 09:18-09:19, volume 0 throughout.
    rows.append(
        _row(
            underlying="IDX",
            timeframe="1m",
            tradingsymbol="IDX21500CE",
            instrument_type="CE",
            expiry=EXPIRY,
            strike=21500.0,
            timestamp=_ts(DAY1, 9, 18),
            open_=102,
            high=102,
            low=102,
            close=102,
            volume=0,
        )
    )
    # A second day's 1m row, so coverage() (measured from 1m) spans two days.
    rows.append(
        _row(
            underlying="IDX",
            timeframe="1m",
            tradingsymbol="IDX21500CE",
            instrument_type="CE",
            expiry=EXPIRY,
            strike=21500.0,
            timestamp=_ts(DAY2, 9, 15),
            open_=103,
            high=105,
            low=102,
            close=104,
            volume=20,
        )
    )

    return rows


@pytest.fixture
def source() -> FrameSource:
    return FrameSource(pd.DataFrame(_synthetic_rows()))


def test_frame_source_requires_raw_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        FrameSource(pd.DataFrame({"underlying": ["IDX"]}))


def test_underlying_bars_shape_and_index(source: FrameSource):
    bars = source.underlying_bars("IDX", "15m", DAY1, DAY2)
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(bars.index, pd.DatetimeIndex)
    assert str(bars.index.tz) == IST
    assert bars.index.is_monotonic_increasing
    assert (bars["volume"] == 0).all()


def test_underlying_bars_daily_normalised_to_midnight(source: FrameSource):
    bars = source.underlying_bars("IDX", "1d", DAY1, DAY2)
    assert all(ts.time() == dt.time(0, 0) for ts in bars.index)


def test_listing_only_shows_strikes_listed_that_day(source: FrameSource):
    today = source.listing("IDX", DAY1)
    assert set(today["strike"]) == {21400.0, 21450.0, 21500.0, 21550.0, 21600.0}
    # The strike that only appears in tomorrow's listing must not leak back.
    assert 21650.0 not in set(today["strike"])

    tomorrow = source.listing("IDX", DAY2)
    assert 21650.0 in set(tomorrow["strike"])


def test_listing_empty_for_a_day_with_no_rows(source: FrameSource):
    out = source.listing("IDX", dt.date(2099, 1, 1))
    assert out.empty


def test_contract_bars_native_1m(source: FrameSource):
    bars = source.contract_bars("IDX", "IDX21500CE", "1m", DAY1, DAY1)
    assert list(bars.columns) == [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "oi",
        "bid",
        "ask",
        "traded",
    ]
    assert len(bars) == 4  # 09:15 traded, 09:16 untraded, 09:17 traded, 09:18 untraded
    assert bars["traded"].tolist() == [True, False, True, False]


def test_contract_bars_empty_for_unknown_symbol(source: FrameSource):
    bars = source.contract_bars("IDX", "NOPE", "1m", DAY1, DAY1)
    assert bars.empty


def test_contract_bars_rejects_non_1m_timeframe(source: FrameSource):
    """Never read the native 15m/1d option quotes for prices -- see the module
    docstring: they are forward-fill-contaminated. Resampling from 1m is the
    caller's job (`nlt.options.execution`)."""
    with pytest.raises(ValueError, match="1m"):
        source.contract_bars("IDX", "IDX21500CE", "15m", DAY1, DAY1)


def test_module_level_contract_bars_also_rejects_non_1m_timeframe():
    """Same rule on `ParquetSource`'s underlying function -- checked before
    any disk access, so this needs no real data on disk."""
    with pytest.raises(ValueError, match="1m"):
        contract_bars("IDX", "IDX21500CE", "15m", DAY1, DAY1)


def test_traded_mask_excludes_bad_ohlc():
    df = pd.DataFrame(
        {
            "volume": [10, 10, 0],
            "bad_ohlc": [False, True, False],
        }
    )
    assert traded_mask(df).tolist() == [True, False, False]


# ----------------------------------------------------------- resample_traded


def test_resample_forward_filled_row_never_changes_the_bar(source: FrameSource):
    """The 09:16 row has volume 0 and a fake high of 9999 -- a 3-minute bucket
    covering 09:15-09:18 must not show that high anywhere."""
    raw = source.contract_bars("IDX", "IDX21500CE", "1m", DAY1, DAY1)
    out = resample_traded(raw, 3)
    bucket = out.loc[_ts(DAY1, 9, 15)]
    assert bucket["high"] < 9999
    assert bucket["high"] == 103  # from the 09:17 traded row
    assert bucket["open"] == 100  # first traded row's open (09:15)
    assert bucket["close"] == 102  # last traded row's close (09:17)
    assert bucket["volume"] == 25  # 10 + 15, the untraded row contributes 0


def test_resample_untraded_bucket_is_marked_untraded(source: FrameSource):
    raw = source.contract_bars("IDX", "IDX21500CE", "1m", DAY1, DAY1)
    out = resample_traded(raw, 1)
    bucket = out.loc[_ts(DAY1, 9, 18)]
    assert bool(bucket["traded"]) is False
    assert pd.isna(bucket["open"])
    assert pd.isna(bucket["close"])


def test_resample_empty_frame_returns_empty():
    out = resample_traded(pd.DataFrame(), 5)
    assert out.empty


# ------------------------------------------------------------------ resample_daily


def test_resample_daily_one_bar_per_calendar_day(source: FrameSource):
    """DAY1 has four 1m rows (09:15 traded, 09:16 untraded fake-high, 09:17
    traded, 09:18 untraded); DAY2 has one traded row. `resample_daily` must
    collapse each day to exactly one bar, open/close from traded rows only."""
    raw = source.contract_bars("IDX", "IDX21500CE", "1m", DAY1, DAY2)
    out = resample_daily(raw)
    assert len(out) == 2

    day1_bar = out.loc[pd.Timestamp(DAY1, tz=IST)]
    assert day1_bar["open"] == 100  # 09:15 traded open
    assert day1_bar["high"] == 103  # 09:17 traded high -- the 9999 fake never wins
    assert day1_bar["close"] == 102  # 09:17 traded close (the last traded row)
    assert day1_bar["volume"] == 25  # 10 + 15, untraded rows contribute 0
    assert day1_bar["traded"]

    day2_bar = out.loc[pd.Timestamp(DAY2, tz=IST)]
    assert day2_bar["open"] == 103
    assert day2_bar["close"] == 104


def test_resample_daily_indexed_at_midnight_ist(source: FrameSource):
    raw = source.contract_bars("IDX", "IDX21500CE", "1m", DAY1, DAY1)
    out = resample_daily(raw)
    assert all(ts.time() == dt.time(0, 0) for ts in out.index)
    assert str(out.index.tz) == IST


def test_resample_daily_untraded_day_has_no_price(source: FrameSource):
    raw = source.contract_bars("IDX", "IDX21500CE", "1m", DAY1, DAY1).copy()
    raw["traded"] = False
    raw["volume"] = 0
    out = resample_daily(raw)
    bar = out.loc[pd.Timestamp(DAY1, tz=IST)]
    assert bool(bar["traded"]) is False
    assert pd.isna(bar["open"])


def test_resample_daily_empty_frame_returns_empty():
    out = resample_daily(pd.DataFrame())
    assert out.empty


# ------------------------------------------------------------------ coverage


def test_coverage_from_frame_source(source: FrameSource):
    start, end = source.coverage("IDX", "1d")
    assert start == DAY1
    assert end == DAY2


def test_coverage_is_measured_from_1m_regardless_of_timeframe_argument(source: FrameSource):
    """Every price timeframe is built from 1m (see module docstring), so
    coverage must report the same span no matter which timeframe is asked
    about -- it is never silently reading a different (and possibly
    forward-fill-contaminated) native file."""
    for tf in ("1m", "15m", "1d", "30m", "1h"):
        assert source.coverage("IDX", tf) == (DAY1, DAY2)


def test_coverage_none_when_underlying_absent(source: FrameSource):
    assert source.coverage("NOPE", "1d") is None


# -------------------------------------------------------- real-data tests


FNO_PRESENT = FNO_ROOT.exists() and any(FNO_ROOT.iterdir())
ONE_M_PRESENT = Path("data/fno/1m/_coverage.json").exists() or (
    Path("data/fno/1m").exists() and any(Path("data/fno/1m").iterdir())
)


@pytest.mark.skipif(not ONE_M_PRESENT, reason="1m data not present in this worktree")
def test_real_sensex_coverage_starts_july_2024_measured_from_1m():
    # Measured directly: SENSEX's options have real traded 1-minute prices
    # from 2024-07-30 onwards. `coverage()` is measured from 1m for every
    # timeframe, so asking about "15m" or "1d" gives the same answer.
    assert coverage("SENSEX", "15m") == (dt.date(2024, 7, 30), dt.date(2025, 12, 31))
    assert coverage("SENSEX", "1d") == coverage("SENSEX", "15m")


@pytest.mark.skipif(not FNO_PRESENT, reason="data/fno not present in this worktree")
def test_real_sensex_listing_exists_earlier_than_1m_price_coverage():
    """The daily file's contract-*listing* rows for SENSEX go back to
    2023-07-20 -- a full year before any real traded 1-minute price exists
    (2024-07-30, see the coverage test above). `listing()` only answers "did
    this contract exist", so it legitimately shows contracts further back
    than `coverage()` (which answers "is there a real price here") does --
    these are different questions, and a contract can be listed with nothing
    to trade in our 1-minute file.
    """
    rows = listing("SENSEX", dt.date(2023, 7, 20))
    assert not rows.empty
    assert listing("SENSEX", dt.date(2023, 7, 19)).empty


@pytest.mark.skipif(not ONE_M_PRESENT, reason="1m data not present in this worktree")
def test_real_banknifty_15m_bar_opens_at_the_first_real_trade_not_the_stale_forward_fill():
    """Pinned regression for the forward-fill bug found in the native 15m/1d
    option quotes files (see `nlt/options/data.py`'s module docstring).

    Hand-checked from the raw 1-minute Parquet rows for
    BANKNIFTY2460540000CE on 2024-06-04: every minute from 11:00 to 11:12 is
    a forward-filled repeat of a stale 10,899.95 print (volume 0); the first
    real trade in the quarter-hour is at 11:13, at 7,735.60 (volume 60). A
    15-minute bar built the old (native-file) way would open at 10,899.95 --
    a price nobody traded at, in a bar whose contract moved over 3,000 points
    that quarter-hour. Rebuilding from 1m traded rows only must open at
    7,735.60.
    """
    from nlt.options.data import ParquetSource
    from nlt.options.execution import resample_traded

    source = ParquetSource()
    day = dt.date(2024, 6, 4)
    raw = source.contract_bars("BANKNIFTY", "BANKNIFTY2460540000CE", "1m", day, day)
    assert not raw.empty

    bars_15m = resample_traded(raw, 15)
    bucket = bars_15m.loc[bars_15m.index.time == dt.time(11, 0)]
    assert len(bucket) == 1
    assert bucket["open"].iloc[0] == pytest.approx(7735.60, abs=0.01)
    assert bucket["traded"].iloc[0]


@pytest.mark.skipif(not FNO_PRESENT, reason="data/fno not present in this worktree")
def test_real_banknifty_listing_has_the_last_weekly_expiry():
    # 13 Nov 2024 is BANKNIFTY's last weekly expiry (verified by scanning
    # data/fno/1d/underlying=BANKNIFTY: every listed expiry from 27 Nov 2024
    # onwards is a calendar month's last expiry only).
    day = dt.date(2024, 11, 13)
    rows = listing("BANKNIFTY", day)
    assert dt.date(2024, 11, 13) in set(rows["expiry"])
    # And the very next listed expiry after it is the month-end, ~2 weeks out.
    upcoming = sorted(e for e in set(rows["expiry"]) if e >= day)
    assert upcoming[1] == dt.date(2024, 11, 27)


@pytest.mark.skipif(not ONE_M_PRESENT, reason="1m data not present in this worktree")
def test_real_1m_coverage_is_measured_not_assumed():
    start, end = coverage("NIFTY", "1m")
    assert start is not None and end is not None
