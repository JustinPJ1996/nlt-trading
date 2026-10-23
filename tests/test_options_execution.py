"""The one-pass candle builder (`execution._aggregate`, 2026-10-16) gives exactly
what the bucket-by-bucket loop it replaced gave -- on made-up minutes with
every awkward case, and on real contracts. The old loop is kept here, verbatim,
as the reference."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest

from nlt.options.execution import BAR_COLUMNS, _bucket_start, resample_daily, resample_traded

IST = "Asia/Kolkata"


def _aggregate_bucket(bucket_start: pd.Timestamp, group: pd.DataFrame) -> dict[str, object]:
    """One bucket's row: traded-only OHLCV, last-row-in-bucket oi/bid/ask.

    Shared by `resample_traded` (minute buckets) and `resample_daily` (whole
    calendar days) -- the aggregation rule is identical, only the grouping
    key differs.
    """
    traded_rows = group[group["traded"]].sort_index()
    last_any = group.sort_index().iloc[-1]
    if traded_rows.empty:
        return {
            "timestamp": bucket_start,
            "open": float("nan"),
            "high": float("nan"),
            "low": float("nan"),
            "close": float("nan"),
            "volume": 0.0,
            "oi": last_any.get("oi", float("nan")),
            "bid": last_any.get("bid", float("nan")),
            "ask": last_any.get("ask", float("nan")),
            "traded": False,
        }
    return {
        "timestamp": bucket_start,
        "open": traded_rows["open"].iloc[0],
        "high": traded_rows["high"].max(),
        "low": traded_rows["low"].min(),
        "close": traded_rows["close"].iloc[-1],
        "volume": traded_rows["volume"].sum(),
        "oi": last_any.get("oi", float("nan")),
        "bid": last_any.get("bid", float("nan")),
        "ask": last_any.get("ask", float("nan")),
        "traded": True,
    }


def _reference(df: pd.DataFrame, bucket_minutes: int | None) -> pd.DataFrame:
    if df.empty:
        return df.reindex(columns=BAR_COLUMNS)
    work = df.copy()
    if bucket_minutes is None:
        work["_bucket"] = pd.DatetimeIndex(work.index).normalize()  # ty: ignore[unresolved-attribute]  # stub lacks DatetimeIndex.normalize
    else:
        work["_bucket"] = _bucket_start(cast(pd.DatetimeIndex, work.index), bucket_minutes)
    rows = [
        _aggregate_bucket(cast(pd.Timestamp, b), g) for b, g in work.groupby("_bucket", sort=True)
    ]
    out = pd.DataFrame(rows).set_index("timestamp").sort_index()
    out.index.name = "timestamp"
    return out[BAR_COLUMNS]


def _same(new: pd.DataFrame, old: pd.DataFrame) -> None:
    assert list(new.index) == list(old.index)
    for column in BAR_COLUMNS:
        a, b = new[column].to_numpy(), old[column].to_numpy()
        if column == "traded":
            assert [bool(x) for x in a] == [bool(x) for x in b]
        else:
            np.testing.assert_array_equal(a.astype(float), b.astype(float), err_msg=column)


def _minutes(seed: int, days: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    stamps = []
    for d in pd.bdate_range("2025-01-01", periods=days):
        start = pd.Timestamp(d.date()).tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        # Gaps: not every minute has a row.
        stamps.extend(
            start + pd.Timedelta(minutes=int(k))
            for k in sorted(rng.choice(375, 200, replace=False))
        )
    n = len(stamps)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.5, n),
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": rng.integers(0, 3, n).astype(float),
            "oi": np.where(rng.random(n) < 0.2, np.nan, rng.integers(1, 1000, n)).astype(float),
            "bid": close - 0.05,
            "ask": close + 0.05,
        },
        index=pd.DatetimeIndex(stamps),
    )
    # Untraded rows (volume 0), some whole buckets of them; a NaN open on a trade.
    df["traded"] = df["volume"] > 0
    df.iloc[5, df.columns.get_loc("open")] = np.nan
    return df


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("bucket", [3, 5, 15, 30, 60])
def test_minute_buckets_match_the_old_loop(seed, bucket) -> None:
    df = _minutes(seed)
    _same(resample_traded(df, bucket), _reference(df, bucket))


@pytest.mark.parametrize("seed", range(4))
def test_daily_buckets_match_the_old_loop(seed) -> None:
    df = _minutes(seed, days=6)
    _same(resample_daily(df), _reference(df, None))


def test_a_missing_quote_column_is_nan_in_both() -> None:
    df = _minutes(9).drop(columns=["bid"])
    _same(resample_traded(df, 15), _reference(df, 15))


def test_a_bucket_with_no_trade_has_no_price_and_no_volume() -> None:
    df = _minutes(1)
    df["volume"] = 0.0
    df["traded"] = False
    out = resample_traded(df, 15)
    assert out["open"].isna().all() and (out["volume"] == 0).all() and not out["traded"].any()


_REAL = Path(__file__).resolve().parents[1] / "data/fno/1m/underlying=NIFTY"


@pytest.mark.skipif(not _REAL.exists(), reason="the F&O 1-minute data is not on this machine")
@pytest.mark.parametrize(
    "symbol", ["NIFTY2510223800CE", "NIFTY2510224000CE", "NIFTY2510223800PE", "NIFTY25JAN24000CE"]
)
def test_real_contracts_match_the_old_loop(symbol) -> None:
    from nlt.options.data import ParquetSource

    raw = ParquetSource().contract_bars(
        "NIFTY", symbol, "1m", dt.date(2024, 12, 20), dt.date(2025, 1, 31)
    )
    assert len(raw) > 100
    from nlt.options.data import traded_mask

    raw = raw.assign(traded=traded_mask(raw)) if "traded" not in raw.columns else raw
    for bucket in (5, 15, 60):
        _same(resample_traded(raw, bucket), _reference(raw, bucket))
    _same(resample_daily(raw), _reference(raw, None))
