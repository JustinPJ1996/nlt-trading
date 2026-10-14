"""Reading the historical F&O Parquet dataset for Phase 2's options engine.

This is the one place that touches `data/fno/` directly. Everything else in
`nlt/options/` works through `ParquetSource` (the real one) or `FrameSource`
(a tiny in-memory stand-in for tests), both implementing the same four
functions, so the selection and execution logic never has to know whether it
is reading a 64-million-row Parquet partition or a five-row synthetic chain.

## Every option price comes from `1m`, never the native `15m`/`1d` files

Measured directly on 2026-10-13 and worth stating loudly: the native `15m`
and `1d` quotes files were themselves built by aggregating **all** 1-minute
rows, including the forward-filled (`volume == 0`) ones. A 1-minute print
repeats the last traded price until the next real trade, so an aggregate over
all of them can report an `open` (or a `high`/`low`) that nobody ever traded
at.

Concrete, checked example: `BANKNIFTY2460540000CE`, 2024-06-04, the native
15-minute bar starting 11:00 opens at 10,899.95. The first *real* trade inside
that quarter-hour was at 11:13, at 7,735.60 -- the native bar's open is a
stale forward-filled price from over an hour earlier, off by 3,164.35. Checked
systematically: native 15-minute option `close`/`volume` match a from-1m
rebuild 100% of the time (both are last-print and summed-print concepts that
survive the forward-fill unchanged), but `open` only matches the first
*traded* minute 65-70% of the time and `high`/`low` only ~88% (some native
`low` values are even 0.00). The daily (`1d`) file's `open`/`high`/`low` do
match a from-1m-traded rebuild, but its `close` is a different, settlement-style
print that matches the last real trade only ~30% of the time.

**The fix: every option contract bar, at every timeframe (1m, 3m, 5m, 15m,
30m, 1h, 1d), is built in this module from `data/fno/1m` traded rows only**
(`traded_mask`: `volume > 0` and not `bad_ohlc`). `contract_bars` therefore
only reads the `1m` partition; `nlt.options.execution` resamples every other
timeframe from it (see that module for the traded-only aggregation rule).
`coverage()` is measured from `1m` for the same reason -- it answers "is
there real option *price* data here", not just "is a contract listed".

**The index's own candles are a different story and are fine as-is.** `EQ`
rows (the underlying's own level) are always `volume 0` -- "traded" has no
meaning for them, so there is no forward-fill contamination to guard against,
and the native per-timeframe EQ files were checked to agree with a full (not
traded-only) resample from `1m` to the rupee on a sample day. `underlying_bars`
keeps reading the native `15m`/`1d` EQ rows directly: cheaper, and verified
equivalent.

**Listing is unaffected.** `listing()` only answers "which contracts existed
that day", not "what did they trade at", so it keeps reading the daily (`1d`)
file, whose contract-existence rows are untouched by this issue.

## Partition layout, and why a recursive-glob-shaped reader

`data/fno/{tf}/underlying=U/year=Y/*.parquet` for `1d`/`15m`; `1m` adds an
extra `month=M` level. PyArrow's hive-partition discovery walks a directory
tree at any depth and adds whatever partition columns it finds, so a single
`pyarrow.dataset.dataset(root, partitioning="hive")` call already tolerates
both layouts without this module caring how deep a given timeframe happens to
nest -- that is what "read with a recursive glob so layout depth doesn't
matter" buys us here, via the dataset API rather than a literal `Path.rglob`.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow.dataset as ds

from nlt.data.source import BAR_COLUMNS, IST, normalise

# Overridable so tests (and a future bigger disk) can point at a different
# export without editing code -- see the module docstring in fno_import.py for
# where the real data comes from.
FNO_ROOT = (
    Path(os.environ.get("NLT_FNO_ROOT", ""))
    if os.environ.get("NLT_FNO_ROOT")
    else (Path(__file__).resolve().parents[2] / "data" / "fno")
)

OPTION_TYPES = ("CE", "PE")

# The only timeframe `contract_bars` reads natively -- see the module
# docstring for why the native 15m/1d *option* quotes files cannot be trusted
# for prices. `nlt.options.execution` resamples every other timeframe from it.
CONTRACT_NATIVE_TIMEFRAME = "1m"

# 09:15 IST, every session -- used to align a resampled bucket's first bar.
SESSION_START = dt.time(9, 15)


def _timeframe_root(timeframe: str) -> Path:
    return FNO_ROOT / timeframe


def _dataset(timeframe: str) -> ds.Dataset | None:
    root = _timeframe_root(timeframe)
    if not root.exists():
        return None
    return ds.dataset(root, format="parquet", partitioning="hive")


def traded_mask(df: pd.DataFrame) -> pd.Series:
    """True where a row represents a real trade.

    One definition, used everywhere: Justin's rule is volume > 0 and not
    `bad_ohlc` -- never `liquid`, which also demands both quote sides be
    priced and would throw away trades that happened without a two-sided
    quote at that instant.
    """
    bad_ohlc = df["bad_ohlc"] if "bad_ohlc" in df.columns else False
    return (df["volume"] > 0) & ~bad_ohlc


def _read_filtered(
    timeframe: str,
    underlying: str,
    years: Iterable[int],
    *,
    columns: list[str] | None = None,
    extra_filter: ds.Expression | None = None,
) -> pd.DataFrame:
    dataset = _dataset(timeframe)
    if dataset is None:
        return pd.DataFrame()

    years = sorted(set(years))
    filt = (ds.field("underlying") == underlying) & (ds.field("year").isin(years))
    if extra_filter is not None:
        filt = filt & extra_filter

    table = dataset.to_table(filter=filt, columns=columns)
    return table.to_pandas()


# ------------------------------------------------------------------ coverage


def _coverage_cache_path() -> Path:
    # Always under `1m` -- see `coverage()`: every price timeframe's
    # availability is measured from the 1-minute file, since that is the only
    # one actually read for a contract's prices.
    return _timeframe_root(CONTRACT_NATIVE_TIMEFRAME) / "_coverage.json"


def _underlyings_on_disk(timeframe: str) -> list[str]:
    """Underlyings actually partitioned under this timeframe, read from the
    `underlying=X` directory names -- not from the Parquet data itself, which
    would mean materialising a column across the whole (300M+ row) dataset
    just to find five distinct strings."""
    root = _timeframe_root(timeframe)
    if not root.exists():
        return []
    return sorted(
        p.name.split("=", 1)[1]
        for p in root.iterdir()
        if p.is_dir() and p.name.startswith("underlying=")
    )


def _compute_coverage() -> dict[str, list[str]]:
    """First/last day with an options (CE/PE) row in the 1-minute file, per
    underlying actually present. Scans the whole `1m` partition once."""
    dataset = _dataset(CONTRACT_NATIVE_TIMEFRAME)
    if dataset is None:
        return {}

    result: dict[str, list[str]] = {}
    for underlying in _underlyings_on_disk(CONTRACT_NATIVE_TIMEFRAME):
        filt = (ds.field("underlying") == underlying) & (
            ds.field("instrument_type").isin(list(OPTION_TYPES))
        )
        table = dataset.to_table(filter=filt, columns=["timestamp"])
        if table.num_rows == 0:
            continue
        ts = table.to_pandas()["timestamp"]
        result[underlying] = [str(ts.min().date()), str(ts.max().date())]
    return result


def coverage(underlying: str, timeframe: str) -> tuple[dt.date, dt.date] | None:
    """First/last calendar day with a real option price for `underlying`.

    Measured from the **1-minute** file regardless of `timeframe` -- every
    timeframe this module serves is built from `1m` (see the module
    docstring), so `1m`'s own coverage is every other timeframe's coverage
    too. `timeframe` is accepted for a stable call signature and is not
    currently used to pick a different file.

    Cached in `data/fno/1m/_coverage.json` because computing it means
    scanning the whole 1-minute file once. If that file already exists it is
    trusted as-is (not merged or refreshed) -- delete it to force a rescan.
    """
    cache_path = _coverage_cache_path()
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
    else:
        cached = _compute_coverage()
        root = _timeframe_root(CONTRACT_NATIVE_TIMEFRAME)
        if root.exists():
            cache_path.write_text(json.dumps(cached, indent=2, sort_keys=True))

    entry = cached.get(underlying)
    if not entry:
        return None
    start = dt.date.fromisoformat(entry[0])
    end = dt.date.fromisoformat(entry[1])
    return start, end


# ------------------------------------------------------------- underlying bars


def underlying_bars(underlying: str, timeframe: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """The index's own candles (the `EQ` rows), shaped like every other bar
    frame in this project -- see `nlt.data.source.normalise`.

    These are what a signal is judged against; they are never themselves
    tradable (`EQ` volume is always 0 in this dataset).
    """
    years = range(start.year, end.year + 1)
    df = _read_filtered(
        timeframe,
        underlying,
        years,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
        extra_filter=ds.field("instrument_type") == "EQ",
    )
    if df.empty:
        return pd.DataFrame(columns=BAR_COLUMNS).set_index(
            pd.DatetimeIndex([], tz=IST, name="timestamp")
        )

    df = df.set_index("timestamp")
    if timeframe == "1d":
        # The daily file's print time wanders (09:15, occasionally 18:15 for a
        # late settlement print) -- normalise to the date, matching every
        # other daily bar frame in this project (see `data/bars/*.parquet`,
        # all indexed at 00:00 IST).
        df.index = df.index.normalize()

    df = normalise(df)
    mask = (df.index.date >= start) & (df.index.date <= end)
    return df[mask]


# ------------------------------------------------------------------- listing


def listing(underlying: str, day: dt.date) -> pd.DataFrame:
    """Options listed for `underlying` on `day`: tradingsymbol, instrument_type
    (CE/PE), expiry, strike.

    Always read from the **daily** partition, regardless of what timeframe a
    caller eventually trades on: the daily file's one-row-per-contract-day
    shape is the cheapest complete listing, and is enough on its own to answer
    "what strikes/expiries existed on day D" without loading any 15-minute
    data. Nothing here can see a day after `day` -- this is the no-look-ahead
    guarantee `resolve()` depends on.
    """
    year = day.year
    df = _read_filtered(
        "1d",
        underlying,
        [year],
        columns=["timestamp", "tradingsymbol", "instrument_type", "expiry", "strike"],
        # The day's own rows only, so the year's file is not read whole for each
        # day; the exact-date check below still decides what counts.
        extra_filter=ds.field("instrument_type").isin(list(OPTION_TYPES))
        & (ds.field("timestamp") >= _day_start(day))
        & (ds.field("timestamp") < _day_start(day + dt.timedelta(days=1))),
    )
    if df.empty:
        return df.assign(expiry=pd.Series(dtype="datetime64[ns]"))

    on_day = df[df["timestamp"].dt.date == day]
    if on_day.empty:
        return on_day.drop(columns=["timestamp"])

    out = on_day.drop(columns=["timestamp"]).drop_duplicates(
        subset=["tradingsymbol", "instrument_type", "expiry", "strike"]
    )
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.date
    return out.reset_index(drop=True)


def _day_start(day: dt.date) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, dt.time(0, 0)), tz=IST)


def _months_filter(start: dt.date, end: dt.date) -> ds.Expression:
    """Only the `month=M` partitions between `start` and `end`: reading one
    contract no longer scans a whole year of every contract's minutes."""
    expr = None
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        one = (ds.field("year") == y) & (ds.field("month") == m)
        expr = one if expr is None else expr | one
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return expr


# -------------------------------------------------------------- contract bars


def contract_bars(
    underlying: str, tradingsymbol: str, timeframe: str, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """One contract's own 1-minute candles: open/high/low/close/volume/oi/bid/ask/traded.

    Only `1m` is read here -- see the module docstring for why the native
    15m/1d *option* quotes cannot be trusted for prices. Every other
    timeframe (3m, 5m, 15m, 30m, 1h, 1d) is resampled from this by
    `nlt.options.execution`, using traded rows only.
    """
    if timeframe != CONTRACT_NATIVE_TIMEFRAME:
        raise ValueError(
            f"contract_bars reads {CONTRACT_NATIVE_TIMEFRAME!r} only; "
            f"{timeframe!r} must be resampled by the caller from 1m."
        )

    years = range(start.year, end.year + 1)
    df = _read_filtered(
        timeframe,
        underlying,
        years,
        columns=[
            "timestamp",
            "tradingsymbol",
            "open",
            "high",
            "low",
            "close",
            "best_buy_price",
            "best_sell_price",
            "volume",
            "oi",
            "bad_ohlc",
        ],
        extra_filter=(ds.field("tradingsymbol") == tradingsymbol) & _months_filter(start, end),
    )
    empty_cols = [
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
    if df.empty:
        return pd.DataFrame(columns=empty_cols).set_index(
            pd.DatetimeIndex([], tz=IST, name="timestamp")
        )

    df = df.rename(columns={"best_buy_price": "bid", "best_sell_price": "ask"})
    df["traded"] = traded_mask(df)
    df = df.set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    mask = (df.index.date >= start) & (df.index.date <= end)
    df = df[mask]
    return df[empty_cols]


# ------------------------------------------------------------------ protocol


class OptionSource(Protocol):
    """What the execution layer needs from a data source -- `ParquetSource`
    (real disk) and `FrameSource` (in-memory, for tests) both implement it."""

    def coverage(self, underlying: str, timeframe: str) -> tuple[dt.date, dt.date] | None: ...

    def underlying_bars(
        self, underlying: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame: ...

    def listing(self, underlying: str, day: dt.date) -> pd.DataFrame: ...

    def contract_bars(
        self, underlying: str, tradingsymbol: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame: ...


class ParquetSource:
    """The real data source: thin wrapper over this module's module-level
    functions, so `OptionExecution` can take either this or `FrameSource`."""

    def coverage(self, underlying: str, timeframe: str) -> tuple[dt.date, dt.date] | None:
        return coverage(underlying, timeframe)

    def underlying_bars(
        self, underlying: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        return underlying_bars(underlying, timeframe, start, end)

    def listing(self, underlying: str, day: dt.date) -> pd.DataFrame:
        return listing(underlying, day)

    def contract_bars(
        self, underlying: str, tradingsymbol: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        return contract_bars(underlying, tradingsymbol, timeframe, start, end)


# -------------------------------------------------------------- FrameSource


_FRAME_RAW_COLUMNS = [
    "underlying",
    "timeframe",
    "tradingsymbol",
    "instrument_type",
    "expiry",
    "strike",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "best_buy_price",
    "best_sell_price",
    "volume",
    "oi",
    "bad_ohlc",
]


class FrameSource:
    """An in-memory stand-in for `ParquetSource`, built from one DataFrame of
    raw rows shaped exactly like the Parquet dataset plus a `timeframe`
    column (since one real Parquet directory per timeframe becomes one `df`
    here). Lets engine and selection tests build a synthetic options chain --
    a handful of strikes, a couple of days -- with no disk access at all.

    Required columns: `underlying, timeframe, tradingsymbol, instrument_type
    (EQ/CE/PE), expiry, strike, timestamp (tz-aware IST), open, high, low,
    close, best_buy_price, best_sell_price, volume, oi, bad_ohlc`.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        missing = [c for c in _FRAME_RAW_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"FrameSource is missing required columns: {missing}")
        self._df = df.copy()
        self._df["expiry"] = pd.to_datetime(self._df["expiry"]).dt.date

    def coverage(self, underlying: str, timeframe: str) -> tuple[dt.date, dt.date] | None:
        # Measured from `1m` regardless of `timeframe` -- matches
        # `ParquetSource.coverage`; see the module docstring for why.
        sub = self._df[
            (self._df["underlying"] == underlying)
            & (self._df["timeframe"] == CONTRACT_NATIVE_TIMEFRAME)
            & (self._df["instrument_type"].isin(OPTION_TYPES))
        ]
        if sub.empty:
            return None
        days = sub["timestamp"].dt.date
        return days.min(), days.max()

    def underlying_bars(
        self, underlying: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        sub = self._df[
            (self._df["underlying"] == underlying)
            & (self._df["timeframe"] == timeframe)
            & (self._df["instrument_type"] == "EQ")
        ].copy()
        if sub.empty:
            return pd.DataFrame(columns=BAR_COLUMNS).set_index(
                pd.DatetimeIndex([], tz=IST, name="timestamp")
            )
        sub = sub.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        if timeframe == "1d":
            sub.index = sub.index.normalize()
        sub = normalise(sub)
        mask = (sub.index.date >= start) & (sub.index.date <= end)
        return sub[mask]

    def listing(self, underlying: str, day: dt.date) -> pd.DataFrame:
        # Always the daily rows, matching ParquetSource -- a synthetic chain
        # that only supplies 15m/1m rows for a day has no listing for it,
        # same as the real dataset would have no listing without a 1d row.
        sub = self._df[
            (self._df["underlying"] == underlying)
            & (self._df["timeframe"] == "1d")
            & (self._df["instrument_type"].isin(OPTION_TYPES))
            & (self._df["timestamp"].dt.date == day)
        ]
        cols = ["tradingsymbol", "instrument_type", "expiry", "strike"]
        return sub[cols].drop_duplicates().reset_index(drop=True)

    def contract_bars(
        self, underlying: str, tradingsymbol: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        # Matches `ParquetSource.contract_bars`: only `1m` is native here too,
        # so a synthetic-chain test exercises exactly the same resampling path
        # as the real data (see the module docstring).
        if timeframe != CONTRACT_NATIVE_TIMEFRAME:
            raise ValueError(
                f"FrameSource.contract_bars reads {CONTRACT_NATIVE_TIMEFRAME!r} only; "
                f"{timeframe!r} must be resampled by the caller from 1m."
            )
        sub = self._df[
            (self._df["underlying"] == underlying)
            & (self._df["timeframe"] == timeframe)
            & (self._df["tradingsymbol"] == tradingsymbol)
        ].copy()
        empty_cols = ["open", "high", "low", "close", "volume", "oi", "bid", "ask", "traded"]
        if sub.empty:
            return pd.DataFrame(columns=empty_cols).set_index(
                pd.DatetimeIndex([], tz=IST, name="timestamp")
            )
        sub = sub.rename(columns={"best_buy_price": "bid", "best_sell_price": "ask"})
        sub["traded"] = traded_mask(sub)
        sub = sub.set_index("timestamp").sort_index()
        sub = sub[~sub.index.duplicated(keep="last")]
        mask = (sub.index.date >= start) & (sub.index.date <= end)
        return sub.loc[mask, empty_cols]
