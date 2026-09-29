"""Importing the historical index F&O dataset into Parquet.

The source (`~/fno-data/`) is a ClickHouse Native export, read here through
`chdb` (an embedded ClickHouse) because nothing else on this machine can open
that format -- DuckDB cannot, and there is no reason to stand up a real
ClickHouse server to read a handful of files once.

## Two conversions that are silently catastrophic if missed

**Prices are in paise.** `open/high/low/close/best_buy_price/best_sell_price/
settle_price` are all integers scaled by 100 -- NIFTY's raw `close` reads
2,621,605 on the day it closed at 26,216.05. `volume`, `oi` and the two
quantity columns are already in real units and must NOT be divided.

**Timestamps are UTC.** The 15-minute grid runs 03:45-09:45 UTC, which is
09:15-15:15 IST -- the 25-bar session this project's engine already assumes.
`nlt/data/source.py`'s `normalise()` *assumes* a naive timestamp is IST rather
than rejecting one that isn't, so a tz-naive Parquet column would be silently
5h30m wrong and would sail through every downstream check. The Parquet written
here always carries a tz-aware `timestamp` column for exactly this reason --
see `test_timestamp_is_tz_aware` in `tests/test_fno_import.py`, which exists
to fail the day someone "simplifies" this to a naive column.

## The master table has to be collapsed, carefully

`daily_scrips` repeats one row per instrument per trading day. 65,047
`(scrip_code, trading_day)` pairs in the file carry two rows rather than one.
Almost all of those pairs are exact duplicates -- the same values written
twice -- but 399 of them are not: they are lot-size-change transition days
(e.g. BANKNIFTY's 25-to-15 change), where the source appears to have captured
both the outgoing and the incoming lot size under the same trading day. Taking
`any()` for most columns is fine for the exact-duplicate case, but `lot_size`
specifically uses `max()`, an order-independent aggregate, so the collapse is
deterministic rather than depending on ClickHouse's internal row order. This
is documented rather than "fixed" because there is no way to know, from the
data alone, which of the two lot sizes actually applied intraday on the
changeover date; `max()` is a stated, arbitrary, repeatable choice, not a
claim of correctness for that one day per contract.

Historical lot size is carried through per contract-day (unlike
`nlt/data/instruments.py`, which deliberately uses *today's* lot size
everywhere for the live engine) because this importer's job is to preserve
what the exchange actually specified, contract by contract, day by day. What
the engine later chooses to do with that history is Phase 2's decision, not
this module's.

**A measured gap, not hidden:** joining `(scrip_code, trading_day)` means a
quote whose day has no matching master snapshot for that exact contract
cannot be given an underlying/instrument_type/lot_size, and is necessarily
left out of the join. Measured directly: of the 15-minute file, 248,221 rows
(0.39%) have no matching master row for their contract-day; 216,913 of those
(0.34% of the whole file) fall inside the master's own 2022-11-2025-12
coverage window, meaning they are a genuine small gap in the source rather
than an artefact of our own date scoping. `check()` reports this count so it
is never a silent loss.

## The bid/ask is not a fill price

44% of *clean* daily option rows (volume > 0, both quote sides priced) have a
`close` that falls outside the quoted `[best_buy_price, best_sell_price]`
spread, because the quote is a snapshot from a different moment than the
close print. **Do not use `best_buy_price`/`best_sell_price` as a fill price**
in any backtest built on top of this data -- use it only to *estimate* spread
cost. Treating it as a tradable price would flatter results with fills that
never actually happened.

## Data quality is flagged, never dropped

56.9% of daily option rows never traded that day (`volume = 0`), and defects
(a `close` outside `[low, high]`, or `close <= 0`) live almost entirely inside
that untraded set. Filtering to `volume > 0 AND best_buy_price > 0 AND
best_sell_price > 0` isolates a clean 694,729-row subset with zero impossible
prices and only 95 residual bad-OHLC rows (0.014%) -- but this module does not
do that filtering. It adds two boolean columns, `liquid` and `bad_ohlc`, and
leaves every row in place. A strategy backtest silently missing 43 million
untraded 15-minute bars because an importer decided they were not interesting
is exactly the kind of confident, wrong number this project exists to avoid.

## Parquet layout

`data/fno/{timeframe}/underlying={U}/year={Y}/*.parquet`, hive-partitioned by
underlying and calendar year (IST). Two reasons: the five underlyings
(NIFTY, BANKNIFTY, SENSEX, BANKEX, INDIAVIX) are almost always queried one at
a time, and partition pruning means a reader asking for one symbol never
touches the other four; and splitting by year keeps any one partition file to
single-digit millions of rows even for the 15-minute grid (64M rows total),
which keeps both the writer and any later reader inside comfortable memory
whether local or on a small VPS. `data/` is gitignored (anchored `/data/`
pattern -- verified in `.gitignore` and guarded by
`tests/test_repo_hygiene.py`), so nothing here is expected to reach git.

## Scope

Daily and 15-minute only, by design -- see `TIMEFRAMES`. The 1-minute file
(`quotes_1m.bin`, decompressed it would be tens of gigabytes) is wired up but
not decompressed or imported by default; `import_timeframe("1m", ...)` raises
a clear `FileNotFoundError` naming the `zstd` command to run first, rather
than silently doing that decompression as a side effect of an import.
"""

from __future__ import annotations

import datetime as dt
import shutil
from dataclasses import dataclass
from pathlib import Path

import chdb
import pandas as pd

IST = "Asia/Kolkata"

RAW_DIR = Path.home() / "fno-data" / "raw"
DAILY_SCRIPS_FILE = "daily_scrips.bin"

# timeframe label -> raw quotes filename. All four share an identical schema;
# only 1d and 15m are decompressed by default (see module docstring).
QUOTES_FILES: dict[str, str] = {
    "1d": "quotes_1d.bin",
    "15m": "quotes_15m.bin",
    "1m": "quotes_1m.bin",
}

# Compressed source for each raw file, named so a FileNotFoundError can tell
# the caller exactly what to run rather than just that something is missing.
COMPRESSED_NAMES: dict[str, str] = {
    "1d": "market.quotes_1d_local.bin.zst",
    "15m": "market.quotes_15m_local.bin.zst",
    "1m": "market.quotes_1m_local.bin.zst",
}

TIMEFRAMES = tuple(QUOTES_FILES)
DEFAULT_TIMEFRAMES = ("1d", "15m")  # 1m is opt-in; see module docstring.

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "fno"

# Index F&O only, matching the source -- no single-stock options, no
# commodities. Measured directly from `daily_scrips` (`is_fno OR
# is_underlying`); not an assumption.
UNDERLYINGS = ("NIFTY", "BANKNIFTY", "SENSEX", "BANKEX", "INDIAVIX")

# Divided by 100 in `_select_sql` to convert paise to rupees: open, high, low,
# close, settle_price, best_buy_price, best_sell_price. `volume`, `oi` and the
# two quantity columns are already in real units and must NOT be divided.


def _quotes_path(raw_dir: Path, timeframe: str) -> Path:
    if timeframe not in QUOTES_FILES:
        raise ValueError(f"unknown timeframe {timeframe!r}; expected one of {TIMEFRAMES}")
    return raw_dir / QUOTES_FILES[timeframe]


def _require_quotes_file(raw_dir: Path, timeframe: str) -> Path:
    """The path to a decompressed quotes file, or a raised error naming the fix.

    Deliberately not auto-decompressing: `quotes_1m.bin`'s compressed source is
    3.7 GB and, going by the 15-minute file's ~12x expansion, decompresses to
    tens of gigabytes. Doing that as a side effect of an import call would
    surprise anyone who typed `--timeframe 1m` expecting an import, not an
    hours-long, disk-filling decompression.
    """
    path = _quotes_path(raw_dir, timeframe)
    if not path.exists():
        compressed = COMPRESSED_NAMES[timeframe]
        raise FileNotFoundError(
            f"{path} does not exist. Decompress it first:\n"
            f"  zstd -dq {raw_dir.parent / compressed} -o {path}"
        )
    return path


def _daily_scrips_path(raw_dir: Path) -> Path:
    path = raw_dir / DAILY_SCRIPS_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; see ~/fno-data/PUT-FILES-HERE.txt")
    return path


def _master_cte(daily_scrips_path: Path) -> str:
    """SQL for one row per (scrip_code, trading_day) -- see module docstring
    for why `lot_size` uses `max()` while everything else uses `any()`."""
    return f"""
    (
        SELECT
            scrip_code,
            trading_day,
            any(tradingsymbol) AS tradingsymbol,
            any(underlying_instrument) AS underlying,
            any(instrument_type) AS instrument_type,
            any(expiry) AS expiry,
            any(strike) AS strike,
            max(lot_size) AS lot_size
        FROM file('{daily_scrips_path}', 'Native')
        WHERE is_fno = 1 OR is_underlying = 1
        GROUP BY scrip_code, trading_day
    )
    """


def _join_sql(quotes_path: Path, daily_scrips_path: Path, *, where: str = "") -> str:
    """The shared join: quotes to the collapsed master, on the contract-day.

    `q.time_stamp + INTERVAL 330 MINUTE` converts the UTC quote timestamp to
    IST before truncating to a date, matching `trading_day` in the master.
    Market hours (03:45-09:45 UTC) never cross an IST midnight, so this is
    always the same calendar day as the eventual tz-aware IST timestamp.
    """
    where_clause = f"WHERE {where}" if where else ""
    return f"""
    FROM file('{quotes_path}', 'Native') AS q
    INNER JOIN {_master_cte(daily_scrips_path)} AS m
        ON q.scrip_code = m.scrip_code
       AND toDate(q.time_stamp + INTERVAL 330 MINUTE) = m.trading_day
    {where_clause}
    """


# Only the daily quotes file carries `settle_price` -- a once-a-day concept.
# `DESCRIBE TABLE` on quotes_15m.bin (and quotes_1h.bin) confirms the column is
# simply absent there, not null; the source's own schema note that "quotes_*
# are all four identical" is not quite right, and this was caught by a failed
# import rather than assumed from the note.
_HAS_SETTLE_PRICE = {"1d": True, "1h": False, "15m": False, "1m": False}


def _select_sql(
    quotes_path: Path, daily_scrips_path: Path, *, timeframe: str, where: str = ""
) -> str:
    join = _join_sql(quotes_path, daily_scrips_path, where=where)
    settle_price_col = (
        "q.settle_price / 100.0 AS settle_price,\n        " if _HAS_SETTLE_PRICE[timeframe] else ""
    )
    return f"""
    SELECT
        m.underlying AS underlying,
        m.tradingsymbol AS tradingsymbol,
        m.instrument_type AS instrument_type,
        m.expiry AS expiry,
        m.strike AS strike,
        m.lot_size AS lot_size,
        q.scrip_code AS scrip_code,
        q.time_stamp AS time_stamp_utc,
        q.open / 100.0 AS open,
        q.high / 100.0 AS high,
        q.low / 100.0 AS low,
        q.close / 100.0 AS close,
        {settle_price_col}q.best_buy_price / 100.0 AS best_buy_price,
        q.best_sell_price / 100.0 AS best_sell_price,
        q.best_buy_qty AS best_buy_qty,
        q.best_sell_qty AS best_sell_qty,
        q.oi AS oi,
        q.volume AS volume
    {join}
    ORDER BY m.underlying, q.scrip_code, q.time_stamp
    """


def _run_query(sql: str) -> pd.DataFrame:
    return chdb.query(sql, "DataFrame")


def _localise(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the IST timestamp column and drop the raw UTC one.

    `chdb` hands back a naive `datetime64[s]` column that is UTC in fact but
    not in name -- `tz_localize("UTC")` records what it already is, and
    `tz_convert(IST)` is the mandatory conversion from the module docstring.
    """
    df = df.copy()
    df["timestamp"] = df["time_stamp_utc"].dt.tz_localize("UTC").dt.tz_convert(IST)
    return df.drop(columns=["time_stamp_utc"])


def add_quality_flags(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Flag liquidity and OHLC defects. Never filters -- see module docstring.

    `liquid` marks rows where a fill was plausible at all: something traded,
    and both sides of the quote were priced. `bad_ohlc` marks a `close`
    outside `[low, high]` or a nonsensical `close <= 0`, regardless of
    liquidity, because a caller filtering on `liquid` alone should not
    discover bad prints living inside the rows they kept.
    """
    df = df.copy()
    liquid = (df["volume"] > 0) & (df["best_buy_price"] > 0) & (df["best_sell_price"] > 0)
    bad_ohlc = (df["close"] > df["high"]) | (df["close"] < df["low"]) | (df["close"] <= 0)
    df["liquid"] = liquid
    df["bad_ohlc"] = bad_ohlc

    rows = len(df)
    notes = []
    if rows:
        notes.append(
            f"{rows:,} rows: {int(liquid.sum()):,} liquid ({100 * liquid.mean():.1f}%), "
            f"{int((~liquid).sum()):,} untraded/unquoted ({100 * (~liquid).mean():.1f}%)"
        )
        notes.append(
            f"{int(bad_ohlc.sum()):,} rows have a close outside [low, high] or <= 0 "
            f"({100 * bad_ohlc.mean():.3f}%); {int((bad_ohlc & ~liquid).sum()):,} of those "
            f"are also untraded, {int((bad_ohlc & liquid).sum()):,} are not"
        )
    return df, notes


@dataclass(frozen=True)
class CoverageReport:
    """What `--check` reports: measured, not assumed. See `check()`."""

    timeframe: str
    rows: int
    start: dt.date | None
    end: dt.date | None
    liquid: int
    untraded: int
    impossible_price: int
    bad_ohlc: int
    unmatched_in_window: int

    def __str__(self) -> str:
        if self.rows == 0:
            return f"{self.timeframe}: no rows"

        def pct(n: int) -> float:
            return 100 * n / self.rows

        lines = [
            f"{self.timeframe}: {self.rows:,} rows, {self.start} to {self.end}",
            f"  liquid (volume>0, both quote sides priced): "
            f"{self.liquid:,} ({pct(self.liquid):.1f}%)",
            f"  untraded (volume=0): {self.untraded:,} ({pct(self.untraded):.1f}%)",
            f"  impossible price (close<=0): {self.impossible_price:,}",
            f"  bad OHLC (close outside [low, high]): "
            f"{self.bad_ohlc:,} ({pct(self.bad_ohlc):.3f}%)",
        ]
        if self.unmatched_in_window:
            lines.append(
                f"  excluded (quote inside coverage window with no matching "
                f"master snapshot for that contract-day): {self.unmatched_in_window:,}"
            )
        return "\n".join(lines)


def check(timeframe: str, raw_dir: Path = RAW_DIR) -> CoverageReport:
    """Report row counts, date range and defect counts. Writes nothing.

    Computed entirely inside chdb (aggregates, not a materialised frame) so
    this is cheap even for the 15-minute file's 64M rows.
    """
    quotes_path = _require_quotes_file(raw_dir, timeframe)
    daily_scrips_path = _daily_scrips_path(raw_dir)
    join = _join_sql(quotes_path, daily_scrips_path)

    sql = f"""
    SELECT
        count() AS rows,
        min(toDate(q.time_stamp + INTERVAL 330 MINUTE)) AS start_date,
        max(toDate(q.time_stamp + INTERVAL 330 MINUTE)) AS end_date,
        sum(q.volume > 0 AND q.best_buy_price > 0 AND q.best_sell_price > 0) AS liquid,
        sum(q.volume = 0) AS untraded,
        sum(q.close <= 0) AS impossible_price,
        sum(q.close > q.high OR q.close < q.low) AS bad_ohlc
    {join}
    """
    result = _run_query(sql)
    row = result.iloc[0]

    unmatched_sql = f"""
    WITH m AS (
        SELECT DISTINCT scrip_code, trading_day
        FROM file('{daily_scrips_path}', 'Native')
        WHERE is_fno = 1 OR is_underlying = 1
    ),
    bounds AS (
        SELECT min(trading_day) AS lo, max(trading_day) AS hi
        FROM file('{daily_scrips_path}', 'Native')
        WHERE is_fno = 1 OR is_underlying = 1
    )
    SELECT count() AS unmatched
    FROM file('{quotes_path}', 'Native') AS q
    LEFT JOIN m ON q.scrip_code = m.scrip_code
        AND toDate(q.time_stamp + INTERVAL 330 MINUTE) = m.trading_day
    CROSS JOIN bounds
    WHERE m.scrip_code = 0
      AND toDate(q.time_stamp + INTERVAL 330 MINUTE) BETWEEN bounds.lo AND bounds.hi
    """
    unmatched = int(_run_query(unmatched_sql).iloc[0]["unmatched"])

    return CoverageReport(
        timeframe=timeframe,
        rows=int(row["rows"]),
        start=row["start_date"].date() if row["rows"] else None,
        end=row["end_date"].date() if row["rows"] else None,
        liquid=int(row["liquid"]),
        untraded=int(row["untraded"]),
        impossible_price=int(row["impossible_price"]),
        bad_ohlc=int(row["bad_ohlc"]),
        unmatched_in_window=unmatched,
    )


def _load_chunk(
    quotes_path: Path,
    daily_scrips_path: Path,
    underlying: str,
    year: int,
    timeframe: str,
) -> pd.DataFrame:
    """One (underlying, year) chunk, small enough to hold in memory even for
    the busiest year of the 15-minute grid (~10M rows for NIFTY 2025)."""
    where = f"m.underlying = '{underlying}' AND toYear(q.time_stamp) = {year}"
    sql = _select_sql(quotes_path, daily_scrips_path, timeframe=timeframe, where=where)
    df = _run_query(sql)
    df = _localise(df)
    df, _ = add_quality_flags(df)
    return df


def _years_for(daily_scrips_path: Path) -> list[int]:
    sql = f"""
    SELECT DISTINCT toYear(trading_day) AS yr
    FROM file('{daily_scrips_path}', 'Native')
    WHERE is_fno = 1 OR is_underlying = 1
    ORDER BY yr
    """
    return [int(y) for y in _run_query(sql)["yr"]]


def import_timeframe(
    timeframe: str,
    *,
    raw_dir: Path = RAW_DIR,
    out_dir: Path = OUTPUT_DIR,
    underlyings: tuple[str, ...] = UNDERLYINGS,
) -> list[str]:
    """Import one timeframe to `out_dir/{timeframe}/underlying=U/year=Y/*.parquet`.

    Re-runnable: the timeframe's output directory is cleared first, so a
    second run replaces rather than accumulates alongside the first.
    Processes one (underlying, year) chunk at a time -- see `_load_chunk` --
    so even the 15-minute file's 64M rows never sit in memory all at once.

    Returns the quality notes gathered across every chunk written.
    """
    quotes_path = _require_quotes_file(raw_dir, timeframe)
    daily_scrips_path = _daily_scrips_path(raw_dir)

    timeframe_dir = out_dir / timeframe
    if timeframe_dir.exists():
        shutil.rmtree(timeframe_dir)
    timeframe_dir.mkdir(parents=True, exist_ok=True)

    notes: list[str] = []
    for underlying in underlyings:
        for year in _years_for(daily_scrips_path):
            chunk = _load_chunk(quotes_path, daily_scrips_path, underlying, year, timeframe)
            if chunk.empty:
                continue
            chunk = chunk.assign(year=year)
            chunk.to_parquet(
                timeframe_dir,
                engine="pyarrow",
                partition_cols=["underlying", "year"],
                index=False,
            )
            notes.append(f"{underlying} {year}: wrote {len(chunk):,} rows")

    return notes
