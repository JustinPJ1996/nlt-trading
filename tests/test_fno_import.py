"""Tests for the historical F&O importer (`nlt/data/fno_import.py`).

Two kinds of test here, deliberately kept apart.

`synthetic_raw_dir` builds a tiny, fully-known ClickHouse Native dataset (via
`chdb` itself, writing the `Native` bytes it hands back straight to disk) with
one row of every case that matters: a normal liquid contract, an untraded
contract with a bad print, a quote whose contract-day has no master row at
all, and a lot-size-change transition day. Every assertion against it is
against a number this file states outright, so a mutation that silently
breaks the paise conversion, the timezone conversion, the join, or the
quality flags has nowhere to hide.

The real dataset under `~/fno-data/raw/` is only used for the known-answer
test and a real coverage sanity check; both are skipped if that data is not
present, matching how `tests/conftest.py`'s `nifty_bars` fixture treats a
cache that has not been populated. Nothing else in this file touches it,
because a 5 GB fixture is not something a contributor should need before the
suite runs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import chdb
import pandas as pd
import pytest

from nlt.data import fno_import as fi

REAL_RAW_DIR = Path.home() / "fno-data" / "raw"
needs_real_data = pytest.mark.skipif(
    not REAL_RAW_DIR.exists(), reason="~/fno-data/raw not present on this machine"
)


def _write_native(sql: str, path: Path) -> None:
    result = chdb.query(sql, "Native")
    path.write_bytes(result.bytes())


def _import_and_read(raw_dir: Path, out_dir: Path, timeframe: str = "1d") -> pd.DataFrame:
    fi.import_timeframe(timeframe, raw_dir=raw_dir, out_dir=out_dir)
    return pd.read_parquet(out_dir / timeframe)


# scrip_code 1: a NIFTY future, liquid, ordinary OHLC. Its master row appears
# twice for the same trading day with different lot sizes (50 then 75) -- the
# lot-size-change-day duplicate case measured in the real data (399 such
# contract-days) -- so this also exercises `max(lot_size)` in `_master_cte`.
#
# scrip_code 2: an option that never traded and has an impossible close
# (negative), i.e. the "untraded, bad print" case that dominates the real
# defect counts.
#
# scrip_code 3 has a quote row but deliberately NO master row at all: the
# "unmatched" case `check()` reports separately rather than silently folding
# into the row count.
_SCRIPS_SQL = """
SELECT * FROM (
    SELECT toInt32(1) AS scrip_code, toInt32(1) instrument_token, toInt32(1) exchange_token,
           'NIFTY24JANFUT' AS tradingsymbol, 'NIFTY' AS underlying_instrument, 'NIFTY' AS name,
           toDate('2024-01-25') AS expiry, toFloat32(0) AS strike, toFloat32(0.05) AS tick_size,
           toInt32(50) AS lot_size, 'FUT' AS instrument_type, 'NFO-FUT' AS segment, 'NFO' AS exchange,
           'monthly' AS expiry_type, toInt32(1) AS multiplier, true AS is_fno, false AS is_underlying,
           toDate('2024-01-02') AS trading_day
    UNION ALL
    SELECT toInt32(1), toInt32(1), toInt32(1),
           'NIFTY24JANFUT', 'NIFTY', 'NIFTY',
           toDate('2024-01-25'), toFloat32(0), toFloat32(0.05),
           toInt32(75), 'FUT', 'NFO-FUT', 'NFO',
           'monthly', toInt32(1), true, false,
           toDate('2024-01-02')
    UNION ALL
    SELECT toInt32(2), toInt32(2), toInt32(2),
           'NIFTY24JAN22000CE', 'NIFTY', 'NIFTY',
           toDate('2024-01-25'), toFloat32(22000), toFloat32(0.05),
           toInt32(50), 'CE', 'NFO-OPT', 'NFO',
           'weekly', toInt32(1), true, false,
           toDate('2024-01-02')
) ORDER BY scrip_code, lot_size
"""

# scrip_code 1: close = 2,160,500 paise = 21,605.00 rupees -- a plain, checkable
# paise conversion. Timestamp 03:45:00 UTC = 09:15:00 IST, same calendar day.
# scrip_code 2: an all-zero quote with close = -5 paise (an impossible price).
# scrip_code 3: an otherwise ordinary row with no master counterpart.
_QUOTES_SQL = """
SELECT * FROM (
    SELECT toInt32(1) AS scrip_code, toDateTime('2024-01-02 03:45:00') AS time_stamp,
           toInt64(2160000) AS open, toInt64(2170000) AS high, toInt64(2150000) AS low,
           toInt64(2160500) AS close, toUInt64(1000) AS oi, toUInt64(500) AS volume,
           toInt64(2160000) AS best_buy_price, toUInt32(10) AS best_buy_qty,
           toInt64(2161000) AS best_sell_price, toUInt32(10) AS best_sell_qty,
           toInt64(2160500) AS settle_price
    UNION ALL
    SELECT toInt32(2), toDateTime('2024-01-02 03:45:00'),
           toInt64(0), toInt64(0), toInt64(0),
           toInt64(-5), toUInt64(0), toUInt64(0),
           toInt64(0), toUInt32(0),
           toInt64(0), toUInt32(0),
           toInt64(0)
    UNION ALL
    SELECT toInt32(3), toDateTime('2024-01-02 03:45:00'),
           toInt64(10000), toInt64(11000), toInt64(9000),
           toInt64(10500), toUInt64(10), toUInt64(5),
           toInt64(10000), toUInt32(1),
           toInt64(10600), toUInt32(1),
           toInt64(10500)
) ORDER BY scrip_code
"""


@pytest.fixture
def synthetic_raw_dir(tmp_path) -> Path:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_native(_SCRIPS_SQL, raw_dir / fi.DAILY_SCRIPS_FILE)
    _write_native(_QUOTES_SQL, raw_dir / fi.QUOTES_FILES["1d"])
    return raw_dir


def test_paise_conversion_matches_a_stated_value(synthetic_raw_dir, tmp_path):
    """2,160,500 paise must become 21,605.00 rupees, not 21,605,00 or 21.605."""
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    fut = df[df.tradingsymbol == "NIFTY24JANFUT"].iloc[0]
    assert fut["close"] == pytest.approx(21605.00)
    assert fut["open"] == pytest.approx(21600.00)
    assert fut["high"] == pytest.approx(21700.00)
    assert fut["low"] == pytest.approx(21500.00)


def test_quantity_columns_are_not_divided(synthetic_raw_dir, tmp_path):
    """volume, oi and the quote quantities must survive untouched -- only the
    seven price columns are in paise."""
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    fut = df[df.tradingsymbol == "NIFTY24JANFUT"].iloc[0]
    assert fut["volume"] == 500
    assert fut["oi"] == 1000
    assert fut["best_buy_qty"] == 10
    assert fut["best_sell_qty"] == 10


def test_timestamp_is_tz_aware(synthetic_raw_dir, tmp_path):
    """A tz-naive Parquet is silently 5h30m wrong -- see the module docstring.

    This must fail if the importer ever stops localising the timestamp, so it
    asserts both that a timezone is present and which one -- a tz-aware column
    in the wrong zone would be just as wrong as a naive one.
    """
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    assert df["timestamp"].dt.tz is not None, "timestamp column lost its timezone"
    assert str(df["timestamp"].dt.tz) == "Asia/Kolkata"


def test_timestamp_reparsed_from_parquet_is_still_tz_aware(synthetic_raw_dir, tmp_path):
    """The tz-awareness must survive an actual Parquet round trip, not just
    the in-memory DataFrame the importer happens to build it from."""
    out = tmp_path / "out"
    fi.import_timeframe("1d", raw_dir=synthetic_raw_dir, out_dir=out)
    reloaded = pd.read_parquet(out / "1d")
    assert reloaded["timestamp"].dt.tz is not None
    assert str(reloaded["timestamp"].dt.tz) == "Asia/Kolkata"


def test_utc_to_ist_offset_is_five_hours_thirty(synthetic_raw_dir, tmp_path):
    """03:45 UTC must land on 09:15 IST, same calendar day -- not shifted by a
    whole day, and not left at the UTC wall-clock time."""
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    fut = df[df.tradingsymbol == "NIFTY24JANFUT"].iloc[0]
    ts = fut["timestamp"]
    assert ts.date() == dt.date(2024, 1, 2)
    assert (ts.hour, ts.minute) == (9, 15)


def test_lot_size_change_day_resolves_deterministically(synthetic_raw_dir, tmp_path):
    """Two conflicting master rows for the same contract-day (50 then 75) --
    the transition-day duplicate seen 399 times in the real data -- must
    collapse to one value, not silently sum, average, or pick whichever row
    ClickHouse happened to read first."""
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    fut = df[df.tradingsymbol == "NIFTY24JANFUT"]
    assert len(fut) == 1, "the duplicate master rows must not fan out into two quote rows"
    assert fut.iloc[0]["lot_size"] == 75, "max(lot_size) must win the collapse deterministically"


def test_liquid_flag_true_for_a_traded_quoted_row(synthetic_raw_dir, tmp_path):
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    fut = df[df.tradingsymbol == "NIFTY24JANFUT"].iloc[0]
    assert bool(fut["liquid"]) is True
    assert bool(fut["bad_ohlc"]) is False


def test_liquid_flag_false_and_bad_ohlc_true_for_the_untraded_bad_print(
    synthetic_raw_dir, tmp_path
):
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    option = df[df.tradingsymbol == "NIFTY24JAN22000CE"].iloc[0]
    assert bool(option["liquid"]) is False, "volume=0 and both quote sides are 0"
    assert bool(option["bad_ohlc"]) is True, "close = -0.05, an impossible price"


def test_no_row_is_silently_dropped_for_being_low_quality(synthetic_raw_dir, tmp_path):
    """The bad row must still be present in the output -- flagged, not removed.

    This is the assertion that would fail if someone "cleaned up" the
    importer to filter on `liquid` or `bad_ohlc` before writing, which the
    module docstring explicitly says not to do.
    """
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    assert "NIFTY24JAN22000CE" in set(df["tradingsymbol"])
    assert len(df) == 2, "both the good and the bad row must survive the import"


def test_unmatched_contract_day_is_excluded_but_counted(synthetic_raw_dir):
    """scrip_code 3 has a quote but no master row for that day at all -- there
    is no underlying/instrument_type/lot_size to attach, so it cannot appear
    in the joined output, but `check()` must still say so rather than let the
    row count quietly come up short with no explanation."""
    report = fi.check("1d", raw_dir=synthetic_raw_dir)
    assert report.rows == 2, "scrip 3's row must not appear in the joined output"
    assert report.unmatched_in_window == 1, "but its absence must be counted, not silent"


def test_check_writes_nothing(synthetic_raw_dir, tmp_path):
    """`--check` (this function) must never create the output directory."""
    out = tmp_path / "out"
    fi.check("1d", raw_dir=synthetic_raw_dir)
    assert not out.exists()


def test_reimport_replaces_rather_than_accumulates(synthetic_raw_dir, tmp_path):
    """Re-running an import must not leave two copies of the same rows lying
    around next to each other in the same partition."""
    out = tmp_path / "out"
    first = _import_and_read(synthetic_raw_dir, out)
    second = _import_and_read(synthetic_raw_dir, out)
    assert len(first) == len(second) == 2


def test_settle_price_absent_for_intraday_timeframes(synthetic_raw_dir, tmp_path):
    """quotes_15m.bin (and 1h) do not carry `settle_price` at all -- confirmed
    directly against the real file's schema, contradicting this project's own
    working assumption that all four quote tables are identical. Reusing the
    daily SQL unchanged for 15m data used to raise `UNKNOWN_IDENTIFIER`; this
    pins the fix so a future "simplification" back to one shared query breaks
    loudly instead of only on the next real 15m import."""
    quotes_15m = synthetic_raw_dir / fi.QUOTES_FILES["15m"]
    quotes_15m.write_bytes((synthetic_raw_dir / fi.QUOTES_FILES["1d"]).read_bytes())
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out", timeframe="15m")
    assert "settle_price" not in df.columns
    assert "close" in df.columns


def test_missing_quotes_file_names_the_decompress_command(tmp_path):
    """`--timeframe 1m` on a machine that has not decompressed it yet must
    fail with an actionable message, not decompress 40GB as a side effect."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="zstd -dq"):
        fi.check("1m", raw_dir=raw_dir)


@needs_real_data
def test_real_nifty_close_matches_the_published_record_close(tmp_path):
    """NIFTY closed above 26,000 for the first time on 25 September 2024, at
    26,004 points (widely reported at the time). The imported close must
    match to within a rupee -- this is the project's known-answer pin against
    a fact from outside the dataset, not a number derived from it."""
    df = _import_and_read(REAL_RAW_DIR, tmp_path / "out")
    spot = df[(df["instrument_type"] == "EQ") & (df["underlying"] == "NIFTY")]
    row = spot[spot["timestamp"].dt.date == dt.date(2024, 9, 25)]
    assert len(row) == 1, "expected exactly one NIFTY spot row for 2024-09-25"
    assert row.iloc[0]["close"] == pytest.approx(26004.15, abs=1.0)


@needs_real_data
def test_real_coverage_matches_measured_figures():
    """A loose sanity check against the real dataset: the numbers measured
    while building this importer (2,901,779 rows, 2022-11-01 to 2025-12-31),
    not numbers invented for the test to pass."""
    report = fi.check("1d")
    assert report.rows == 2_901_779
    assert report.start == dt.date(2022, 11, 1)
    assert report.end == dt.date(2025, 12, 31)


# ---------------------------------------------------------------------------
# Contract identity: strike, expiry, and quote orientation
#
# Added after the conversion tests above all passed while three deliberate
# breakages went unnoticed: dividing `strike` by 100 as if it were a paise
# price, blanking `expiry` entirely, and swapping bid with ask so every quote
# came out crossed. The suite covered the columns the importer *transforms* and
# not the ones it merely carries through.
#
# These are the columns Phase 3 resolves a contract BY. A strike that is off by
# 100x or an expiry that is null does not produce an error -- it produces a
# backtest of a contract nobody asked for, reported with the same confidence as
# any other. That is the failure this project cares about most.
# ---------------------------------------------------------------------------


def test_strike_is_carried_through_in_rupees(synthetic_raw_dir, tmp_path):
    """`strike` is Float32 in rupees in the source, NOT paise like the prices.

    The paise conversion sits two lines away in the same SELECT, so applying it
    one column too far is an easy and entirely silent mistake: a 22000 strike
    becomes 220, still a plausible-looking number.
    """
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    ce = df[df["tradingsymbol"] == "NIFTY24JAN22000CE"]
    assert len(ce) == 1
    assert ce.iloc[0]["strike"] == pytest.approx(22000.0), (
        "strike must stay in rupees; the paise divisor must not reach it"
    )


def test_expiry_is_carried_through(synthetic_raw_dir, tmp_path):
    """Without an expiry there is no way to tell one contract from the next."""
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    ce = df[df["tradingsymbol"] == "NIFTY24JAN22000CE"]
    assert ce["expiry"].notna().all(), "expiry must not be blank"
    assert pd.Timestamp(ce.iloc[0]["expiry"]).date() == dt.date(2024, 1, 25)


def test_bid_never_exceeds_ask(synthetic_raw_dir, tmp_path):
    """A crossed quote -- bid above ask -- cannot happen in a real market.

    If it appears, the two columns have been swapped somewhere. Phase 5 uses the
    spread width to estimate what trading costs, and a negative spread would
    quietly turn that cost into a subsidy.
    """
    df = _import_and_read(synthetic_raw_dir, tmp_path / "out")
    quoted = df[(df["best_buy_price"] > 0) & (df["best_sell_price"] > 0)]
    assert len(quoted) > 0, "the fixture must contain at least one quoted row"
    crossed = quoted[quoted["best_sell_price"] < quoted["best_buy_price"]]
    assert crossed.empty, f"{len(crossed)} rows have bid above ask:\n{crossed.head()}"


@needs_real_data
def test_real_liquid_rows_are_never_crossed(tmp_path):
    """The same guarantee on the real dataset.

    Measured before this test was written: zero crossed quotes among the liquid
    rows. So this asserts a property the data genuinely has, and would fail if
    the importer ever swapped the columns rather than if the market misbehaved.
    """
    import glob

    files = glob.glob("data/fno/1d/underlying=NIFTY/**/*.parquet", recursive=True)
    if not files:
        pytest.skip("data/fno/1d not imported on this machine")
    df = pd.concat([pd.read_parquet(f) for f in files])
    liquid = df[df["liquid"]]
    crossed = liquid[liquid["best_sell_price"] < liquid["best_buy_price"]]
    assert crossed.empty, f"{len(crossed)} liquid rows have bid above ask"
