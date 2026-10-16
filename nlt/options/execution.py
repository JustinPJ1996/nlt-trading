"""Reading one resolved contract's prices once it has been chosen.

`OptionExecution` is the object the engine (not built by this agent -- see
`HANDOFF.md` Phase 2) will hold for the life of a backtest: `resolve()` picks
the contract at signal time, and the rest of its methods answer "what did this
contract do" without ever looking past the moment being asked about.

## Every option timeframe is resampled from 1-minute traded rows

`nlt.options.data.contract_bars` only ever reads `1m` -- the module docstring
there has the measured reason: the native `15m`/`1d` *option* quotes files
were themselves aggregated over every 1-minute row including the
forward-filled (`volume == 0`) ones, so their `open`/`high`/`low` can be a
stale repeated price from well before the bar's first real trade (checked
example: a native 15-minute bar opening at 10,899.95 when the first actual
trade inside it was at 7,735.60, over an hour later).

So **every** timeframe this module serves -- `1m` itself, `3m`, `5m`, `15m`,
`30m`, `1h`, and `1d` -- is built here from `1m` traded rows (`_RESAMPLE_MINUTES`
plus the dedicated daily path, `resample_daily`). The rule, Justin's: a
forward-filled row never contributes a price -- not to open, not to
high/low, not to close -- because in the 1-minute file those rows are a
repeat of the last real print, not a new one. A bucket with no traded row at
all gets `traded=False` and `NaN` prices rather than inheriting a stale
number from its only (untraded) row. Volume is summed over the bucket's
traded rows (untraded rows contribute 0 anyway, so summing everything gives
the identical answer).

Intraday buckets are aligned to the session's 09:15 IST open, not to the
clock: a `1h` bucket runs 09:15-10:15, not 09:00-10:00, because the session
never traded in the first 15 minutes of a clock hour anyway. A `1d` bar is
one bucket per calendar day (`resample_daily`), indexed at 00:00 IST like
every other daily bar frame in this project.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from nlt.data.source import IST
from nlt.options.contracts import Contract, Resolution
from nlt.options.contracts import resolve as _resolve_contract
from nlt.options.contracts import resolve_fixed as _resolve_fixed
from nlt.options.contracts import resolve_points as _resolve_points
from nlt.options.data import CONTRACT_NATIVE_TIMEFRAME, OptionSource

SESSION_OPEN = dt.time(9, 15)

# Every non-native target timeframe -> minutes per bucket, resampled from 1m.
# "1d" is handled separately by `resample_daily` (a calendar day, not a fixed
# number of minutes).
_RESAMPLE_MINUTES: dict[str, int] = {
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}

BAR_COLUMNS = ["open", "high", "low", "close", "volume", "oi", "bid", "ask", "traded"]
SUPPORTED_TIMEFRAMES = (CONTRACT_NATIVE_TIMEFRAME, "1d", *_RESAMPLE_MINUTES)


def _base_timeframe(timeframe: str) -> tuple[str, int | str | None]:
    """(native timeframe to read, resample spec).

    The spec is `None` for `1m` itself (no resampling), the string `"1d"` for
    a calendar-day bucket (`resample_daily`), or an `int` bucket size in
    minutes (`resample_traded`). Every spec reads `1m` as its base -- there is
    no other native option timeframe any more, see the module docstring.
    """
    if timeframe == CONTRACT_NATIVE_TIMEFRAME:
        return CONTRACT_NATIVE_TIMEFRAME, None
    if timeframe == "1d":
        return CONTRACT_NATIVE_TIMEFRAME, "1d"
    if timeframe in _RESAMPLE_MINUTES:
        return CONTRACT_NATIVE_TIMEFRAME, _RESAMPLE_MINUTES[timeframe]
    raise ValueError(
        f"unsupported option timeframe {timeframe!r}; expected one of {SUPPORTED_TIMEFRAMES}"
    )


def _bucket_start(index: pd.DatetimeIndex, bucket_minutes: int) -> pd.DatetimeIndex:
    """Each timestamp's bucket start, aligned to that day's 09:15 session open."""
    day_open = index.normalize() + pd.Timedelta(
        hours=SESSION_OPEN.hour, minutes=SESSION_OPEN.minute
    )
    minutes_since_open = (index - day_open) / pd.Timedelta(minutes=1)
    bucket_index = np.floor(minutes_since_open / bucket_minutes).astype("int64")
    return day_open + pd.to_timedelta(bucket_index * bucket_minutes, unit="m")


def _aggregate(work: pd.DataFrame) -> pd.DataFrame:
    """One row per `_bucket`, all at once (2026-10-16: the bucket-by-bucket
    loop it replaced was a third of an option backtest's time).

    The same rule, column by column: open/close are the first/last TRADED
    row's (whatever their value, as `.iloc` takes them), high/low/volume over
    traded rows only, and oi/bid/ask from the bucket's last row of any kind.
    A bucket with no trade has NaN prices, volume 0 and traded False.
    `tests/test_options_execution.py` checks it against the loop, row for row.
    """
    work = work.sort_index(kind="stable")
    buckets = pd.Index(work["_bucket"].unique()).sort_values()
    traded = work[work["traded"].astype(bool)]
    by = traded.groupby("_bucket", sort=True)
    out = pd.DataFrame(index=buckets)
    out["open"] = by.head(1).set_index("_bucket")["open"]
    out["high"] = by["high"].max()
    out["low"] = by["low"].min()
    out["close"] = by.tail(1).set_index("_bucket")["close"]
    out["volume"] = by["volume"].sum()
    out["volume"] = out["volume"].fillna(0.0)
    last_any = work.groupby("_bucket", sort=True).tail(1).set_index("_bucket")
    for column in ("oi", "bid", "ask"):
        out[column] = last_any[column] if column in last_any.columns else float("nan")
    out["traded"] = out.index.isin(traded["_bucket"].unique())
    out.index.name = "timestamp"
    return out[BAR_COLUMNS]


def resample_traded(df: pd.DataFrame, bucket_minutes: int) -> pd.DataFrame:
    """Resample 1-minute bars to `bucket_minutes`, using only `traded` rows for
    every price and for volume. See module docstring for the exact rule."""
    if df.empty:
        return df.reindex(columns=BAR_COLUMNS)

    work = df.copy()
    work["_bucket"] = _bucket_start(work.index, bucket_minutes)
    return _aggregate(work)


def resample_daily(df: pd.DataFrame) -> pd.DataFrame:
    """One bar per calendar day, from 1-minute traded rows only.

    Indexed at 00:00 IST, matching every other daily bar frame in this
    project (see `data/bars/*.parquet`) -- never the native daily option
    file's close, which the module docstring shows matches the last real
    trade only about 30% of the time.
    """
    if df.empty:
        return df.reindex(columns=BAR_COLUMNS)

    work = df.copy()
    work["_bucket"] = pd.DatetimeIndex(work.index).normalize()
    return _aggregate(work)


class OptionExecution:
    """Resolves a contract at signal time, then answers questions about its
    price -- all of them bounded by the timestamp asked, never later."""

    def __init__(
        self,
        source: OptionSource,
        underlying: str,
        timeframe: str,
        option_type: str = "CE",
        offset: int = 0,
        expiry_kind: str = "nearest_weekly",
        strike: float | None = None,
    ) -> None:
        """`option_type` / `offset` / `expiry_kind` describe batch 1's single
        bought option, for `resolve`; `strike` is a strike the user named
        ("the 24,000 call"), used exactly or not at all. A multi-leg run
        passes each leg to `resolve_leg` instead and can leave them at their
        defaults."""
        self.source = source
        self.underlying = underlying
        self.timeframe = timeframe
        self.option_type = option_type
        self.offset = offset
        self.expiry_kind = expiry_kind
        self.strike = strike
        self._bars_cache: dict[str, pd.DataFrame] = {}
        # Remembers the signal day each contract was resolved on, so the
        # bars cache always spans *that* day through expiry -- never a later
        # day's, which would silently amputate the history a square-off
        # lookback (`last_trade`) needs to see past its own calling day.
        self._signal_day: dict[str, dt.date] = {}
        self._listing_cache: dict[dt.date, pd.DataFrame] = {}
        self._rows: dict[str, tuple[pd.DataFrame, dict]] = {}

    def listing(self, day: dt.date) -> pd.DataFrame:
        """The options listed on `day` (and nothing later), read once per day."""
        cached = self._listing_cache.get(day)
        if cached is None:
            cached = self.source.listing(self.underlying, day)
            self._listing_cache[day] = cached
        return cached

    def resolve(self, signal_time: dt.datetime, spot_close: float) -> Resolution:
        day = signal_time.date()
        chain = self.listing(day)
        if self.strike is not None:
            resolution = _resolve_fixed(
                chain, self.underlying, signal_time, self.option_type, self.strike, self.expiry_kind
            )
        else:
            resolution = _resolve_contract(
                chain,
                self.underlying,
                signal_time,
                spot_close,
                self.option_type,
                self.offset,
                self.expiry_kind,
            )
        if resolution.contract is not None:
            self._signal_day.setdefault(resolution.contract.tradingsymbol, day)
        return resolution

    def resolve_leg(self, signal_time: dt.datetime, spot_close: float, leg) -> Resolution:
        """Pick one leg's contract (`nlt.spec.models.OptionLeg`) at the signal.

        `leg.strike_points` set: the listed strike nearest spot + points
        (`resolve_points`); otherwise `leg.strike_offset` listed strikes out
        of the money for the leg's own type (`resolve`, batch 1's rule).
        """
        day = signal_time.date()
        chain = self.listing(day)
        if leg.strike is not None:
            resolution = _resolve_fixed(
                chain, self.underlying, signal_time, leg.option_type, leg.strike, leg.expiry
            )
        elif leg.strike_points is not None:
            resolution = _resolve_points(
                chain,
                self.underlying,
                signal_time,
                spot_close,
                leg.option_type,
                leg.strike_points,
                leg.expiry,
            )
        else:
            resolution = _resolve_contract(
                chain,
                self.underlying,
                signal_time,
                spot_close,
                leg.option_type,
                leg.strike_offset,
                leg.expiry,
            )
        if resolution.contract is not None:
            self._signal_day.setdefault(resolution.contract.tradingsymbol, day)
        return resolution

    def _contract_bars(self, contract: Contract, from_day: dt.date) -> pd.DataFrame:
        """Load once per contract per run (signal day..expiry) and cache it.

        The window always starts at the day `resolve()` chose this contract
        on, regardless of which day a later call happens to ask about first
        -- `from_day` is only a fallback for callers that never went through
        `resolve()` (e.g. a unit test driving this contract directly).
        """
        cached = self._bars_cache.get(contract.tradingsymbol)
        if cached is not None:
            return cached

        start = self._signal_day.get(contract.tradingsymbol, from_day)
        base_timeframe, spec = _base_timeframe(self.timeframe)
        raw = self.source.contract_bars(
            contract.underlying, contract.tradingsymbol, base_timeframe, start, contract.expiry
        )
        if spec is None:
            bars = raw
        elif spec == "1d":
            bars = resample_daily(raw)
        else:
            bars = resample_traded(raw, spec)
        self._bars_cache[contract.tradingsymbol] = bars
        return bars

    def bar(self, contract: Contract, ts: dt.datetime) -> dict | None:
        """The bar starting exactly at `ts` (its columns by name), or None if
        the contract has none there.

        A plain row, looked up in a table built once per contract: the
        engines ask this for every open option on every candle (170,000 times
        in a year of 1-minute straddles), and a pandas row each time was a
        third of the run (2026-10-16)."""
        bars = self._contract_bars(contract, ts.date())
        rows = self._rows.get(contract.tradingsymbol)
        if rows is None or rows[0] is not bars:
            rows = (bars, dict(zip(bars.index, bars.to_dict("records"), strict=True)))
            self._rows[contract.tradingsymbol] = rows
        return rows[1].get(pd.Timestamp(ts))

    def bar_times(
        self, contract: Contract, after: dt.datetime, until: dt.time
    ) -> list[pd.Timestamp]:
        """Start times of the contract's candles later than `after` and before
        `until` on the same day -- the option candles after the index's last."""
        bars = self._contract_bars(contract, after.date())
        after = pd.Timestamp(after)
        end = pd.Timestamp(dt.datetime.combine(after.date(), until), tz=after.tz)
        return list(bars.index[(bars.index > after) & (bars.index < end)])

    def traded(self, contract: Contract, ts: dt.datetime) -> bool:
        row = self.bar(contract, ts)
        return bool(row is not None and row["traded"])

    def price_at_signal(self, contract: Contract, signal_ts: dt.datetime) -> float | None:
        """The option's last traded close at or before `signal_ts`, same
        trading session (day) only -- never a price from a different day, and
        never a bar that starts after `signal_ts`."""
        bars = self._contract_bars(contract, signal_ts.date())
        ts = pd.Timestamp(signal_ts)
        same_day = bars[bars.index.date == ts.date()]
        traded_up_to_ts = same_day[(same_day.index <= ts) & same_day["traded"]]
        if traded_up_to_ts.empty:
            return None
        return float(traded_up_to_ts["close"].iloc[-1])

    def last_trade(self, contract: Contract, ts: dt.datetime) -> tuple[float, dt.datetime] | None:
        """The last traded close strictly before `ts`, any earlier day included
        -- used to square off a contract that is not trading right now."""
        bars = self._contract_bars(contract, ts.date())
        pts = pd.Timestamp(ts)
        traded_before = bars[(bars.index < pts) & bars["traded"]]
        if traded_before.empty:
            return None
        row = traded_before.iloc[-1]
        return float(row["close"]), row.name.to_pydatetime()

    def expiry_session_close(self, contract: Contract) -> tuple[float, dt.datetime] | None:
        """The contract's last traded close on or before its expiry day --
        `last_trade` as of just past the expiry day's end."""
        boundary = pd.Timestamp(
            dt.datetime.combine(contract.expiry + dt.timedelta(days=1), dt.time(0, 0)), tz=IST
        )
        return self.last_trade(contract, boundary.to_pydatetime())
