"""Live option prices for paper trading, from Kite, kept on disk as they arrive.

`LiveOptionSource` answers the same four questions as `ParquetSource` (the
historical F&O dataset), so a paper run picks contracts and fills trades with
exactly the code a backtest uses (`nlt.options.execution`).

## Why it has to keep what it reads

Kite keeps an option's candles only while the contract is live. Once a weekly
option expires its history is gone from Kite -- and its instrument token is
later reused for a different contract. The list of contracts is the same: Kite
publishes only today's. But a paper run re-derives its whole history on every
pass (`nlt.paper.runner`), so it needs, for every past day of the run, the
options that were listed that day and the prices of every contract it chose.

So this module saves, under `data/fno_live/`:

  * `listing/<UNDERLYING>/<YYYY-MM-DD>.csv` -- that day's listed options, taken
    from Kite's list the first time a pass asks on that day;
  * `bars/<UNDERLYING>/<TRADINGSYMBOL>/<YYYY-MM-DD>.parquet` -- one contract's
    1-minute candles for one finished day. An empty file means "asked for, and
    nothing was there" (a holiday, or a day it never traded).

Justin's choice (2026-10-15): only the contracts paper runs actually use are
kept, not the whole chain. A contract's day is saved by the first pass after
that day has finished, and -- since 2026-10-16 -- every pass during the day
also keeps the minutes so far (`<YYYY-MM-DD>.partial.parquet`). If no pass
runs between the close and Kite dropping the contract, the day is kept up to
the last pass and the source says where it stops (`cut_short`); with no pass
at all that day it is lost (`lost`). Nothing is ever invented.

## The same traded-only rule as the dataset

Kite's 1-minute candles include minutes with zero volume, priced at the last
trade (checked 2026-10-15: NIFTY26O1924500CE had 22 of 375 minutes at volume
0). They are kept but marked untraded, exactly like the dataset's
forward-filled rows (`nlt.options.data.traded_mask`), and every longer candle
is built from traded minutes only.

## Never a minute that has not finished

Every candle returned ends at or before `now`. The engine is only ever shown
closed index candles, but a square-off lookback (`last_trade`) or a daily
strategy's 15:15 price reads the minute prices directly; clipping here means
none of them can see a minute that is still forming.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pandas as pd

from nlt.data.source import IST
from nlt.options.execution import BAR_COLUMNS

LIVE_ROOT = (
    Path(os.environ["NLT_FNO_LIVE_ROOT"])
    if os.environ.get("NLT_FNO_LIVE_ROOT")
    else Path(__file__).resolve().parents[2] / "data" / "fno_live"
)

UNDERLYINGS = ("NIFTY", "BANKNIFTY", "SENSEX")

# A day's minutes are complete once its last minute (15:39-15:40, see
# `nlt.data.session.NSE_FNO`) has closed; two minutes' margin for Kite.
DAY_COMPLETE_AT = dt.time(15, 42)

_LISTING_COLUMNS = ["tradingsymbol", "instrument_type", "expiry", "strike"]
_MINUTE = pd.Timedelta(minutes=1)


class KiteOptionClient(Protocol):
    """What this module needs from Kite. `KiteClient` is the real one; tests
    pass one built from the historical dataset."""

    def instruments(self, now: pd.Timestamp) -> pd.DataFrame:
        """Today's live options for every index, shaped like
        `nlt.data.kite.option_contracts` plus a `name` (underlying) column."""

    def minute_candles(self, token: int, start: dt.date, end: dt.date) -> pd.DataFrame:
        """Kite's 1-minute candles (open, high, low, close, volume), IST index."""

    def index_bars(
        self, symbol: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame: ...


class KiteClient:
    """The real `KiteOptionClient`: `nlt.data.kite`, read-only like everything there."""

    def __init__(self) -> None:
        from nlt.data.kite import KiteSource

        self._source = KiteSource()

    def instruments(self, now: pd.Timestamp) -> pd.DataFrame:
        from nlt.data import kite

        everything = kite.todays_instruments(now)
        frames = []
        for underlying in UNDERLYINGS:
            rows = kite.option_contracts(underlying, everything)
            frames.append(rows.assign(name=underlying))
        return pd.concat(frames, ignore_index=True)

    def minute_candles(self, token: int, start: dt.date, end: dt.date) -> pd.DataFrame:
        from nlt.data.session import NSE_FNO

        # F&O minutes run to 15:40; the equity session would cut the last ten.
        return self._source.contract_bars(token, "1m", start, end, session=NSE_FNO)

    def index_bars(self, symbol: str, timeframe: str, start: dt.date, end: dt.date):
        return self._source.bars(symbol, timeframe, start, end)


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=BAR_COLUMNS).set_index(
        pd.DatetimeIndex([], tz=IST, name="timestamp")
    )


def _shape(candles: pd.DataFrame) -> pd.DataFrame:
    """Kite's candles in the dataset's contract-bar shape (`BAR_COLUMNS`).

    A minute counts as traded only if something traded in it (volume above
    zero) and its prices make sense -- the dataset's own rule. Kite gives no
    bid, ask or open interest here; they are left empty (nothing reads them).
    """
    if candles is None or candles.empty:
        return _empty_bars()
    out = candles[["open", "high", "low", "close", "volume"]].astype(float).copy()
    sane = (
        (out["low"] > 0)
        & (out["high"] >= out["low"])
        & out["open"].between(out["low"], out["high"])
        & out["close"].between(out["low"], out["high"])
    )
    out["oi"] = float("nan")
    out["bid"] = float("nan")
    out["ask"] = float("nan")
    out["traded"] = (out["volume"] > 0) & sane
    out.index = pd.DatetimeIndex(out.index).tz_convert(IST)
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out[BAR_COLUMNS]


def _weekdays(start: dt.date, end: dt.date) -> list[dt.date]:
    days = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += dt.timedelta(days=1)
    return days


class LiveOptionSource:
    """`OptionSource` over Kite's live options, as of `now`, saving as it reads.

    One is made per paper pass: `now` fixes what "finished" means for the pass.
    `lost` and `missing_listing_days` collect what could not be answered, so the
    runner can say so in plain words.
    """

    def __init__(
        self, client: KiteOptionClient, now: pd.Timestamp, root: Path | None = None
    ) -> None:
        self.client = client
        self.now = pd.Timestamp(now).tz_convert(IST)
        self.today = self.now.date()
        self.root = Path(root) if root is not None else LIVE_ROOT
        self.lost: set[tuple[str, dt.date]] = set()
        # Days kept only up to a minute (saved part-way through, then Kite
        # dropped the contract): (tradingsymbol, day, last saved minute's end).
        self.cut_short: set[tuple[str, dt.date, dt.time]] = set()
        self.missing_listing_days: set[dt.date] = set()
        self._live: pd.DataFrame | None = None
        self._listings: dict[tuple[str, dt.date], pd.DataFrame] = {}
        self._bars: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------- listing

    def _listing_path(self, underlying: str, day: dt.date) -> Path:
        return self.root / "listing" / underlying / f"{day.isoformat()}.csv"

    def _live_contracts(self) -> pd.DataFrame:
        """Today's Kite list, read once per pass; saved as today's listing."""
        if self._live is None:
            live = self.client.instruments(self.now)
            for underlying, rows in live.groupby("name"):
                path = self._listing_path(str(underlying), self.today)
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    rows[_LISTING_COLUMNS].to_csv(tmp, index=False)
                    tmp.replace(path)
            self._live = live
        return self._live

    def save_todays_listing(self) -> None:
        """Make sure today's list of options is on disk (one public download)."""
        self._live_contracts()

    def listing(self, underlying: str, day: dt.date) -> pd.DataFrame:
        key = (underlying, day)
        if key in self._listings:
            return self._listings[key]
        path = self._listing_path(underlying, day)
        if day == self.today and not path.exists():
            self._live_contracts()
        if path.exists():
            df = pd.read_csv(path)
            df["expiry"] = pd.to_datetime(df["expiry"]).dt.date
            df["strike"] = df["strike"].astype(float)
            df = df[_LISTING_COLUMNS]
        else:
            if _weekdays(day, day) and day <= self.today:
                self.missing_listing_days.add(day)
            df = pd.DataFrame(columns=_LISTING_COLUMNS)
        self._listings[key] = df
        return df

    def coverage(self, underlying: str, timeframe: str) -> tuple[dt.date, dt.date] | None:
        folder = self.root / "listing" / underlying
        days = sorted(p.stem for p in folder.glob("*.csv")) if folder.exists() else []
        if not days:
            return None
        return dt.date.fromisoformat(days[0]), self.today

    def underlying_bars(
        self, underlying: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        return self.client.index_bars(underlying, timeframe, start, min(end, self.today))

    # ---------------------------------------------------------------- bars

    def _bars_path(self, underlying: str, tradingsymbol: str, day: dt.date) -> Path:
        return self.root / "bars" / underlying / tradingsymbol / f"{day.isoformat()}.parquet"

    def _partial_path(self, underlying: str, tradingsymbol: str, day: dt.date) -> Path:
        """Today's minutes so far, rewritten by every pass (2026-10-16): if
        the box is off from the close until Kite drops an expiring option,
        only the minutes after the last pass are lost, not the whole day."""
        return self._bars_path(underlying, tradingsymbol, day).with_suffix(".partial.parquet")

    def _day_complete(self, day: dt.date) -> bool:
        return day < self.today or (day == self.today and self.now.time() >= DAY_COMPLETE_AT)

    def _token(self, underlying: str, tradingsymbol: str) -> int | None:
        """Kite's token for a contract that is live today, else None.

        Looked up in TODAY's list every time, never remembered: Kite reuses an
        expired contract's token for a new one, so an old token could fetch
        some other contract's prices.
        """
        live = self._live_contracts()
        rows = live[(live["name"] == underlying) & (live["tradingsymbol"] == tradingsymbol)]
        return int(rows["instrument_token"].iloc[0]) if len(rows) == 1 else None

    def contract_bars(
        self, underlying: str, tradingsymbol: str, timeframe: str, start: dt.date, end: dt.date
    ) -> pd.DataFrame:
        if timeframe != "1m":
            raise ValueError("LiveOptionSource reads 1-minute candles only, like the dataset")
        end = min(end, self.today)
        days = _weekdays(start, end)
        if not days:
            return _empty_bars()

        cached = self._bars.get(tradingsymbol)
        if cached is None or cached.attrs["from"] > days[0] or cached.attrs["to"] < days[-1]:
            first = min(days[0], cached.attrs["from"]) if cached is not None else days[0]
            last = max(days[-1], cached.attrs["to"]) if cached is not None else days[-1]
            cached = self._load(underlying, tradingsymbol, first, last)
            cached.attrs["from"], cached.attrs["to"] = first, last
            self._bars[tradingsymbol] = cached
        lo = pd.Timestamp(dt.datetime.combine(start, dt.time(0, 0)), tz=IST)
        hi = pd.Timestamp(dt.datetime.combine(end, dt.time(23, 59, 59)), tz=IST)
        return cached[(cached.index >= lo) & (cached.index <= hi)]

    def _load(
        self, underlying: str, tradingsymbol: str, first: dt.date, last: dt.date
    ) -> pd.DataFrame:
        """Every minute from `first` to `last` (never past now): saved days from
        disk, the rest from Kite. Callers stop at the contract's expiry, so a
        day after it is never asked for."""
        days = _weekdays(first, last)
        frames: list[pd.DataFrame] = []
        wanted: list[dt.date] = []
        for day in days:
            path = self._bars_path(underlying, tradingsymbol, day)
            if path.exists():
                frames.append(pd.read_parquet(path))
            else:
                wanted.append(day)

        if wanted:
            token = self._token(underlying, tradingsymbol)
            if token is None:
                # Expired (or never seen live): Kite no longer has these days.
                # What an earlier pass saved part-way through the day is used,
                # and said to stop where it stops; nothing is filled in.
                for day in wanted:
                    partial = self._partial_path(underlying, tradingsymbol, day)
                    if partial.exists():
                        part = pd.read_parquet(partial)
                        frames.append(part)
                        end = (part.index.max() + _MINUTE).time() if len(part) else dt.time(9, 15)
                        self.cut_short.add((tradingsymbol, day, end))
                    else:
                        self.lost.add((tradingsymbol, day))
            else:
                fetched = _shape(self.client.minute_candles(token, wanted[0], wanted[-1]))
                for day in wanted:
                    part = fetched[fetched.index.date == day]
                    if self._day_complete(day):
                        path = self._bars_path(underlying, tradingsymbol, day)
                        _write(part, path)
                        self._partial_path(underlying, tradingsymbol, day).unlink(missing_ok=True)
                    else:
                        _write(
                            part[part.index + _MINUTE <= self.now],
                            self._partial_path(underlying, tradingsymbol, day),
                        )
                    frames.append(part)

        frames = [f for f in frames if not f.empty]
        if not frames:
            return _empty_bars()
        out = pd.concat(frames).sort_index()
        out = out[~out.index.duplicated(keep="last")]
        # Only minutes that have finished by `now`.
        out = out[out.index + _MINUTE <= self.now]
        out["traded"] = out["traded"].astype(bool)
        return out[BAR_COLUMNS]


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)


# ------------------------------------------------------------- India VIX


def vix_iv_fn(
    client: KiteOptionClient, now: pd.Timestamp
) -> Callable[[str, pd.DataFrame, int], pd.Series]:
    """An `iv_fn` for the legs engine that ranks India VIX instead of our own IV.

    Justin's choice (2026-10-15): there are no option prices from January to
    mid-October 2026, so the past year of our own ATM implied volatility cannot
    be built for a paper run. India VIX is published every day and measures
    NIFTY's implied volatility, so a NIFTY strategy's "IV percentile" is read
    from it instead -- with the same rule as `nlt.options.iv.iv_percentile`:
    day D uses D-1's close, ranked against the sessions before it. Justin
    (2026-10-15, later the same day): for BANKNIFTY and SENSEX too, though VIX
    is built from NIFTY's options -- `app.logic.paper_notes` says so.
    """
    from nlt.options.iv import iv_feature

    now = pd.Timestamp(now).tz_convert(IST)

    def iv_fn(underlying: str, bars: pd.DataFrame, lookback_days: int) -> pd.Series:
        first = bars.index.min().date() if len(bars) else now.date()
        start = first - dt.timedelta(days=int(lookback_days * 1.6) + 30)
        vix = client.index_bars("INDIA VIX", "1d", start, now.date())
        closes = pd.Series(
            vix["close"].to_numpy(dtype=float),
            index=pd.DatetimeIndex(vix.index).tz_convert(IST).normalize(),
        )
        # Today's VIX candle is still forming until the close; it is never used
        # for today anyway (D uses D-1), so it is left out rather than trusted.
        closes = closes[closes.index.date < now.date()]
        sessions = sorted(set(closes.index.date) | set(bars.index.date))
        daily = closes.reindex(pd.DatetimeIndex([pd.Timestamp(d, tz=IST) for d in sessions]))
        return iv_feature(underlying, bars, lookback_days, daily_iv=daily)

    return iv_fn
