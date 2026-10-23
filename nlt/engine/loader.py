"""Turn `spec.instrument.symbol` into concrete bars, whatever kind of name it is.

`Instrument.symbol` is one of three shapes (see `nlt/spec/models.py`): an index
("NIFTY"), a single NSE ticker ("RELIANCE"), or a universe name ("NIFTY 100").
Each is served by a different data source -- `YahooSource` for the index,
`StockSource` for everything else -- and a universe additionally has to be
expanded to its member tickers before `StockSource` can load anything. This
module is the one place that dispatch happens, so `run_basket_backtest` (and
anything else that wants bars for a spec) never has to know which source a
given symbol lives on.

Every data-quality observation -- artefact cleaning, a load failure, a
survivorship caveat -- is returned as a plain string in `notes`, never printed
or swallowed. A basket backtest that quietly dropped a fifth of its universe
to load failures, or that silently ran on a heavily-cleaned feed, would look
like an ordinary result; the caller has to be able to find out and say so.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from nlt.data.stocks import StockSource
from nlt.data.universe import get_universe, resolve_symbols, survivorship_warning
from nlt.data.yahoo import YahooSource
from nlt.spec.models import INDICES, StrategySpec


def load_for_spec(
    spec: StrategySpec,
    start: dt.date | None = None,
    end: dt.date | None = None,
    *,
    tested_from: dt.date | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Resolve `spec.instrument.symbol` to bars, plus every data-quality note.

    Returns `({symbol: bars}, notes)`. A universe expands to its member
    tickers and is loaded through `StockSource.bars_many`, which never lets
    one bad symbol abort the rest of the basket -- a failed symbol is simply
    absent from the returned dict, and its reason is folded into `notes`
    (also available structured, per-symbol, from `StockSource.failures()` if
    a caller wants it). An index or a bare ticker resolves to a single-entry
    dict, so this function works uniformly whether `run_basket_backtest` is
    given one symbol or five hundred.
    """
    notes: list[str] = []
    symbol = spec.instrument.symbol

    # Options are refused here rather than backtested on the wrong series.
    #
    # Until this check existed, `trade_as="option"` fell straight through to the
    # index branch below: a spec whose readback said "NIFTY, at-the-money call
    # option, nearest weekly expiry" was handed NIFTY *spot* bars and bought 65
    # units of the index at ~24,000 each -- about Rs 15.6 lakh on a Rs 1 lakh
    # account -- while "+2% target" silently meant 2% of the index rather than
    # 2% of the premium. The number that came back looked like every other
    # number the dashboard produces.
    #
    # That is the readback stating something untrue, which HANDOFF.md names as
    # the most serious bug class in this project. A missing feature that says so
    # is recoverable; a wrong answer delivered confidently is not.
    #
    # This is the loading path only. `run_backtest` still accepts an option spec
    # with bars passed in directly, which is what the sizing, lot-cap and
    # small-account tests exercise -- those are about position mechanics and do
    # not depend on the series being real option premiums.
    #
    # Removed when the F&O source lands and this function can resolve a real
    # contract. See the plan's Phase 4.
    #
    # Now that the F&O data exists, an option strategy is loaded by
    # `load_option_bars` and run by `nlt.engine.options_backtest`; this
    # function still refuses one, so nothing can reach the index-bars path.
    if spec.instrument.trade_as == "option":
        raise NotImplementedError(
            "An option strategy is tested on the option's own price data, not the "
            "index's – testing it against the index's price would produce a "
            "confident, wrong answer."
        )

    if spec.instrument.is_future:
        return _load_futures(spec, start, end)

    if spec.instrument.timeframe != "1d":
        return _load_intraday(spec, start, end)

    if symbol in INDICES:
        source = YahooSource()
        bars = source.bars(symbol, interval=spec.instrument.timeframe, start=start, end=end)
        return {symbol: bars}, notes

    source = StockSource()

    if spec.instrument.is_universe:
        universe = get_universe(symbol)
        members = resolve_symbols(universe.name)
        # Notes speak of the test the user chose, not of the extra history
        # loaded before it to warm indicators up.
        since = tested_from or start
        if since is not None:
            msg = survivorship_warning(universe, since)
            if msg:
                notes.append(msg)
        bars_by_symbol = source.bars_many(
            members, interval=spec.instrument.timeframe, start=start, end=end
        )
    else:
        bars_by_symbol = source.bars_many(
            [symbol], interval=spec.instrument.timeframe, start=start, end=end
        )

    for sym, reason in source.failures().items():
        notes.append(f"{sym}: failed to load ({reason})")

    # Corporate-action cleaning is summarised rather than listed per symbol.
    # A hundred-stock basket produced thirty-eight notes, all reporting history
    # the 2020 floor removes anyway, which buried the two that actually matter
    # (survivorship, and signals dropped for want of capital). A warning nobody
    # reads is worse than none: it trains the reader to skip the whole section.
    notes.extend(source.stale_notes().values())

    cleaned = source.quality_notes()
    if cleaned:
        affected = sorted(cleaned)
        shown = ", ".join(affected[:5])
        more = f" and {len(affected) - 5} more" if len(affected) > 5 else ""
        notes.append(
            f"Removed unadjusted corporate-action history from "
            f"{len(affected)} symbol(s) ({shown}{more}). Those splits and bonuses "
            "appear in the raw data as real overnight crashes, which would "
            "otherwise be traded as genuine price moves."
        )

    return bars_by_symbol, notes


def _load_intraday(
    spec: StrategySpec, start: dt.date | None, end: dt.date | None
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Intraday candles come from Kite, for one index or one share.

    A basket stays daily-only: `run_basket_backtest` has no square-off, so an
    intraday basket would hold positions overnight. Refused here too, so the
    user hears it before a hundred symbols are downloaded for nothing.
    """
    from nlt.data import kite

    symbol = spec.instrument.symbol
    if spec.instrument.is_universe:
        raise NotImplementedError(
            f"{symbol} is a basket of stocks, and baskets can only be tested on daily "
            "candles for now – the end-of-day square-off is not built for baskets yet."
        )
    if not kite.is_connected():
        raise kite.KiteNotConnected(
            f"{spec.instrument.timeframe} candles come from Kite, and Kite is not connected. "
            "Save your Kite login on the Paper trading page, then run this again."
        )
    source = kite.KiteSource()
    bars = source.bars(symbol, spec.instrument.timeframe, start=start, end=end)
    notes = [
        f"{spec.instrument.timeframe} candles for {symbol} from Kite.",
        *source.last_notes,
    ]
    if spec.instrument.trade_as == "stock":
        notes.append(
            "Intraday share prices from Kite are not adjusted for splits or bonuses; "
            "a split inside the test window would look like a real price move."
        )
    return {symbol: bars}, notes


def _load_futures(
    spec: StrategySpec, start: dt.date | None, end: dt.date | None
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Futures candles come from Kite, daily or intraday, for one contract.

    Daily candles are Kite's continuous series of the nearest-month contract,
    from 2020 like every other history here. Intraday candles exist only for
    the live contract, and `KiteSource` says so in its notes.
    """
    from nlt.data import kite
    from nlt.data.futures import contract, data_key
    from nlt.data.source import DATA_FLOOR

    symbol = spec.instrument.symbol
    c = contract(symbol)
    if not kite.is_connected():
        raise kite.KiteNotConnected(
            f"{c.title} futures prices come from Kite, and Kite is not connected. "
            "Save your Kite login on the Paper trading page, then run this again."
        )
    start = max(start or DATA_FLOOR, DATA_FLOOR)
    source = kite.KiteSource()
    bars = source.bars(data_key(symbol), spec.instrument.timeframe, start=start, end=end)
    notes = [
        f"{c.title} futures ({c.market}), {c.describe_lot}. Prices from Kite.",
        *source.last_notes,
    ]
    return {symbol: bars}, notes


def load_minute_bars(
    spec: StrategySpec, bars: pd.DataFrame
) -> tuple[pd.DataFrame | None, list[str]]:
    """The same instrument's 1-minute candles over `bars`' days, for an
    intraday square-off at the square-off minute's close (Justin, 2026-10-15:
    "assume 1-minute data for all instruments").

    None (with a note) when there is no square-off to price or Kite cannot
    supply them; the engine then squares off at the candle's close and counts
    the days. 1-minute strategies are their own minutes.
    """
    i = spec.instrument
    if i.timeframe == "1d" or not spec.schedule.intraday or bars.empty or i.is_universe:
        return None, []
    if i.timeframe == "1m":
        return bars, []
    from nlt.data import kite

    key = i.symbol
    if i.is_future:
        from nlt.data.futures import data_key

        key = data_key(i.symbol)
    try:
        minutes = kite.KiteSource().bars(
            key, "1m", start=bars.index[0].date(), end=bars.index[-1].date()
        )
    except kite.KiteError as exc:
        return None, [
            f"1-minute prices could not be loaded ({exc}), so each square-off sold at the "
            f"close of the first candle starting at or after {spec.schedule.square_off:%H:%M}."
        ]
    return minutes, []


# --------------------------------------------------------------------- options

# Index timeframes the F&O dataset holds directly, and what the others are
# built from. Unlike an option's, the index's own candles carry no stale
# forward-filled prices (an index ticks every minute), so every row counts.
_OPTION_INDEX_NATIVE = ("1m", "15m", "1d")
_OPTION_INDEX_RESAMPLE = {"3m": ("1m", 3), "5m": ("1m", 5), "30m": ("15m", 30), "1h": ("15m", 60)}


def load_option_bars(spec: StrategySpec, start: dt.date, end: dt.date, source=None) -> pd.DataFrame:
    """The underlying index's candles for an option strategy's signals.

    Read from the same F&O dataset as the option prices, so the index and the
    option share one clock -- and SENSEX, which has no other source here, works.
    """
    from nlt.options.data import ParquetSource
    from nlt.options.execution import _bucket_start

    source = source or ParquetSource()
    symbol, timeframe = spec.instrument.symbol, spec.instrument.timeframe
    if timeframe in _OPTION_INDEX_NATIVE:
        return source.underlying_bars(symbol, timeframe, start, end)
    base, minutes = _OPTION_INDEX_RESAMPLE[timeframe]
    raw = source.underlying_bars(symbol, base, start, end)
    if raw.empty:
        return raw
    grouped = raw.groupby(_bucket_start(raw.index, minutes))  # ty: ignore[invalid-argument-type]
    out = pd.DataFrame(
        {
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    )
    out.index.name = raw.index.name
    return out


def option_window(
    spec: StrategySpec, start: dt.date | None, end: dt.date | None, source=None
) -> tuple[dt.date, dt.date, list[str]]:
    """Clamp a test window to the days with real option prices, saying so.

    Option prices exist only for the dataset's dates (measured from the
    1-minute files, never typed in). A test asking for more is cut back to
    them, and the note says exactly what was tested -- never silently.
    """
    from nlt.options.data import ParquetSource

    source = source or ParquetSource()
    symbol = spec.instrument.symbol
    cover = source.coverage(symbol, spec.instrument.timeframe)
    if cover is None:
        raise ValueError(f"There are no option prices for {symbol} to test against.")
    first, last = cover
    notes: list[str] = []
    start = start or first
    end = end or last
    if end > last:
        notes.append(
            f"Option prices are only available up to {last:%d %b %Y}, so this test ends there."
        )
        end = last
    if start < first:
        notes.append(
            f"{symbol} option prices start on {first:%d %b %Y}, so this test starts there."
        )
        start = first
    if start > end:
        raise ValueError(
            f"There are no {symbol} option prices for those dates: they cover "
            f"{first:%d %b %Y} to {last:%d %b %Y}."
        )
    return start, end, notes
