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

    if symbol in INDICES:
        source = YahooSource()
        bars = source.bars(symbol, interval=spec.instrument.timeframe, start=start, end=end)
        return {symbol: bars}, notes

    source = StockSource()

    if spec.instrument.is_universe:
        universe = get_universe(symbol)
        members = resolve_symbols(universe.name)
        if start is not None:
            msg = survivorship_warning(universe, start)
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
    for sym, symbol_notes in source.quality_notes().items():
        for note in symbol_notes:
            notes.append(f"{sym}: {note}")

    return bars_by_symbol, notes
