#!/usr/bin/env python
"""Populate the local bar cache.

Run this once before backtesting, and whenever you want fresh data:

    .venv/bin/python scripts/fetch_data.py

Yahoo is a stopgap for daily bars. Once Kite Connect is subscribed, a KiteSource
implementing the same `BarSource` protocol replaces it and unlocks intraday.
"""

from __future__ import annotations

import argparse

from nlt.data.yahoo import SYMBOL_MAP, YahooSource, refresh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbols", nargs="*", default=list(SYMBOL_MAP))
    parser.add_argument("--force", action="store_true", help="re-download, ignoring the cache")
    args = parser.parse_args()

    for symbol in args.symbols or list(SYMBOL_MAP):
        df = refresh(symbol) if args.force else YahooSource().bars(symbol)
        first, last = df.index[0].date(), df.index[-1].date()
        print(f"{symbol:10s} {len(df):6,d} bars   {first} -> {last}")


if __name__ == "__main__":
    main()
