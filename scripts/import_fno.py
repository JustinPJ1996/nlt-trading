#!/usr/bin/env python
"""Import the historical index F&O dataset to Parquet under data/fno/.

    .venv/bin/python scripts/import_fno.py --check      # report only, writes nothing
    .venv/bin/python scripts/import_fno.py               # import 1d + 15m (the default)
    .venv/bin/python scripts/import_fno.py --timeframe 1m   # opt-in; needs decompressing first

See `nlt/data/fno_import.py` for the two mandatory unit conversions (paise,
UTC->IST), the data-quality flags this deliberately does not filter on, and
why the bid/ask must never be used as a fill price.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from nlt.data import fno_import as fi


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeframe",
        action="append",
        choices=fi.TIMEFRAMES,
        dest="timeframes",
        help="repeatable; defaults to 1d and 15m (1m is opt-in, see module docstring)",
    )
    parser.add_argument(
        "--underlying",
        action="append",
        choices=fi.UNDERLYINGS,
        dest="underlyings",
        help="repeatable; defaults to all five",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report row counts, date ranges and defect counts; write nothing",
    )
    parser.add_argument("--raw-dir", type=Path, default=fi.RAW_DIR)
    parser.add_argument("--out", type=Path, default=fi.OUTPUT_DIR)
    args = parser.parse_args()

    timeframes = tuple(args.timeframes) if args.timeframes else fi.DEFAULT_TIMEFRAMES
    underlyings = tuple(args.underlyings) if args.underlyings else fi.UNDERLYINGS

    if args.check:
        for timeframe in timeframes:
            print(fi.check(timeframe, raw_dir=args.raw_dir))
            print()
        return

    for timeframe in timeframes:
        print(f"Importing {timeframe} -> {args.out / timeframe}")
        notes = fi.import_timeframe(
            timeframe, raw_dir=args.raw_dir, out_dir=args.out, underlyings=underlyings
        )
        for note in notes:
            print(f"  {note}")
        print()


if __name__ == "__main__":
    main()
