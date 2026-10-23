#!/usr/bin/env python
"""Work out each index's daily at-the-money IV and save it for the engine.

    .venv/bin/python scripts/build_iv.py                      # NIFTY, BANKNIFTY, SENSEX
    .venv/bin/python scripts/build_iv.py --underlying NIFTY   # just one
    .venv/bin/python scripts/build_iv.py --check-vix          # also compare NIFTY to India VIX

Writes `data/fno/iv/underlying=U.parquet` (git-ignored, like all of
`data/`). Reads the 1-minute option file, so it takes a few minutes per index.
Rebuild it after the dataset changes or `nlt.options.iv.METHOD_VERSION` is
bumped -- an out-of-date file is ignored, and the IV is then worked out from
scratch on every run, slowly. See `nlt/options/iv.py` for every rule.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nlt.options import data as option_data  # noqa: E402
from nlt.options import iv  # noqa: E402

UNDERLYINGS = ("NIFTY", "BANKNIFTY", "SENSEX")


def build(underlying: str) -> None:
    span = option_data.coverage(underlying, "1m")
    if span is None:
        print(f"{underlying}: no option prices on disk, skipped")
        return
    started = time.monotonic()
    frame = iv.daily_atm_iv(underlying, span[0], span[1], use_cache=False)
    path = iv.write_cache(underlying, frame)
    took = time.monotonic() - started

    missing = frame["iv"].isna()
    print(f"{underlying}: {len(frame)} sessions {span[0]} to {span[1]} in {took:.0f}s -> {path}")
    print(f"  legs: {frame['legs'].value_counts().to_dict()}")
    print(f"  sessions with no IV: {int(missing.sum())}")
    for date, row in frame[missing | (frame["legs"] != iv.LEGS_BOTH)].iterrows():
        print(f"    {date.date()} {row['legs']}: {row['reason']}")
    first = iv.iv_percentile(frame["iv"], 252).first_valid_index()
    print(f"  first 252-day percentile: {first.date() if first is not None else 'none'}")  # ty: ignore[unresolved-attribute]


def check_vix() -> None:
    span = option_data.coverage("NIFTY", "1m")
    if span is None:
        print("NIFTY: no option prices on disk, so there is nothing to compare to India VIX")
        return
    frame = iv.daily_atm_iv("NIFTY", span[0], span[1])
    first, last = frame.index[0].date(), frame.index[-1].date()
    vix = option_data.underlying_bars("INDIAVIX", "1d", first, last)["close"]
    both = frame[["iv"]].join(vix.rename("vix"), how="inner").dropna()
    both["iv"] *= 100
    gap = both["iv"] - both["vix"]
    print(f"NIFTY ATM IV vs India VIX, {len(both)} sessions:")
    print(f"  correlation of levels: {both['iv'].corr(both['vix']):.3f}")
    print(f"  correlation of daily changes: {both['iv'].diff().corr(both['vix'].diff()):.3f}")
    print(f"  mean IV {both['iv'].mean():.2f}, mean VIX {both['vix'].mean():.2f}")
    print(
        f"  gap IV - VIX: median {gap.median():+.2f}, middle half "
        f"{gap.quantile(0.25):+.2f} to {gap.quantile(0.75):+.2f}, "
        f"median absolute {gap.abs().median():.2f} points"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--underlying", action="append", choices=UNDERLYINGS, dest="underlyings")
    parser.add_argument("--check-vix", action="store_true")
    args = parser.parse_args()
    for underlying in args.underlyings or UNDERLYINGS:
        build(underlying)
    if args.check_vix:
        check_vix()


if __name__ == "__main__":
    main()
