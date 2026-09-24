"""Detecting price history that is wrong rather than merely surprising.

Yahoo's Indian equity history is not adjusted for corporate actions before
roughly 2010. A 1:10 bonus in 2005 appears as a genuine 90% overnight collapse,
and 21 of the 50 NIFTY 50 constituents carry at least one such artefact --
BAJFINANCE shows -99.1% on 2005-07-29, as do several unrelated stocks on that
same date, which is a bulk data fault rather than a real event.

This matters more than it sounds. A backtest does not know the crash is fake.
RSI plunges, a "buy the dip" rule fires, a percentage stop is blown through, and
the resulting trade is pure fiction -- reported with the same confidence as
every real one. Worse, it is *profitable* fiction: the price recovers to its
pre-artefact level the next day, so the strategy appears to have caught a
once-in-a-decade bounce.

`auto_adjust=True` does not help; the underlying split data is simply absent.
So the only honest options are to refuse the affected history or to flag it
loudly. We do the first by default: a backtest silently including these bars is
worse than a shorter backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# A liquid large cap does not move more than this in one session without it
# being a corporate action or a data fault. India's 20% circuit limit on
# individual stocks makes anything beyond ~35% structurally impossible, so a
# larger move is evidence of bad data rather than a dramatic day.
ARTEFACT_THRESHOLD = 0.35


@dataclass(frozen=True)
class Artefact:
    """A single-session move too large to be real."""

    timestamp: pd.Timestamp
    change_pct: float
    ratio: float  # implied corporate-action ratio, e.g. ~2.0 for a 1:1 bonus

    def __str__(self) -> str:
        return (
            f"{self.timestamp.date()}: {self.change_pct:+.1f}% in one session "
            f"(looks like an unadjusted {self.ratio:.0f}:1 corporate action)"
        )


def detect_artefacts(
    bars: pd.DataFrame, threshold: float = ARTEFACT_THRESHOLD
) -> list[Artefact]:
    """Sessions whose close-to-close move is too large to be a real price move."""
    if bars.empty or len(bars) < 2:
        return []

    change = bars["close"].pct_change()
    flagged = change[(change < -threshold) | (change > threshold / (1 - threshold))]

    out = []
    for ts, pct in flagged.items():
        ratio = 1.0 / (1.0 + pct) if pct < 0 else 1.0 + pct
        out.append(Artefact(ts, 100.0 * float(pct), float(ratio)))
    return out


def usable_from(
    bars: pd.DataFrame, threshold: float = ARTEFACT_THRESHOLD
) -> pd.Timestamp | None:
    """The first timestamp after the last artefact, or None if there are none.

    Everything before the final artefact is suspect: an unadjusted split rescales
    the whole series before it, so the bars are not merely interrupted, they are
    on a different scale entirely.
    """
    artefacts = detect_artefacts(bars, threshold)
    if not artefacts:
        return None

    last = max(a.timestamp for a in artefacts)
    after = bars.index[bars.index > last]
    return after[0] if len(after) else None


def clean(
    bars: pd.DataFrame, symbol: str = "", threshold: float = ARTEFACT_THRESHOLD
) -> tuple[pd.DataFrame, list[str]]:
    """Drop history up to and including the last artefact. Returns bars and notes.

    Never drops silently: a caller that loses fifteen years of history must be
    able to find out why, and say so to the user.
    """
    artefacts = detect_artefacts(bars, threshold)
    if not artefacts:
        return bars, []

    start = usable_from(bars, threshold)
    label = f"{symbol}: " if symbol else ""

    if start is None:
        return bars.iloc[0:0], [
            f"{label}every bar precedes a corporate-action artefact "
            f"({artefacts[-1]}); no usable history"
        ]

    dropped = int((bars.index < start).sum())
    notes = [
        f"{label}dropped {dropped:,} bars before {start.date()} -- "
        f"{len(artefacts)} unadjusted corporate action(s) make the earlier "
        f"history a different price scale. Last one was {artefacts[-1]}"
    ]
    return bars[bars.index >= start], notes
