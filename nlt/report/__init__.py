"""Turns a raw `BacktestResult` into something a non-technical user can act on.

`nlt.engine` answers "what happened if you ran this strategy." This package
answers the two questions a beginner actually needs answered before risking
money on that result: "is that good, compared to doing nothing clever at
all?" (`benchmark`) and "which of the well-known ways a backtest lies is this
one exhibiting?" (`verdict`). `text` renders both into a plain-text report.
"""

from __future__ import annotations
