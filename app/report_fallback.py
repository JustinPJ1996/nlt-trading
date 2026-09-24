"""Stand-in for `nlt.report` until the concurrent agent building it lands.

`app/logic.py` imports `buy_and_hold`, `compare`, `assess` and `render` from
`nlt.report.*` first, and only falls back to the plain functions in this file
on `ImportError`. The public shapes here (`Benchmark`, `Comparison`, `Verdict`,
`Flag`) mirror the interface documented in the build brief exactly, so the
dashboard code that consumes them (`app/logic.py`, `app/pages_*`) never has to
change once the real package exists -- the `try/except ImportError` in
`app/logic.py` just stops choosing this module.

Nothing here is trying to out-engineer the real implementation. The benchmark
is a single buy-and-hold of the same instrument over the same bars, and the
verdict is a handful of the most beginner-dangerous checks (no trades, lost
money, badly lagged a passive buy-and-hold, a single trade decided everything,
a stop loss that never existed in the data window). It is enough to make the
dashboard honest with an empty `nlt.report`; it is not the final word on what
"proven" means.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nlt.engine.backtest import BacktestResult


@dataclass(frozen=True)
class Benchmark:
    """A plain buy-and-hold of the same instrument, over the same bars."""

    equity: pd.Series
    total_return_pct: float
    max_drawdown_pct: float


@dataclass(frozen=True)
class Comparison:
    """How the strategy stacks up against the benchmark."""

    excess_return_pct: float
    beat_benchmark: bool
    time_in_market_pct: float


@dataclass(frozen=True)
class Flag:
    severity: str  # "critical" | "warning" | "info"
    code: str
    headline: str
    detail: str


@dataclass(frozen=True)
class Verdict:
    summary: str
    passed: bool
    flags: list[Flag] = field(default_factory=list)


def buy_and_hold(
    bars: pd.DataFrame,
    capital: float,
    *,
    charge_fn: Callable[[float, int, str], float] | None = None,
) -> Benchmark:
    """Buy as much as `capital` affords at the first bar's open, hold to the last close."""
    if bars.empty:
        flat = pd.Series([capital], index=[pd.Timestamp.now()], name="equity")
        return Benchmark(equity=flat, total_return_pct=0.0, max_drawdown_pct=0.0)

    entry_price = float(bars.iloc[0].open)
    quantity = int(capital // entry_price) if entry_price > 0 else 0
    charges = 0.0
    if charge_fn is not None and quantity > 0:
        exit_price = float(bars.iloc[-1].close)
        charges = charge_fn(entry_price, quantity, "buy") + charge_fn(exit_price, quantity, "sell")

    cash_left = capital - quantity * entry_price
    equity = (cash_left + quantity * bars["close"]).astype("float64")
    equity.iloc[-1] -= charges
    equity.name = "equity"

    total_return_pct = 100.0 * (equity.iloc[-1] - capital) / capital if capital else 0.0
    running_max = equity.cummax()
    drawdown_pct = 100.0 * (equity - running_max) / running_max.replace(0, np.nan)
    max_drawdown_pct = float(-drawdown_pct.min()) if len(drawdown_pct) else 0.0

    return Benchmark(
        equity=equity,
        total_return_pct=float(total_return_pct),
        max_drawdown_pct=max_drawdown_pct,
    )


def compare(result: BacktestResult, benchmark: Benchmark, bars: pd.DataFrame) -> Comparison:
    strategy_return = float(result.metrics.get("total_return_pct") or 0.0)
    excess = strategy_return - benchmark.total_return_pct

    bars_in_market = 0
    for t in result.trades:
        bars_in_market += max(int(t.bars_held), 0)
    time_in_market_pct = 100.0 * bars_in_market / len(bars) if len(bars) else 0.0

    return Comparison(
        excess_return_pct=excess,
        beat_benchmark=excess > 0,
        time_in_market_pct=min(time_in_market_pct, 100.0),
    )


def assess(result: BacktestResult, comparison: Comparison) -> Verdict:
    flags: list[Flag] = []
    m = result.metrics
    total_trades = int(m.get("total_trades") or 0)
    total_return = float(m.get("total_return_pct") or 0.0)

    if total_trades == 0:
        flags.append(
            Flag(
                severity="critical",
                code="no_trades",
                headline="This strategy never traded",
                detail=(
                    "Over the whole period tested, the entry condition never fired. "
                    "There is nothing here to judge yet -- try a longer date range or "
                    "a less restrictive entry rule."
                ),
            )
        )
    elif total_trades < 10:
        flags.append(
            Flag(
                severity="warning",
                code="few_trades",
                headline=f"Only {total_trades} trade(s) in this test",
                detail=(
                    "A handful of trades can look great or terrible by luck alone. "
                    "Treat this result as a rough sketch, not a proven edge."
                ),
            )
        )

    if total_trades and total_return < 0:
        flags.append(
            Flag(
                severity="critical",
                code="lost_money",
                headline="This strategy lost money over the period tested",
                detail=f"Total return was {total_return:.1f}%.",
            )
        )

    if total_trades and not comparison.beat_benchmark:
        flags.append(
            Flag(
                severity="warning",
                code="underperformed_benchmark",
                headline="Just buying and holding did better than this strategy",
                detail=(
                    f"This strategy was {abs(comparison.excess_return_pct):.1f} percentage "
                    "points behind simply buying the index and doing nothing."
                ),
            )
        )

    max_dd = float(m.get("max_drawdown_pct") or 0.0)
    if max_dd >= 20:
        flags.append(
            Flag(
                severity="warning",
                code="deep_drawdown",
                headline="This strategy had a large drop from its peak at some point",
                detail=f"The biggest drop from a peak was {max_dd:.1f}%.",
            )
        )

    passed = not any(f.severity == "critical" for f in flags)
    if not total_trades:
        summary = "Not enough happened to judge this strategy."
    elif passed and comparison.beat_benchmark:
        summary = "This strategy made money and beat just buying and holding."
    elif passed:
        summary = "This strategy made money, but so did doing nothing clever."
    else:
        summary = "This strategy needs work before you should trust it with real money."

    return Verdict(summary=summary, passed=passed, flags=flags)


def render(result: BacktestResult, comparison: Comparison, verdict: Verdict) -> str:
    lines = [verdict.summary, ""]
    for f in verdict.flags:
        lines.append(f"[{f.severity.upper()}] {f.headline}")
        lines.append(f"  {f.detail}")
    lines.append("")
    lines.append(f"Strategy return: {result.metrics.get('total_return_pct', 0):.2f}%")
    lines.append(f"Excess return vs. buy-and-hold: {comparison.excess_return_pct:.2f}%")
    return "\n".join(lines)
