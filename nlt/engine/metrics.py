"""Summary statistics for a finished backtest.

Every ratio here divides by something that can be zero -- no trades, no losers,
a flat equity curve -- so each guard exists because an earlier draft actually
hit that path. A strategy that never fired a single trade is not a bug, it is
one of the most common results a beginner will get, and it must produce a
readable dict of zeros, not a stack trace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nlt.engine.backtest import Trade

METRIC_KEYS = (
    "total_return_pct",
    "cagr_pct",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "max_drawdown_duration_days",
    "win_rate_pct",
    "profit_factor",
    "avg_win",
    "avg_loss",
    "expectancy",
    "total_trades",
    "winning_trades",
    "losing_trades",
    "avg_bars_held",
    "total_charges",
    "exposure_pct",
    "best_trade",
    "worst_trade",
    "max_consecutive_losses",
)


def _empty_metrics() -> dict:
    zeros = {k: 0 for k in METRIC_KEYS}
    zeros["profit_factor"] = None
    zeros["sharpe"] = None
    zeros["sortino"] = None
    zeros["max_drawdown_duration_days"] = 0
    return zeros


def compute_metrics(
    trades: list["Trade"],
    equity: pd.Series,
    capital: float,
    bars_per_year: int,
) -> dict:
    if not trades:
        metrics = _empty_metrics()
        if len(equity) and capital > 0:
            metrics["total_return_pct"] = 100.0 * (equity.iloc[-1] - capital) / capital
        return metrics

    pnls = np.array([t.net_pnl for t in trades], dtype="float64")
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    winning_trades = int(len(wins))
    losing_trades = int(len(losses))
    total_trades = len(trades)

    win_rate_pct = 100.0 * winning_trades / total_trades if total_trades else 0.0
    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(-losses.sum()) if len(losses) else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else float("inf"))

    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    expectancy = float(pnls.mean())

    total_return_pct = 100.0 * (equity.iloc[-1] - capital) / capital if capital > 0 and len(equity) else 0.0

    years = len(equity) / bars_per_year if bars_per_year > 0 else 0.0
    if years > 0 and capital > 0 and equity.iloc[-1] > 0:
        cagr_pct = 100.0 * ((equity.iloc[-1] / capital) ** (1.0 / years) - 1.0)
    else:
        cagr_pct = 0.0

    bar_returns = equity.pct_change().dropna()
    if len(bar_returns) > 1 and bar_returns.std(ddof=0) > 0:
        sharpe = float(bar_returns.mean() / bar_returns.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sharpe = None

    downside = bar_returns[bar_returns < 0]
    if len(bar_returns) > 1 and len(downside) and downside.std(ddof=0) > 0:
        sortino = float(bar_returns.mean() / downside.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sortino = None

    running_max = equity.cummax()
    drawdown_pct = 100.0 * (equity - running_max) / running_max.replace(0.0, np.nan)
    max_drawdown_pct = float(drawdown_pct.min()) if len(drawdown_pct) else 0.0
    max_drawdown_duration_days = _max_drawdown_duration_days(equity, running_max)

    avg_bars_held = float(np.mean([t.bars_held for t in trades]))
    total_charges = float(sum(t.charges for t in trades))

    bars_in_position = sum(t.bars_held for t in trades)
    exposure_pct = 100.0 * bars_in_position / len(equity) if len(equity) else 0.0

    best_trade = float(pnls.max())
    worst_trade = float(pnls.min())
    max_consecutive_losses = _max_consecutive_losses(pnls)

    return {
        "total_return_pct": total_return_pct,
        "cagr_pct": cagr_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_duration_days": max_drawdown_duration_days,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": expectancy,
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "avg_bars_held": avg_bars_held,
        "total_charges": total_charges,
        "exposure_pct": exposure_pct,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "max_consecutive_losses": max_consecutive_losses,
    }


def _max_drawdown_duration_days(equity: pd.Series, running_max: pd.Series) -> int:
    """Longest stretch, in calendar days, spent below a prior equity high."""
    underwater = equity < running_max
    if not underwater.any():
        return 0

    longest = 0
    start = None
    for ts, is_under in underwater.items():
        if is_under and start is None:
            start = ts
        elif not is_under and start is not None:
            longest = max(longest, (ts - start).days)
            start = None
    if start is not None:
        longest = max(longest, (equity.index[-1] - start).days)
    return int(longest)


def _max_consecutive_losses(pnls: np.ndarray) -> int:
    longest = current = 0
    for pnl in pnls:
        if pnl < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)
