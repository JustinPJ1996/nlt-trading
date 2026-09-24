""""Would I have done better just buying the index?" -- the question every
backtest result must be able to answer before anyone risks money on it.

A strategy's own numbers, read in isolation, cannot tell a beginner whether
they are good. +50% total return sounds wonderful until you learn the index
did +419% over the same stretch while you took on strategy risk, trading
costs and your own nerve holding through drawdowns the index never asked you
to sit through. `buy_and_hold` builds that reference line honestly -- same
capital, same charges, same calendar -- so `compare` can put the two side by
side without either number being flattered.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from nlt.engine.backtest import BacktestResult

# A calendar year, not a trading year: unlike the engine's own CAGR (which
# divides by *bar count* / 252, because a strategy's trades are anchored to
# trading days), the benchmark holds continuously through weekends and
# holidays, so its natural clock is wall time. Using the same 252-bars/year
# convention here would understate CAGR every time the data includes non-1d
# bars or gaps, and it is also the unit a beginner already thinks in ("held
# for 19 years").
_DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Benchmark:
    """A buy-and-hold equity curve, computed the same way the strategy's is.

    `equity` shares its index with the bars it was built from, so it can be
    plotted directly against `BacktestResult.equity` with no realignment.
    """

    name: str
    total_return_pct: float
    cagr_pct: float
    max_drawdown_pct: float
    sharpe: float | None
    equity: pd.Series


def buy_and_hold(
    bars: pd.DataFrame,
    capital: float,
    *,
    name: str = "Buy and hold",
    charge_fn: Callable[[float, int, str], float] | None = None,
    bars_per_year: int = 252,
) -> Benchmark:
    """Buy at the first bar's open, hold to the last bar's close.

    This mirrors what a strategy's own equity curve represents -- money
    deployed at a real, tradeable price, marked to market bar by bar -- so
    the two curves are comparable rather than one being a clean theoretical
    line and the other a costed simulation:

    * Quantity is whole units affordable with `capital` (``floor(capital /
      entry_price)``). Whatever capital that leaves over sits idle in cash
      for the whole holding period and is carried through untouched in
      `equity` -- it is not reinvested, is not "put to work" some other way,
      and is not silently dropped. This is the same affordability rule the
      engine itself uses for strategy fills.
    * If `charge_fn` is given, entry and exit charges are subtracted from
      that cash exactly once each, at the bar they are incurred (entry
      charges reduce cash immediately; exit charges are only realised on the
      final bar, since the position is held, not round-tripped, until then).
      Omitting `charge_fn` compares the strategy's costed numbers against a
      benchmark that also excludes costs -- passing it makes the comparison
      like-for-like instead of flattering the index.
    """
    if bars.empty:
        raise ValueError("buy_and_hold requires at least one bar")
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital!r}")

    entry_price = float(bars["open"].iloc[0])
    quantity = math.floor(capital / entry_price) if entry_price > 0 else 0

    entry_charges = charge_fn(entry_price, quantity, "buy") if charge_fn is not None else 0.0
    cash = capital - quantity * entry_price - entry_charges

    equity_values = cash + quantity * bars["close"].to_numpy(dtype="float64")

    exit_price = float(bars["close"].iloc[-1])
    exit_charges = charge_fn(exit_price, quantity, "sell") if charge_fn is not None else 0.0
    equity_values[-1] -= exit_charges

    equity = pd.Series(equity_values, index=bars.index, name="equity")

    total_return_pct = float(100.0 * (equity.iloc[-1] - capital) / capital)

    years = (bars.index[-1] - bars.index[0]).days / _DAYS_PER_YEAR
    if years > 0 and equity.iloc[-1] > 0:
        cagr_pct = float(100.0 * ((equity.iloc[-1] / capital) ** (1.0 / years) - 1.0))
    else:
        cagr_pct = 0.0

    running_max = equity.cummax()
    drawdown_pct = 100.0 * (equity - running_max) / running_max.replace(0.0, np.nan)
    max_drawdown_pct = float(drawdown_pct.min()) if len(drawdown_pct) else 0.0

    bar_returns = equity.pct_change().dropna()
    if len(bar_returns) > 1 and bar_returns.std(ddof=0) > 0:
        sharpe = float(bar_returns.mean() / bar_returns.std(ddof=0) * np.sqrt(bars_per_year))
    else:
        sharpe = None

    return Benchmark(
        name=name,
        total_return_pct=total_return_pct,
        cagr_pct=cagr_pct,
        max_drawdown_pct=max_drawdown_pct,
        sharpe=sharpe,
        equity=equity,
    )


@dataclass(frozen=True)
class Comparison:
    """Strategy vs. benchmark, on the numbers a beginner actually needs."""

    strategy_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    beat_benchmark: bool
    strategy_max_drawdown_pct: float
    benchmark_max_drawdown_pct: float
    time_in_market_pct: float


def compare(result: BacktestResult, benchmark: Benchmark, bars: pd.DataFrame) -> Comparison:
    """Put the strategy's own metrics next to the benchmark's.

    `time_in_market_pct` (the fraction of bars the strategy held a position,
    already computed by the engine as `metrics["exposure_pct"]`) is carried
    through deliberately: a strategy that was in the market 3% of the time
    and made 5% is doing something completely different from a buy-and-hold
    line that was in the market 100% of the time to make the same 5% --
    the first took far less market risk for the same reward, the second took
    far more. Without this number the headline return comparison alone would
    be unfair in both directions, and a beginner has no way to know which
    way it is unfair without being told.
    """
    if len(bars) != len(result.equity):
        raise ValueError(
            f"bars ({len(bars)} rows) and result.equity ({len(result.equity)} rows) "
            "must cover the same backtest run"
        )

    strategy_return_pct = float(result.metrics["total_return_pct"])
    benchmark_return_pct = benchmark.total_return_pct
    excess_return_pct = strategy_return_pct - benchmark_return_pct

    return Comparison(
        strategy_return_pct=strategy_return_pct,
        benchmark_return_pct=benchmark_return_pct,
        excess_return_pct=excess_return_pct,
        beat_benchmark=bool(excess_return_pct > 0),
        strategy_max_drawdown_pct=float(result.metrics["max_drawdown_pct"]),
        benchmark_max_drawdown_pct=benchmark.max_drawdown_pct,
        time_in_market_pct=float(result.metrics["exposure_pct"]),
    )
