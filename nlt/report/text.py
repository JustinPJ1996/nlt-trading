"""Plain-text rendering of a verdict -- what a CLI prints, and what the
dashboard falls back to when it cannot show a chart.

This is the artifact a beginner actually reads. It leads with the verdict's
one-sentence summary and the pass/fail call, states every flag in the order
the verdict ranked them (most severe first), then backs all of that up with
the strategy-vs-benchmark numbers and the engine's own operational warnings
(fill assumptions, sizing fallbacks) so nothing that shaped the result is
hidden below the fold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlt.engine.backtest import BacktestResult
    from nlt.report.benchmark import Comparison
    from nlt.report.verdict import Verdict

_RULE = "-" * 72


def render(result: BacktestResult, comparison: Comparison, verdict: Verdict) -> str:
    lines: list[str] = []

    lines.append(_RULE)
    lines.append(verdict.summary)
    lines.append("VERDICT: " + ("PASS" if verdict.passed else "DO NOT TRADE THIS YET"))
    lines.append(_RULE)

    if verdict.flags:
        lines.append("")
        lines.append("Flags:")
        for flag in verdict.flags:
            lines.append(f"  [{flag.severity.upper()}] {flag.headline}")
            lines.append(f"      {flag.detail}")
    else:
        lines.append("")
        lines.append("No flags raised.")

    lines.append("")
    lines.append(_RULE)
    lines.append("Strategy vs. buy-and-hold NIFTY")
    lines.append(_RULE)
    lines.append(f"{'':24}{'Strategy':>20}{'Buy and hold':>20}")
    lines.append(
        f"{'Total return':24}{comparison.strategy_return_pct:>19.1f}%"
        f"{comparison.benchmark_return_pct:>19.1f}%"
    )
    lines.append(
        f"{'Worst drawdown':24}{comparison.strategy_max_drawdown_pct:>19.1f}%"
        f"{comparison.benchmark_max_drawdown_pct:>19.1f}%"
    )
    lines.append(f"{'Time in market':24}{comparison.time_in_market_pct:>19.1f}%{'100.0':>19}%")
    beat = "beat" if comparison.beat_benchmark else "did NOT beat"
    lines.append("")
    lines.append(
        f"The strategy {beat} buy-and-hold by "
        f"{abs(comparison.excess_return_pct):.1f} percentage points."
    )

    lines.append("")
    lines.append(_RULE)
    lines.append("Key metrics")
    lines.append(_RULE)
    m = result.metrics
    lines.append(f"Total trades:            {m['total_trades']}")
    lines.append(f"Win rate:                {m['win_rate_pct']:.1f}%")
    lines.append(f"Winning / losing trades: {m['winning_trades']} / {m['losing_trades']}")
    lines.append(f"Average win / loss:      {_rupees(m['avg_win'])} / {_rupees(m['avg_loss'])}")
    lines.append(f"Best / worst trade:      {_rupees(m['best_trade'])} / {_rupees(m['worst_trade'])}")
    lines.append(f"Max consecutive losses:  {m['max_consecutive_losses']}")
    lines.append(f"Total charges paid:      {_rupees(m['total_charges'])}")
    lines.append(f"CAGR:                    {m['cagr_pct']:.1f}%")
    lines.append(f"Max drawdown:            {m['max_drawdown_pct']:.1f}%")
    lines.append(
        f"Max drawdown length:     {m['max_drawdown_duration_days']} day(s)"
    )
    lines.append(f"Sharpe ratio:            {_fmt_ratio(m['sharpe'])}")
    lines.append(f"Sortino ratio:           {_fmt_ratio(m['sortino'])}")
    lines.append(f"Profit factor:           {_fmt_ratio(m['profit_factor'])}")
    lines.append(f"Expectancy per trade:    {_rupees(m['expectancy'])}")

    if result.warnings:
        lines.append("")
        lines.append(_RULE)
        lines.append("Engine warnings (how this backtest was actually run)")
        lines.append(_RULE)
        for warning in result.warnings:
            lines.append(f"  - {warning}")

    lines.append(_RULE)
    return "\n".join(lines)


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _rupees(amount: float) -> str:
    """Indian digit grouping: Rs 1,00,000 -- lakh/crore groups of two after
    the first group of three, not the thousands-only grouping most number
    formatting libraries default to.
    """
    sign = "-" if amount < 0 else ""
    whole = int(round(abs(amount)))
    s = str(whole)
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        groups: list[str] = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ",".join(groups) + "," + last3
    return f"Rs {sign}{grouped}"
