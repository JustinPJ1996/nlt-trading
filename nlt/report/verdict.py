"""Plain-English judgment on a backtest, for someone who is not going to read
the trade log.

A backtest is a data-generating process with a small, well-known set of ways
it can lie to the person reading it: too few trades to mean anything, one
lucky trade carrying the whole result, costs excluded from the number that
gets remembered, a drawdown nobody would actually sit through, a comparison
against doing nothing that the strategy quietly loses. None of these are
bugs in the engine -- they are true facts about the run that a beginner has
no way to know to look for. `assess` looks for all of them and says so in
words that do not require knowing what "Sharpe ratio" means.

Every threshold below is a judgment call, not a law of finance, and is
documented as one at the point it is used. They are deliberately set on the
strict side: a false alarm here costs a beginner a moment of re-reading a
flag; a missed one costs them real money.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nlt.engine.backtest import BacktestResult
    from nlt.report.benchmark import Comparison

_SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}

# -- thresholds, named so the reasoning in `assess` can refer to them by name
# instead of a bare number floating in an `if`. See the module docstring and
# the per-flag comments below for why each value sits where it does.
TOO_FEW_TRADES_CRITICAL = 30
TOO_FEW_TRADES_WARNING = 100
ONE_TRADE_SHARE_CRITICAL = 0.50
CHARGES_SHARE_WARNING = 0.30
SEVERE_DRAWDOWN_WARNING = 25.0
SEVERE_DRAWDOWN_CRITICAL = 50.0
SHORT_BACKTEST_YEARS = 3.0
SHORT_BACKTEST_BARS = 250
LOW_WIN_RATE_INFO = 40.0
RARELY_IN_MARKET_INFO = 10.0
LONG_LOSING_STREAK_WARNING = 8


@dataclass(frozen=True)
class Flag:
    """One specific, named way this backtest might be misleading."""

    severity: str
    code: str
    headline: str
    detail: str


@dataclass(frozen=True)
class Verdict:
    """The one-paragraph judgment a non-technical user reads first."""

    summary: str
    passed: bool
    flags: list[Flag]


def assess(result: BacktestResult, comparison: Comparison) -> Verdict:
    """Run every check and produce the verdict.

    `passed` is False the instant any *critical* flag fires -- warnings and
    info flags are things to know, not reasons to block. This is a
    deliberately low bar (a strategy can clear it while still being a bad
    idea for reasons a warning flag names) but it is the one thing this
    module can say with confidence from the numbers alone: "at least one of
    the well-known ways a backtest lies is present here."
    """
    flags = _all_flags(result, comparison)
    flags.sort(key=lambda f: _SEVERITY_RANK[f.severity])
    passed = not any(f.severity == "critical" for f in flags)
    summary = _summary(result, comparison, flags, passed)
    return Verdict(summary=summary, passed=passed, flags=flags)


def _all_flags(result: BacktestResult, comparison: Comparison) -> list[Flag]:
    flags: list[Flag] = []
    metrics = result.metrics
    trades = result.trades

    lost_money = metrics["total_return_pct"] < 0
    if lost_money:
        flags.append(
            Flag(
                severity="critical",
                code="lost_money",
                headline="This strategy lost money.",
                detail=(
                    f"Total return was {metrics['total_return_pct']:.1f}% over the backtest "
                    "period -- the account ended with less than it started with."
                ),
            )
        )

    if not comparison.beat_benchmark:
        flags.append(
            Flag(
                severity="critical" if lost_money else "warning",
                code="underperformed_benchmark",
                headline=f"Simply {comparison.benchmark_name} would have done better.",
                detail=(
                    f"The strategy returned {comparison.strategy_return_pct:.1f}% versus "
                    f"{comparison.benchmark_return_pct:.1f}% for {comparison.benchmark_name} over "
                    f"the same period -- {abs(comparison.excess_return_pct):.1f} percentage "
                    "points worse. All the extra effort and risk of trading bought nothing "
                    "over doing nothing."
                ),
            )
        )

    total_trades = metrics["total_trades"]
    if total_trades < TOO_FEW_TRADES_CRITICAL:
        flags.append(
            Flag(
                severity="critical",
                code="too_few_trades",
                headline="There are far too few trades to trust this result.",
                detail=(
                    f"Only {total_trades} trade(s) closed in this backtest. A handful of "
                    "trades cannot distinguish a genuine edge from a lucky run -- the same "
                    "strategy re-run on a slightly different stretch of history could show "
                    f"the opposite result. {TOO_FEW_TRADES_CRITICAL} is a rough floor for "
                    "even starting to trust the numbers, not a guarantee at that count either."
                ),
            )
        )
    elif total_trades < TOO_FEW_TRADES_WARNING:
        flags.append(
            Flag(
                severity="warning",
                code="too_few_trades",
                headline="This is still a fairly small number of trades.",
                detail=(
                    f"{total_trades} trades is enough to start drawing a conclusion from, but "
                    "not enough to be confident -- luck plays a real role at this sample size. "
                    "Treat the numbers as a first look, not a final answer."
                ),
            )
        )

    if trades:
        gross_profits = [t.gross_pnl for t in trades if t.gross_pnl > 0]
        total_gross_profit = sum(gross_profits)
        if total_gross_profit > 0:
            best_trade_profit = max(gross_profits)
            share = best_trade_profit / total_gross_profit
            if share > ONE_TRADE_SHARE_CRITICAL:
                flags.append(
                    Flag(
                        severity="critical",
                        code="driven_by_one_trade",
                        headline="Almost all the profit came from a single trade.",
                        detail=(
                            f"One trade made Rs {best_trade_profit:,.0f} before costs, which is "
                            f"{share * 100:.0f}% of the Rs {total_gross_profit:,.0f} total gross "
                            "profit across every winning trade. Remove that one trade and this "
                            "strategy would look completely different -- possibly a loser. A "
                            "single outlier is not a repeatable edge."
                        ),
                    )
                )

    total_charges = metrics["total_charges"]
    if trades and total_charges > 0:
        total_gross_profit = sum(t.gross_pnl for t in trades if t.gross_pnl > 0)
        if total_gross_profit > 0:
            charges_ratio = total_charges / total_gross_profit
            if charges_ratio > CHARGES_SHARE_WARNING:
                flags.append(
                    Flag(
                        severity="warning",
                        code="charges_dominate",
                        headline="A large share of the profit went to brokerage and taxes.",
                        detail=(
                            f"Rs {total_charges:,.0f} was paid in charges, which is "
                            f"{charges_ratio * 100:.0f}% of the Rs {total_gross_profit:,.0f} "
                            "gross profit before costs. A strategy that only looks good before "
                            "costs is not a strategy worth trading."
                        ),
                    )
                )
        else:
            total_gross_loss = -sum(t.gross_pnl for t in trades if t.gross_pnl < 0)
            if total_gross_loss > 0 and total_charges / total_gross_loss > CHARGES_SHARE_WARNING:
                flags.append(
                    Flag(
                        severity="warning",
                        code="charges_dominate",
                        headline="Brokerage and taxes made the losses meaningfully worse.",
                        detail=(
                            f"Rs {total_charges:,.0f} was paid in charges on top of "
                            f"Rs {total_gross_loss:,.0f} of trading losses before costs. Even a "
                            "strategy that is roughly break-even on price alone can lose money "
                            "once costs like these are included."
                        ),
                    )
                )

    max_dd = abs(metrics["max_drawdown_pct"])
    if max_dd > SEVERE_DRAWDOWN_CRITICAL:
        flags.append(
            Flag(
                severity="critical",
                code="severe_drawdown",
                headline="At its worst, this strategy lost more than half its value.",
                detail=(
                    f"At the worst point in the backtest, the account was down {max_dd:.0f}% "
                    "from its peak. Very few people hold on through a loss that size without "
                    "abandoning the strategy or the market entirely -- and a loss this deep "
                    "needs a gain more than twice as large just to break even."
                ),
            )
        )
    elif max_dd > SEVERE_DRAWDOWN_WARNING:
        flags.append(
            Flag(
                severity="warning",
                code="severe_drawdown",
                headline="This strategy had a deep dip along the way.",
                detail=(
                    f"At the worst point in the backtest, the account was down {max_dd:.0f}% "
                    "from its peak before recovering. Ask honestly whether you would have kept "
                    "trading through a loss that size, or sold out near the bottom."
                ),
            )
        )

    equity = result.equity
    years = (equity.index[-1] - equity.index[0]).days / 365.25 if len(equity) > 1 else 0.0
    bars_count = len(equity)
    if years < SHORT_BACKTEST_YEARS or bars_count < SHORT_BACKTEST_BARS:
        flags.append(
            Flag(
                severity="warning",
                code="short_backtest",
                headline="This was only tested over a short stretch of history.",
                detail=(
                    f"The backtest covers about {years:.1f} year(s) ({bars_count} bars). "
                    "Markets go through very different conditions -- trending, sideways, "
                    "crashing -- and a strategy that only saw one of those has not really been "
                    "tested yet."
                ),
            )
        )

    win_rate = metrics["win_rate_pct"]
    if trades and win_rate < LOW_WIN_RATE_INFO:
        avg_win = metrics["avg_win"]
        avg_loss = abs(metrics["avg_loss"])
        # Comparing the two averages directly is both unreadable and wrong. When
        # they are close they round to the same rupee, producing "Rs 155 is bigger
        # than Rs 155"; and "bigger" is not the question anyway. What decides
        # whether a low win rate is survivable is the break-even win rate implied
        # by the two sizes -- win 155 / lose 155 needs 50% to break even, so 30%
        # is a losing strategy no matter how the averages compare.
        total_avg = avg_win + avg_loss
        breakeven_rate = 100.0 * avg_loss / total_avg if total_avg > 0 else None
        big_winners = breakeven_rate is not None and win_rate > breakeven_rate

        if breakeven_rate is None:
            detail = (
                f"Only {win_rate:.0f}% of trades were winners, and there were no losing "
                "trades to weigh that against."
            )
            severity = "info"
        elif big_winners:
            detail = (
                f"Only {win_rate:.0f}% of trades were winners, which is fine when the "
                f"winners are bigger. The average winner is Rs {avg_win:,.0f} against an "
                f"average loser of Rs {avg_loss:,.0f}, so a win rate above "
                f"{breakeven_rate:.0f}% is enough to come out ahead -- and this won "
                f"{win_rate:.0f}%."
            )
            severity = "info"
        else:
            detail = (
                f"The average winner is Rs {avg_win:,.0f} against an average loser of "
                f"Rs {avg_loss:,.0f}. At those sizes the strategy has to win "
                f"{breakeven_rate:.0f}% of its trades just to break even, and it won only "
                f"{win_rate:.0f}%. Winning less often than that, with winners no bigger "
                "than the losers, does not add up to a profit."
            )
            # A win rate below the break-even line is not a footnote, it is the
            # reason the strategy loses. Reporting it as "info" buries it.
            severity = "warning"

        flags.append(
            Flag(
                severity=severity,
                code="low_win_rate_needs_big_winners",
                headline=(
                    "This strategy loses more often than it wins."
                    if big_winners
                    else "This strategy does not win often enough to cover its losses."
                ),
                detail=detail,
            )
        )

    exposure = metrics["exposure_pct"]
    if exposure < RARELY_IN_MARKET_INFO:
        flags.append(
            Flag(
                severity="info",
                code="rarely_in_market",
                headline="This strategy was barely ever in a trade.",
                detail=(
                    f"A position was open only {exposure:.1f}% of the time in this backtest. "
                    "Comparing its return directly to buying and holding is not really "
                    "apples-to-apples: most of the time this strategy's money was sitting in "
                    "cash, not exposed to the market at all, which is a very different bet."
                ),
            )
        )

    # Raised from an engine warning to a critical flag deliberately. It belongs
    # above the numbers, not below them: the point is not that the result is
    # poor but that the instrument is unsuitable at this account size, which is
    # a conclusion the return figure cannot express and might actively hide by
    # looking good.
    if any("OPTIONS ON A SMALL ACCOUNT" in w for w in result.warnings):
        flags.append(
            Flag(
                severity="critical",
                code="options_need_a_bigger_account",
                headline="Options are not suitable for an account this size.",
                detail=(
                    "One NIFTY option lot is 75 units and cannot be split, so a small "
                    "account either cannot afford a single lot or has to put far too "
                    "much of itself into one contract -- and an option can lose its "
                    "entire premium in a session. Roughly Rs 10,00,000 is where options "
                    "start to make sense. Below that, the same idea is usually better "
                    "expressed on the index or on shares."
                ),
            )
        )

    if any("no charge_fn supplied" in w for w in result.warnings):
        flags.append(
            Flag(
                severity="warning",
                code="untested_costs",
                headline="These numbers do not include brokerage or taxes.",
                detail=(
                    "This backtest was run without a cost model, so every number here is "
                    "before brokerage, STT and other charges. Real trading always has these "
                    "costs, and they fall hardest on strategies that trade often -- re-run with "
                    "a charge model before trusting the return."
                ),
            )
        )

    max_losing_streak = metrics["max_consecutive_losses"]
    if max_losing_streak > LONG_LOSING_STREAK_WARNING:
        flags.append(
            Flag(
                severity="warning",
                code="long_losing_streak",
                headline="This strategy had a long run of losing trades in a row.",
                detail=(
                    f"At one point, {max_losing_streak} trades lost money back to back. Most "
                    "people give up on a strategy partway through a losing streak that long, "
                    "long before it has a chance to recover -- the backtest's return assumes "
                    "you stuck it out."
                ),
            )
        )

    return flags


def _summary(
    result: BacktestResult,
    comparison: Comparison,
    flags: list[Flag],
    passed: bool,
) -> str:
    """The single sentence shown above everything else.

    Deliberately free of jargon: no "Sharpe", "Sortino" or "profit factor"
    (those belong in `Flag.detail` at most). "Drawdown" is allowed -- it is
    plain enough in context ("a drawdown") and appears in the brief's own
    example summaries, unlike the three ratio names above which mean nothing
    without already knowing statistics.
    """
    codes = {f.code for f in flags}
    metrics = result.metrics
    lost_money = "lost_money" in codes
    underperformed = "underperformed_benchmark" in codes

    if lost_money and underperformed:
        return (
            "This strategy lost money, and it also did far worse than "
            f"{comparison.benchmark_name}."
        )
    if lost_money:
        return (
            f"This strategy lost money over the backtest period "
            f"({metrics['total_return_pct']:.1f}% total return)."
        )
    if underperformed:
        return (
            f"This made money, but did worse than {comparison.benchmark_name} over the "
            "same period."
        )
    if "too_few_trades" in codes and any(
        f.code == "too_few_trades" and f.severity == "critical" for f in flags
    ):
        return (
            f"This made money, but on only {metrics['total_trades']} trades -- far too few to "
            "tell skill from luck."
        )
    if "driven_by_one_trade" in codes:
        return "This made money, but almost all of it came from a single lucky trade."
    if "severe_drawdown" in codes and any(
        f.code == "severe_drawdown" and f.severity == "critical" for f in flags
    ):
        return (
            "This beat buy-and-hold, but with a drawdown severe enough that most people would "
            "have abandoned it."
        )
    if metrics["total_trades"] == 0:
        return "This strategy never took a single trade in the backtest period."

    years = (
        (result.equity.index[-1] - result.equity.index[0]).days / 365.25
        if len(result.equity) > 1
        else 0.0
    )
    return (
        f"This beat buy-and-hold with a {'modest' if abs(metrics['max_drawdown_pct']) < 25 else 'notable'} "
        f"drawdown, across {metrics['total_trades']} trades over {years:.0f} years."
    )
