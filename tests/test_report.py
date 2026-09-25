"""Tests for `nlt.report`: the benchmark comparison and the plain-English verdict.

Every flag test comes in a pair -- a case built to trigger it, and a mirror
case built so it should not. The mirror half is not optional decoration: a
flag that never fires on realistic-looking clean input is a false alarm
waiting to erode trust in every other flag next to it.

Where a numeric result is asserted, the expected value is computed by hand in
the test body (not by calling the code under test a second time), so a
mutation that breaks the arithmetic actually gets caught.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nlt.costs.charges import NseEquityDeliveryCharges
from nlt.costs.charges import charge_fn as make_charge_fn
from nlt.engine.backtest import BacktestResult, Trade, run_backtest
from nlt.engine.metrics import compute_metrics
from nlt.report.benchmark import Benchmark, Comparison, buy_and_hold, compare
from nlt.report.text import render
from nlt.report.verdict import Verdict, assess
from nlt.spec.models import (
    Compare as CompareCond,
)
from nlt.spec.models import (
    Const,
    ExitRules,
    IndicatorSpec,
    Ref,
    StrategySpec,
)

BANNED_JARGON = ("sharpe", "sortino", "profit factor")

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _bdate_index(n: int, start: str = "2010-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, tz="Asia/Kolkata")


def _mkbars(n: int = 5, start_price: float = 100.0, step: float = 2.0) -> pd.DataFrame:
    """A simple, strictly-rising OHLCV series: close[i] = start_price + step*i."""
    idx = _bdate_index(n)
    closes = np.array([start_price + step * i for i in range(n)])
    opens = closes - step / 2.0
    opens[0] = start_price  # first bar's open is the entry price
    highs = closes + 1.0
    lows = opens - 1.0
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": [1_000.0] * n},
        index=idx,
    )


def _spec() -> StrategySpec:
    return StrategySpec(
        name="test",
        description="test",
        entry=CompareCond(op="gt", left=Ref(name="close"), right=Const(value=-1.0)),
        exit=ExitRules(stop_pct=5.0),
    )


def _mk_trade(
    idx: pd.DatetimeIndex,
    entry_i: int,
    net_pnl: float,
    charges: float = 0.0,
    bars_held: int = 5,
    gross_pnl: float | None = None,
) -> Trade:
    gross = gross_pnl if gross_pnl is not None else net_pnl + charges
    exit_i = min(entry_i + bars_held, len(idx) - 1)
    return Trade(
        entry_time=idx[entry_i],
        entry_price=100.0,
        exit_time=idx[exit_i],
        exit_price=100.0 + net_pnl,
        direction="long",
        quantity=1,
        gross_pnl=gross,
        charges=charges,
        net_pnl=net_pnl,
        exit_reason="target" if net_pnl >= 0 else "stop",
        bars_held=bars_held,
        entry_reason="close > -1",
    )


def _flat_equity(n: int, capital: float = 100_000.0, end_capital: float | None = None) -> pd.Series:
    idx = _bdate_index(n)
    end = capital if end_capital is None else end_capital
    return pd.Series(np.linspace(capital, end, n), index=idx, name="equity")


def _equity_with_drawdown(n: int, capital: float, trough_pct: float, end_pct: float) -> pd.Series:
    """A curve that falls `trough_pct`% below `capital` partway through, then
    recovers to `end_pct`% of `capital` by the end. `trough_pct`/`end_pct` are
    negative/positive percentages relative to `capital`, e.g. -60 and +10.
    """
    idx = _bdate_index(n)
    trough_i = n // 3
    trough_val = capital * (1.0 + trough_pct / 100.0)
    end_val = capital * (1.0 + end_pct / 100.0)
    vals = np.empty(n)
    vals[:trough_i] = np.linspace(capital, trough_val, trough_i)
    vals[trough_i:] = np.linspace(trough_val, end_val, n - trough_i)
    return pd.Series(vals, index=idx, name="equity")


def _build_result(
    trades: list[Trade],
    equity: pd.Series,
    *,
    capital: float = 100_000.0,
    warnings: list[str] | None = None,
) -> BacktestResult:
    metrics = compute_metrics(trades, equity, capital, bars_per_year=252)
    return BacktestResult(
        spec=_spec(),
        trades=trades,
        equity=equity,
        metrics=metrics,
        features=pd.DataFrame(index=equity.index),
        warnings=warnings or [],
    )


def _comparison(
    result: BacktestResult,
    *,
    beats: bool = True,
    benchmark_return_pct: float = 20.0,
    benchmark_name: str = "holding NIFTY",
) -> Comparison:
    strategy_return = result.metrics["total_return_pct"]
    if beats:
        benchmark_return_pct = min(benchmark_return_pct, strategy_return - 5.0)
    else:
        benchmark_return_pct = max(benchmark_return_pct, strategy_return + 5.0)
    excess = strategy_return - benchmark_return_pct
    return Comparison(
        strategy_return_pct=strategy_return,
        benchmark_return_pct=benchmark_return_pct,
        excess_return_pct=excess,
        beat_benchmark=excess > 0,
        strategy_max_drawdown_pct=result.metrics["max_drawdown_pct"],
        benchmark_max_drawdown_pct=-15.0,
        time_in_market_pct=result.metrics["exposure_pct"],
        benchmark_name=benchmark_name,
    )


def _baseline_good_result(n_trades: int = 150, n_bars: int = 1000) -> BacktestResult:
    """A clean, unremarkable, profitable result: alternating wins/losses,
    even profit distribution, modest charges, no long losing streak, no
    severe drawdown, plenty of history -- nothing here should raise a flag.
    """
    idx = _bdate_index(n_bars)
    trades = []
    for i in range(n_trades):
        entry_i = min(int(i * n_bars / n_trades), n_bars - 6)
        if i % 2 == 0:
            trades.append(_mk_trade(idx, entry_i, net_pnl=1000.0, charges=20.0, bars_held=3))
        else:
            trades.append(_mk_trade(idx, entry_i, net_pnl=-400.0, charges=20.0, bars_held=3))
    equity = _equity_with_drawdown(n_bars, 100_000.0, trough_pct=-8.0, end_pct=50.0)
    return _build_result(trades, equity)


# --------------------------------------------------------------------------
# 1. benchmark arithmetic
# --------------------------------------------------------------------------


def test_buy_and_hold_return_cagr_and_quantity_by_hand():
    bars = _mkbars(n=5, start_price=100.0, step=2.0)
    capital = 1_000.0

    benchmark = buy_and_hold(bars, capital)

    entry_price = 100.0  # first bar's open
    expected_quantity = 10  # floor(1000 / 100)
    expected_cash_remainder = capital - expected_quantity * entry_price  # 0.0
    exit_close = 108.0  # closes = 100,102,104,106,108
    expected_final_equity = expected_cash_remainder + expected_quantity * exit_close
    expected_return_pct = 100.0 * (expected_final_equity - capital) / capital

    years = (bars.index[-1] - bars.index[0]).days / 365.25
    expected_cagr_pct = 100.0 * ((expected_final_equity / capital) ** (1.0 / years) - 1.0)

    assert benchmark.equity.iloc[0] == pytest.approx(capital)
    assert benchmark.equity.iloc[-1] == pytest.approx(expected_final_equity)
    assert benchmark.total_return_pct == pytest.approx(expected_return_pct)
    assert benchmark.cagr_pct == pytest.approx(expected_cagr_pct)
    assert benchmark.equity.index.equals(bars.index)


def test_buy_and_hold_leaves_unaffordable_remainder_in_cash():
    bars = _mkbars(n=3, start_price=300.0, step=10.0)
    capital = 1_000.0  # 1000 / 300 = 3 units, Rs 100 left in cash

    benchmark = buy_and_hold(bars, capital)

    quantity = 3
    cash = capital - quantity * 300.0
    assert cash == pytest.approx(100.0)
    expected_first_equity = cash + quantity * bars["close"].iloc[0]
    assert benchmark.equity.iloc[0] == pytest.approx(expected_first_equity)


def test_charges_reduce_the_benchmark_return():
    bars = _mkbars(n=5, start_price=100.0, step=2.0)
    capital = 1_000.0

    no_cost = buy_and_hold(bars, capital)
    cfn = make_charge_fn(NseEquityDeliveryCharges())
    costed = buy_and_hold(bars, capital, charge_fn=cfn)

    assert costed.total_return_pct < no_cost.total_return_pct
    assert costed.equity.iloc[0] < no_cost.equity.iloc[0]  # entry charges hit cash immediately
    assert costed.equity.iloc[-1] < no_cost.equity.iloc[-1]  # exit charges hit the last bar too


def test_compare_time_in_market_and_excess_return():
    n = 20
    idx = _bdate_index(n)
    trades = [_mk_trade(idx, 0, net_pnl=5000.0, bars_held=4)]
    equity = _flat_equity(n, 100_000.0, end_capital=105_000.0)
    result = _build_result(trades, equity)

    bars = pd.DataFrame(
        {"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n, "close": [100.0] * n},
        index=idx,
    )
    benchmark = Benchmark(
        name="bm", total_return_pct=2.0, cagr_pct=1.0, max_drawdown_pct=-3.0, sharpe=0.5, equity=equity
    )

    comparison = compare(result, benchmark, bars)

    assert comparison.strategy_return_pct == pytest.approx(5.0)
    assert comparison.benchmark_return_pct == pytest.approx(2.0)
    assert comparison.excess_return_pct == pytest.approx(3.0)
    assert comparison.beat_benchmark is True
    # exactly one trade held 4 bars out of 20 equity bars = 20% exposure
    assert comparison.time_in_market_pct == pytest.approx(20.0)


def test_compare_rejects_mismatched_lengths():
    n = 10
    idx = _bdate_index(n)
    equity = _flat_equity(n)
    result = _build_result([], equity)
    bars = pd.DataFrame({"open": [1.0] * (n - 1)}, index=idx[:-1])
    benchmark = Benchmark(name="bm", total_return_pct=0.0, cagr_pct=0.0, max_drawdown_pct=0.0, sharpe=None, equity=equity)

    with pytest.raises(ValueError):
        compare(result, benchmark, bars)


# --------------------------------------------------------------------------
# 2. baseline sanity: clean result raises nothing critical
# --------------------------------------------------------------------------


def test_baseline_good_result_passes_clean():
    result = _baseline_good_result()
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)

    codes = {f.code for f in verdict.flags}
    assert codes == set(), f"unexpected flags on a clean baseline: {codes}"
    assert verdict.passed is True


# --------------------------------------------------------------------------
# 3 & 4. each flag fires when it should, and does not when it shouldn't
# --------------------------------------------------------------------------


def test_lost_money_fires_on_negative_return():
    result = _baseline_good_result()
    result.equity.iloc[:] = _equity_with_drawdown(len(result.equity), 100_000.0, trough_pct=-20.0, end_pct=-5.0)
    result.metrics.update(
        compute_metrics(result.trades, result.equity, 100_000.0, bars_per_year=252)
    )
    comparison = _comparison(result, beats=False)
    verdict = assess(result, comparison)
    assert "lost_money" in {f.code for f in verdict.flags}


def test_lost_money_absent_when_profitable():
    result = _baseline_good_result()
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "lost_money" not in {f.code for f in verdict.flags}


def test_underperformed_benchmark_fires_and_is_critical_when_also_lost_money():
    result = _baseline_good_result()
    result.equity.iloc[:] = _equity_with_drawdown(len(result.equity), 100_000.0, trough_pct=-20.0, end_pct=-5.0)
    result.metrics.update(
        compute_metrics(result.trades, result.equity, 100_000.0, bars_per_year=252)
    )
    comparison = _comparison(result, beats=False)
    verdict = assess(result, comparison)
    flags = {f.code: f for f in verdict.flags}
    assert "underperformed_benchmark" in flags
    assert flags["underperformed_benchmark"].severity == "critical"


def test_underperformed_benchmark_is_warning_when_still_profitable():
    result = _baseline_good_result()
    comparison = _comparison(result, beats=False)
    verdict = assess(result, comparison)
    flags = {f.code: f for f in verdict.flags}
    assert "underperformed_benchmark" in flags
    assert flags["underperformed_benchmark"].severity == "warning"


def test_underperformed_benchmark_absent_when_strategy_beats_it():
    result = _baseline_good_result()
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "underperformed_benchmark" not in {f.code for f in verdict.flags}


def test_too_few_trades_critical_below_30():
    result = _baseline_good_result(n_trades=10)
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    flags = {f.code: f for f in verdict.flags}
    assert flags["too_few_trades"].severity == "critical"


def test_too_few_trades_warning_between_30_and_100():
    result = _baseline_good_result(n_trades=60)
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    flags = {f.code: f for f in verdict.flags}
    assert flags["too_few_trades"].severity == "warning"


def test_too_few_trades_absent_at_150():
    result = _baseline_good_result(n_trades=150)
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "too_few_trades" not in {f.code for f in verdict.flags}


def test_driven_by_one_trade_fires_when_one_winner_is_80_percent_of_profit():
    idx = _bdate_index(200)
    trades = [_mk_trade(idx, 0, net_pnl=8000.0, charges=0.0, bars_held=2)]
    for i in range(1, 10):
        trades.append(_mk_trade(idx, i * 10, net_pnl=222.0, charges=0.0, bars_held=2))
    equity = _flat_equity(200, 100_000.0, end_capital=110_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "driven_by_one_trade" in {f.code for f in verdict.flags}


def test_driven_by_one_trade_absent_when_profit_spread_evenly():
    idx = _bdate_index(200)
    trades = [_mk_trade(idx, i * 3, net_pnl=200.0, charges=0.0, bars_held=2) for i in range(50)]
    equity = _flat_equity(200, 100_000.0, end_capital=110_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "driven_by_one_trade" not in {f.code for f in verdict.flags}


def test_charges_dominate_fires_when_costs_exceed_30_percent_of_profit():
    idx = _bdate_index(200)
    trades = [_mk_trade(idx, i * 3, net_pnl=100.0, charges=80.0, bars_held=2) for i in range(40)]
    equity = _flat_equity(200, 100_000.0, end_capital=104_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "charges_dominate" in {f.code for f in verdict.flags}


def test_charges_dominate_absent_when_costs_are_small():
    idx = _bdate_index(200)
    trades = [_mk_trade(idx, i * 3, net_pnl=170.0, charges=10.0, bars_held=2) for i in range(40)]
    equity = _flat_equity(200, 100_000.0, end_capital=107_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "charges_dominate" not in {f.code for f in verdict.flags}


def test_severe_drawdown_critical_above_50_percent():
    n = 500
    equity = _equity_with_drawdown(n, 100_000.0, trough_pct=-60.0, end_pct=10.0)
    idx = equity.index
    trades = [_mk_trade(idx, i * 3, net_pnl=100.0, charges=5.0, bars_held=2) for i in range(150)]
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    flags = {f.code: f for f in verdict.flags}
    assert flags["severe_drawdown"].severity == "critical"


def test_severe_drawdown_warning_between_25_and_50_percent():
    n = 500
    equity = _equity_with_drawdown(n, 100_000.0, trough_pct=-35.0, end_pct=10.0)
    idx = equity.index
    trades = [_mk_trade(idx, i * 3, net_pnl=100.0, charges=5.0, bars_held=2) for i in range(150)]
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    flags = {f.code: f for f in verdict.flags}
    assert flags["severe_drawdown"].severity == "warning"


def test_severe_drawdown_absent_when_shallow():
    result = _baseline_good_result()  # baseline uses an 8% trough
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "severe_drawdown" not in {f.code for f in verdict.flags}


def test_short_backtest_fires_under_3_years_or_250_bars():
    n = 100  # well under both the year and bar floors
    equity = _flat_equity(n, 100_000.0, end_capital=110_000.0)
    idx = equity.index
    trades = [_mk_trade(idx, i, net_pnl=200.0, charges=5.0, bars_held=1) for i in range(40)]
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "short_backtest" in {f.code for f in verdict.flags}


def test_short_backtest_absent_over_3_years():
    result = _baseline_good_result(n_bars=1000)  # ~4 calendar years of bdays
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "short_backtest" not in {f.code for f in verdict.flags}


def test_low_win_rate_needs_big_winners_fires_under_40_percent():
    idx = _bdate_index(300)
    trades = []
    for i in range(10):
        trades.append(_mk_trade(idx, i * 4, net_pnl=100.0, charges=5.0, bars_held=2))
    for i in range(30):
        trades.append(_mk_trade(idx, 50 + i * 4, net_pnl=-40.0, charges=5.0, bars_held=2))
    equity = _flat_equity(300, 100_000.0, end_capital=101_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "low_win_rate_needs_big_winners" in {f.code for f in verdict.flags}


def test_low_win_rate_needs_big_winners_absent_at_50_percent():
    result = _baseline_good_result()  # 50% win rate
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "low_win_rate_needs_big_winners" not in {f.code for f in verdict.flags}


def test_rarely_in_market_fires_under_10_percent_exposure():
    n = 1000
    idx = _bdate_index(n)
    # 30 trades, 1 bar held each = 30 bars in position out of 1000 = 3% exposure
    trades = [_mk_trade(idx, i * 30, net_pnl=100.0, charges=5.0, bars_held=1) for i in range(30)]
    equity = _flat_equity(n, 100_000.0, end_capital=103_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "rarely_in_market" in {f.code for f in verdict.flags}


def test_rarely_in_market_absent_when_mostly_invested():
    result = _baseline_good_result()  # bars_held=3 per trade over 150 trades, well above 10%
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "rarely_in_market" not in {f.code for f in verdict.flags}


def test_untested_costs_fires_when_engine_warns_no_charge_fn():
    result = _baseline_good_result()
    result.warnings.append("no charge_fn supplied; results exclude transaction costs")
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "untested_costs" in {f.code for f in verdict.flags}


def test_untested_costs_absent_when_no_such_warning():
    result = _baseline_good_result()
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert "untested_costs" not in {f.code for f in verdict.flags}


def test_long_losing_streak_fires_above_8_in_a_row():
    idx = _bdate_index(300)
    trades = [_mk_trade(idx, i * 4, net_pnl=-100.0, charges=5.0, bars_held=2) for i in range(9)]
    trades += [_mk_trade(idx, 100 + i * 4, net_pnl=500.0, charges=5.0, bars_held=2) for i in range(20)]
    equity = _flat_equity(300, 100_000.0, end_capital=105_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "long_losing_streak" in {f.code for f in verdict.flags}


def test_long_losing_streak_absent_at_8_or_fewer():
    idx = _bdate_index(300)
    trades = [_mk_trade(idx, i * 4, net_pnl=-100.0, charges=5.0, bars_held=2) for i in range(8)]
    trades += [_mk_trade(idx, 100 + i * 4, net_pnl=500.0, charges=5.0, bars_held=2) for i in range(20)]
    equity = _flat_equity(300, 100_000.0, end_capital=105_000.0)
    result = _build_result(trades, equity)
    comparison = _comparison(result, beats=True)

    verdict = assess(result, comparison)
    assert "long_losing_streak" not in {f.code for f in verdict.flags}


# --------------------------------------------------------------------------
# 5. `passed`
# --------------------------------------------------------------------------


def test_passed_false_when_any_critical_flag_present():
    result = _baseline_good_result(n_trades=10)  # too_few_trades -> critical
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert any(f.severity == "critical" for f in verdict.flags)
    assert verdict.passed is False


def test_passed_true_on_clean_good_result():
    result = _baseline_good_result()
    comparison = _comparison(result, beats=True)
    verdict = assess(result, comparison)
    assert verdict.passed is True


# --------------------------------------------------------------------------
# 7. zero trades
# --------------------------------------------------------------------------


def test_zero_trades_produces_sensible_verdict_not_a_crash():
    equity = _flat_equity(300, 100_000.0)
    result = _build_result([], equity)
    comparison = _comparison(result, beats=False, benchmark_return_pct=10.0)

    verdict = assess(result, comparison)

    assert isinstance(verdict, Verdict)
    assert verdict.passed is False  # too_few_trades (0) is critical
    assert any(f.code == "too_few_trades" for f in verdict.flags)
    for flag in verdict.flags:
        assert "nan" not in flag.detail.lower()
        assert "nan" not in flag.headline.lower()
    assert "nan" not in verdict.summary.lower()


# --------------------------------------------------------------------------
# 8. no jargon in summaries
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_verdict",
    [
        lambda: assess(*_lost_and_underperformed()),
        lambda: assess(*_profitable_but_underperformed()),
        lambda: assess(*_clean_pass()),
        lambda: assess(*_zero_trades_case()),
    ],
)
def test_summary_contains_no_jargon(make_verdict):
    verdict = make_verdict()
    lowered = verdict.summary.lower()
    for word in BANNED_JARGON:
        assert word not in lowered, f"summary leaked jargon {word!r}: {verdict.summary!r}"


def _lost_and_underperformed():
    result = _baseline_good_result()
    result.equity.iloc[:] = _equity_with_drawdown(len(result.equity), 100_000.0, trough_pct=-20.0, end_pct=-5.0)
    result.metrics.update(compute_metrics(result.trades, result.equity, 100_000.0, bars_per_year=252))
    return result, _comparison(result, beats=False)


def _profitable_but_underperformed():
    result = _baseline_good_result()
    return result, _comparison(result, beats=False)


def _clean_pass():
    result = _baseline_good_result()
    return result, _comparison(result, beats=True)


def _zero_trades_case():
    equity = _flat_equity(300, 100_000.0)
    result = _build_result([], equity)
    return result, _comparison(result, beats=False, benchmark_return_pct=10.0)


# --------------------------------------------------------------------------
# 9. real-data integration
# --------------------------------------------------------------------------


def test_rsi_strategy_underperforms_nifty_buy_and_hold(nifty_bars):
    spec = StrategySpec(
        name="RSI cracks 30",
        description="buy nifty when rsi cracks 30 exit at 2% sl 1%",
        entry=CompareCond(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30.0)),
        exit=ExitRules(stop_pct=1.0, target_pct=2.0),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
    )
    cfn = make_charge_fn(NseEquityDeliveryCharges())
    capital = 100_000.0

    result = run_backtest(spec, nifty_bars, capital=capital, charge_fn=cfn)
    benchmark = buy_and_hold(nifty_bars, capital=capital, name="NIFTY buy and hold", charge_fn=cfn)
    comparison = compare(result, benchmark, nifty_bars)
    verdict = assess(result, comparison)

    assert comparison.beat_benchmark is False
    # The index made roughly 400+% over this data; the RSI strategy is
    # roughly flat, so it underperforms by hundreds of percentage points.
    assert comparison.excess_return_pct < -300.0
    assert "underperformed_benchmark" in {f.code for f in verdict.flags}
    assert verdict.passed is False

    report_text = render(result, comparison, verdict)
    assert "buy-and-hold" in report_text.lower() or "buy and hold" in report_text.lower()
    assert "NIFTY" in report_text


# ---------------------------------------------------------------------------
# The verdict must name the benchmark it actually measured
#
# The flag text said "Simply holding NIFTY would have done better" regardless of
# what was compared. On a stock basket the benchmark is an equal-weight hold of
# those same stocks, so the sentence was plainly untrue -- and a verdict a
# beginner cannot trust literally is worse than no verdict at all.
# ---------------------------------------------------------------------------


def test_verdict_names_the_benchmark_it_was_given():
    from nlt.report.verdict import assess

    idx = _bdate_index(400)
    result = _build_result(
        trades=[_mk_trade(idx, i * 6, net_pnl=-500.0) for i in range(60)],
        equity=_flat_equity(400, end_capital=70_000.0),
    )
    comparison = _comparison(
        result, beats=False, benchmark_name="holding all 100 of these stocks equally"
    )
    verdict = assess(result, comparison)

    text = verdict.summary + " ".join(f.headline + f.detail for f in verdict.flags)
    assert "all 100 of these stocks" in text
    assert "NIFTY" not in text, "the verdict names NIFTY when the benchmark was a basket"


def test_benchmark_name_flows_from_the_benchmark_into_the_comparison():
    from nlt.report.benchmark import buy_and_hold, compare

    bars = _mkbars(120)
    result = _build_result(
        trades=[_mk_trade(bars.index, 0, net_pnl=100.0)],
        equity=_flat_equity(120, end_capital=101_000.0),
    )
    benchmark = buy_and_hold(bars, 100_000.0, name="holding TCS")
    assert compare(result, benchmark, bars).benchmark_name == "holding TCS"


# ---------------------------------------------------------------------------
# Options on a small account
#
# Not a preference but arithmetic: a NIFTY lot is 75 units and indivisible, so a
# small account either cannot afford one or must put far too much of itself into
# a single contract that can lose its whole premium in a session.
#
# This sits above the numbers rather than below them, because the conclusion is
# that the instrument is unsuitable -- something a return figure cannot express
# and might actively hide by looking good.
# ---------------------------------------------------------------------------


def _with_warning(warning: str) -> BacktestResult:
    idx = _bdate_index(400)
    return _build_result(
        trades=[_mk_trade(idx, i * 6, net_pnl=250.0) for i in range(60)],
        equity=_flat_equity(400, end_capital=115_000.0),
        warnings=[warning],
    )


def test_small_account_options_warning_becomes_a_critical_flag():
    from nlt.report.verdict import assess

    result = _with_warning("OPTIONS ON A SMALL ACCOUNT: this was tested with Rs 100,000.")
    verdict = assess(result, _comparison(result))

    flag = next(
        (f for f in verdict.flags if f.code == "options_need_a_bigger_account"), None
    )
    assert flag is not None, "an unsuitable instrument must be flagged"
    assert flag.severity == "critical"
    assert verdict.passed is False, (
        "a profitable backtest must still not pass when the instrument is "
        "unsuitable for the account"
    )


def test_no_small_account_flag_without_the_warning():
    from nlt.report.verdict import assess

    result = _with_warning("no charge_fn supplied; results exclude transaction costs")
    verdict = assess(result, _comparison(result))
    assert not any(f.code == "options_need_a_bigger_account" for f in verdict.flags)
