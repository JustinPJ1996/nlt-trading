"""Tests for `nlt.engine`: the backtest loop, condition evaluator and metrics.

Every test here is built to be capable of failing -- either by doing the fill
arithmetic explicitly and comparing against the engine's output, or by hand
constructing inputs whose correct answer is known before the engine runs.
`test_no_lookahead.py` already covers indicator prefix-invariance in isolation;
`test_full_backtest_matches_prefix_backtest` below is the equivalent check for
the whole engine (signal -> fill -> exit), which is where a look-ahead bug in
the *loop* itself -- as opposed to an indicator -- would actually show up.
"""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from nlt.data.session import NSE_EQUITY, session_date
from nlt.engine.backtest import Trade, run_backtest
from nlt.engine.conditions import evaluate
from nlt.engine.metrics import compute_metrics
from nlt.spec.models import (
    All,
    Any_,
    Compare,
    Const,
    ExitRules,
    IndicatorSpec,
    Instrument,
    Not,
    Ref,
    RiskLimits,
    Schedule,
    Sizing,
    StrategySpec,
)

# --------------------------------------------------------------------- helpers


def _mkbars(opens, highs, lows, closes, volumes=None) -> pd.DataFrame:
    """A minimal OHLCV frame over consecutive business days, starting a Monday.

    Business days only, so every bar's weekday falls in the default
    `schedule.weekdays` (Mon-Fri) without the test having to think about it.
    """
    n = len(opens)
    volumes = volumes if volumes is not None else [1_000_000.0] * n
    idx = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes},
        index=idx,
    )


def _always_true_entry() -> Compare:
    """An entry condition true on every bar, for tests that just need *a* signal
    and never let the position exit within the test window (so re-entry can't
    happen and confuse the trade count).
    """
    return Compare(op="gt", left=Ref(name="close"), right=Const(value=-1.0))


def _signal_once(bars: pd.DataFrame, at: int = 0) -> Compare:
    """An entry condition true on exactly one bar of `bars` (index `at`).

    For tests where the position is expected to exit *within* the test
    window: an always-true condition would let the engine correctly re-enter
    the instant that position closes (the signal is still true), which is
    real behaviour, not a bug -- but it would throw off a test that expects
    exactly one trade. `bars["close"].iloc[at]` must not repeat elsewhere in
    `bars["close"]`, which the caller is responsible for arranging.
    """
    value = float(bars["close"].iloc[at])
    assert (bars["close"] == value).sum() == 1, "signal bar's close must be unique"
    return Compare(op="eq", left=Ref(name="close"), right=Const(value=value))


def _spec(entry, exit_rules, **kwargs) -> StrategySpec:
    kwargs.setdefault("risk", RiskLimits(max_lots=100))
    return StrategySpec(
        name="test strategy",
        description="test",
        entry=entry,
        exit=exit_rules,
        **kwargs,
    )


# ----------------------------------------------------------------- fill timing


def test_entry_fills_on_bar_after_signal_at_that_bars_open():
    """The signal fires on bar 4 (close crosses above 120); the fill must be
    bar 5's open, adjusted for slippage -- never bar 4's close, and never
    bar 5's close.
    """
    opens = [100, 100, 100, 100, 150, 160]
    highs = [101, 101, 101, 101, 151, 165]
    lows = [99, 99, 99, 99, 149, 155]
    closes = [100, 100, 100, 100, 150, 155]
    bars = _mkbars(opens, highs, lows, closes)

    spec = _spec(
        entry=Compare(op="gt", left=Ref(name="close"), right=Const(value=120.0)),
        exit_rules=ExitRules(stop_pct=50.0, target_pct=50.0),
    )

    slippage_pct = 1.0
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=slippage_pct)

    assert len(res.trades) == 1
    trade = res.trades[0]

    # Signal is first true at bar index 4 (close=150 > 120); bars index 0-3
    # never satisfy it, so the only possible fill bar is index 5.
    expected_entry_time = bars.index[5]
    expected_entry_price = bars["open"].iloc[5] * (1.0 + slippage_pct / 100.0)

    assert trade.entry_time == expected_entry_time
    assert trade.entry_price == pytest.approx(expected_entry_price)
    # Not the signal bar's close, and not the fill bar's own close.
    assert trade.entry_price != pytest.approx(bars["close"].iloc[4])
    assert trade.entry_price != pytest.approx(bars["close"].iloc[5])


# -------------------------------------------------------------- no look-ahead


def test_full_backtest_matches_prefix_backtest(synthetic_bars):
    """The headline check: every trade that closes before the cut only ever
    depended on data available up to that point, so running on a longer
    history must reproduce it exactly.
    """
    n = 400
    spec = _spec(
        entry=Compare(op="crosses_below", left=Ref(name="close"), right=Ref(name="sma10")),
        exit_rules=ExitRules(stop_pct=2.0, target_pct=3.0),
        indicators=[IndicatorSpec(id="sma10", type="sma", params={"length": 10})],
    )

    res_full = run_backtest(spec, synthetic_bars, capital=100_000.0, lot_size=1)
    res_part = run_backtest(spec, synthetic_bars.iloc[:n], capital=100_000.0, lot_size=1)

    # A trade closing on the prefix's very last bar may only exist there
    # because the prefix forced it closed early (end_of_data); anything that
    # closed strictly before that is not a truncation artifact and must match.
    cutoff = synthetic_bars.index[n - 1]
    full_trades = [t for t in res_full.trades if t.exit_time < cutoff]
    part_trades = [t for t in res_part.trades if t.exit_time < cutoff]

    assert len(full_trades) > 0, "test is vacuous if no trade closes before the cut"
    assert len(full_trades) == len(part_trades)
    for tf, tp in zip(full_trades, part_trades):
        assert tf.entry_time == tp.entry_time
        assert tf.entry_price == pytest.approx(tp.entry_price)
        assert tf.exit_time == tp.exit_time
        assert tf.exit_price == pytest.approx(tp.exit_price)
        assert tf.exit_reason == tp.exit_reason
        assert tf.net_pnl == pytest.approx(tp.net_pnl)


# ------------------------------------------------------------------ tie-break


def test_stop_beats_target_when_both_hit_same_bar():
    opens = [100, 100, 95]
    highs = [101, 105, 96]
    lows = [99, 90, 94]
    closes = [100, 95, 95]
    bars = _mkbars(opens, highs, lows, closes)

    spec = _spec(
        entry=_signal_once(bars, at=0),
        exit_rules=ExitRules(stop_pct=1.0, target_pct=2.0),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    # entry_price = 100 (bar1 open, no slippage). Stop and target are both
    # measured from entry_price: stop = 99.0 (-1%), target = 102.0 (+2%).
    # Bar1's range [90, 105] reaches both, so the engine must report the stop.
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(99.0)
    assert "stop was assumed to have filled first" in " ".join(res.warnings)


# --------------------------------------------------------------- trailing stop


def test_trailing_stop_ratchets_and_never_moves_down():
    """Price rises to a high of 120, dips, then falls through the trail.

    The trail must ratchet up to the 120 high (not the dip, not the final
    bar's own high) and never retreat, so the exit reflects 120 * (1 - 5%),
    not some lower level a non-ratcheting or backward-moving trail would give.
    """
    opens = [100, 100, 100, 105, 108, 118]
    highs = [100, 100, 110, 107, 120, 116]
    lows = [100, 99, 105, 106, 115, 110]
    closes = [100, 100, 108, 104, 118, 112]
    bars = _mkbars(opens, highs, lows, closes)

    spec = _spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(trailing_stop_pct=5.0),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.entry_price == pytest.approx(100.0)
    assert trade.exit_reason == "trailing_stop"
    # Highest high seen over the trade's life is 120 (bar index 4), not the
    # exit bar's own high of 116 and not the pre-dip high of 110.
    assert trade.exit_price == pytest.approx(120.0 * 0.95)


# --------------------------------------------------------------------- warm-up


def test_no_trade_before_indicator_warm_up(synthetic_bars):
    spec = _spec(
        entry=Compare(op="gt", left=Ref(name="close"), right=Ref(name="sma200")),
        exit_rules=ExitRules(stop_pct=5.0, target_pct=5.0),
        indicators=[IndicatorSpec(id="sma200", type="sma", params={"length": 200, "source": "close"})],
    )
    res = run_backtest(spec, synthetic_bars, capital=100_000.0, lot_size=1)

    # SMA(200) is NaN for the first 199 bars (index 0..198); a NaN comparison
    # evaluates False (see conditions.py), so the earliest possible signal is
    # bar 199, and the earliest possible fill is bar 200.
    warm_up_cutoff = synthetic_bars.index[200]
    for trade in res.trades:
        assert trade.entry_time >= warm_up_cutoff


# -------------------------------------------------------------------- shorting


def test_short_direction_profits_when_price_falls():
    opens = [100, 100, 90, 80]
    highs = [101, 100, 91, 81]
    lows = [100, 99, 89, 79]
    closes = [101, 100, 90, 80]
    bars = _mkbars(opens, highs, lows, closes)

    spec = _spec(
        entry=_signal_once(bars, at=0),
        exit_rules=ExitRules(max_bars_held=2),
        direction="short",
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.direction == "short"
    assert trade.net_pnl > 0


# ---------------------------------------------------------------------- charges


def test_charges_are_subtracted_symmetrically(synthetic_bars):
    def flat_charge(price: float, quantity: int, side: str) -> float:
        return 20.0

    spec = _spec(
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=35.0)),
        exit_rules=ExitRules(stop_pct=1.5, target_pct=2.5),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
    )
    res = run_backtest(
        spec, synthetic_bars, capital=100_000.0, lot_size=1, charge_fn=flat_charge
    )

    assert len(res.trades) > 0, "test is vacuous with zero trades"
    for trade in res.trades:
        assert trade.charges == pytest.approx(40.0)
        assert trade.net_pnl == pytest.approx(trade.gross_pnl - trade.charges)

    assert res.metrics["total_charges"] == pytest.approx(40.0 * len(res.trades))
    assert "no charge_fn supplied" not in " ".join(res.warnings)


# ---------------------------------------------------------------------- sizing


def test_fixed_lots_sizing_uses_lots_times_lot_size():
    opens = [100, 100, 100]
    highs = [101, 101, 101]
    lows = [99, 99, 99]
    closes = [100, 100, 100]
    bars = _mkbars(opens, highs, lows, closes)

    spec = _spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(stop_pct=50.0),
        sizing=Sizing(mode="fixed_lots", lots=3),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=10, slippage_pct=0.0)

    assert len(res.trades) == 1
    assert res.trades[0].quantity == 30


def test_fixed_value_sizing_never_exceeds_value():
    opens = [100, 100, 100]
    highs = [101, 101, 101]
    lows = [99, 99, 99]
    closes = [100, 100, 100]
    bars = _mkbars(opens, highs, lows, closes)

    value = 5_000.0
    spec = _spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(stop_pct=50.0),
        sizing=Sizing(mode="fixed_value", value=value),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=10, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.quantity * trade.entry_price <= value
    # floor(5000 / 100 / 10) = 5 lots of 10 = 50 units, and it must not have
    # rounded up to 6 lots (60 units, worth 6000 -- over the stated value).
    assert trade.quantity == 50


def test_risk_based_sizing_targets_risk_pct_of_capital():
    opens = [100, 100, 100]
    highs = [101, 101, 101]
    lows = [99, 99, 99]
    closes = [100, 100, 100]
    bars = _mkbars(opens, highs, lows, closes)

    capital = 100_000.0
    risk_pct = 1.0
    stop_pct = 2.0
    lot_size = 10
    spec = _spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(stop_pct=stop_pct),
        sizing=Sizing(mode="risk_based", risk_pct=risk_pct),
    )
    res = run_backtest(spec, bars, capital=capital, lot_size=lot_size, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    stop_distance = trade.entry_price * stop_pct / 100.0
    risk_amount = risk_pct / 100.0 * capital
    actual_risk = trade.quantity * stop_distance
    # Quantity is floored to whole lots, so the actual risk can undershoot the
    # target by at most one lot's worth -- never overshoot it.
    assert actual_risk <= risk_amount + stop_distance
    assert actual_risk == pytest.approx(risk_amount, rel=0.2)


# ----------------------------------------------------------------- affordability


def test_unaffordable_position_is_not_opened():
    opens = [200.0, 200.0, 200.0]
    highs = [201.0, 201.0, 201.0]
    lows = [199.0, 199.0, 199.0]
    closes = [200.0, 200.0, 200.0]
    bars = _mkbars(opens, highs, lows, closes)

    # 1 lot of 1000 units at price ~200 costs ~Rs 200,000 against Rs 100,000
    # of capital -- not even one lot is affordable.
    spec = _spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(stop_pct=50.0),
        sizing=Sizing(mode="fixed_lots", lots=1),
    )
    res = run_backtest(spec, bars, capital=100_000.0, lot_size=1000, slippage_pct=0.0)

    assert res.trades == []
    assert any("not enough spare capital" in w for w in res.warnings)


def test_unaffordable_position_is_capped_to_what_capital_allows():
    opens = [200.0, 200.0, 200.0]
    highs = [201.0, 201.0, 201.0]
    lows = [199.0, 199.0, 199.0]
    closes = [200.0, 200.0, 200.0]
    bars = _mkbars(opens, highs, lows, closes)

    # 8 lots of 100 units at price ~200 would cost Rs 160,000 against Rs
    # 100,000 of capital -- unaffordable, but 5 lots (Rs 100,000) fit exactly,
    # so the engine should reduce to 5 lots rather than skip the trade.
    spec = _spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(stop_pct=50.0),
        sizing=Sizing(mode="fixed_lots", lots=8),
        risk=RiskLimits(max_lots=100),
    )
    res = run_backtest(spec, bars, capital=100_000.0, lot_size=100, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.quantity == 500
    assert trade.quantity < 800  # less than the 8 lots requested
    assert trade.quantity * trade.entry_price <= 100_000.0
    assert any("sized down" in w for w in res.warnings)


# -------------------------------------------------------------------- equity floor


def test_equity_never_goes_negative_long_only(synthetic_bars):
    spec = _spec(
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30.0)),
        exit_rules=ExitRules(stop_pct=1.0, target_pct=2.0),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        direction="long",
    )
    res = run_backtest(spec, synthetic_bars, capital=100_000.0)

    assert not res.equity.isna().any()
    assert (res.equity >= 0).all()


# ------------------------------------------------------------------------ metrics


def _trade(net_pnl: float, gross_pnl: float | None = None, bars_held: int = 1) -> Trade:
    gross_pnl = net_pnl if gross_pnl is None else gross_pnl
    ts = pd.Timestamp("2024-01-01", tz="Asia/Kolkata")
    return Trade(
        entry_time=ts,
        entry_price=100.0,
        exit_time=ts + pd.Timedelta(days=bars_held),
        exit_price=100.0 + net_pnl,
        direction="long",
        quantity=1,
        gross_pnl=gross_pnl,
        charges=gross_pnl - net_pnl,
        net_pnl=net_pnl,
        exit_reason="target",
        bars_held=bars_held,
        entry_reason="test",
    )


def test_metrics_on_hand_built_trades():
    trades = [_trade(500.0, bars_held=5), _trade(-200.0, bars_held=3), _trade(300.0, bars_held=4)]
    equity = pd.Series(
        [100_000.0, 100_500.0, 100_300.0, 100_600.0],
        index=pd.date_range("2024-01-01", periods=4, tz="Asia/Kolkata"),
    )
    metrics = compute_metrics(trades, equity, capital=100_000.0, bars_per_year=252)

    assert metrics["total_trades"] == 3
    assert metrics["winning_trades"] == 2
    assert metrics["losing_trades"] == 1
    assert metrics["win_rate_pct"] == pytest.approx(100.0 * 2 / 3)
    assert metrics["profit_factor"] == pytest.approx((500.0 + 300.0) / 200.0)
    assert metrics["expectancy"] == pytest.approx((500.0 - 200.0 + 300.0) / 3)
    assert metrics["avg_win"] == pytest.approx((500.0 + 300.0) / 2)
    assert metrics["avg_loss"] == pytest.approx(-200.0)
    assert metrics["best_trade"] == pytest.approx(500.0)
    assert metrics["worst_trade"] == pytest.approx(-200.0)
    assert metrics["avg_bars_held"] == pytest.approx((5 + 3 + 4) / 3)
    assert metrics["total_return_pct"] == pytest.approx(0.6)


def test_metrics_on_zero_trades_is_all_zero_and_never_nan():
    equity = pd.Series(
        [100_000.0] * 5, index=pd.date_range("2024-01-01", periods=5, tz="Asia/Kolkata")
    )
    metrics = compute_metrics([], equity, capital=100_000.0, bars_per_year=252)

    assert metrics["total_trades"] == 0
    assert metrics["win_rate_pct"] == 0
    assert metrics["profit_factor"] is None
    assert metrics["sharpe"] is None
    assert metrics["total_return_pct"] == pytest.approx(0.0)
    for key, value in metrics.items():
        if isinstance(value, float):
            assert not math.isnan(value), f"{key} is NaN"


# -------------------------------------------------------------- real-data smoke


def test_real_nifty_rsi_strategy_gives_believable_metrics(nifty_bars):
    """This is the exact shape of strategy that produced -113% total return
    and -108.8% max drawdown before the affordability and lot_size fixes.
    """
    spec = _spec(
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30.0)),
        exit_rules=ExitRules(stop_pct=1.0, target_pct=2.0),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        instrument=Instrument(symbol="NIFTY", trade_as="index"),
    )
    res = run_backtest(spec, nifty_bars, capital=100_000.0)

    assert 10 < res.metrics["total_trades"] < 2000
    assert not res.equity.isna().any()
    for trade in res.trades:
        # Stops/targets can trigger intrabar on the very bar a position fills
        # (see backtest.py's module docstring), so entry_time == exit_time is
        # a legitimate same-bar stop-out, not a bug -- exit strictly *before*
        # entry is what would indicate a look-ahead violation.
        assert trade.exit_time >= trade.entry_time
    assert -100.0 <= res.metrics["total_return_pct"] <= 10_000.0


# ------------------------------------------------------------- condition evaluator


def _features(**columns) -> pd.DataFrame:
    return pd.DataFrame(columns)


def test_crosses_above_fires_exactly_at_the_crossover():
    features = _features(fast=[1.0, 3.0, 2.0, 4.0, 5.0], slow=[2.0, 2.0, 2.0, 2.0, 2.0])
    condition = Compare(op="crosses_above", left=Ref(name="fast"), right=Ref(name="slow"))
    result = evaluate(condition, features)
    assert result.tolist() == [False, True, False, True, False]


def test_nan_operand_yields_false_not_error():
    features = _features(a=[np.nan, 1.0, 2.0])
    condition = Compare(op="gt", left=Ref(name="a"), right=Const(value=0.0))
    result = evaluate(condition, features)
    assert result.tolist() == [False, True, True]


def test_all_any_not_truth_tables():
    features = _features(x=[1.0, 0.0, 1.0, 0.0], y=[1.0, 1.0, 0.0, 0.0])
    c1 = Compare(op="gt", left=Ref(name="x"), right=Const(value=0.5))
    c2 = Compare(op="gt", left=Ref(name="y"), right=Const(value=0.5))

    assert evaluate(All(conditions=[c1, c2]), features).tolist() == [True, False, False, False]
    assert evaluate(Any_(conditions=[c1, c2]), features).tolist() == [True, True, True, False]
    assert evaluate(Not(condition=c1), features).tolist() == [False, True, False, True]


def test_bars_ago_shifts_the_referenced_series():
    features = _features(close=[5.0, 3.0, 4.0, 2.0, 6.0])
    condition = Compare(op="gt", left=Ref(name="close"), right=Ref(name="close", bars_ago=1))
    result = evaluate(condition, features)
    # close[i] > close[i-1]; index 0 has no prior bar (NaN -> False).
    assert result.tolist() == [False, False, True, False, True]


# ---------------------------------------------------------------------------
# Concurrency and capital netting
#
# The -113% bug was a single position sized beyond the account. The same bug
# returns through a different door when several positions are open at once: if
# each one is sized against the full account balance rather than against what is
# actually free, three "affordable" positions can still add up to 3x leverage.
#
# The engine nets out committed capital correctly, but nothing tested it -- I
# removed the netting and the whole suite stayed green. These tests close that.
# ---------------------------------------------------------------------------

def _always_entering_spec(max_concurrent: int, hold_bars: int = 30) -> StrategySpec:
    """A strategy that wants to enter on every bar and holds for a long time."""
    return StrategySpec.model_validate(
        {
            "name": "stacker",
            "description": "enters constantly so positions accumulate",
            "instrument": {"symbol": "NIFTY", "timeframe": "1d", "trade_as": "index"},
            "indicators": [],
            "entry": {
                "kind": "compare",
                "op": "gt",
                "left": {"kind": "ref", "name": "close"},
                "right": {"kind": "const", "value": 0},
            },
            "exit": {"max_bars_held": hold_bars},
            "sizing": {"mode": "fixed_lots", "lots": 1},
            "risk": {"max_concurrent_positions": max_concurrent, "max_lots": 10},
        }
    )


def _rising_bars(n: int = 60) -> pd.DataFrame:
    close = pd.Series(np.linspace(1000.0, 1200.0, n))
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="Asia/Kolkata")
    close.index = idx
    return pd.DataFrame(
        {"open": close, "high": close * 1.02, "low": close * 0.99,
         "close": close, "volume": 1e6},
        index=idx,
    )


def _peak_exposure(trades) -> float:
    """Largest simultaneous position value across the whole run."""
    events = []
    for t in trades:
        events.append((t.entry_time, t.quantity * t.entry_price))
        events.append((t.exit_time, -t.quantity * t.entry_price))
    events.sort()

    current = peak = 0.0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def test_concurrent_positions_never_exceed_capital_in_aggregate():
    """Several positions open at once must still fit inside the account.

    Capital is deliberately set to fund only about four units, while the spec
    asks for five concurrent positions -- so the limit has to bind.
    """
    capital = 5_000.0
    result = run_backtest(_always_entering_spec(max_concurrent=5), _rising_bars(),
                          capital=capital, lot_size=1)

    assert len(result.trades) > 1, "test is vacuous unless positions actually stack"
    peak = _peak_exposure(result.trades)
    assert peak <= capital * 1.001, (
        f"peak simultaneous exposure {peak:,.0f} exceeds capital {capital:,.0f} "
        f"({peak / capital:.2f}x leverage) -- committed capital is not being netted out"
    )


def test_concurrency_limit_is_respected():
    """Never more open positions than the spec allows."""
    result = run_backtest(_always_entering_spec(max_concurrent=2), _rising_bars(),
                          capital=1_000_000.0, lot_size=1)

    events = []
    for t in result.trades:
        events.append((t.entry_time, 1))
        events.append((t.exit_time, -1))
    events.sort()

    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    assert peak <= 2, f"{peak} positions open at once, limit was 2"


def test_single_position_case_still_bounded():
    """The original -113% shape: one position, index priced far above capital."""
    result = run_backtest(_always_entering_spec(max_concurrent=1), _rising_bars(),
                          capital=500.0, lot_size=1)
    assert _peak_exposure(result.trades) <= 500.0 * 1.001


# ---------------------------------------------------------------------------
# Current-bar look-ahead
#
# The prefix-invariance test compares a truncated run against a full one, so it
# only detects reads of *future* bars. Reading the entry bar's own close -- which
# has not happened when the order fills at that bar's open -- is invisible to it.
# The engine did exactly that when setting stop distances. These test it directly.
# ---------------------------------------------------------------------------


def test_stop_is_measured_from_entry_price_not_the_entry_bars_close():
    """Two runs identical up to the entry bar's close must place the same stop.

    The bars differ only in what happens AFTER the fill at bar 1's open. If the
    stop distance depended on that bar's close, the two stops would differ.
    """
    common_open = [100.0, 100.0]

    # Same open, wildly different closes on the entry bar.
    fell = _mkbars(common_open + [95.0], [101.0, 105.0, 96.0], [99.0, 90.0, 94.0],
                   [100.0, 91.0, 95.0])
    rose = _mkbars(common_open + [95.0], [101.0, 105.0, 96.0], [99.0, 90.0, 94.0],
                   [100.0, 104.0, 95.0])

    stops = []
    for bars in (fell, rose):
        spec = _spec(entry=_signal_once(bars, at=0),
                     exit_rules=ExitRules(stop_pct=1.0, target_pct=50.0))
        res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)
        assert res.trades and res.trades[0].exit_reason == "stop"
        stops.append(res.trades[0].exit_price)

    assert stops[0] == pytest.approx(stops[1]), (
        f"stop moved from {stops[0]} to {stops[1]} purely because the entry bar "
        "closed differently -- the stop is being set with knowledge of the future"
    )
    assert stops[0] == pytest.approx(99.0), "stop should be 1% below the 100.0 fill"


def test_stop_and_target_are_measured_from_the_signal_close():
    """"1% stop, 2% target" must be two percentages of the SAME number.

    That number is the signal bar's close -- the price at which the user's
    condition became true. Bar 0 closes at 100 and the fill happens at bar 1's
    open of 110, so the two bases are distinguishable: a 10% stop is 10 points
    (10% of 100), not 11 (10% of 110).
    """
    bars = _mkbars([100.0, 110.0, 110.0], [101.0, 140.0, 111.0],
                   [99.0, 80.0, 109.0], [100.0, 110.0, 110.0])

    spec = _spec(entry=_signal_once(bars, at=0),
                 exit_rules=ExitRules(stop_pct=10.0, target_pct=20.0))
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    trade = res.trades[0]
    signal_close = 100.0
    # Bar 1 spans [80, 140] so both levels are reachable; the stop wins the tie-break.
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(trade.entry_price - signal_close * 0.10), (
        f"stop at {trade.exit_price} is not 10% of the signal close ({signal_close}) "
        f"below the {trade.entry_price} fill"
    )


def test_stop_basis_is_the_signal_close_not_the_fill():
    """A gap between signal and fill must not silently rescale the stop distance.

    Two runs share an identical signal bar but gap to different opens. The stop
    DISTANCE must be the same in both, because it is derived from the signal bar
    they share -- even though the resulting levels differ.
    """
    distances = []
    for gap_open in (110.0, 130.0):
        bars = _mkbars(
            [100.0, gap_open, gap_open],
            [101.0, gap_open + 40.0, gap_open + 1.0],
            [99.0, 10.0, gap_open - 1.0],
            [100.0, gap_open, gap_open],
        )
        spec = _spec(entry=_signal_once(bars, at=0),
                     exit_rules=ExitRules(stop_pct=10.0, target_pct=90.0))
        res = run_backtest(spec, bars, capital=10_000_000.0, lot_size=1, slippage_pct=0.0)
        trade = res.trades[0]
        assert trade.exit_reason == "stop"
        distances.append(trade.entry_price - trade.exit_price)

    assert distances[0] == pytest.approx(distances[1]), (
        f"stop distance changed from {distances[0]} to {distances[1]} because the fill "
        "gapped -- it should be fixed by the signal bar the two runs share"
    )
    assert distances[0] == pytest.approx(10.0), "10% of the 100.0 signal close"


# ---------------------------------------------------------------------------
# Gaps through stops and targets
#
# The engine used to fill a stop at the stop level whenever the bar's range
# touched it -- including when the bar OPENED well beyond it. That reports a
# fill at a price which never traded, and it is the most flattering lie a
# backtester can tell: it makes every stop-loss strategy look as though its
# worst case is bounded by the stop, when overnight gaps are precisely when
# large losses occur.
#
# On the real NIFTY RSI strategy this understated one loss by a factor of ten.
# ---------------------------------------------------------------------------


def _gap_spec() -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "gap", "description": "x",
            "instrument": {"symbol": "NIFTY", "timeframe": "1d", "trade_as": "index"},
            "indicators": [],
            "entry": {"kind": "compare", "op": "gt",
                      "left": {"kind": "ref", "name": "close"},
                      "right": {"kind": "const", "value": 99.9}},
            "exit": {"stop_pct": 1.0, "target_pct": 5.0},
            "sizing": {"mode": "fixed_lots", "lots": 1},
            "risk": {"max_concurrent_positions": 1},
        }
    )


def test_gap_down_through_stop_fills_at_the_open_not_the_stop():
    """Entry at 100, stop at 99, market opens at 90. The fill must be 90."""
    bars = _mkbars(
        [100.0, 100.0, 90.0, 90.0],
        [100.5, 100.5, 90.5, 90.5],
        [99.5, 99.5, 89.0, 89.0],
        [100.0, 100.0, 90.0, 90.0],
    )
    res = run_backtest(_gap_spec(), bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    trade = res.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(90.0), (
        f"filled at {trade.exit_price}, a price that never traded -- the bar "
        "opened at 90.0, below the 99.0 stop"
    )
    assert trade.gross_pnl == pytest.approx(-10.0)


def test_stop_inside_the_bar_still_fills_at_the_stop_level():
    """The ordinary case must not regress: a bar that trades down through the
    stop, having opened above it, fills at the stop."""
    bars = _mkbars(
        [100.0, 100.0, 99.5, 99.5],
        [100.5, 100.5, 100.0, 100.0],
        [99.5, 98.0, 99.0, 99.0],
        [100.0, 99.5, 99.5, 99.5],
    )
    res = run_backtest(_gap_spec(), bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    trade = res.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(99.0), "opened above the stop, so fills at it"


def test_gap_up_past_target_fills_at_the_open_in_our_favour():
    """The mirror case. Modelling only adverse gaps would bias results the other way."""
    bars = _mkbars(
        [100.0, 100.0, 120.0, 120.0],
        [100.5, 100.5, 121.0, 121.0],
        [99.5, 99.9, 119.0, 119.0],
        [100.0, 100.0, 120.0, 120.0],
    )
    res = run_backtest(_gap_spec(), bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    trade = res.trades[0]
    assert trade.exit_reason == "target"
    assert trade.exit_price == pytest.approx(120.0), (
        f"filled at {trade.exit_price}; the bar opened at 120.0, already past the "
        "105.0 target, so that is the realistic fill"
    )


def test_short_gap_up_through_stop_fills_at_the_open():
    """Shorts gap the other way, and the engine must be symmetric about it."""
    bars = _mkbars(
        [100.0, 100.0, 115.0, 115.0],
        [100.5, 100.5, 116.0, 116.0],
        [99.5, 99.5, 114.0, 114.0],
        [100.0, 100.0, 115.0, 115.0],
    )
    spec = _gap_spec().model_copy(update={"direction": "short"})
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    trade = res.trades[0]
    assert trade.exit_reason == "stop"
    assert trade.exit_price == pytest.approx(115.0)
    assert trade.gross_pnl == pytest.approx(-15.0)


# ---------------------------------------------------------------------------
# Session-aware intraday backtesting
#
# NSE cash/F&O runs 09:15-15:30, so a 15-minute bar timeframe has 25 bars per
# session (09:15, 09:30, ... 15:15 -- each labelled by its start, covering up
# to the close). These helpers build that shape directly rather than through
# any resampling, so a test's inputs are exactly the bars the assertions
# reason about.
# ---------------------------------------------------------------------------


def _session_bar_times(last_time: dt.time = dt.time(15, 15)) -> list[dt.time]:
    """15-minute bar start times from NSE's 09:15 open through `last_time` inclusive.

    A full session (`last_time=15:15`) yields 25 times; a shorter `last_time`
    models an early-close/muhurat day with fewer bars, without needing a
    second code path.
    """
    times = []
    minutes = 9 * 60 + 15
    end_minutes = last_time.hour * 60 + last_time.minute
    while minutes <= end_minutes:
        times.append(dt.time(minutes // 60, minutes % 60))
        minutes += 15
    return times


def _mkintraday_bars(day_specs: list[tuple[dt.date, dt.time, list[float]]]) -> pd.DataFrame:
    """Build 15m OHLCV bars for several trading days.

    `day_specs` is a list of `(date, last_bar_time, closes)`. `closes` must have
    exactly as many entries as `_session_bar_times(last_bar_time)` produces, so a
    short session is expressed simply by giving it fewer closes and an earlier
    `last_bar_time` -- there is no separate "half day" flag to get wrong.
    OHLC is a tiny flat range around each close (open == close, high/low +-1):
    these tests are about signal timing and session boundaries, not price
    action shape, and a flat range keeps stop/target arithmetic out of the way
    unless a test deliberately widens a bar's high/low.
    """
    rows = []
    idx = []
    for day, last_time, closes in day_specs:
        times = _session_bar_times(last_time)
        assert len(times) == len(closes), (
            f"{day}: expected {len(times)} closes for a session ending {last_time}, "
            f"got {len(closes)}"
        )
        for t, close in zip(times, closes):
            ts = pd.Timestamp.combine(day, t).tz_localize("Asia/Kolkata")
            idx.append(ts)
            rows.append(
                {
                    "open": close,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 100_000.0,
                }
            )
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx, name="ts"))


def _full_day(day: dt.date, closes: list[float]) -> tuple[dt.date, dt.time, list[float]]:
    return (day, dt.time(15, 15), closes)


def _flat_day(day: dt.date, price: float = 100.0, n: int = 25) -> tuple[dt.date, dt.time, list[float]]:
    return (day, dt.time(15, 15), [price] * n)


def _intraday_spec(entry, exit_rules, **kwargs) -> StrategySpec:
    kwargs.setdefault("risk", RiskLimits(max_lots=100))
    kwargs.setdefault("instrument", Instrument(symbol="NIFTY", trade_as="index", timeframe="15m"))
    return StrategySpec(
        name="intraday test strategy", description="test", entry=entry, exit=exit_rules, **kwargs
    )


def test_square_off_fires_and_beats_a_reachable_target():
    """Square-off must win even when a target is also reachable on that same bar."""
    day = dt.date(2024, 1, 8)  # a Monday
    closes = [50.0] + [100.0] * 24
    bars = _mkintraday_bars([_full_day(day, closes)])
    # Bump the final bar's high above where the 1% target sits, so the target
    # is reachable on the exact bar the square-off clock also fires on.
    bars.iloc[-1, bars.columns.get_loc("high")] = 105.0

    spec = _intraday_spec(
        entry=_signal_once(bars, at=0),
        # 5% of the 50.0 signal close = 2.5, so the 102.5 target is out of
        # reach of every bar's flat high (101.0) except the one we bumped.
        exit_rules=ExitRules(target_pct=5.0),
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.exit_reason == "square_off"
    assert trade.exit_time.time() == dt.time(15, 15)
    # The square-off fill is the bar's close (100.0), not the ~102.5 target.
    assert trade.exit_price == pytest.approx(100.0)


def test_square_off_on_short_session_fires_on_its_last_bar():
    """A day that ends at 12:15 (muhurat/early close) must still force the exit.

    `square_off` defaults to 15:15, which this session never reaches -- the
    engine has to notice this is the session's *last* bar and force the exit
    there instead of leaving the position open into a session that, in this
    test, never comes.
    """
    day = dt.date(2024, 1, 8)
    closes = [50.0] + [100.0] * (len(_session_bar_times(dt.time(12, 15))) - 1)
    bars = _mkintraday_bars([(day, dt.time(12, 15), closes)])

    spec = _intraday_spec(
        entry=_signal_once(bars, at=0),
        exit_rules=ExitRules(stop_pct=50.0),  # never reachable; only square-off can close this
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert len(res.trades) == 1
    trade = res.trades[0]
    assert trade.exit_reason == "square_off"
    assert trade.exit_time.time() == dt.time(12, 15)
    assert trade.exit_time.time() < spec.schedule.square_off


def test_no_entry_after_cutoff_blocks_a_signal_at_the_cutoff():
    """A signal exactly AT `no_entry_after` (default 15:00) must not open a position.

    `no_entry_after` is documented as an "at or after" cutoff; the engine used
    to accept entries up to and including that exact time (`<=` instead of
    `<`), which let a 15:00 signal open a position 15:00 was supposed to rule
    out.
    """
    day = dt.date(2024, 1, 8)
    closes = [100.0] * 23 + [77.0, 100.0]
    bars = _mkintraday_bars([_full_day(day, closes)])
    assert bars.index[23].time() == dt.time(15, 0)

    spec = _intraday_spec(
        entry=_signal_once(bars, at=23),
        exit_rules=ExitRules(stop_pct=50.0),
        schedule=Schedule(),  # default no_entry_after == 15:00
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert res.trades == []


def test_no_overnight_carry_across_multiple_sessions():
    """Over a multi-day intraday run, every trade's entry and exit share a session date."""
    days = [d.date() for d in pd.bdate_range(dt.date(2024, 1, 8), periods=5)]
    day_specs = [_flat_day(day, price=100.0 + i) for i, day in enumerate(days)]
    bars = _mkintraday_bars(day_specs)

    spec = _intraday_spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(max_bars_held=3),
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert len(res.trades) > 1, "test is vacuous with too few trades"
    for trade in res.trades:
        assert session_date(trade.entry_time, NSE_EQUITY) == session_date(
            trade.exit_time, NSE_EQUITY
        ), f"trade {trade} carried a position across a session boundary"


def test_signal_on_last_bar_of_session_does_not_fill_at_next_mornings_open():
    """The session-boundary entry problem: pick (a), skip the entry outright.

    A signal on a session's last bar has no valid intraday fill -- the only bar
    left to fill it on is the next morning's open, 14+ hours (a full overnight
    gap) away. This asserts the chosen behaviour: no trade is opened, and the
    skip is counted in `warnings` rather than silently dropped.
    """
    # day1 is a deliberately short session ending well before the default
    # `no_entry_after` (15:00), so `no_entry_after` cannot be what blocks this
    # entry -- if the trade is skipped here, it is the session-boundary check
    # doing it, not the unrelated time-of-day cutoff (the two would otherwise
    # be indistinguishable, since a *full* session's last bar is 15:15, always
    # at or after `no_entry_after` by the model's own validator).
    day1 = dt.date(2024, 1, 8)
    day2 = dt.date(2024, 1, 9)
    last_time = dt.time(11, 0)
    n1 = len(_session_bar_times(last_time))
    closes_day1 = [100.0] * (n1 - 1) + [42.0]
    closes_day2 = [200.0] * 25
    bars = _mkintraday_bars([(day1, last_time, closes_day1), _full_day(day2, closes_day2)])
    assert bars.index[n1 - 1].time() < dt.time(15, 0)

    spec = _intraday_spec(
        entry=_signal_once(bars, at=n1 - 1),  # day1's last bar
        exit_rules=ExitRules(stop_pct=50.0),
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert res.trades == []
    assert any("last bar of a trading session" in w for w in res.warnings)


def test_weekend_gap_not_treated_as_adjacent_bars():
    """Friday's last bar and Monday's first bar are adjacent ROWS but not adjacent
    session time -- the boundary-entry skip must fire across the weekend too,
    not just overnight. Friday is again a short session ending before
    `no_entry_after`, isolating the session-boundary check from the unrelated
    time-of-day cutoff (see the comment in the test above)."""
    friday = dt.date(2024, 1, 12)
    monday = dt.date(2024, 1, 15)
    last_time = dt.time(11, 0)
    n1 = len(_session_bar_times(last_time))
    closes_fri = [100.0] * (n1 - 1) + [17.0]
    closes_mon = [300.0] * 25
    bars = _mkintraday_bars([(friday, last_time, closes_fri), _full_day(monday, closes_mon)])

    assert bars.index[n1 - 1].date() == friday
    assert bars.index[n1].date() == monday

    spec = _intraday_spec(
        entry=_signal_once(bars, at=n1 - 1),
        exit_rules=ExitRules(stop_pct=50.0),
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    # If the engine treated row `n1` as simply "the next bar" without regard
    # for the 3-calendar-day gap behind it, this would open a Monday-morning
    # position off a Friday signal -- a weekend held as if it were the usual
    # 15 minutes between bars.
    assert res.trades == []


def test_partial_last_bar_is_dropped_and_warned_about():
    """`drop_partial_last_bar=True` (the default) must ignore a still-forming final bar."""
    future_day = dt.date(2099, 1, 5)  # a Monday, safely after "now"
    closes = [100.0, 100.0, 100.0]
    bars = _mkintraday_bars([(future_day, dt.time(9, 45), closes)])

    spec = _intraday_spec(
        entry=_always_true_entry(), exit_rules=ExitRules(stop_pct=50.0), schedule=Schedule()
    )

    res_dropped = run_backtest(
        spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0,
        drop_partial_last_bar=True,
    )
    assert len(res_dropped.equity) == 2, "the 3rd (still-forming) bar should have been dropped"
    assert any("still-forming" in w for w in res_dropped.warnings)

    res_kept = run_backtest(
        spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0,
        drop_partial_last_bar=False,
    )
    assert len(res_kept.equity) == 3, "with the guard off, all 3 bars should remain"
    assert not any("still-forming" in w for w in res_kept.warnings)


def test_daily_nifty_rsi_regression_pinned(nifty_bars):
    """Pins the exact daily-bar numbers the session-awareness change must not move.

    `test_real_nifty_rsi_strategy_gives_believable_metrics` above only checks
    loose bounds, which would not catch a shift caused by this change. Every
    piece of new session logic is gated behind `instrument.timeframe != "1d"`
    (`is_intraday_tf`), so a daily-bar run should take none of the new code
    paths -- these exact numbers are the proof.
    """
    spec = _spec(
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30.0)),
        exit_rules=ExitRules(stop_pct=1.0, target_pct=2.0),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        instrument=Instrument(symbol="NIFTY", trade_as="index"),
    )
    res = run_backtest(spec, nifty_bars, capital=100_000.0)

    assert res.metrics["total_trades"] == 56
    assert res.metrics["total_return_pct"] == pytest.approx(-2.1177329341105504)


def test_long_lookback_indicator_warns_how_many_sessions_it_spans():
    """A 200-period SMA on 15m bars spans 200 / 25 = 8 trading sessions.

    Users reason in "200 periods" as roughly "200 days"; on 15m bars it is
    really 8 trading days, smoothed across every overnight gap in between as
    though they were consecutive minutes. This warning is informational only
    (indicators themselves are out of bounds for this change) -- it exists so
    the surprise is visible instead of silent.
    """
    day = dt.date(2024, 1, 8)
    bars = _mkintraday_bars([_flat_day(day, price=100.0)])

    spec = _intraday_spec(
        entry=_always_true_entry(),
        exit_rules=ExitRules(stop_pct=50.0),
        indicators=[IndicatorSpec(id="sma200", type="sma", params={"length": 200})],
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)

    assert any("200 bars" in w and "8 trading session" in w for w in res.warnings)


def test_intraday_end_to_end_produces_sane_trades():
    """A full run on constructed 15m bars: trades happen, equity is never NaN,
    and no trade exits before it entered."""
    rng = np.random.default_rng(2026)
    days = [d.date() for d in pd.bdate_range(dt.date(2024, 1, 8), periods=10)]
    day_specs = []
    price = 100.0
    for day in days:
        closes = []
        for _ in range(25):
            price += rng.normal(0, 0.5)
            closes.append(price)
        day_specs.append(_full_day(day, closes))
    bars = _mkintraday_bars(day_specs)

    spec = _intraday_spec(
        entry=Compare(op="crosses_below", left=Ref(name="close"), right=Ref(name="sma5")),
        exit_rules=ExitRules(stop_pct=1.0, target_pct=2.0, max_bars_held=10),
        indicators=[IndicatorSpec(id="sma5", type="sma", params={"length": 5})],
        schedule=Schedule(),
    )
    res = run_backtest(spec, bars, capital=1_000_000.0)

    assert len(res.trades) > 0, "test is vacuous with zero trades"
    assert not res.equity.isna().any()
    for trade in res.trades:
        # Same-bar stop-outs are legitimate (see the NIFTY smoke test's
        # comment above): a position can fill and hit its stop within the
        # very bar it opens on. What must never happen is an exit strictly
        # BEFORE its entry.
        assert trade.exit_time >= trade.entry_time


# ---------------------------------------------------------------------------
# The overnight-carry safety net, tested at its trigger
#
# Every other intraday test exercises this invariant as a *passing* condition:
# square-off forces the exit, so it never fires. That leaves the net itself
# untested -- if the exit precedence ever broke, we would be relying on a check
# nobody had ever seen work.
#
# These two tests call the invariant directly and then verify the engine wires
# it in, which together prove the net catches what it claims to.
# ---------------------------------------------------------------------------


def test_overnight_carry_invariant_raises_on_a_boundary_crossing():
    """Called directly with a trade that spans two sessions, it must raise."""
    from nlt.data.session import NSE_EQUITY
    from nlt.engine.backtest import _assert_no_overnight_carry

    entry = pd.Timestamp("2026-09-21 14:45", tz="Asia/Kolkata")
    exit_ = pd.Timestamp("2026-09-22 09:15", tz="Asia/Kolkata")

    with pytest.raises(RuntimeError, match="carry a position across a session boundary"):
        _assert_no_overnight_carry(entry, exit_, NSE_EQUITY)


def test_overnight_carry_invariant_permits_a_same_session_trade():
    """The mirror: two bars on one trading day must not raise."""
    from nlt.data.session import NSE_EQUITY
    from nlt.engine.backtest import _assert_no_overnight_carry

    entry = pd.Timestamp("2026-09-21 09:30", tz="Asia/Kolkata")
    exit_ = pd.Timestamp("2026-09-21 15:15", tz="Asia/Kolkata")

    _assert_no_overnight_carry(entry, exit_, NSE_EQUITY)  # must not raise


def test_engine_actually_calls_the_overnight_carry_invariant(monkeypatch):
    """Break square-off, and the invariant must fire rather than the engine
    quietly producing an overnight trade.

    This is what makes the net real: with the forced exit disabled, an intraday
    run over several sessions must raise, not return a trade that spans days.
    """
    import nlt.engine.backtest as bt

    bars = _mkintraday_bars([
        _flat_day(dt.date(2026, 9, 21), 100.0),
        _flat_day(dt.date(2026, 9, 22), 100.0),
    ])
    # Enter on the first bar and never exit on merit, so only the forced
    # square-off can close the position -- which is what we are disabling.
    spec = _intraday_spec(
        entry=Compare(op="gt", left=Ref(name="close"), right=Const(value=0.0)),
        exit_rules=ExitRules(target_pct=100.0, stop_pct=99.0),
    )

    # Disable every square-off path: nothing forces the position shut at the close.
    monkeypatch.setattr(bt, "is_last_bar_of_session",
                        lambda index, session: pd.Series(False, index=index))
    original = bt._check_exit

    def no_square_off(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is not None and result[0] == "square_off":
            return None
        return result

    monkeypatch.setattr(bt, "_check_exit", no_square_off)

    with pytest.raises(RuntimeError, match="session boundary"):
        run_backtest(spec, bars, capital=1_000_000.0, lot_size=1, slippage_pct=0.0)


# ---------------------------------------------------------------------------
# The concentration rail
#
# Risk-based sizing with a tight stop produces a large position for the same
# rupee risk: a 1% stop on a 1% risk budget put 91% of the account into one
# NIFTY trade. The risk arithmetic was correct and the outcome was still
# reckless -- no room for a second position, and full exposure to an overnight
# gap that jumps straight past the stop that made the size look safe.
#
# The rail binds whatever the sizing mode computes, because a rail that yields
# to the thing it restrains is not a rail.
# ---------------------------------------------------------------------------


def _flat_price_bars(price: float, n: int = 40) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    px = pd.Series([price] * n, index=idx)
    return pd.DataFrame(
        {"open": px, "high": px * 1.02, "low": px * 0.98, "close": px, "volume": 1e6},
        index=idx,
    )


def _capped_spec(pct: float | None, *, lots: int = 10, trade_as: str = "stock") -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": "cap", "description": "x",
            "instrument": {"symbol": "TCS", "trade_as": trade_as},
            "indicators": [],
            "entry": {"kind": "compare", "op": "gt",
                      "left": {"kind": "ref", "name": "close"},
                      "right": {"kind": "const", "value": 0}},
            "exit": {"target_pct": 50.0, "stop_pct": 40.0},
            "sizing": {"mode": "fixed_lots", "lots": lots},
            "risk": {"max_position_pct": pct},
        }
    )


def test_concentration_rail_caps_an_explicitly_requested_size():
    """Asking for more than the rail allows must be cut down, not honoured."""
    # Rs 500 a share and 100 lots asks for Rs 50,000, well past the Rs 20,000 rail.
    bars = _flat_price_bars(500.0)
    result = run_backtest(_capped_spec(20.0, lots=100), bars, capital=100_000.0,
                          lot_size=1, slippage_pct=0.0)

    biggest = max(t.quantity * t.entry_price for t in result.trades)
    assert biggest <= 20_000.0 * 1.001, (
        f"largest position was Rs {biggest:,.0f}, above the 20% of Rs 1,00,000 rail"
    )


def test_without_the_rail_the_same_request_is_honoured():
    """The mirror -- proves the cap, not something else, is doing the work."""
    bars = _flat_price_bars(500.0)
    result = run_backtest(_capped_spec(None, lots=100), bars, capital=100_000.0,
                          lot_size=1, slippage_pct=0.0)

    biggest = max(t.quantity * t.entry_price for t in result.trades)
    assert biggest == pytest.approx(50_000.0), "the full request should go through"


def test_rail_rounds_down_to_a_whole_lot():
    """One lot over the limit is still over it."""
    bars = _flat_price_bars(150.0)
    # 10% of 1,00,000 = 10,000; a 75-unit lot at 150 costs 11,250, so nothing fits.
    result = run_backtest(_capped_spec(10.0, lots=1, trade_as="option"), bars,
                          capital=100_000.0, lot_size=75, slippage_pct=0.0)
    assert result.trades == []


def test_a_blocked_trade_says_it_was_the_rail_not_a_shortage_of_money():
    """Being told "not enough capital" while the balance sits untouched is baffling."""
    bars = _flat_price_bars(250.0)
    result = run_backtest(_capped_spec(10.0, lots=1, trade_as="option"), bars,
                          capital=100_000.0, lot_size=75, slippage_pct=0.0)

    skipped = [w for w in result.warnings if "skipped" in w and "signal" in w]
    assert skipped, "signals were dropped with no warning at all"
    assert "10% of the account" in skipped[0], skipped[0]


def test_rail_is_proportional_to_capital():
    """A bigger account may take a bigger position, in rupees, for the same rail."""
    # Rs 5,000 a share x 100 lots asks for Rs 5,00,000 -- past the rail at both
    # account sizes, so the rail alone decides the answer in each case.
    bars = _flat_price_bars(5_000.0)
    sizes = []
    for capital in (100_000.0, 500_000.0):
        result = run_backtest(_capped_spec(20.0, lots=100), bars, capital=capital,
                              lot_size=1, slippage_pct=0.0)
        sizes.append(max(t.quantity * t.entry_price for t in result.trades))

    assert sizes[1] == pytest.approx(sizes[0] * 5, rel=0.01)


def test_options_on_a_small_account_are_warned_about():
    """A NIFTY lot is indivisible, so a small account cannot size proportionately."""
    bars = _flat_price_bars(100.0)
    spec = _capped_spec(10.0, lots=1, trade_as="option")

    small = run_backtest(spec, bars, capital=100_000.0, lot_size=75, slippage_pct=0.0)
    assert any("SMALL ACCOUNT" in w for w in small.warnings)

    large = run_backtest(spec, bars, capital=2_000_000.0, lot_size=75, slippage_pct=0.0)
    assert not any("SMALL ACCOUNT" in w for w in large.warnings), (
        "an account well above the threshold must not be warned"
    )


def test_stocks_on_a_small_account_are_not_warned_about():
    """The warning is about lot indivisibility, which shares do not have."""
    bars = _flat_price_bars(100.0)
    result = run_backtest(_capped_spec(20.0, lots=1, trade_as="stock"), bars,
                          capital=50_000.0, lot_size=1, slippage_pct=0.0)
    assert not any("SMALL ACCOUNT" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Lot sizes come from one table, and it is today's
# ---------------------------------------------------------------------------


def test_engine_and_readback_agree_on_lot_size():
    """Two sources of truth would let the readback describe a different size
    from the one traded -- the exact failure the readback exists to prevent."""
    from nlt.data.instruments import lot_size
    from nlt.engine.backtest import _default_lot_size
    from nlt.translate.readback import _lot_size_for

    for trade_as in ("option", "stock", "index"):
        spec = _capped_spec(10.0, lots=1, trade_as=trade_as).model_copy(
            update={
                "instrument": Instrument(symbol="NIFTY", trade_as=trade_as),
            }
        )
        expected = lot_size("NIFTY", trade_as)
        assert _default_lot_size(spec) == expected
        assert _lot_size_for(spec) == expected


def test_current_nifty_lot_size_is_used():
    """NSE cut NIFTY from 75 to 65 for the January 2026 series. A stale number
    here silently mis-sizes every options backtest."""
    from nlt.data.instruments import lot_size

    assert lot_size("NIFTY", "option") == 65
    assert lot_size("BANKNIFTY", "option") == 30


def test_options_backtest_reports_which_lot_size_it_used():
    """The choice to use today's lot for historical years is a real trade-off,
    so it is stated rather than assumed."""
    bars = _flat_price_bars(100.0)
    spec = _capped_spec(10.0, lots=1, trade_as="option").model_copy(
        update={"instrument": Instrument(symbol="NIFTY", trade_as="option")}
    )
    result = run_backtest(spec, bars, capital=5_000_000.0, slippage_pct=0.0)

    assert any("65 units" in w for w in result.warnings), result.warnings
    assert result.trades and result.trades[0].quantity % 65 == 0
