"""Tests for `nlt.engine.basket`: shared capital, portfolio concurrency,
deterministic contention, per-symbol alignment and attribution.

Every basket built here is small (3-5 symbols, a handful of bars) so the
expected trades and equity can be worked out by hand before the assertion is
written -- the same discipline `tests/test_engine.py` documents for itself.
Price levels are chosen per symbol (100s, 200s, 300s, ...) wherever cross-
contamination between symbols would otherwise be invisible: a bug that reads
symbol B's bar while processing symbol A shows up immediately as a price from
the wrong century.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.engine.backtest import run_backtest
from nlt.engine.basket import run_basket_backtest
from nlt.spec.models import (
    Compare,
    Const,
    ExitRules,
    IndicatorSpec,
    Instrument,
    Ref,
    RiskLimits,
    Sizing,
    StrategySpec,
)

# --------------------------------------------------------------------- helpers


def _always_true_entry() -> Compare:
    return Compare(op="gt", left=Ref(name="close"), right=Const(value=-1.0))


def _one_shot_bars(base: float, n: int = 2, start: str = "2024-01-01") -> pd.DataFrame:
    """`n` bars for one symbol: bar0 is a throwaway signal bar, bar1's OPEN is
    exactly `base` (the clean, hand-computable entry price with slippage=0),
    and bar1 is also (with n=2) the symbol's own last bar, so an always-true
    entry combined with a wide stop/target force-closes the position on that
    same bar via the end-of-data path -- giving one fully self-contained trade
    per symbol with an entry price of exactly `base`.
    """
    idx = pd.bdate_range(start, periods=n, tz="Asia/Kolkata")
    opens = [base - 5.0] + [base + 2.0 * i for i in range(n - 1)]
    closes = [base - 5.0] + [base + 2.0 + 2.0 * i for i in range(n - 1)]
    highs = [o + 3 for o in opens]
    lows = [c - 3 for c in closes]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000.0] * n},
        index=idx,
    )


def _spec(**kwargs) -> StrategySpec:
    kwargs.setdefault("risk", RiskLimits(max_lots=100, max_concurrent_positions=10))
    kwargs.setdefault("entry", _always_true_entry())
    kwargs.setdefault("exit", ExitRules(stop_pct=50.0, target_pct=50.0))
    return StrategySpec(name="basket test", description="test", **kwargs)


def _max_concurrent(trades) -> int:
    """Peak number of simultaneously-open positions implied by a trade list,
    computed from entry/exit events rather than assumed -- so a bug that lets
    the engine hold one too many shows up as a number, not a hand-wave.
    """
    events = []
    for t in trades:
        events.append((t.entry_time, 1))
        events.append((t.exit_time, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # exits (-1) before entries (+1) at a tie
    running = peak = 0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


# ------------------------------------------------------------- shared capital


def test_shared_capital_caps_concurrent_exposure_and_counts_the_shortfall():
    bases = [100.0, 200.0, 300.0, 400.0, 500.0]
    bars_by_symbol = {f"S{i + 1}": _one_shot_bars(b) for i, b in enumerate(bases)}

    spec = _spec()
    # Enough for S1 (100) + S2 (200) = 300, not enough for a third.
    res = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=350.0,
                               slippage_pct=0.0)

    filled_symbols = {t.symbol for t in res.trades}
    assert filled_symbols == {"S1", "S2"}
    assert len(res.trades) == 2

    exposure = sum(t.entry_price * t.quantity for t in res.trades)
    assert exposure <= 350.0

    assert any("3" in w and "capital" in w for w in res.warnings)


def test_portfolio_concurrency_never_exceeds_the_limit():
    bases = [100.0, 200.0, 300.0, 400.0, 500.0]
    bars_by_symbol = {f"S{i + 1}": _one_shot_bars(b) for i, b in enumerate(bases)}

    spec = _spec(risk=RiskLimits(max_lots=100, max_concurrent_positions=2))
    res = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                               slippage_pct=0.0)

    assert _max_concurrent(res.trades) <= 2
    filled_symbols = {t.symbol for t in res.trades}
    assert len(filled_symbols) == 2  # only the first two in contention order filled
    assert any("max_concurrent_positions" in w for w in res.warnings)


# --------------------------------------------------------- deterministic contention


def test_contention_is_deterministic_across_runs():
    bases = [500.0, 100.0, 300.0, 200.0, 400.0]
    bars_by_symbol = {f"S{i + 1}": _one_shot_bars(b) for i, b in enumerate(bases)}

    spec = _spec(risk=RiskLimits(max_lots=100, max_concurrent_positions=2))
    res1 = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                                slippage_pct=0.0)
    res2 = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                                slippage_pct=0.0)

    assert res1.trades == res2.trades
    assert len(res1.trades) > 0


def test_ranked_selection_orders_by_signal_bar_move_then_symbol_name():
    """`selection == "ranked"` should prefer the symbol whose signal bar moved
    the most, breaking ties by symbol name.

    The entry condition fires on exactly one bar (index 1, close == the
    shared marker 12345.0) for every symbol, so the fill (index 2) is
    unambiguous and its ranking is driven purely by how big a move each
    symbol made getting to that marker from its own bar-0 close -- distinct
    for AAA/BBB, tied for AAA/CCC. `max_concurrent_positions=2` admits two of
    the three: BBB (biggest move) must always win a slot, and the tie between
    AAA and CCC must always resolve to AAA (alphabetically first), never CCC.
    """
    idx = pd.bdate_range("2024-01-01", periods=3, tz="Asia/Kolkata")
    marker = 12345.0

    def _moving_bars(pct_move_to_marker: float, fill_open: float) -> pd.DataFrame:
        close0 = marker / (1.0 + pct_move_to_marker / 100.0)
        closes = [close0, marker, marker]
        opens = [close0, fill_open, fill_open]
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [1_000_000.0] * len(closes)},
            index=idx,
        )

    bars_by_symbol = {
        "AAA": _moving_bars(1.0, fill_open=150.0),  # small move, ties with CCC
        "BBB": _moving_bars(5.0, fill_open=250.0),  # biggest move -> always wins a slot
        "CCC": _moving_bars(1.0, fill_open=350.0),  # ties with AAA, loses the tiebreak
    }
    spec = _spec(
        entry=Compare(op="eq", left=Ref(name="close"), right=Const(value=marker)),
        instrument=Instrument(symbol="NIFTY", selection="ranked"),
        risk=RiskLimits(max_lots=100, max_concurrent_positions=2),
    )
    res = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                               slippage_pct=0.0)

    filled_symbols = {t.symbol for t in res.trades}
    assert filled_symbols == {"BBB", "AAA"}


# ------------------------------------------------------------------- alignment


def test_alignment_no_forward_fill_and_no_cross_contamination():
    """A missing middle day, a late listing, and no strategy ever seeing
    another symbol's price.
    """
    full_idx = pd.bdate_range("2024-01-01", periods=6, tz="Asia/Kolkata")
    gappy_idx = full_idx.delete(3)  # A misses its 4th trading day entirely
    late_idx = full_idx[2:]  # C only starts trading on day index 2

    def _frame(idx, base) -> pd.DataFrame:
        n = len(idx)
        closes = [base + i for i in range(n)]
        opens = closes
        highs = [c + 1 for c in closes]
        lows = [c - 1 for c in closes]
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes,
             "volume": [1_000_000.0] * n},
            index=idx,
        )

    bars_by_symbol = {
        "AAA": _frame(full_idx, 100.0),
        "BBB": _frame(gappy_idx, 200.0),
        "CCC": _frame(late_idx, 300.0),
    }

    spec = _spec(exit=ExitRules(max_bars_held=1))
    res = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                               slippage_pct=0.0)

    # No forward-fill: BBB's missing day never appears in its own bars, and
    # the union clock does not manufacture a row for it there either.
    assert full_idx[3] not in bars_by_symbol["BBB"].index

    # Every symbol's trades use ONLY its own price band -- if the loop ever
    # read a neighbour's bar, a price from the wrong century would show up.
    by_symbol = {"AAA": (100, 200), "BBB": (200, 300), "CCC": (300, 400)}
    assert res.trades, "test is vacuous with zero trades"
    for t in res.trades:
        lo, hi = by_symbol[t.symbol]
        assert lo <= t.entry_price < hi
        assert lo <= t.exit_price < hi

    # CCC (late listing) must not have any trade dated before it actually
    # started trading.
    ccc_trades = [t for t in res.trades if t.symbol == "CCC"]
    assert all(t.entry_time >= late_idx[0] for t in ccc_trades)


# --------------------------------------------------------------- attribution


def test_per_symbol_pnl_sums_to_portfolio_total():
    bases = [100.0, 200.0, 300.0]
    bars_by_symbol = {f"S{i + 1}": _one_shot_bars(b, n=3) for i, b in enumerate(bases)}

    spec = _spec()
    res = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                               slippage_pct=0.0)

    per_symbol_total = sum(v["net_pnl"] for v in res.per_symbol.values())
    trades_total = sum(t.net_pnl for t in res.trades)
    assert per_symbol_total == pytest.approx(trades_total)

    per_symbol_count = sum(v["trades"] for v in res.per_symbol.values())
    assert per_symbol_count == len(res.trades)


# ------------------------------------------------------ single-symbol equivalence


def test_matches_single_symbol_run_backtest_exactly():
    """With one symbol, ample capital and `max_concurrent_positions=1`, the
    basket engine must reuse `run_backtest`'s own fill/exit logic closely
    enough to produce byte-identical trades -- this is the test that proves
    the basket loop calls the same helpers rather than a re-derived copy.
    """
    idx = pd.bdate_range("2024-01-01", periods=40, tz="Asia/Kolkata")
    import numpy as np
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1.0, len(idx)))
    opens = closes - rng.normal(0, 0.3, len(idx))
    highs = np.maximum(opens, closes) + np.abs(rng.normal(0, 1.0, len(idx)))
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0, 1.0, len(idx)))
    bars = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [1_000_000.0] * len(idx)},
        index=idx,
    )

    spec = StrategySpec(
        name="equiv test",
        description="test",
        instrument=Instrument(symbol="ONESYM"),
        entry=Compare(op="crosses_below", left=Ref(name="close"), right=Ref(name="sma5")),
        exit=ExitRules(stop_pct=1.5, target_pct=2.5),
        indicators=[IndicatorSpec(id="sma5", type="sma", params={"length": 5})],
        risk=RiskLimits(max_lots=100, max_concurrent_positions=1),
    )

    res_single = run_backtest(spec, bars, capital=1_000_000.0, slippage_pct=0.0)
    res_basket = run_basket_backtest(
        bars_by_symbol={"ONESYM": bars}, spec=spec, capital=1_000_000.0, slippage_pct=0.0
    )

    assert len(res_single.trades) > 0, "test is vacuous with zero trades"
    assert res_basket.trades == res_single.trades


# ----------------------------------------------------------------- failures


def test_a_bad_symbol_is_skipped_not_fatal():
    good = _one_shot_bars(100.0, n=3)
    bad = _one_shot_bars(200.0, n=3).drop(columns=["volume"])  # missing PRICE_FIELDS column

    spec = _spec()
    res = run_basket_backtest(
        bars_by_symbol={"GOOD": good, "BAD": bad, "EMPTY": pd.DataFrame()},
        spec=spec, capital=1_000_000.0, slippage_pct=0.0,
    )

    assert "BAD" in res.skipped
    assert "EMPTY" in res.skipped
    assert any(t.symbol == "GOOD" for t in res.trades)


# ------------------------------------------------------------- survivorship


def test_survivorship_warning_fires_for_a_universe_started_long_before_snapshot(monkeypatch):
    def fail(name):
        raise RuntimeError("no network in tests")

    monkeypatch.setattr("nlt.data.universe.refresh_from_nse", fail)

    bars_by_symbol = {
        "RELIANCE": _one_shot_bars(100.0, n=3, start="2005-01-03"),
        "TCS": _one_shot_bars(200.0, n=3, start="2005-01-03"),
    }
    spec = _spec(instrument=Instrument(symbol="NIFTY 50"))
    res = run_basket_backtest(bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
                               slippage_pct=0.0)

    assert any("snapshot" in w.lower() for w in res.warnings)


# ------------------------------------------------------------------- charges


def test_charges_applied_per_trade_and_sum_matches_portfolio():
    bases = [100.0, 200.0, 300.0]
    bars_by_symbol = {f"S{i + 1}": _one_shot_bars(b, n=3) for i, b in enumerate(bases)}

    def charge_fn(price, quantity, side):
        return price * quantity * 0.001

    spec = _spec()
    res = run_basket_backtest(
        bars_by_symbol=bars_by_symbol, spec=spec, capital=1_000_000.0,
        charge_fn=charge_fn, slippage_pct=0.0,
    )

    assert res.trades, "test is vacuous with zero trades"
    for t in res.trades:
        expected_charges = charge_fn(t.entry_price, t.quantity, "buy") + charge_fn(
            t.exit_price, t.quantity, "sell"
        )
        assert t.charges == pytest.approx(expected_charges)

    assert res.metrics["total_charges"] == pytest.approx(sum(t.charges for t in res.trades))


# ------------------------------------------------------------------ real data


@pytest.mark.network
def test_real_nifty50_rsi_basket_smoke():
    """NIFTY 50, RSI(14) crosses below 30, 2% target, 1% stop, since 2020.

    Loose bounds only (real market data, not a hand-built fixture): a
    plausible trade count, no NaN equity, no trade exiting before it entered,
    and peak exposure never over capital.
    """
    from nlt.data.stocks import StockSource
    from nlt.data.universe import get_universe

    universe = get_universe("NIFTY 50")
    source = StockSource()
    bars_by_symbol = source.bars_many(
        list(universe.symbols), start=dt.date(2020, 1, 1)
    )
    assert bars_by_symbol, "no NIFTY 50 data could be loaded; check network/cache"

    spec = StrategySpec(
        name="nifty50 rsi basket",
        description="Buy NIFTY 50 stocks when RSI drops below 30",
        instrument=Instrument(symbol="NIFTY 50", trade_as="stock"),
        entry=Compare(op="crosses_below", left=Ref(name="rsi14"), right=Const(value=30.0)),
        exit=ExitRules(stop_pct=1.0, target_pct=2.0),
        indicators=[IndicatorSpec(id="rsi14", type="rsi", params={"length": 14})],
        sizing=Sizing(mode="fixed_value", value=5_000.0),
        risk=RiskLimits(max_lots=100, max_concurrent_positions=10),
    )

    res = run_basket_backtest(spec=spec, bars_by_symbol=bars_by_symbol, capital=200_000.0)

    assert not res.equity.isna().any()
    assert 0 < res.metrics["total_trades"] < 5000
    for t in res.trades:
        # Same-bar stop-outs are legitimate (see test_engine.py's equivalent
        # comment): a position can fill and hit its stop within the very bar
        # it opens on. What must never happen is exiting strictly BEFORE entry.
        assert t.exit_time >= t.entry_time

    events = []
    for t in res.trades:
        events.append((t.entry_time, t.entry_price * t.quantity))
        events.append((t.exit_time, -t.entry_price * t.quantity))
    events.sort(key=lambda e: e[0])
    running = 0.0
    for _, delta in events:
        running += delta
        assert running <= 200_000.0 + 1e-6


# ---------------------------------------------------------------------------
# Capital committed on an EARLIER bar
#
# The existing shared-capital test opens every position on one bar, so the
# netting done inside the fill loop covers for the netting done at the start of
# it. Replacing the start-of-bar computation with the raw balance left the whole
# suite green -- meaning a position opened yesterday and still open did not
# reduce today's available capital, which is the leverage bug wearing a
# different hat.
#
# Here the two entries are deliberately on different bars, so only the
# start-of-bar netting can prevent the second one.
# ---------------------------------------------------------------------------


def _staggered_bars(base: float, signal_on: int, n: int = 8) -> pd.DataFrame:
    """Flat bars at `base`, with a one-bar dip that fires an RSI-free trigger.

    The entry condition used below is `close < base`, so the single dipped bar
    at index `signal_on` is the only signal, and it fills on the bar after it.
    Everything else is flat so nothing else can move the position.
    """
    idx = pd.bdate_range("2024-01-01", periods=n, tz="Asia/Kolkata")
    closes = [base] * n
    closes[signal_on] = base - 1.0
    opens = [base] * n
    return pd.DataFrame(
        {"open": opens, "high": [c + 0.5 for c in opens], "low": [c - 1.5 for c in closes],
         "close": closes, "volume": [1_000_000.0] * n},
        index=idx,
    )


def test_capital_committed_on_an_earlier_bar_still_blocks_a_later_entry():
    """Two symbols signalling on different bars must share one capital pool.

    A alone fits the account. B signals three bars later, while A is still open.
    With the carried-over position netted out, B cannot be afforded; without
    that netting, both open and the portfolio is leveraged.
    """
    bars_by_symbol = {
        "AAA": _staggered_bars(100.0, signal_on=1),
        "BBB": _staggered_bars(100.0, signal_on=4),
    }
    # Room for exactly one 100-unit position, and both positions run to the end
    # of the data because the stop and target are far away.
    spec = _spec(
        entry=Compare(op="lt", left=Ref(name="close"), right=Const(value=100.0)),
        exit=ExitRules(stop_pct=90.0, target_pct=90.0),
        risk=RiskLimits(max_lots=100, max_concurrent_positions=5),
    )

    result = run_basket_backtest(
        spec, bars_by_symbol, capital=120.0, slippage_pct=0.0
    )

    peak = _peak_exposure(result.trades)
    assert peak <= 120.0 * 1.001, (
        f"peak simultaneous exposure {peak:,.2f} exceeds the 120.00 account "
        f"({peak / 120.0:.2f}x) -- capital committed on an earlier bar is not "
        "being netted out at the start of the next one"
    )


def _peak_exposure(trades) -> float:
    """Largest simultaneous position value across the run."""
    events = []
    for t in trades:
        events.append((t.entry_time, t.quantity * t.entry_price))
        events.append((t.exit_time, -t.quantity * t.entry_price))
    events.sort(key=lambda e: (e[0], e[1]))
    running = peak = 0.0
    for _, delta in events:
        running += delta
        peak = max(peak, running)
    return peak


def test_peak_exposure_never_exceeds_capital_in_the_existing_basket_case():
    """A direct exposure assertion on the original five-symbol basket.

    The existing test asserts concurrency and the dropped-signal count; this one
    asserts the rupee invariant those are proxies for.
    """
    bases = [100.0, 200.0, 300.0, 400.0, 500.0]
    bars_by_symbol = {f"S{i + 1}": _one_shot_bars(b) for i, b in enumerate(bases)}
    spec = _spec(sizing=Sizing(mode="fixed_lots", lots=1))

    result = run_basket_backtest(spec, bars_by_symbol, capital=350.0, slippage_pct=0.0)
    assert _peak_exposure(result.trades) <= 350.0 * 1.001


def test_intraday_basket_is_refused_rather_than_run_without_session_rules():
    """The basket loop has no square-off and no overnight-carry invariant.

    Running an intraday spec here would hold positions through the night on a
    strategy whose author wrote "square off at 3:15" -- which the single-symbol
    engine treats as a hard error. Refusing is the only consistent answer, and
    it costs nothing while every stock basket is on daily bars.
    """
    idx = pd.date_range("2024-01-01 09:15", periods=50, freq="15min", tz="Asia/Kolkata")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e6},
        index=idx,
    )
    spec = _spec(instrument=Instrument(symbol="NIFTY 50", trade_as="stock", timeframe="15m"))

    with pytest.raises(NotImplementedError, match="daily-only"):
        run_basket_backtest(spec, {"AAA": bars}, capital=1_000_000.0)


def test_daily_basket_is_not_refused():
    """The mirror: the guard must not block the case it exists to protect."""
    bars_by_symbol = {"AAA": _one_shot_bars(100.0), "BBB": _one_shot_bars(200.0)}
    result = run_basket_backtest(_spec(), bars_by_symbol, capital=100_000.0)
    assert result.trades
