"""The engine's decision log (`Decision`, `BacktestResult.decisions` /
`BasketResult.decisions`) and the `paper_decision` activity feed built on it.

Every test either ties a decision count to an *existing* warning counter (so
the two cannot quietly disagree) or forces a specific skip reason on
synthetic data whose correct answer is known before the engine runs. Nothing
here reads real market data -- see `tests/test_paper.py` and
`tests/test_futures_trading.py` for the equivalent real-data checks.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from nlt.costs.slippage import LEGACY as LEGACY_SLIPPAGE
from nlt.costs.slippage import ZERO as ZERO_SLIPPAGE
from nlt.engine.backtest import run_backtest
from nlt.engine.basket import run_basket_backtest
from nlt.paper import runner
from nlt.spec.models import (
    Compare,
    Const,
    ExitRules,
    Instrument,
    Ref,
    RiskLimits,
    Schedule,
    StrategySpec,
)
from nlt.store.db import Store
from nlt.translate.readback import describe

IST = "Asia/Kolkata"


def _zero(price, qty, side):
    return 0.0


def _mkbars(
    n: int, start: str = "2024-01-01", freq: str | None = None, **overrides
) -> pd.DataFrame:
    """A flat OHLCV frame, `n` bars, every field overridable by index-aligned arrays."""
    if freq is None:
        idx = pd.bdate_range(start, periods=n, tz=IST)
    else:
        idx = pd.date_range(start, periods=n, freq=freq, tz=IST)
    base = {
        "open": np.full(n, 100.0),
        "high": np.full(n, 110.0),
        "low": np.full(n, 90.0),
        "close": np.full(n, 100.0),
        "volume": np.full(n, 1_000_000.0),
    }
    base.update(overrides)
    return pd.DataFrame(base, index=idx)


def _always_true() -> Compare:
    return Compare(op="gt", left=Ref(name="close"), right=Const(value=-1.0))


def _signal_once(bars: pd.DataFrame, at: int) -> Compare:
    value = float(bars["close"].iloc[at])
    assert (bars["close"] == value).sum() == 1, "signal bar's close must be unique"
    return Compare(op="eq", left=Ref(name="close"), right=Const(value=value))


def _spec(entry, exit_rules=None, **kwargs) -> StrategySpec:
    kwargs.setdefault("risk", RiskLimits(max_lots=100))
    return StrategySpec(
        name="decision test",
        description="test",
        entry=entry,
        exit=exit_rules or ExitRules(stop_pct=90.0, target_pct=90.0),
        **kwargs,
    )


# ------------------------------------------------------- generic trade fields


def test_every_trade_has_a_signal_time_before_its_entry_time():
    bars = _mkbars(20, close=np.concatenate([[100.0], np.full(19, 101.0)]))
    spec = _spec(_signal_once(bars, 0), ExitRules(stop_pct=5.0, target_pct=5.0))
    result = run_backtest(
        spec, bars, capital=1_000_000.0, charge_fn=_zero, slippage=LEGACY_SLIPPAGE
    )
    assert result.trades
    for t in result.trades:
        assert t.signal_time is not None
        assert t.signal_time < t.entry_time


def test_raw_prices_with_slippage_reproduce_the_recorded_fill_prices():
    bars = _mkbars(20, close=np.concatenate([[100.0], np.full(19, 101.0)]))
    spec = _spec(_signal_once(bars, 0), ExitRules(stop_pct=5.0, target_pct=5.0))
    result = run_backtest(
        spec, bars, capital=1_000_000.0, charge_fn=_zero, slippage=LEGACY_SLIPPAGE
    )
    assert result.trades
    for t in result.trades:
        assert t.raw_entry_price is not None
        assert t.raw_exit_price is not None
        expected_entry = t.raw_entry_price * (1 + 0.02 / 100.0)  # a long's buy side
        assert t.entry_price == pytest.approx(expected_entry)
        if t.direction == "long":
            expected_exit = t.raw_exit_price * (1 - 0.02 / 100.0)
        else:
            expected_exit = t.raw_exit_price * (1 + 0.02 / 100.0)
        assert t.exit_price == pytest.approx(expected_exit)


def test_both_hit_matches_the_stop_beat_target_warning_count():
    # Bar 0 signals; bar 1's open fills the entry; bar 2's range swallows both
    # the stop (95) and the target (105) -- the ambiguous case the engine
    # always resolves in the stop's favour.
    opens = [100.0, 100.0, 100.0]
    highs = [100.0, 100.0, 110.0]
    lows = [100.0, 100.0, 90.0]
    closes = [100.0, 100.0, 100.0]
    volume = [2000.0, 1000.0, 1000.0]  # the signal fires on bar 0 only
    bars = _mkbars(
        3,
        close=np.array(closes),
        open=np.array(opens),
        high=np.array(highs),
        low=np.array(lows),
        volume=np.array(volume),
    )
    spec = _spec(
        Compare(op="gt", left=Ref(name="volume"), right=Const(value=1500)),
        ExitRules(stop_pct=5.0, target_pct=5.0),
    )
    result = run_backtest(spec, bars, capital=1_000_000.0, charge_fn=_zero, slippage=ZERO_SLIPPAGE)
    ambiguous_warning = next(w for w in result.warnings if "could have hit both" in w)
    counted = int(ambiguous_warning.split()[0])
    both_hit_trades = [t for t in result.trades if t.both_hit]
    assert len(both_hit_trades) == counted == 1
    assert both_hit_trades[0].exit_reason == "stop"  # the pessimistic assumption


# ----------------------------------------------- counted reasons, forced individually


def test_expiry_skip_decision_matches_the_expiry_counter():
    from nlt.spec.models import Instrument as Instr

    oct_expiry = dt.date(2026, 10, 19)
    idx = pd.bdate_range("2026-10-12", "2026-10-30", tz=IST)
    volume = np.full(len(idx), 100.0)
    volume[idx.date == oct_expiry] = 2000.0  # ty: ignore[unresolved-attribute]  # pandas stub omits DatetimeIndex.date
    bars = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": volume,
        },
        index=idx,
    )
    spec = _spec(
        Compare(op="gt", left=Ref(name="volume"), right=Const(value=1500)),
        ExitRules(stop_pct=50.0, target_pct=50.0),
        instrument=Instr(symbol="CRUDEOIL", trade_as="future", timeframe="1d"),
        risk=RiskLimits(max_lots=100, max_position_pct=None),
        schedule=Schedule(intraday=False),
    )
    result = run_backtest(spec, bars, capital=10_000_000.0, charge_fn=_zero)
    assert result.trades == []
    expiry_warning = next(w for w in result.warnings if "expiring contract" in w)
    counted = int(expiry_warning.split()[0])
    expiry_decisions = [d for d in result.decisions if d.reason_code == "expiry"]
    assert len(expiry_decisions) == counted == 1
    assert expiry_decisions[0].kind == "signal_skipped"
    assert expiry_decisions[0].time.date() == oct_expiry


def test_boundary_skip_decision_matches_the_boundary_counter():
    # One full NSE session's worth of 5m bars (75), entry true only on the
    # last one -- the only bar with no valid intraday fill.
    idx = pd.date_range("2024-03-04 09:15", periods=75, freq="5min", tz=IST)
    close = np.full(75, 100.0)
    close[-1] = 123.0
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=idx,
    )
    spec = _spec(
        _signal_once(bars, 74),
        ExitRules(stop_pct=50.0, target_pct=50.0),
        instrument=Instrument(symbol="NIFTY", trade_as="index", timeframe="5m"),
        # A 15:30 cut-off: the 15:25 candle would fill at 15:30, not after it,
        # so the cut-off lets it through and the boundary rule is what skips it.
        schedule=Schedule(
            intraday=True, no_entry_after=dt.time(15, 30), square_off=dt.time(15, 30)
        ),
    )
    result = run_backtest(
        spec, bars, capital=1_000_000.0, charge_fn=_zero, drop_partial_last_bar=False
    )
    assert result.trades == []
    boundary_warning = next(w for w in result.warnings if "last bar of a trading session" in w)
    counted = int(boundary_warning.split()[0])
    boundary_decisions = [d for d in result.decisions if d.reason_code == "boundary"]
    assert len(boundary_decisions) == counted == 1
    assert boundary_decisions[0].time == idx[-1]


def test_no_entry_after_is_emitted_though_never_counted_before():
    idx = pd.date_range("2024-03-04 09:15", periods=4, freq="5min", tz=IST)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}, index=idx
    )
    spec = _spec(
        _always_true(),
        ExitRules(stop_pct=90.0, target_pct=90.0),
        instrument=Instrument(symbol="NIFTY", trade_as="index", timeframe="5m"),
        schedule=Schedule(intraday=True, no_entry_after=dt.time(9, 20)),
        risk=RiskLimits(max_lots=100, max_concurrent_positions=10),
    )
    result = run_backtest(
        spec, bars, capital=1_000_000.0, charge_fn=_zero, drop_partial_last_bar=False
    )
    # Nothing counts this today -- no warning exists for it.
    assert not any("no_entry_after" in w for w in result.warnings)
    skipped = [d for d in result.decisions if d.reason_code == "no_entry_after"]
    assert [d.time for d in skipped] == list(idx[1:])
    assert all(d.kind == "signal_skipped" for d in skipped)


def test_weekday_skip_is_emitted_though_never_counted_before():
    bars = _mkbars(5, start="2024-01-01")  # Mon..Fri
    assert bars.index[-1].weekday() == 4  # Friday
    spec = _spec(
        _always_true(),
        ExitRules(stop_pct=90.0, target_pct=90.0),
        schedule=Schedule(weekdays=[0, 1, 2, 3]),  # no Friday
        risk=RiskLimits(max_lots=100, max_concurrent_positions=10),
    )
    result = run_backtest(spec, bars, capital=10_000_000.0, charge_fn=_zero)
    assert not any("weekday" in w for w in result.warnings)
    skipped = [d for d in result.decisions if d.reason_code == "weekday"]
    assert len(skipped) == 1
    assert skipped[0].time == bars.index[-1]


def test_concurrency_at_signal_is_emitted_though_never_counted_before():
    n = 10
    bars = _mkbars(n)
    spec = _spec(
        _always_true(),
        ExitRules(stop_pct=90.0, target_pct=90.0),  # never hit within the test window
        risk=RiskLimits(max_lots=100, max_concurrent_positions=1),
    )
    result = run_backtest(spec, bars, capital=10_000_000.0, charge_fn=_zero, slippage=ZERO_SLIPPAGE)
    assert len(result.trades) == 1  # only ever one open position
    assert not any("concurrency_at_signal" in w for w in result.warnings)
    skipped = [d for d in result.decisions if d.reason_code == "concurrency_at_signal"]
    taken = [d for d in result.decisions if d.kind == "signal_taken"]
    assert len(taken) == 1
    assert len(skipped) == n - 1


@pytest.mark.parametrize(
    "max_position_pct,expected_reason", [(None, "capital"), (50.0, "position_cap")]
)
def test_affordability_skip_reason_matches_whether_a_position_cap_is_set(
    max_position_pct, expected_reason
):
    bars = _mkbars(5, close=np.concatenate([[100.0], np.full(4, 101.0)]))
    spec = _spec(
        _signal_once(bars, 0),
        ExitRules(stop_pct=50.0, target_pct=50.0),
        risk=RiskLimits(max_lots=100, max_position_pct=max_position_pct),
    )
    # Rs 10 cannot buy a single unit at a price of 100.
    result = run_backtest(spec, bars, capital=10.0, charge_fn=_zero)
    assert result.trades == []
    affordability_warning = next(
        w for w in result.warnings if "not enough" in w or "may not exceed" in w
    )
    counted = int(affordability_warning.split()[0])
    decisions = [d for d in result.decisions if d.reason_code == expected_reason]
    assert len(decisions) == counted == 1
    assert decisions[0].kind == "signal_skipped"


# --------------------------------------------------------------- blocked windows


def test_blocked_window_skips_the_entry_but_not_an_open_positions_exit():
    bars = _mkbars(10, close=np.concatenate([[100.0], np.full(9, 101.0)]))
    spec = _spec(_signal_once(bars, 0), ExitRules(stop_pct=1.0, target_pct=50.0))
    window = [(bars.index[0], bars.index[5])]
    result = run_backtest(spec, bars, capital=1_000_000.0, charge_fn=_zero, blocked_windows=window)
    assert result.trades == []
    paused = [d for d in result.decisions if d.reason_code == "paused"]
    assert len(paused) == 1
    assert paused[0].time == bars.index[0]


def test_an_open_positions_stop_still_fires_inside_a_blocked_window():
    # Signal on bar 0 opens a position (window starts AFTER it); its stop is
    # then hit on bar 2, which is inside a window -- the exit must still fire.
    close = [100.0, 101.0, 50.0, 50.0, 50.0]
    low = [99.0, 100.0, 40.0, 50.0, 50.0]
    bars = _mkbars(5, close=np.array(close), low=np.array(low))
    spec = _spec(_signal_once(bars, 0), ExitRules(stop_pct=5.0, target_pct=90.0))
    window = [(bars.index[1], bars.index[-1] + pd.Timedelta(days=1))]
    unwindowed = run_backtest(spec, bars, capital=1_000_000.0, charge_fn=_zero)
    windowed = run_backtest(
        spec, bars, capital=1_000_000.0, charge_fn=_zero, blocked_windows=window
    )
    # The one trade taken (entered before the window) still exits on its stop.
    assert len(unwindowed.trades) == 1
    assert [t.exit_reason for t in windowed.trades] == [t.exit_reason for t in unwindowed.trades]
    assert [t.exit_time for t in windowed.trades] == [t.exit_time for t in unwindowed.trades]


def test_basket_blocked_window_matches_single_symbol_behaviour():
    bars = _mkbars(10, close=np.concatenate([[100.0], np.full(9, 101.0)]))
    spec = _spec(
        _signal_once(bars, 0),
        ExitRules(stop_pct=50.0, target_pct=50.0),
        instrument=Instrument(symbol="ONESYM", trade_as="stock", timeframe="1d"),
        risk=RiskLimits(max_lots=100, max_concurrent_positions=1),
    )
    window = [(bars.index[0], bars.index[5])]
    result = run_basket_backtest(
        spec, {"ONESYM": bars}, capital=1_000_000.0, charge_fn=_zero, blocked_windows=window
    )
    assert result.trades == []
    paused = [d for d in result.decisions if d.reason_code == "paused"]
    assert len(paused) == 1


# --------------------------------------------------------------------- basket


def test_basket_slots_decision_matches_the_dropped_for_slots_counter():
    bars_a = _mkbars(5, close=np.concatenate([[100.0], np.full(4, 101.0)]))
    bars_b = _mkbars(5, close=np.concatenate([[100.0], np.full(4, 101.0)]))
    spec = _spec(
        _signal_once(bars_a, 0),
        ExitRules(stop_pct=90.0, target_pct=90.0),
        instrument=Instrument(symbol="BASKET", trade_as="stock", timeframe="1d"),
        risk=RiskLimits(max_lots=100, max_concurrent_positions=1),
    )
    result = run_basket_backtest(
        spec, {"A": bars_a, "B": bars_b}, capital=10_000_000.0, charge_fn=_zero
    )
    assert len(result.trades) == 1  # only one slot, two symbols signal together
    slots_warning = next(w for w in result.warnings if "positions were already open" in w)
    counted = int(slots_warning.split()[0])
    decisions = [d for d in result.decisions if d.reason_code == "slots"]
    assert len(decisions) == counted == 1


def test_basket_capital_decision_matches_the_dropped_for_capital_counter():
    bars = _mkbars(5, close=np.concatenate([[100.0], np.full(4, 101.0)]))
    spec = _spec(
        _signal_once(bars, 0),
        ExitRules(stop_pct=90.0, target_pct=90.0),
        instrument=Instrument(symbol="BASKET", trade_as="stock", timeframe="1d"),
        risk=RiskLimits(max_lots=100, max_position_pct=None),
    )
    result = run_basket_backtest(spec, {"A": bars}, capital=10.0, charge_fn=_zero)
    assert result.trades == []
    capital_warning = next(w for w in result.warnings if "spare money" in w)
    counted = int(capital_warning.split()[0])
    decisions = [d for d in result.decisions if d.reason_code == "capital"]
    assert len(decisions) == counted == 1


# -------------------------------------------------------------------- runner


IST_ = IST


def _runner_spec(entry, **kwargs) -> StrategySpec:
    kwargs.setdefault("risk", RiskLimits(max_lots=100, max_concurrent_positions=10))
    return StrategySpec(
        name="runner decision test",
        description="test",
        entry=entry,
        exit=ExitRules(stop_pct=90.0, target_pct=90.0),
        **kwargs,
    )


def test_expiry_day_signal_gives_a_skip_decision_and_no_signal_row(tmp_path):
    """A signal on an expiry day's last candle must be visible as a skip, and
    must NEVER produce the "signal: fills at the next candle's open" row --
    that row claims a fill the engine did not schedule.
    """
    from nlt.data.futures import data_key

    idx = pd.bdate_range("2026-10-12", "2026-10-19", tz=IST)
    volume = np.full(len(idx), 100.0)
    volume[-1] = 2000.0  # the expiry day itself
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": volume}, index=idx
    )
    spec = _runner_spec(
        Compare(op="gt", left=Ref(name="volume"), right=Const(value=1500)),
        instrument=Instrument(symbol="CRUDEOIL", trade_as="future", timeframe="1d"),
        risk=RiskLimits(max_lots=100, max_position_pct=None),
        schedule=Schedule(intraday=False),
    )
    store = Store(tmp_path / "test.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    started_at = idx[0] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 10_000_000.0, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)

    def fetch(symbol, timeframe, start, end):
        assert symbol == data_key("CRUDEOIL")
        return bars

    now = idx[-1] + pd.Timedelta(days=1)  # safely after MCX's close, whatever the season
    assert run is not None
    result = runner.step(store, run, fetch=fetch, charge_fn=_zero, now=now)
    assert result.health == "ok"

    decisions = store.paper_decisions(run_id)
    expiry_decisions = [d for d in decisions if d["reason_code"] == "expiry"]
    assert len(expiry_decisions) == 1
    assert expiry_decisions[0]["kind"] == "signal_skipped"

    signal_rows = [e for e in store.paper_events(run_id) if e["kind"] == "signal"]
    assert signal_rows == []


def test_a_signal_after_no_entry_after_gives_a_skip_decision_and_no_signal_row(tmp_path):
    """Companion to the expiry-day case above: a signal past the cut-off time
    must show as skipped, never as a "fills at the next candle's open" row.

    This is the test that must FAIL against the old hand-rolled signal check
    (which re-evaluated `spec.entry` on the newest candle and ignored
    `no_entry_after` entirely) and PASS once the "signal" row is sourced only
    from the engine's own `signal_taken` decision -- see the module docstring
    and `nlt.paper.runner.step`.
    """
    idx = pd.date_range("2024-03-04 09:15", periods=5, freq="5min", tz=IST)
    close = np.full(5, 100.0)
    close[-1] = 123.0  # the entry rule is true ONLY on the newest candle
    bars = pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1000.0},
        index=idx,
    )
    spec = _runner_spec(
        _signal_once(bars, 4),
        instrument=Instrument(symbol="NIFTY", trade_as="index", timeframe="5m"),
        schedule=Schedule(intraday=True, no_entry_after=dt.time(9, 30)),
    )
    store = Store(tmp_path / "test.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    started_at = idx[0] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 1_000_000.0, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)
    now = idx[-1] + runner.close_offset("5m") + pd.Timedelta(minutes=1)
    assert run is not None
    result = runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    assert result.health == "ok"

    decisions = store.paper_decisions(run_id)
    skipped = [d for d in decisions if d["reason_code"] == "no_entry_after"]
    assert len(skipped) == 1
    assert skipped[0]["kind"] == "signal_skipped"
    assert skipped[0]["bar_time"] == idx[-1].isoformat()

    signal_rows = [e for e in store.paper_events(run_id) if e["kind"] == "signal"]
    assert signal_rows == []

    signal_rows = [e for e in store.paper_events(run_id) if e["kind"] == "signal"]
    assert signal_rows == []


def test_a_real_signal_row_still_comes_from_the_engines_own_decision(tmp_path):
    bars = _mkbars(5, close=np.concatenate([np.full(4, 100.0), [123.0]]))
    spec = _runner_spec(_signal_once(bars, 4))
    store = Store(tmp_path / "test.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    started_at = bars.index[0] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 1_000_000.0, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)
    now = bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=1)
    assert run is not None
    result = runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    assert result.health == "ok"
    signal_rows = [e for e in store.paper_events(run_id) if e["kind"] == "signal"]
    assert len(signal_rows) == 1
    assert signal_rows[0]["bar_time"] == bars.index[-1].isoformat()
    decisions = [d for d in store.paper_decisions(run_id) if d["kind"] == "signal_taken"]
    assert len(decisions) == 1
    assert decisions[0]["bar_time"] == bars.index[-1].isoformat()


def test_decisions_are_idempotent_across_repeated_passes(tmp_path):
    bars = _mkbars(5, close=np.concatenate([np.full(4, 100.0), [123.0]]))
    spec = _runner_spec(_signal_once(bars, 4))
    store = Store(tmp_path / "test.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    started_at = bars.index[0] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 1_000_000.0, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)
    now = bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=1)
    assert run is not None
    runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    before = store.paper_decisions(run_id)
    result = runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    assert store.paper_decisions(run_id) == before
    assert not any(n.startswith("signal_taken") for n in result.new_events)


# ----------------------------------------------------------------- append-only


def test_paper_decision_cannot_be_edited_or_deleted(tmp_path):
    import sqlite3

    bars = _mkbars(5, close=np.concatenate([np.full(4, 100.0), [123.0]]))
    spec = _runner_spec(_signal_once(bars, 4))
    store = Store(tmp_path / "test.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    started_at = bars.index[0] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 1_000_000.0, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)
    now = bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=1)
    assert run is not None
    runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    assert store.paper_decisions(run_id)

    conn = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE paper_decision SET reason_code = 'x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM paper_decision")
    conn.close()


def test_a_queued_signal_turned_down_at_the_open_is_never_described_as_a_fill(tmp_path):
    """The money check happens at the fill bar, not the signal bar. So when a
    signal is queued on the newest candle, the app cannot yet know it will be
    bought. Nothing it writes may promise a fill; once the next candle arrives
    and the entry is turned down for lack of money, the feed must say so.
    """
    from app.logic import describe_decision, describe_paper_event

    idx = pd.bdate_range("2024-01-01", periods=4, tz=IST)
    close = np.array([100.0, 100.0, 123.0, 100.0])
    opens = np.array([100.0, 100.0, 100.0, 5_000_000.0])  # next open far too dear
    bars = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close) + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )
    spec = _runner_spec(
        _signal_once(bars, 2),
        instrument=Instrument(symbol="RELIANCE", trade_as="stock", timeframe="1d"),
        risk=RiskLimits(max_lots=100, max_position_pct=None),
    )
    store = Store(tmp_path / "test.sqlite")
    sid = store.save_strategy(spec, describe(spec))
    started_at = idx[0] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": 1_000_000.0, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)

    # Pass 1: the signal candle is the newest one; the entry is only queued.
    first = bars.iloc[:3]
    now = idx[2] + pd.Timedelta(days=1, hours=-6)
    assert run is not None
    assert runner.step(store, run, fetch=lambda *a: first, charge_fn=_zero, now=now).health == "ok"
    # Pass 2: the fill candle arrives and its open is unaffordable.
    now = idx[3] + pd.Timedelta(days=1, hours=-6)
    assert runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now).health == "ok"

    events = store.paper_events(run_id)
    decisions = store.paper_decisions(run_id)
    assert [e["kind"] for e in events if e["kind"] in ("signal", "entry")] == ["signal"]
    assert {(d["kind"], d["reason_code"]) for d in decisions} == {
        ("signal_taken", "queued"),
        ("signal_skipped", "capital"),
    }
    texts = [describe_paper_event(e, spec) for e in events] + [
        describe_decision(d, spec) for d in decisions
    ]
    for text in texts:
        assert "fills" not in text and "will buy" not in text, text
    assert any("skipped" in t for t in texts)
