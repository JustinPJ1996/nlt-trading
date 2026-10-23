"""Pausing a paper run, and the global paper kill switch becoming "missed" too.

The central property is the same shape as `test_paper.py`'s: pausing must
change exactly what the rules say it should and nothing else -- a signal whose
bar becomes known inside a pause is skipped forever (never recorded late), an
open position's stop or target keeps firing and getting recorded through a
pause, and the two ways a run can be paused (a hand pause, and the global
kill switch) must behave identically from the engine's point of view because
both turn into the same `blocked_windows` the engine already understands.

No test here touches Kite or the network -- same discipline as `test_paper.py`.
"""

from __future__ import annotations

import sqlite3
from typing import cast

import pandas as pd
import pytest

from nlt.costs.slippage import LEGACY as LEGACY_SLIPPAGE
from nlt.data.session import NSE_EQUITY
from nlt.engine.backtest import _in_blocked_window, run_backtest
from nlt.paper import runner
from nlt.store.db import Store
from nlt.translate.readback import describe
from tests.test_paper import CAPITAL, DAILY_SPEC, _daily_bars, _events, _replay, _start, _zero

IST = "Asia/Kolkata"


def _insert_pause(store: Store, run_id: int, kind: str, at, reason: str = "") -> None:
    """Writes a `paper_pause` row with a chosen timestamp, bypassing
    `Store.pause_paper_run`/`resume_paper_run` (which always use the real wall
    clock) so a test can align a pause with a specific historical bar's
    "signal became known" moment."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO paper_pause (run_id, kind, at, reason) VALUES (?, ?, ?, ?)",
            (run_id, kind, pd.Timestamp(at).isoformat(), reason),
        )


def _insert_kill_switch_window(store: Store, start, end) -> None:
    """Writes the engage/release pair straight into `audit_log`, which is where
    `Store.paper_kill_switch_windows` reads from -- again bypassing the real
    methods (which stamp the real wall clock) so the window can be placed over
    a chosen historical bar."""
    with store.tx() as conn:
        conn.execute(
            "INSERT INTO audit_log (at, strategy_id, run_id, event, detail_json) "
            "VALUES (?, NULL, NULL, 'paper_kill_switch_engaged', '{}')",
            (pd.Timestamp(start).isoformat(),),
        )
        conn.execute(
            "INSERT INTO audit_log (at, strategy_id, run_id, event, detail_json) "
            "VALUES (?, NULL, NULL, 'paper_kill_switch_released', '{}')",
            (pd.Timestamp(end).isoformat(),),
        )


def _unpaused_backtest(bars, start_index):
    trade_from = bars.index[start_index] - pd.Timedelta(minutes=1) + pd.Timedelta(microseconds=1)
    return run_backtest(DAILY_SPEC, bars, capital=CAPITAL, charge_fn=_zero, trade_from=trade_from)


def _entry_key(spec, trade) -> str:
    return f"{trade.symbol or spec.instrument.symbol}|{trade.entry_time.isoformat()}"


# ---------------------------------------------------------- the timing trap


def test_daily_bar_timing_pausing_0900_to_1600_blocks_that_days_signal():
    """A daily bar is stamped at 00:00 but its signal is only known at the
    session close (15:30 for NSE). Pausing 09:00-16:00 spans that close, so
    the day's signal must be blocked."""
    day = cast(pd.Timestamp, pd.Timestamp("2024-05-02", tz=IST))
    window = cast(
        list[tuple[pd.Timestamp, pd.Timestamp]],
        [(day + pd.Timedelta(hours=9), day + pd.Timedelta(hours=16))],
    )
    shifted = runner.blocked_windows_for_engine(window, "1d", NSE_EQUITY)
    assert _in_blocked_window(day, shifted)


def test_daily_bar_timing_pausing_1600_to_2300_does_not_block_that_days_signal():
    """The same day's close (15:30) is BEFORE this window opens, so the
    signal was already known before the pause started and must not be
    blocked."""
    day = cast(pd.Timestamp, pd.Timestamp("2024-05-02", tz=IST))
    window = cast(
        list[tuple[pd.Timestamp, pd.Timestamp]],
        [(day + pd.Timedelta(hours=16), day + pd.Timedelta(hours=23))],
    )
    shifted = runner.blocked_windows_for_engine(window, "1d", NSE_EQUITY)
    assert not _in_blocked_window(day, shifted)


# --------------------------------------------------------- pause vs replay


def test_replay_with_a_pause_equals_a_backtest_given_the_same_shifted_window(tmp_path_factory):
    bars = _daily_bars()
    store = Store(tmp_path_factory.mktemp("pause_equiv") / "paper.sqlite")
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run = _start(store, DAILY_SPEC, started_at)

    pause_at, resume_at = bars.index[330], bars.index[345]
    _insert_pause(store, run["id"], "pause", pause_at)
    _insert_pause(store, run["id"], "resume", resume_at)
    _replay(store, run, bars, "1d", from_index=300)

    shifted = runner.blocked_windows_for_engine([(pause_at, resume_at)], "1d", NSE_EQUITY)
    expected = run_backtest(
        DAILY_SPEC,
        bars,
        capital=CAPITAL,
        charge_fn=_zero,
        trade_from=started_at + pd.Timedelta(microseconds=1),
        blocked_windows=shifted,
        # `_start` (from tests.test_paper) builds this run's params with no
        # "slippage" key -- exactly a run started before 2026-10-13 -- so the
        # runner fills it at the old hidden 0.02%, not the new zero default.
        slippage=LEGACY_SLIPPAGE,
    )
    recorded = {e["event_key"] for e in _events(store, run["id"], "entry")}
    expected_keys = {_entry_key(DAILY_SPEC, t) for t in expected.trades}
    assert recorded == expected_keys
    assert recorded, "the window picked should not have erased every trade"


def test_no_entry_from_a_signal_known_inside_a_pause(tmp_path_factory):
    bars = _daily_bars()
    bt = _unpaused_backtest(bars, 300)
    assert bt.trades
    trade = min(bt.trades, key=lambda t: t.entry_time)
    known_at = trade.signal_time + runner.close_offset("1d")

    store = Store(tmp_path_factory.mktemp("pause_blocks") / "paper.sqlite")
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run = _start(store, DAILY_SPEC, started_at)
    _insert_pause(store, run["id"], "pause", known_at - pd.Timedelta(hours=1))
    _insert_pause(store, run["id"], "resume", known_at + pd.Timedelta(hours=1))
    _replay(store, run, bars, "1d", from_index=300)

    key = _entry_key(DAILY_SPEC, trade)
    assert not any(e["event_key"] == key for e in _events(store, run["id"], "entry"))
    skips = [
        d
        for d in store.paper_decisions(run["id"])
        if d["kind"] == "signal_skipped" and d["reason_code"] == "paused"
    ]
    assert any(d["bar_time"] == trade.signal_time.isoformat() for d in skips)


def test_an_open_positions_stop_still_fires_and_is_recorded_during_a_pause(tmp_path_factory):
    """The pause is never resumed in this test, on purpose: `is_paper_run_paused`
    stays True for every remaining pass, so if the exit shows up at all, it was
    necessarily recorded WHILE paused, not deferred until some later resume.
    """
    bars = _daily_bars()
    bt = _unpaused_backtest(bars, 300)
    trade = next(t for t in bt.trades if t.exit_reason != "end_of_data")
    exit_known_at = trade.exit_time + runner.close_offset("1d")

    store = Store(tmp_path_factory.mktemp("pause_exit") / "paper.sqlite")
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run = _start(store, DAILY_SPEC, started_at)
    _insert_pause(store, run["id"], "pause", exit_known_at - pd.Timedelta(hours=2))
    _replay(store, run, bars, "1d", from_index=300)

    assert store.is_paper_run_paused(run["id"])
    key = _entry_key(DAILY_SPEC, trade)
    exits = [e for e in _events(store, run["id"], "exit") if e["event_key"] == key]
    assert exits, "a stop/target already in flight must still be recorded during a pause"
    assert exits[0]["bar_time"] == trade.exit_time.isoformat()


def test_kill_switch_engaged_then_released_skips_the_entry_rather_than_recording_it_late(
    tmp_path_factory,
):
    """Write this test FIRST and confirm it FAILS before `kill_switch_windows_for_run`
    exists -- without it, replaying history after a release finds the entry
    normally and records it, just late and at whatever price the next pass
    happens to run at. That is exactly the behaviour this feature removes."""
    bars = _daily_bars()
    bt = _unpaused_backtest(bars, 300)
    assert bt.trades
    trade = min(bt.trades, key=lambda t: t.entry_time)
    known_at = trade.signal_time + runner.close_offset("1d")

    store = Store(tmp_path_factory.mktemp("kill_switch") / "paper.sqlite")
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run = _start(store, DAILY_SPEC, started_at)
    _insert_kill_switch_window(
        store, known_at - pd.Timedelta(hours=1), known_at + pd.Timedelta(hours=1)
    )
    # The switch is OFF right now (paper_kill_switch.engaged is untouched), so
    # `step` does not take its early-return path -- it replays history as
    # normal and must treat the past window as blocked on every pass.
    assert not store.paper_kill_switch_engaged()
    _replay(store, run, bars, "1d", from_index=300)

    key = _entry_key(DAILY_SPEC, trade)
    assert not any(e["event_key"] == key for e in _events(store, run["id"], "entry"))
    skips = [
        d
        for d in store.paper_decisions(run["id"])
        if d["kind"] == "signal_skipped" and d["reason_code"] == "paused"
    ]
    assert any(d["bar_time"] == trade.signal_time.isoformat() for d in skips)


# ------------------------------------------------------- pause/resume rules


def test_pause_and_resume_refuse_when_already_in_that_state(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    sid = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    run_id = store.start_run(
        sid, "paper", {"capital": CAPITAL, "started_at": pd.Timestamp.now(tz=IST).isoformat()}
    )
    assert not store.is_paper_run_paused(run_id)
    with pytest.raises(ValueError):
        store.resume_paper_run(run_id)

    store.pause_paper_run(run_id, "lunch break")
    assert store.is_paper_run_paused(run_id)
    with pytest.raises(ValueError):
        store.pause_paper_run(run_id)

    store.resume_paper_run(run_id, "back")
    assert not store.is_paper_run_paused(run_id)
    with pytest.raises(ValueError):
        store.resume_paper_run(run_id)


def test_pause_resume_write_audit_log(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    sid = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    run_id = store.start_run(
        sid, "paper", {"capital": CAPITAL, "started_at": pd.Timestamp.now(tz=IST).isoformat()}
    )
    store.pause_paper_run(run_id, "because")
    store.resume_paper_run(run_id, "because")
    events = [a["event"] for a in store.audit_trail()]
    assert "paper_run_paused" in events
    assert "paper_run_resumed" in events


def test_paper_pause_is_append_only(tmp_path):
    store = Store(tmp_path / "s.sqlite")
    sid = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    run_id = store.start_run(
        sid, "paper", {"capital": CAPITAL, "started_at": pd.Timestamp.now(tz=IST).isoformat()}
    )
    store.pause_paper_run(run_id)
    conn = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE paper_pause SET reason = 'edited'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM paper_pause")
    conn.close()


# ----------------------------------------------------------- final result


def test_stop_summary_states_the_pause_time_and_skipped_count(tmp_path_factory):
    from app import logic

    bars = _daily_bars()
    bt = _unpaused_backtest(bars, 300)
    trade = min(bt.trades, key=lambda t: t.entry_time)
    known_at = trade.signal_time + runner.close_offset("1d")

    store = Store(tmp_path_factory.mktemp("pause_summary") / "paper.sqlite")
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run = _start(store, DAILY_SPEC, started_at)
    pause_at, resume_at = known_at - pd.Timedelta(hours=1), known_at + pd.Timedelta(hours=1)
    _insert_pause(store, run["id"], "pause", pause_at)
    _insert_pause(store, run["id"], "resume", resume_at)
    _replay(store, run, bars, "1d", from_index=300)

    logic.stop_paper(store, run["id"])
    finished = store.get_run(run["id"])
    assert finished is not None
    import json

    metrics = json.loads(finished["metrics_json"])
    assert metrics["paused_signal_count"] >= 1
    assert metrics["paused_seconds"] == pytest.approx((resume_at - pause_at).total_seconds())
    assert "paused" in metrics["paused_summary"].lower()
    assert str(metrics["paused_signal_count"]) in metrics["paused_summary"]
