"""Store tests, concentrated on the two things that protect real money:
the proving gate and the kill switch.
"""

from __future__ import annotations

import pytest

from nlt.spec.models import StrategySpec
from nlt.store.db import Store, spec_hash


def make_spec(name="RSI Dip Buyer", length=14, target=2.0) -> StrategySpec:
    return StrategySpec.model_validate(
        {
            "name": name,
            "description": "Buy NIFTY when RSI cracks 30, exit at +2%",
            "indicators": [{"id": "rsi14", "type": "rsi", "params": {"length": length}}],
            "entry": {
                "kind": "compare",
                "op": "crosses_below",
                "left": {"kind": "ref", "name": "rsi14"},
                "right": {"kind": "const", "value": 30},
            },
            "exit": {"target_pct": target, "stop_pct": 1.0},
        }
    )


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.sqlite")


def test_spec_hash_is_order_independent():
    """Field ordering from the model must not change the identity of a strategy."""
    a, b = make_spec(), make_spec()
    assert spec_hash(a) == spec_hash(b)


def test_spec_hash_changes_when_rules_change():
    assert spec_hash(make_spec(length=14)) != spec_hash(make_spec(length=21))
    assert spec_hash(make_spec(target=2.0)) != spec_hash(make_spec(target=3.0))


def test_saving_identical_spec_twice_reuses_the_version(store):
    first = store.save_strategy(make_spec(), "readback")
    second = store.save_strategy(make_spec(), "readback")
    assert first == second
    assert len(store.list_strategies()) == 1


def test_editing_a_strategy_creates_a_new_version(store):
    v1 = store.save_strategy(make_spec(target=2.0), "rb")
    v2 = store.save_strategy(make_spec(target=3.0), "rb")
    assert v1 != v2

    rows = {r["id"]: r for r in store.list_strategies()}
    assert rows[v1]["version"] == 1
    assert rows[v2]["version"] == 2


def test_round_trips_the_spec(store):
    sid = store.save_strategy(make_spec(), "rb")
    loaded = store.load_spec(sid)
    assert loaded.name == "RSI Dip Buyer"
    assert loaded.indicators[0].params["length"] == 14
    assert loaded.exit.target_pct == 2.0


# ------------------------------------------------------------- proving gate

def test_new_strategy_is_not_proven(store):
    assert store.is_proven(store.save_strategy(make_spec(), "rb")) is False


def test_backtest_alone_does_not_prove(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.finish_run(store.start_run(sid, "backtest"), {})
    assert store.is_proven(sid) is False


def test_backtest_plus_paper_proves(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.finish_run(store.start_run(sid, "backtest"), {})
    store.finish_run(store.start_run(sid, "paper"), {})
    assert store.is_proven(sid) is True


def test_incomplete_runs_do_not_prove(store):
    """A paper run that was started and abandoned must not count as proof."""
    sid = store.save_strategy(make_spec(), "rb")
    store.finish_run(store.start_run(sid, "backtest"), {})
    store.start_run(sid, "paper")  # left running
    assert store.is_proven(sid) is False


def test_failed_run_does_not_prove(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.finish_run(store.start_run(sid, "backtest"), {})
    store.fail_run(store.start_run(sid, "paper"), "broker disconnected")
    assert store.is_proven(sid) is False


def test_cannot_go_live_unproven(store):
    sid = store.save_strategy(make_spec(), "rb")
    with pytest.raises(PermissionError, match="paper traded"):
        store.set_state(sid, "live")
    assert store.get_strategy(sid)["state"] == "draft"


def test_override_allows_live_and_is_recorded(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.set_allow_unproven(sid, True, reason="I know what I am doing")
    store.set_state(sid, "live")

    assert store.get_strategy(sid)["state"] == "live"
    events = [e["event"] for e in store.audit_trail()]
    assert "proving_gate_override" in events


def test_proving_does_not_transfer_to_an_edited_strategy(store):
    """The whole point of content-addressing: changing the rules resets the gate."""
    v1 = store.save_strategy(make_spec(target=2.0), "rb")
    store.finish_run(store.start_run(v1, "backtest"), {})
    store.finish_run(store.start_run(v1, "paper"), {})
    assert store.is_proven(v1) is True

    v2 = store.save_strategy(make_spec(target=3.0), "rb")
    assert store.is_proven(v2) is False


# -------------------------------------------------------------- kill switch

def test_kill_switch_halts_live_strategies(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.set_allow_unproven(sid, True, "test")
    store.set_state(sid, "live")

    store.engage_kill_switch("panic")

    assert store.kill_switch_engaged() is True
    assert store.get_strategy(sid)["state"] == "halted"


def test_releasing_kill_switch_does_not_rearm(store):
    """Clearing the flag must never put strategies back into live by itself."""
    sid = store.save_strategy(make_spec(), "rb")
    store.set_allow_unproven(sid, True, "test")
    store.set_state(sid, "live")

    store.engage_kill_switch("panic")
    store.release_kill_switch("all clear")

    assert store.kill_switch_engaged() is False
    assert store.get_strategy(sid)["state"] == "halted"


# -------------------------------------------------------------- daily pnl

def test_daily_pnl_accumulates(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.record_daily_pnl("2026-09-22", sid, "live", -500.0, 40.0)
    store.record_daily_pnl("2026-09-22", sid, "live", -300.0, 40.0)
    assert store.daily_pnl("2026-09-22", sid, "live") == pytest.approx(-800.0)


def test_daily_pnl_is_scoped_by_day_and_mode(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.record_daily_pnl("2026-09-22", sid, "live", -500.0, 40.0)
    assert store.daily_pnl("2026-09-23", sid, "live") == 0.0
    assert store.daily_pnl("2026-09-22", sid, "paper") == 0.0


# ------------------------------------------------------------------- audit

def test_audit_log_is_append_only_in_practice(store):
    sid = store.save_strategy(make_spec(), "rb")
    store.audit("signal_evaluated", strategy_id=sid, rsi=29.4, decision="enter")

    trail = store.audit_trail()
    assert [e["event"] for e in trail][:1] == ["signal_evaluated"]
    assert "29.4" in trail[0]["detail_json"]
    # Creating the strategy logged its own event, which is still present.
    assert "strategy_created" in [e["event"] for e in trail]


def test_unknown_strategy_raises(store):
    with pytest.raises(KeyError):
        store.load_spec(999)
