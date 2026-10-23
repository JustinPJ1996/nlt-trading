"""The merged Activity feed (`/api/activity`), the one-trade and one-order
timelines built on recorded rows, and the pure functions behind them
(`logic.trade_timeline`, `logic.activity_feed`, `orders_view.order_timeline`).

Every number a timeline states must come from the row that was actually
recorded -- never re-derived -- so each test below either reads the real
`paper_event`/`trade` rows a real paper run produced, or hand-builds event
dicts whose numbers are then checked back out verbatim.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from starlette.testclient import TestClient

from app import api, logic, orders_view
from nlt.orders.models import ConditionalOrder, PriceTrigger, TradeAction, Watched
from nlt.paper import runner
from nlt.store.db import Store
from nlt.translate.readback import describe, format_inr
from nlt.translate.rules import parse

IST = "Asia/Kolkata"
LOGIN_ID = "tester"
PASSWORD = "correct horse"
CAPITAL = 1_000_000.0


def _zero(price, qty, side):
    return 0.0


def _daily_bars(n: int = 500, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 20000 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2024-01-01", periods=n, tz=IST)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 0.0}, index=idx
    )


DAILY_SPEC = parse("buy nifty when rsi crosses below 40, target 3%, stop 2%").spec


@pytest.fixture(scope="module")
def replayed(tmp_path_factory):
    """One full daily paper run, replayed in a single pass (every pass reruns
    the whole engine, so one call produces the same rows incremental replay
    would)."""
    store = Store(tmp_path_factory.mktemp("activity") / "paper.sqlite")
    bars = _daily_bars().iloc[:400]  # cut here: the known scenario leaves one trade open
    assert DAILY_SPEC is not None
    sid = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": CAPITAL, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)
    now = bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=1)
    assert run is not None
    runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    return store, run_id, bars


def _closed_trade_key(store, run_id) -> str:
    exits = [e for e in store.paper_events(run_id) if e["kind"] == "exit"]
    assert exits, "the scenario must produce a closed trade for this test to mean anything"
    return exits[0]["event_key"]


def _open_trade_key(store, run_id) -> str:
    entries = {e["event_key"]: e for e in store.paper_events(run_id) if e["kind"] == "entry"}
    exits = {e["event_key"] for e in store.paper_events(run_id) if e["kind"] == "exit"}
    open_keys = [k for k in entries if k not in exits]
    assert open_keys, "the scenario must leave a trade open for this test to mean anything"
    return open_keys[0]


# ------------------------------------------------------------- trade_timeline


def test_closed_trade_numbers_match_the_trade_table_and_events(replayed):
    store, run_id, _bars = replayed
    key = _closed_trade_key(store, run_id)
    events = [e for e in store.paper_events(run_id) if e["event_key"] == key]
    entry = next(e for e in events if e["kind"] == "entry")
    exit_ = next(e for e in events if e["kind"] == "exit")
    exit_detail = json.loads(exit_["detail_json"])
    trade_row = next(
        t
        for t in store.trades(run_id)
        if pd.Timestamp(t["entry_time"]) == pd.Timestamp(entry["bar_time"])
        and round(t["entry_price"], 6) == round(entry["price"], 6)
    )

    steps = logic.trade_timeline(events, DAILY_SPEC)
    assert steps[-1].startswith("Net result:")
    assert format_inr(abs(exit_detail["net_pnl"])) in steps[-1]
    assert format_inr(abs(trade_row["net_pnl"])) in steps[-1]
    assert format_inr(entry["price"]) in steps[2]
    assert format_inr(exit_["price"]) in steps[4]
    assert format_inr(trade_row["exit_price"]) in steps[4]


def test_both_hit_sentence_appears_only_when_both_hit(replayed):
    store, run_id, _bars = replayed
    key = _closed_trade_key(store, run_id)
    events = [e for e in store.paper_events(run_id) if e["event_key"] == key]
    exit_ = next(e for e in events if e["kind"] == "exit")
    detail = json.loads(exit_["detail_json"])
    sentence = "The stop and the target were both touched"

    steps = logic.trade_timeline(events, DAILY_SPEC)
    exit_step = next(s for s in steps if s.startswith("Exited"))
    assert (sentence in exit_step) == bool(detail.get("both_hit"))

    # Force the other branch by hand, on a copy of the same recorded row --
    # the fingerprint this test exists to check, not a guess at wording.
    toggled = dict(exit_)
    flipped_detail = {**detail, "both_hit": not detail.get("both_hit")}
    toggled["detail_json"] = json.dumps(flipped_detail)
    other_events = [e if e["kind"] != "exit" else toggled for e in events]
    other_steps = logic.trade_timeline(other_events, DAILY_SPEC)
    other_exit_step = next(s for s in other_steps if s.startswith("Exited"))
    assert (sentence in other_exit_step) != (sentence in exit_step)


def test_an_open_trade_never_shows_an_exit_step(replayed):
    store, run_id, _bars = replayed
    key = _open_trade_key(store, run_id)
    events = [
        e for e in store.paper_events(run_id) if e["event_key"] == key and e["kind"] == "entry"
    ]
    steps = logic.trade_timeline(events, DAILY_SPEC, open_pnl=1234.5)
    assert not any(s.startswith("Exited") for s in steps)
    assert not any(s.startswith("Net result:") for s in steps)
    assert steps[-1].startswith("Still open.")
    assert format_inr(1234.5) in steps[-1]


def test_no_step_text_promises_a_fill(replayed):
    """A queued entry can still be turned down at the open for lack of money;
    nothing in a trade's timeline may say 'fills at'."""
    store, run_id, _bars = replayed
    for key in [_closed_trade_key(store, run_id), _open_trade_key(store, run_id)]:
        events = [e for e in store.paper_events(run_id) if e["event_key"] == key]
        steps = logic.trade_timeline(events, DAILY_SPEC)
        assert not any("fills at" in s for s in steps)


def test_trade_timeline_of_an_unknown_trade_is_empty():
    assert logic.trade_timeline([], DAILY_SPEC) == []


# --------------------------------------------------------------- order_timeline


def test_order_timeline_uses_the_same_words_as_describe_event():
    events = [
        {
            "order_id": 7,
            "kind": "triggered",
            "detail": {"reason": "price", "low": 100.0, "candle": "2024-01-01T09:16:00+05:30"},
            "bar_time": "2024-01-01T09:16:00+05:30",
        },
        {
            "order_id": 7,
            "kind": "filled",
            "side": "buy",
            "price": 101.0,
            "detail": {"lots": 1, "contract": "NIFTY FUT", "raw_price": 100.9, "charges": 20.0},
            "bar_time": "2024-01-01T09:21:00+05:30",
        },
    ]
    steps = orders_view.order_timeline(events, 7)
    assert steps == [orders_view.describe_event(e, 7, frozenset()) for e in events]


def test_order_timeline_excludes_the_finished_row():
    events = [
        {
            "order_id": 7,
            "kind": "filled",
            "side": "buy",
            "price": 1,
            "bar_time": "2024-01-01T09:21:00+05:30",
            "detail": {"lots": 1, "contract": "NIFTY FUT", "raw_price": 0.99, "charges": 1.0},
        },
        {"order_id": 7, "kind": "finished", "detail": {}},
    ]
    assert len(orders_view.order_timeline(events, 7)) == 1


# ------------------------------------------------------------------- the API


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv(api.auth.LOGIN_ID_ENV_VAR, LOGIN_ID)
    monkeypatch.setenv(api.auth.ENV_VAR, PASSWORD)


@pytest.fixture
def client(configured, tmp_path) -> TestClient:
    app = api.create_app(Store(tmp_path / "web.sqlite"), secret="test-secret")
    return TestClient(app, base_url="https://testserver")


@pytest.fixture
def authed(client) -> TestClient:
    r = client.post("/api/login", json={"login_id": LOGIN_ID, "password": PASSWORD})
    assert r.status_code == 200
    return client


@pytest.mark.parametrize(
    "path",
    [
        "/api/activity",
        "/api/paper/1/trade/x",
        "/api/orders/1/timeline",
    ],
)
def test_the_new_routes_all_require_a_login(client, path) -> None:
    r = client.get(path)
    assert r.status_code == 401
    assert "log in" in r.json()["error"].lower()


def _seed_two_paper_runs(store: Store):
    """Two strategies, each with one recorded signal_skipped decision at a
    different, known time, so ordering and filtering can be checked without
    running the engine."""
    assert DAILY_SPEC is not None
    sid_a = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    other = DAILY_SPEC.model_copy(update={"name": "second"})
    sid_b = store.save_strategy(other, describe(other))
    rid_a = store.start_run(sid_a, "paper", {"capital": CAPITAL, "started_at": "2024-01-01"})
    rid_b = store.start_run(sid_b, "paper", {"capital": CAPITAL, "started_at": "2024-01-01"})
    store.record_paper_decision(
        rid_a,
        "signal_skipped",
        "k1",
        bar_time="2024-01-01T00:00:00+00:00",
        symbol="NIFTY",
        reason_code="capital",
    )
    store.record_paper_decision(
        rid_b,
        "signal_taken",
        "k2",
        bar_time="2024-01-02T00:00:00+00:00",
        symbol="NIFTY",
    )
    return rid_a, rid_b


def test_activity_merges_and_sorts_newest_first(authed) -> None:
    store = authed.app.state.store
    rid_a, rid_b = _seed_two_paper_runs(store)
    r = authed.get("/api/activity")
    assert r.status_code == 200
    items = r.json()["items"]
    assert len(items) == 2
    # k2 (2 Jan) is newer than k1 (1 Jan)
    assert items[0]["source_id"] == rid_b
    assert items[1]["source_id"] == rid_a
    assert [it["time"] for it in items] == sorted((it["time"] for it in items), reverse=True)


def test_activity_skips_only_filter(authed) -> None:
    store = authed.app.state.store
    _seed_two_paper_runs(store)
    r = authed.get("/api/activity", params={"skips_only": "1"})
    items = r.json()["items"]
    assert items
    assert all(it["skip"] for it in items)


def test_activity_source_id_filter(authed) -> None:
    store = authed.app.state.store
    rid_a, _rid_b = _seed_two_paper_runs(store)
    r = authed.get("/api/activity", params={"source_id": rid_a})
    items = r.json()["items"]
    assert items
    assert all(it["source_id"] == rid_a for it in items)


def test_paper_trade_endpoint_returns_the_same_steps_as_the_pure_function(authed) -> None:
    store = authed.app.state.store
    bars = _daily_bars()
    assert DAILY_SPEC is not None
    sid = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    started_at = bars.index[300] - pd.Timedelta(minutes=1)
    run_id = store.start_run(
        sid, "paper", {"capital": CAPITAL, "started_at": started_at.isoformat()}
    )
    run = store.get_run(run_id)
    now = bars.index[-1] + runner.close_offset("1d") + pd.Timedelta(minutes=1)
    runner.step(store, run, fetch=lambda *a: bars, charge_fn=_zero, now=now)
    key = _closed_trade_key(store, run_id)
    r = authed.get(f"/api/paper/{run_id}/trade/{key}")
    assert r.status_code == 200
    events = [e for e in store.paper_events(run_id) if e["event_key"] == key]
    assert r.json()["steps"] == logic.trade_timeline(events, DAILY_SPEC)


def test_paper_trade_unknown_key_is_a_clear_error(authed) -> None:
    store = authed.app.state.store
    assert DAILY_SPEC is not None
    sid = store.save_strategy(DAILY_SPEC, describe(DAILY_SPEC))
    run_id = store.start_run(sid, "paper", {"capital": CAPITAL, "started_at": "2024-01-01"})
    r = authed.get(f"/api/paper/{run_id}/trade/NIFTY|2099-01-01T00:00:00")
    assert r.status_code == 400


def test_orders_timeline_endpoint(authed) -> None:
    store = authed.app.state.store
    order = ConditionalOrder(
        trigger=PriceTrigger(
            watch=Watched(kind="index", symbol="NIFTY"), direction="below", level=1.0
        ),
        action=TradeAction(side="buy", symbol="NIFTY", lots=1),
    )
    oid = store.save_one_off_order(
        "if nifty goes below 1, buy 1 lot nifty futures",
        order.model_dump_json(),
        "readback",
        "2024-01-01T00:00:00+00:00",
    )
    store.record_one_off_event(
        oid,
        "triggered",
        "ek1",
        bar_time="2024-01-01T09:16:00+05:30",
        detail={"reason": "price", "low": 0.5, "candle": "2024-01-01T09:16:00+05:30"},
    )
    r = authed.get(f"/api/orders/{oid}/timeline")
    assert r.status_code == 200
    assert r.json()["steps"] == ["Condition met: traded as low as 0.5 in the 01 Jan 09:16 minute."]


def test_orders_timeline_unknown_order_is_a_clear_error(authed) -> None:
    r = authed.get("/api/orders/999/timeline")
    assert r.status_code == 400


def test_an_open_trades_profit_says_which_candle_it_is_as_of(replayed):
    """The open profit comes from the last successful check. A paused run or a
    dead feed leaves it hours old, so the line must name that candle and must
    not claim to be "right now"."""
    store, run_id, _bars = replayed
    key = _open_trade_key(store, run_id)
    events = [
        e for e in store.paper_events(run_id) if e["event_key"] == key and e["kind"] == "entry"
    ]
    steps = logic.trade_timeline(
        events, DAILY_SPEC, open_pnl=1234.5, open_pnl_as_of="2024-03-04T09:45:00+05:30"
    )
    assert "as of the 04 Mar 09:45 candle" in steps[-1]
    assert "right now" not in " ".join(steps)


def test_one_unreadable_order_does_not_blank_the_activity_feed(authed, monkeypatch) -> None:
    from app import orders_view

    store = authed.app.state.store
    _seed_two_paper_runs(store)

    def broken(_store):
        raise KeyError("expiry")

    monkeypatch.setattr(orders_view.runner, "current_book", broken)
    r = authed.get("/api/activity")
    assert r.status_code == 200
    items = r.json()["items"]
    assert sum(it["source"] == "strategy" for it in items) == 2
    assert any("could not be read" in it["text"] for it in items)
