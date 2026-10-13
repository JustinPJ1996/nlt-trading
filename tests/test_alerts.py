"""Alerts: the notification inbox, one-off alerts, and alert-only strategy runs.

Three things this app must never do, each defended by a test below:

  * An alert must never create a fill, a position, a trade or any paper money
    -- for a one-off alert or for an alert-only strategy run.
  * The same notification must never be delivered twice.
  * An alert-only run must never count toward the proving gate.

Every rule here was checked by breaking the code it covers and watching the
test fail -- see the mutation table in the PR description.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

from nlt.engine.backtest import run_backtest
from nlt.notify import notify
from nlt.orders import runner as orders_runner
from nlt.orders.engine import Contract
from nlt.orders.models import AlertAction, ConditionalOrder, PnlTrigger
from nlt.orders.parse import parse_order
from nlt.orders.readback import alert_fired_text, describe
from nlt.paper import runner as paper_runner
from nlt.spec.models import StrategySpec
from nlt.store.db import Store
from nlt.translate.readback import describe as strategy_describe
from nlt.translate.rules import TranslationResult
from nlt.translate.rules import parse as translate_parse

IST = "Asia/Kolkata"
CAPITAL = 1_000_000.0


def t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=IST)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "alerts.sqlite")


# ----------------------------------------------------------- one-off alerts: parsing


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("Tell me when NIFTY goes below 22000", "price"),
        ("Alert me when banknifty goes above 52000", "price"),
        ("Notify me if nifty has had 4 red weekly candles in a row", "streak"),
        ("Let me know when nifty goes below 22000", "price"),
    ],
)
def test_alert_phrases_parse_to_the_right_trigger(sentence, kind):
    r = parse_order(sentence)
    assert r.order is not None, [q.text for q in r.questions]
    assert isinstance(r.order.action, AlertAction)
    assert r.order.trigger.kind == kind


def test_crosses_below_asks_rather_than_guesses_like_orders_do():
    """One-off orders do not read "crosses below" (a crossing needs the
    previous candle on the other side, which is not what "goes below" means).
    An alert must not quietly read it as "goes below" either."""
    r = parse_order("Let me know when nifty crosses below 22000")
    assert r.order is None
    assert r.questions


def test_tell_me_now_is_refused():
    r = parse_order("Tell me now")
    assert r.order is None
    assert r.questions


def test_ambiguous_pnl_alert_asks_loss_or_profit():
    s = "Tell me when my P&L falls below 50000"
    r = parse_order(s)
    assert r.order is None
    assert [q.field for q in r.questions] == ["pnl.sign"]
    loss = parse_order(s, {"pnl.sign": "a loss"})
    assert isinstance(loss.order.action, AlertAction)
    assert loss.order.trigger.threshold == -50000


def test_a_stated_pnl_alert_needs_no_question():
    r = parse_order("Tell me if my loss exceeds 50000")
    assert r.questions == [], [q.text for q in r.questions]
    assert isinstance(r.order.action, AlertAction)
    assert r.order.trigger.threshold == -50000


def test_alert_cannot_have_a_stop_or_target():
    r = parse_order("Tell me when nifty goes below 22000, stop loss 1%")
    assert r.order is None
    assert any(q.field == "trade.stop_loss" for q in r.questions)


def test_the_word_alert_alone_does_not_make_an_order():
    """The parser must not over-match on the mere word "alert" appearing somewhere."""
    r = parse_order("Set an alert on my phone if nifty goes below 22000")
    # "an alert on my phone if ..." is not one of the recognised alert verbs
    # ("tell me" / "alert me" / "notify me" / "let me know"), so this must be
    # refused like any other unrecognised action, never silently accepted.
    assert r.order is None


def test_none_of_the_70_user_sentences_parses_as_an_alert():
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "user_strategies.txt"
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    assert len(lines) == 70
    alerts = [
        x
        for x in lines
        if (r := parse_order(x)).order is not None and isinstance(r.order.action, AlertAction)
    ]
    assert alerts == []


# ----------------------------------------------------------- one-off alerts: readback


def test_alert_readback_always_says_nothing_is_bought_or_sold():
    for sentence in (
        "Tell me when NIFTY goes below 22000",
        "Notify me if nifty has had 4 red weekly candles in a row",
    ):
        order = parse_order(sentence).order
        assert order is not None, sentence
        lines = describe(order)
        assert any("Nothing is bought or sold" in line for line in lines), lines


def test_alert_readback_has_no_stop_loss_warning():
    order = parse_order("Tell me when NIFTY goes below 22000").order
    lines = describe(order)
    assert not any("stop loss" in line.lower() for line in lines)


def test_alert_fired_text_says_nothing_was_bought_or_sold():
    order = parse_order("Tell me when NIFTY goes below 22000").order
    text = alert_fired_text(order)
    assert "Nothing was bought or sold." in text
    assert "22,000" in text


# ----------------------------------------------------------------- the engine


OCT = Contract("NIFTY", "NIFTY26OCTFUT", pd.Timestamp("2026-10-27").date(), 101)
DAY = "2026-10-05"


def bars(start: str, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(t(start), periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(
        volume=0.0
    )


def flat(start: str, n: int, price: float) -> list[tuple[float, float, float, float]]:
    return [(price, price, price, price)] * n


def session(day: str, price: float, changes=None) -> pd.DataFrame:
    df = bars(f"{day} 09:15", flat(f"{day} 09:15", 375, price))
    for hhmm, row in (changes or {}).items():
        df.loc[t(f"{day} {hhmm}"), ["open", "high", "low", "close"]] = row
    return df


def dip_feed(low_at="10:05", low=21990.0, before_low=21950.0) -> orders_runner.Feed:
    index = session(
        DAY,
        22100,
        {"09:50": (22100, 22100, before_low, 22100), low_at: (22100, 22100, low, 22050)},
    )
    empty = bars("2026-01-01 09:15", [])
    return orders_runner.Feed(
        index_minutes=lambda s, a, b: index if s == "NIFTY" else empty,
        contract_minutes=lambda c, a, b: empty,
        daily=lambda name, a, b: empty,
        contracts=lambda s: [],
    )


def save_alert(store: Store, sentence: str, at: str, answers=None) -> int:
    r = parse_order(sentence, answers)
    assert r.order is not None, [q.text for q in r.questions]
    assert isinstance(r.order.action, AlertAction)
    return store.save_one_off_order(
        sentence, r.order.model_dump_json(), "\n".join(describe(r.order)), t(at).isoformat()
    )


def zero(symbol):
    return lambda price, qty, side: 0.0


def test_a_fired_alert_creates_exactly_one_notification_and_nothing_else(store):
    oid = save_alert(store, "Tell me when Nifty spot goes below 22000", f"{DAY} 10:00:30")
    orders_runner.step(store, dip_feed(), now=t(f"{DAY} 10:30"), charges=zero)

    notes = store.notifications()
    assert len(notes) == 1
    assert notes[0]["source"] == "one_off_alert"
    assert notes[0]["source_id"] == oid
    assert "Nothing was bought or sold" in notes[0]["text"]

    events = [e for e in store.one_off_events() if e["order_id"] == oid]
    assert {e["kind"] for e in events} == {"triggered", "finished"}
    assert "filled" not in {e["kind"] for e in events}
    assert "exited" not in {e["kind"] for e in events}

    book = orders_runner.current_book(store)
    assert book.positions == {}
    assert store.trades(0) == []  # no run id was ever created to hold a trade either


def test_repeated_passes_do_not_duplicate_the_notification(store):
    save_alert(store, "Tell me when Nifty spot goes below 22000", f"{DAY} 10:00:30")
    feed = dip_feed()
    for _ in range(3):
        orders_runner.step(store, feed, now=t(f"{DAY} 10:30"), charges=zero)
    assert len(store.notifications()) == 1


def test_notify_is_idempotent_on_its_dedupe_key(store):
    first = notify(store, "one_off_alert", 1, "k1", "hello")
    second = notify(store, "one_off_alert", 1, "k1", "a different message")
    assert first is True
    assert second is False
    notes = store.notifications()
    assert len(notes) == 1
    assert notes[0]["text"] == "hello"


def test_pnl_alert_watches_the_chosen_positions_and_never_closes_them(store):
    # Build one open paper position with a real trade order, then an alert
    # watching its P&L -- the alert must notify but must never close it.
    trade_order = parse_order("If Nifty goes below 22000, buy 1 lot Nifty Futures").order
    a = store.save_one_off_order(
        "If Nifty goes below 22000, buy 1 lot Nifty Futures",
        trade_order.model_dump_json(),
        "",
        t(f"{DAY} 10:00:30").isoformat(),
    )
    feed = dip_feed()
    feed_with_fut = orders_runner.Feed(
        index_minutes=feed.index_minutes,
        contract_minutes=lambda c, a_, b: session(
            DAY, 22150, {"10:06": (22140, 22160, 22130, 22150)}
        ),
        daily=feed.daily,
        contracts=lambda s: [OCT],
    )
    orders_runner.step(store, feed_with_fut, now=t(f"{DAY} 10:15"), charges=zero)
    assert len(orders_runner.current_book(store).positions) == 1

    alert_order = ConditionalOrder(
        trigger=PnlTrigger(threshold=-1), action=AlertAction(positions=(a,))
    )
    guard = store.save_one_off_order(
        "Tell me if my loss exceeds 1",
        alert_order.model_dump_json(),
        "\n".join(describe(alert_order)),
        t(f"{DAY} 10:15:30").isoformat(),
    )
    # Drop the price so the position is at a loss.
    fut = session(DAY, 22150, {"10:06": (22140, 22160, 22130, 22150)})
    fut.loc[t(f"{DAY} 10:40"), ["open", "high", "low", "close"]] = (21900, 21900, 21800, 21800)
    feed_dropped = orders_runner.Feed(
        index_minutes=feed.index_minutes,
        contract_minutes=lambda c, a_, b: fut,
        daily=feed.daily,
        contracts=lambda s: [OCT],
    )
    orders_runner.step(store, feed_dropped, now=t(f"{DAY} 11:00"), charges=zero)

    notes = [n for n in store.notifications() if n["source_id"] == guard]
    assert len(notes) == 1
    assert (
        len(orders_runner.current_book(store).positions) == 1
    )  # still open, alert never closes it


# ------------------------------------------------------------ notification table


def test_notification_text_cannot_be_updated_and_rows_cannot_be_deleted(store):
    notify(store, "one_off_alert", 1, "k1", "hello")
    conn = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="only read_at"):
        conn.execute("UPDATE notification SET text = 'edited'")
    with pytest.raises(sqlite3.DatabaseError, match="cannot be deleted"):
        conn.execute("DELETE FROM notification")
    conn.close()


def test_read_at_is_the_only_column_that_may_change(store):
    store.create_notification("one_off_alert", 1, "k1", "hello")
    assert store.mark_notification_read(1) is True
    row = store.notifications()[0]
    assert row["read_at"] is not None
    assert row["text"] == "hello"
    assert store.mark_notification_read(1) is False  # already read


def test_unread_count_and_mark_all_read(store):
    store.create_notification("one_off_alert", 1, "k1", "one")
    store.create_notification("one_off_alert", 2, "k2", "two")
    assert store.unread_notification_count() == 2
    assert store.mark_all_notifications_read() == 2
    assert store.unread_notification_count() == 0


# ---------------------------------------------------------- alert-only strategies


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


DAILY_SPEC = translate_parse("buy nifty when rsi crosses below 40, target 3%, stop 2%").spec


def _zero(price, qty, side):
    return 0.0


def _start_alert_only(store: Store, spec: StrategySpec, started_at: pd.Timestamp) -> dict:
    sid = store.save_strategy(spec, strategy_describe(spec))
    run_id = store.start_run(
        sid,
        "paper",
        {"capital": CAPITAL, "started_at": started_at.isoformat(), "alert_only": True},
        alert_only=True,
    )
    return store.get_run(run_id)


def _replay(store, run, bars_, timeframe, *, from_index):
    offset = paper_runner.close_offset(timeframe)
    for ts in bars_.index[from_index:]:
        paper_runner.step(
            store,
            run,
            fetch=lambda s, tf, a, b: bars_,
            charge_fn=_zero,
            now=ts + offset + pd.Timedelta(minutes=1),
        )


@pytest.fixture(scope="module")
def alert_only_replay(tmp_path_factory):
    bars_ = _daily_bars()
    store = Store(tmp_path_factory.mktemp("alert_only") / "paper.sqlite")
    started_at = bars_.index[300] - pd.Timedelta(minutes=1)
    run = _start_alert_only(store, DAILY_SPEC, started_at)
    _replay(store, run, bars_, "1d", from_index=300)
    return store, run, bars_, started_at


def test_alert_only_run_notifies_once_per_signal_and_exit_never_for_skips(alert_only_replay):
    store, run, bars_, started_at = alert_only_replay
    expected = run_backtest(
        DAILY_SPEC,
        bars_,
        capital=CAPITAL,
        charge_fn=_zero,
        drop_partial_last_bar=False,
        trade_from=paper_runner.trade_from_for(started_at, "1d"),
    )
    taken = [d for d in expected.decisions if d.kind == "signal_taken"]
    closed = [tr for tr in expected.trades if tr.exit_reason != "end_of_data"]
    assert len(taken) >= 3 and len(closed) >= 3, (
        "the scenario must trade for this test to mean anything"
    )

    notes = [n for n in store.notifications() if n["source_id"] == run["id"]]
    signal_notes = [
        n
        for n in notes
        if "would buy" in n["text"].lower() or "would sell short" in n["text"].lower()
    ]
    exit_notes = [n for n in notes if n not in signal_notes]
    assert len(signal_notes) == len(taken)
    assert len(exit_notes) == len(closed)
    for n in notes:
        assert n["text"].endswith("Nothing is bought or sold."), n["text"]
    for n in signal_notes:
        assert "would buy NIFTY at the next candle's open" in n["text"], n["text"]
        # The user's own confirmed wording, not the engine's internal names,
        # and a daily candle is a date, not midnight.
        assert "RSI(14)" in n["text"] and "rsi14" not in n["text"], n["text"]
        assert "00:00" not in n["text"], n["text"]

    # No trade or paper money was ever written for this run.
    assert store.trades(run["id"]) == []


def test_alert_only_run_shows_no_money_and_is_never_proven(alert_only_replay):
    from app import logic

    store, run, bars_, started_at = alert_only_replay
    sid = run["strategy_id"]
    logic.stop_paper(store, run["id"])
    metrics = store.get_run(run["id"])["metrics_json"]
    assert "equity" not in metrics
    assert "realised_pnl" not in metrics
    assert store.is_proven(sid) is False


def test_is_proven_requires_a_non_alert_only_paper_run(store):
    from app import logic

    bt = run_backtest(DAILY_SPEC, _daily_bars(), capital=CAPITAL, charge_fn=_zero)
    result = logic.PipelineResult(
        translation=TranslationResult(spec=DAILY_SPEC), spec=DAILY_SPEC, backtest=bt
    )
    sid = logic.save_strategy(
        store, DAILY_SPEC, result, capital=CAPITAL, cost_model_label="Index futures"
    )
    assert not store.is_proven(sid)

    alert_run_id = logic.start_paper(
        store, sid, capital=CAPITAL, cost_model_label="Index futures", alert_only=True
    )
    logic.stop_paper(store, alert_run_id)
    assert not store.is_proven(sid)  # an alert-only run, even completed, never counts

    real_run_id = logic.start_paper(store, sid, capital=CAPITAL, cost_model_label="Index futures")
    assert not store.is_proven(sid)  # still running
    logic.stop_paper(store, real_run_id)
    assert store.is_proven(sid)


def test_a_short_strategys_alert_says_sell_short_not_buy(tmp_path):
    spec = translate_parse("short nifty when rsi crosses above 60, target 3%, stop 2%").spec
    assert spec.direction == "short"
    bars_ = _daily_bars()
    store = Store(tmp_path / "short.sqlite")
    run = _start_alert_only(store, spec, bars_.index[300] - pd.Timedelta(minutes=1))
    _replay(store, run, bars_, "1d", from_index=300)
    texts = [n["text"] for n in store.notifications()]
    signals = [t for t in texts if "at the next candle's open" in t]
    assert signals, "the scenario must signal for this test to mean anything"
    assert all("would sell short NIFTY" in t for t in signals)
    assert not any("would buy NIFTY" in t for t in texts)


def test_an_alert_only_run_is_marked_in_the_same_step_it_is_created(tmp_path):
    """The every-minute job must never see an alert-only run unmarked: it would
    record paper money for it."""
    store = Store(tmp_path / "start.sqlite")
    sid = store.save_strategy(DAILY_SPEC, strategy_describe(DAILY_SPEC))
    seen = []
    real_tx = store.tx

    from contextlib import contextmanager

    @contextmanager
    def watching_tx():
        with real_tx() as conn:
            yield conn
        for r in store.active_paper_runs():
            seen.append(store.is_alert_only(r["id"]))

    store.tx = watching_tx
    store.start_run(sid, "paper", {}, alert_only=True)
    assert seen and all(seen)


@pytest.mark.parametrize(
    "sentence",
    [
        "Tell me when NIFTY goes below 22000",
        "Notify me if nifty has had 4 red weekly candles in a row",
    ],
)
def test_alert_readback_never_promises_an_instant_alert(sentence):
    """Checks run once a minute (a day or week candle an hour after it ends),
    and only while the job runs. "The moment it does" overpromised."""
    from nlt.orders.readback import describe as order_describe

    text = " ".join(order_describe(parse_order(sentence).order))
    assert "moment" not in text
    assert "goes through" not in text
    assert "after that candle ends" in text
