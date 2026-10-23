"""One-off conditional orders: reading the sentence, the readback, and paper fills.

Nothing here touches Kite: the feed is a set of made-up candles, and the clock
is whatever `now` each test passes. Every rule asserted here was checked by
breaking the code it covers and watching the test fail.
"""

from __future__ import annotations

import datetime as dt
import sqlite3

import pandas as pd
import pytest

from nlt.costs.slippage import LEGACY as LEGACY_SLIPPAGE
from nlt.costs.slippage import Slippage
from nlt.orders import runner
from nlt.orders.engine import SLIPPAGE_PCT, Contract, SavedOrder, book_from_record
from nlt.orders.models import (
    ConditionalOrder,
    ExitAction,
    NowTrigger,
    PnlTrigger,
    PriceTrigger,
    StreakTrigger,
    TradeAction,
)
from nlt.orders.parse import parse_order
from nlt.orders.readback import describe
from nlt.store.db import Store

IST = "Asia/Kolkata"
OCT = Contract("NIFTY", "NIFTY26OCTFUT", dt.date(2026, 10, 27), 101)
NOV = Contract("NIFTY", "NIFTY26NOVFUT", dt.date(2026, 11, 24), 102)
BUY = 1 + SLIPPAGE_PCT / 100
SELL = 1 - SLIPPAGE_PCT / 100


def t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz=IST)  # ty: ignore[invalid-return-type]  # pandas stubs: Timestamp(str) is never NaT


# ------------------------------------------------------------------ reading


def test_justins_first_sentence():
    r = parse_order("If Nifty spot goes below 22000, buy 1 lot Nifty Futures")
    assert r.questions == []
    assert r.order is not None
    d = r.order.model_dump()
    assert d["trigger"] == {
        "kind": "price",
        "watch": {"symbol": "NIFTY", "kind": "index"},
        "direction": "below",
        "level": 22000.0,
    }
    assert d["action"] == {
        "kind": "trade",
        "side": "buy",
        "symbol": "NIFTY",
        "lots": 1,
        "stop_pct": None,
        "target_pct": None,
    }
    assert any(w.startswith("No stop loss") for w in r.warnings)


def test_justins_second_sentence_reads_the_brackets():
    r = parse_order(
        "Buy 1 lot Futures if Nifty has been going down for 4 consecutive weeks "
        "(4 consecutive red candles)"
    )
    assert r.questions == []
    assert r.order is not None
    assert r.order.trigger.model_dump() == {
        "kind": "streak",
        "watch": {"symbol": "NIFTY", "kind": "index"},
        "candle": "week",
        "count": 4,
        "rule": "red",
    }
    assert r.order is not None
    assert isinstance(r.order.action, TradeAction)
    assert r.order.action.symbol == "NIFTY" and r.order.action.lots == 1


def test_going_down_without_the_brackets_is_a_question():
    r = parse_order("Buy 1 lot Futures if Nifty has been going down for 4 consecutive weeks")
    assert r.order is None
    assert [q.field for q in r.questions] == ["streak.rule"]
    red = parse_order(
        "Buy 1 lot Futures if Nifty has been going down for 4 consecutive weeks",
        {"streak.rule": "red candles"},
    )
    assert red.order is not None
    assert isinstance(red.order.trigger, StreakTrigger)
    assert red.order.trigger.rule == "red"
    lower = parse_order(
        "Buy 1 lot Futures if Nifty has been going down for 4 consecutive weeks",
        {"streak.rule": "lower than the week before"},
    )
    assert lower.order is not None
    assert isinstance(lower.order.trigger, StreakTrigger)
    assert lower.order.trigger.rule == "lower_close"


def test_brackets_that_disagree_are_a_question_not_ignored():
    for clash in ("5 consecutive red candles", "4 consecutive green candles"):
        r = parse_order(f"Buy 1 lot Futures if Nifty has been going down for 4 weeks ({clash})")
        assert r.order is None, clash
        assert r.questions[0].field == "trigger.streak"
    r = parse_order("If nifty goes below 22000 buy 1 lot nifty futures (urgent)")
    assert r.order is None and '"urgent"' in r.questions[0].text


def test_justins_third_sentence_asks_loss_or_profit():
    s = "If P&L has gone below Rs. 50000, exit positions"
    r = parse_order(s)
    assert r.order is None
    assert [q.field for q in r.questions] == ["pnl.sign"]
    loss = parse_order(s, {"pnl.sign": "a loss"})
    assert loss.order is not None
    assert isinstance(loss.order.trigger, PnlTrigger)
    assert loss.order.trigger.threshold == -50000
    assert loss.order.action == ExitAction()
    profit = parse_order(s, {"pnl.sign": "profit falling under it"})
    assert profit.order is not None
    assert isinstance(profit.order.trigger, PnlTrigger)
    assert profit.order.trigger.threshold == 50000


@pytest.mark.parametrize(
    "sentence",
    [
        "If P&L has gone below Rs. -50000, exit positions",
        "If my loss exceeds 50k, square off all positions",
        "if mtm falls below minus 50000 then close all positions",
        "Exit all positions if I lose more than Rs 50,000",
    ],
)
def test_a_stated_loss_needs_no_question(sentence):
    r = parse_order(sentence)
    assert r.questions == [], [q.text for q in r.questions]
    assert r.order is not None
    assert isinstance(r.order.trigger, PnlTrigger)
    assert r.order.trigger.threshold == -50000


@pytest.mark.parametrize(
    ("sentence", "field"),
    [
        ("When banknifty crosses 52000, sell 2 lots banknifty futures", "price.direction"),
        ("If nifty has 3 red candles in a row, buy 1 lot nifty futures", "streak.candle"),
        ("buy 1 lot nifty if nifty falls below 22000", "trade.symbol"),
        ("If Nifty goes below 22000, buy nifty futures", "trade.lots"),
        ("If Nifty goes below 22000, buy 1 lot nifty futures, stop loss 100", "trade.stop_loss"),
    ],
)
def test_ambiguities_become_questions(sentence, field):
    r = parse_order(sentence)
    assert r.order is None
    assert [q.field for q in r.questions] == [field]


@pytest.mark.parametrize(
    "sentence",
    [
        "Exit everything if VIX crosses 20",
        "Whenever NIFTY falls below 22000, buy 1 lot nifty futures",
        "If nifty goes below 22000, buy 1 lot of the 22000 put",
        "If sensex goes below 70000, buy 1 lot nifty futures",
        "If P&L falls below -50000 buy 1 lot nifty futures",
        "If nifty goes below 22000 in the next week, buy 1 lot nifty futures",
        "If reliance goes below 1200, buy 1 lot reliance futures",
    ],
)
def test_refused(sentence):
    r = parse_order(sentence)
    assert r.order is None and r.questions


def test_refusals_say_why():
    """Each refusal names its real reason, not a generic "didn't understand"."""
    repeat = parse_order("Whenever NIFTY falls below 22000, buy 1 lot nifty futures").questions
    assert [q.field for q in repeat] == ["kind"] and "Build page" in repeat[0].text
    later = parse_order("Buy 1 lot nifty futures if nifty falls below 22000 every time")
    assert "Build page" in later.questions[0].text
    option = parse_order("If nifty goes below 22000, buy 1 lot nifty 22000 pe").questions
    assert [q.field for q in option] == ["trade.symbol"] and "options" in option[0].text
    vix = parse_order("Exit everything if VIX crosses 20").questions
    assert "VIX" in vix[0].text


def test_none_of_the_70_user_sentences_becomes_an_order():
    """The 0-wrong property, for this reader too: none of the 70 is a one-off order."""
    from pathlib import Path

    path = Path(__file__).parent / "fixtures" / "user_strategies.txt"
    lines = [x.strip() for x in path.read_text().splitlines() if x.strip()]
    assert len(lines) == 70
    assert [x for x in lines if parse_order(x).order is not None] == []


def test_stop_and_target_are_read_and_mcx_contracts_are_kept_apart():
    r = parse_order("if crude oil goes above 6500 buy 1 lot crude oil mini, sl 1%, target 2%")
    assert r.order is not None
    assert isinstance(r.order.trigger, PriceTrigger)
    assert isinstance(r.order.action, TradeAction)
    assert r.order.trigger.watch.symbol == "CRUDEOIL"
    assert r.order.action.symbol == "CRUDEOILM"
    assert (r.order.action.stop_pct, r.order.action.target_pct) == (1.0, 2.0)
    assert r.warnings == []


def test_a_stop_can_be_added_by_answer():
    r = parse_order(
        "If Nifty goes below 22000, buy 1 lot Nifty Futures", {"trade.stop_loss": "1.5%"}
    )
    assert r.order is not None
    assert isinstance(r.order.action, TradeAction)
    assert r.order.action.stop_pct == 1.5
    assert r.warnings == []


# ------------------------------------------------------------------ readback


def test_readback_says_what_it_will_and_will_not_do():
    r = parse_order("If Nifty spot goes below 22000, buy 1 lot Nifty Futures")
    assert r.order is not None
    lines = describe(r.order)
    assert lines[0] == "When the NIFTY 50 index (spot) trades below 22,000"
    assert lines[1] == "Buy 1 lot of NIFTY futures (65 units), at the market price"
    assert lines[2].startswith("No stop loss.")
    assert "Paper only: nothing is sent to Zerodha." in lines
    assert any("never rolled over" in x for x in lines)


def test_readback_for_a_short_with_stop_and_mcx_lot():
    r = parse_order("If gold goes above 75000, sell 2 lots gold futures, stop loss 1%, target 3%")
    assert r.order is not None
    lines = describe(r.order)
    assert lines[1].startswith("Sell (go short) 2 lots of Gold futures (each lot = ")
    assert "Stop loss: 1% above the price it is sold at" in lines
    assert "Target: 3% below the price it is sold at" in lines


def test_readback_for_streak_and_pnl():
    s = describe(
        parse_order("Buy 1 lot Futures if Nifty has had 4 red weekly candles in a row").order  # ty: ignore[invalid-argument-type]  # parse_order(...).order is Optional; a None fails here
    )
    assert s[0] == (
        "When the NIFTY 50 index (spot) has 4 red weekly candles in a row "
        "(each week closed below where it opened)"
    )
    p = describe(parse_order("If my loss exceeds 50000, exit positions").order, {})  # ty: ignore[invalid-argument-type]  # parse_order(...).order is Optional; a None fails here
    assert (
        p[0] == "When the combined P&L of the chosen positions falls to a loss of Rs 50,000 or more"
    )
    assert any("before charges" in x for x in p)


# --------------------------------------------------------------- the engine


def bars(start: str, rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.date_range(t(start), periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(
        volume=0.0
    )


def flat(start: str, n: int, price: float) -> list[tuple[float, float, float, float]]:
    return [(price, price, price, price)] * n


def session(day: str, price: float, changes: dict[str, tuple] | None = None) -> pd.DataFrame:
    df = bars(f"{day} 09:15", flat(f"{day} 09:15", 375, price))
    for hhmm, row in (changes or {}).items():
        df.loc[t(f"{day} {hhmm}"), ["open", "high", "low", "close"]] = row
    return df


class Feed:
    def __init__(self, index=None, futures=None, daily=None, contracts=None):
        self.index = index or {}
        self.futures = futures or {}
        self.daily_frames = daily or {}
        self.listed = contracts or {"NIFTY": [OCT, NOV]}

    def feed(self) -> runner.Feed:
        empty = bars("2026-01-01 09:15", [])
        return runner.Feed(
            index_minutes=lambda s, a, b: self.index.get(s, empty),
            contract_minutes=lambda c, a, b: self.futures.get(c.tradingsymbol, empty),
            daily=lambda name, a, b: self.daily_frames[name],
            contracts=lambda s: self.listed.get(s, []),
        )


def zero(symbol):
    return lambda price, qty, side: 0.0


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "orders.db")


def save(store: Store, sentence: str, at: str, *, answers=None, positions=(), slippage=None) -> int:
    r = parse_order(sentence, answers)
    assert r.order is not None, [q.text for q in r.questions]
    order = r.order
    if positions:
        order = ConditionalOrder(
            trigger=order.trigger, action=ExitAction(positions=tuple(positions))
        )
    return store.save_one_off_order(
        sentence, order.model_dump_json(), "", t(at).isoformat(), slippage=slippage
    )


def run(store, feed, now: str, charges=zero):
    return runner.step(store, feed.feed(), now=t(now), charges=charges)


def events(store, kind=None, order_id=None):
    return [
        e
        for e in store.one_off_events()
        if (kind is None or e["kind"] == kind) and (order_id is None or e["order_id"] == order_id)
    ]


DAY = "2026-10-05"


def dip_feed(low_at="10:05", low=21990.0, before_low=21950.0) -> Feed:
    index = session(
        DAY,
        22100,
        {"09:50": (22100, 22100, before_low, 22100), low_at: (22100, 22100, low, 22050)},
    )
    fut = session(DAY, 22150, {"10:06": (22140, 22160, 22130, 22150)})
    return Feed(index={"NIFTY": index}, futures={OCT.tradingsymbol: fut})


def test_price_order_fills_at_the_next_minutes_open_with_slippage(store):
    oid = save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    run(store, dip_feed(), f"{DAY} 10:30")
    (trig,) = events(store, "triggered")
    assert pd.Timestamp(trig["bar_time"]) == t(f"{DAY} 10:06")
    assert trig["detail"]["low"] == 21990
    (fill,) = events(store, "filled")
    assert fill["order_id"] == oid
    assert pd.Timestamp(fill["bar_time"]) == t(f"{DAY} 10:06")
    assert fill["price"] == pytest.approx(22140 * BUY)
    assert fill["quantity"] == 65
    assert fill["symbol"] == OCT.tradingsymbol


def test_a_dip_before_the_order_was_saved_does_not_count(store):
    # The 09:50 dip is before the order; so is the 10:00 candle it was saved during.
    feed = dip_feed(low_at="10:00")
    save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    run(store, feed, f"{DAY} 10:30")
    assert events(store, "triggered") == []


def test_below_means_below_not_touching(store):
    save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    run(store, dip_feed(low=22000.0), f"{DAY} 10:30")
    assert events(store, "triggered") == []


def test_a_minute_is_judged_only_after_it_has_settled(store):
    save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    feed = dip_feed()
    run(store, feed, f"{DAY} 10:06:30")  # the 10:05 candle has only just finished
    assert events(store, "triggered") == []
    run(store, feed, f"{DAY} 10:07:00")
    assert len(events(store, "triggered")) == 1
    assert events(store, "filled") == []  # the 10:06 candle has not settled yet
    run(store, feed, f"{DAY} 10:08:00")
    assert len(events(store, "filled")) == 1


def test_stop_is_assumed_first_when_one_candle_hits_both(store):
    oid = save(
        store,
        "If Nifty goes below 22000, buy 1 lot Nifty Futures, stop loss 1%, target 1%",
        f"{DAY} 10:00:30",
    )
    feed = dip_feed()
    fut = feed.futures[OCT.tradingsymbol]
    fut.loc[t(f"{DAY} 10:10"), ["open", "high", "low", "close"]] = (22150, 22400, 21900, 22000)
    run(store, feed, f"{DAY} 10:30")
    (ex,) = events(store, "exited", oid)
    stop = 22140 * 0.99  # from the fill's open, before slippage
    assert ex["detail"]["reason"] == "stop"
    assert ex["detail"]["raw_price"] == pytest.approx(stop)
    assert ex["price"] == pytest.approx(stop * SELL)
    assert ex["detail"]["both_hit"] is True


def test_a_gap_through_the_stop_fills_at_the_open(store):
    oid = save(
        store, "If Nifty goes below 22000, buy 1 lot Nifty Futures, stop loss 1%", f"{DAY} 10:00:30"
    )
    feed = dip_feed()
    fut = feed.futures[OCT.tradingsymbol]
    fut.loc[t(f"{DAY} 10:10"), ["open", "high", "low", "close"]] = (21800, 21850, 21700, 21800)
    run(store, feed, f"{DAY} 10:30")
    (ex,) = events(store, "exited", oid)
    assert ex["detail"]["raw_price"] == 21800


def test_target_on_a_short(store):
    oid = save(
        store,
        "If Nifty goes below 22000, sell 1 lot Nifty Futures, stop loss 2%, target 1%",
        f"{DAY} 10:00:30",
    )
    feed = dip_feed()
    fut = feed.futures[OCT.tradingsymbol]
    fut.loc[t(f"{DAY} 10:12"), ["open", "high", "low", "close"]] = (22100, 22110, 21900, 21950)
    run(store, feed, f"{DAY} 10:30")
    (fill,) = events(store, "filled", oid)
    assert fill["side"] == "sell" and fill["price"] == pytest.approx(22140 * SELL)
    (ex,) = events(store, "exited", oid)
    assert ex["detail"]["reason"] == "target"
    assert ex["detail"]["raw_price"] == pytest.approx(22140 * 0.99)
    assert ex["side"] == "buy"
    assert ex["detail"]["gross_pnl"] == pytest.approx((22140 * SELL - 22140 * 0.99 * BUY) * 65)


def test_a_new_order_saved_with_zero_slippage_fills_at_the_exact_price(store):
    """A new order defaults to zero slippage (see `app.orders_view.create`); a
    legacy order with nothing stored (every other test in this file) still
    uses `SLIPPAGE_PCT`/`LEGACY`."""
    oid = save(
        store,
        "If Nifty spot goes below 22000, buy 1 lot Nifty Futures",
        f"{DAY} 10:00:30",
        slippage={"kind": "pct", "value": 0.0},
    )
    run(store, dip_feed(), f"{DAY} 10:30")
    (fill,) = events(store, "filled", oid)
    assert fill["price"] == pytest.approx(22140)  # no BUY factor: zero slippage


def test_a_custom_slippage_is_stored_and_applied_to_both_the_fill_and_the_exit(store):
    oid = save(
        store,
        "If Nifty goes below 22000, sell 1 lot Nifty Futures, stop loss 2%, target 1%",
        f"{DAY} 10:00:30",
        slippage={"kind": "rs", "value": 5.0},
    )
    feed = dip_feed()
    fut = feed.futures[OCT.tradingsymbol]
    fut.loc[t(f"{DAY} 10:12"), ["open", "high", "low", "close"]] = (22100, 22110, 21900, 21950)
    # Two separate passes, deliberately: the fill happens in the first, and
    # the exit only in the second, which can only see the position again by
    # rebuilding it from the recorded "filled" event (`book_from_record`) --
    # a single pass would build the Position in memory and never exercise
    # that reconstruction path at all.
    run(store, feed, f"{DAY} 10:08")
    (fill,) = events(store, "filled", oid)
    # A sell receives Rs 5 less than the raw open, not a percentage.
    assert fill["side"] == "sell" and fill["price"] == pytest.approx(22140 - 5.0)
    run(store, feed, f"{DAY} 10:30")
    (ex,) = events(store, "exited", oid)
    assert ex["detail"]["reason"] == "target"
    raw_exit = 22140 * 0.99
    # The exit is a buy (closing a short): Rs 5 MORE than the raw price.
    assert ex["side"] == "buy" and ex["price"] == pytest.approx(raw_exit + 5.0)
    assert ex["detail"]["gross_pnl"] == pytest.approx(((22140 - 5.0) - (raw_exit + 5.0)) * 65)


def _fill_event(order_id: int, *, slippage_detail=None) -> dict:
    """A minimal "filled" event dict, as `Store.one_off_events` would hand one
    back -- enough for `book_from_record` to rebuild a `Position` from it.
    """
    detail = {
        "contract": "NIFTY26OCTFUT",
        "symbol": "NIFTY",
        "expiry": "2026-10-27",
        "token": 101,
        "lots": 1,
        "raw_price": 22140.0,
        "charges": 0.0,
        "stop_level": None,
        "target_level": None,
    }
    if slippage_detail is not None:
        detail["slippage"] = slippage_detail
    return {
        "order_id": order_id,
        "kind": "filled",
        "bar_time": t(f"{DAY} 10:06").isoformat(),
        "side": "buy",
        "quantity": 65,
        "price": 22140.0,
        "detail": detail,
    }


def test_book_from_record_treats_a_fill_with_no_stored_slippage_as_legacy():
    """A fill recorded before 2026-10-13 has no "slippage" key in its detail at
    all -- rebuilding the book from it must use the old hidden 0.02%, not the
    new zero default, so a later exit re-prices it the same way it always was.
    """
    order = SavedOrder(
        1,
        parse_order("If Nifty spot goes below 22000, buy 1 lot Nifty Futures").order,  # ty: ignore[invalid-argument-type]  # parse_order(...).order is Optional; a None fails here
        t(f"{DAY} 09:00"),
    )
    book = book_from_record([order], [_fill_event(1)])
    assert book.positions[1].slippage == LEGACY_SLIPPAGE


def test_book_from_record_uses_the_slippage_actually_stored_on_the_fill():
    order = SavedOrder(
        1,
        parse_order("If Nifty spot goes below 22000, buy 1 lot Nifty Futures").order,  # ty: ignore[invalid-argument-type]  # parse_order(...).order is Optional; a None fails here
        t(f"{DAY} 09:00"),
    )
    chosen = Slippage("rs", 3.0)
    book = book_from_record([order], [_fill_event(1, slippage_detail=chosen.to_dict())])
    assert book.positions[1].slippage == chosen


def test_expiry_closes_at_the_last_candle_and_a_fill_on_expiry_day_goes_to_next_month(store):
    eve, expiry = "2026-10-26", "2026-10-27"
    index = pd.concat(
        [
            session(eve, 22100, {"10:05": (22100, 22100, 21990, 22050)}),
            session(expiry, 22100, {"10:05": (22100, 22100, 21990, 22050)}),
        ]
    )
    oct_ = pd.concat(
        [session(eve, 22150), session(expiry, 22200, {"15:29": (22200, 22210, 22190, 22500)})]
    )
    nov = pd.concat([session(eve, 22300), session(expiry, 22350)])
    feed = Feed(index={"NIFTY": index}, futures={OCT.tradingsymbol: oct_, NOV.tradingsymbol: nov})
    first = save(store, "If Nifty goes below 22000, buy 1 lot Nifty Futures", f"{eve} 10:00:30")
    second = save(
        store, "If Nifty goes below 22000, buy 2 lots Nifty Futures", f"{expiry} 10:00:30"
    )
    run(store, feed, f"{eve} 15:40")
    # F&O closes at 15:40 since 3 Aug 2026; the expiry close waits 5 minutes more.
    run(store, feed, f"{expiry} 15:43")  # the session is over, but not the grace period
    assert events(store, "exited", first) == []
    run(store, feed, f"{expiry} 15:46")
    (ex,) = events(store, "exited", first)
    assert ex["detail"]["reason"] == "expiry"
    assert ex["price"] == 22500  # the expiry close, no slippage
    assert pd.Timestamp(ex["bar_time"]) == t(f"{expiry} 15:29")
    (fill,) = events(store, "filled", second)
    assert fill["symbol"] == NOV.tradingsymbol
    assert fill["quantity"] == 130
    assert events(store, "exited", second) == []
    assert {e["order_id"] for e in events(store, "finished")} == {first}


def weekly_daily(weeks: list[tuple[float, float]], first_monday: str) -> pd.DataFrame:
    """Daily candles, five a week; each week opens at its first value and closes at its second.

    Every single day is green -- so a weekly test can only pass if weeks are
    built from days, not judged day by day.
    """
    rows, idx = [], []
    day = pd.Timestamp(first_monday, tz=IST)
    for w_open, w_close in weeks:
        # Gaps down between days, green within each day, net change w_close - w_open.
        step = (w_close - w_open) / 5
        for k in range(5):
            o = w_open + step * k - 10
            c = o + 5 if k < 4 else w_close
            if k == 0:
                o = w_open
            rows.append((o, max(o, c) + 1, min(o, c) - 1, c))
            idx.append(day + pd.Timedelta(days=k))
        day += pd.Timedelta(days=7)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx), columns=["open", "high", "low", "close"])


def streak_feed() -> Feed:
    # Weeks of 7, 14, 21 Sep green; 28 Sep, 5 Oct, 12 Oct, 19 Oct red.
    weeks: list[tuple[float, float]] = [
        (100, 110),
        (110, 120),
        (120, 130),
        (130, 120),
        (120, 110),
        (110, 100),
        (100, 90),
    ]
    daily = weekly_daily(weeks, "2026-09-07")
    fut = pd.concat(
        [session(d, 22000 + i) for i, d in enumerate(pd.bdate_range("2026-10-05", "2026-10-30"))]
    )
    return Feed(daily={"NIFTY": daily}, futures={OCT.tradingsymbol: fut, NOV.tradingsymbol: fut})


def test_weekly_red_run_counts_only_weeks_ending_after_the_order(store):
    feed = streak_feed()
    sentence = "Buy 1 lot Futures if Nifty has had 3 red weekly candles in a row"
    oid = save(store, sentence, "2026-10-14 11:00")  # mid-week: weeks of 28 Sep and 5 Oct red
    run(store, feed, "2026-10-16 16:00")  # the week ended at 15:30; judged after 16:30
    assert events(store, "triggered") == []
    run(store, feed, "2026-10-16 16:31")
    (trig,) = events(store, "triggered", oid)
    assert pd.Timestamp(trig["bar_time"]) == t("2026-10-16 15:30")
    run(store, feed, "2026-10-19 09:20")
    (fill,) = events(store, "filled", oid)
    assert pd.Timestamp(fill["bar_time"]) == t("2026-10-19 09:15")  # Monday's open


def test_a_run_already_complete_when_saved_waits_for_the_next_week(store):
    feed = streak_feed()
    oid = save(
        store,
        "Buy 1 lot Futures if Nifty has had 2 red weekly candles in a row",
        "2026-10-10 10:00",
    )
    run(store, feed, "2026-10-16 17:00")
    (trig,) = events(store, "triggered", oid)
    assert pd.Timestamp(trig["bar_time"]) == t("2026-10-16 15:30")


def test_lower_closes_are_not_red_candles(store):
    # Weeks closing lower than the week before, but each week green.
    weeks: list[tuple[float, float]] = [(100, 110), (90, 105), (80, 100), (70, 95)]
    feed = streak_feed()
    feed.daily_frames["NIFTY"] = weekly_daily(weeks, "2026-09-28")
    red = save(
        store,
        "Buy 1 lot Futures if Nifty has had 3 red weekly candles in a row",
        "2026-09-28 10:00",
    )
    lower = save(
        store,
        "Buy 1 lot Futures if Nifty has closed lower for 3 weeks in a row",
        "2026-09-28 10:00",
    )
    run(store, feed, "2026-10-23 17:00")  # the third lower close is the week of 19 Oct
    (trig,) = events(store, "triggered")
    assert trig["order_id"] == lower
    assert pd.Timestamp(trig["bar_time"]) == t("2026-10-23 15:30")
    assert red not in {e["order_id"] for e in events(store, "triggered")}


def test_pnl_exit_watches_only_the_chosen_positions(store):
    fut = session(
        DAY,
        22140,
        {
            **{f"10:{m:02d}": (22100, 22100, 22000, 22050) for m in range(20, 30)},
            **{f"10:{m:02d}": (22010, 22010, 22000, 22010) for m in range(30, 40)},
        },
    )
    feed = dip_feed()
    feed.futures[OCT.tradingsymbol] = fut
    a = save(store, "If Nifty goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    b = save(store, "If Nifty goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:40")
    run(store, feed, f"{DAY} 10:15")
    assert len(events(store, "filled")) == 2
    guard = save(store, "If my loss exceeds 8000, exit positions", f"{DAY} 10:15:30", positions=[a])
    # At 10:20 each position is 90 points down (Rs 5,850): together past Rs 8,000,
    # the chosen one alone not. At 10:30 the chosen one is 130 down: Rs 8,450.
    run(store, feed, f"{DAY} 10:28")
    assert events(store, "triggered", guard) == []
    run(store, feed, f"{DAY} 10:45")
    (trig,) = events(store, "triggered", guard)
    assert pd.Timestamp(trig["bar_time"]) == t(f"{DAY} 10:31")
    (ex,) = events(store, "exited", a)
    assert ex["detail"]["by_order"] == guard
    assert pd.Timestamp(ex["bar_time"]) == t(f"{DAY} 10:31")
    assert ex["detail"]["raw_price"] == 22010
    assert events(store, "exited", b) == []
    assert {e["order_id"] for e in events(store, "finished")} == {a, guard}


def test_close_now_exits_at_the_next_minute(store):
    a = save(store, "If Nifty goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    run(store, dip_feed(), f"{DAY} 10:15")
    now_order = ConditionalOrder(trigger=NowTrigger(), action=ExitAction(positions=(a,)))
    store.save_one_off_order(
        "close", now_order.model_dump_json(), "", t(f"{DAY} 10:20:10").isoformat()
    )
    run(store, dip_feed(), f"{DAY} 10:30")
    (ex,) = events(store, "exited", a)
    assert pd.Timestamp(ex["bar_time"]) == t(f"{DAY} 10:21")


def test_minute_by_minute_records_exactly_what_one_pass_does(tmp_path):
    """Replaying live, a minute at a time, must equal judging it all at once."""
    sentences = [
        ("If Nifty goes below 22000, buy 1 lot Nifty Futures, stop loss 0.5%", f"{DAY} 10:00:30"),
        ("If Nifty goes above 22080, sell 1 lot Nifty Futures, target 0.2%", f"{DAY} 09:20:30"),
    ]

    def feed():
        f = dip_feed()
        fut = f.futures[OCT.tradingsymbol]
        fut.loc[t(f"{DAY} 10:40"), ["open", "high", "low", "close"]] = (22100, 22110, 21990, 22000)
        f.index["NIFTY"].loc[t(f"{DAY} 09:30"), "high"] = 22090
        return f

    def record(store):
        return [
            (e["order_id"], e["kind"], e["event_key"], e["bar_time"], e["price"])
            for e in store.one_off_events()
        ]

    once = Store(tmp_path / "once.db")
    live = Store(tmp_path / "live.db")
    for s, at in sentences:
        save(once, s, at)
        save(live, s, at)
    run(once, feed(), f"{DAY} 11:00")
    for minute in pd.date_range(t(f"{DAY} 09:21"), t(f"{DAY} 11:00"), freq="1min"):
        runner.step(live, feed().feed(), now=minute + pd.Timedelta(seconds=5), charges=zero)
    assert record(live) == record(once)
    assert len(record(once)) >= 6


def test_the_paper_kill_switch_skips_candles_rather_than_acting_on_them_late(store):
    save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    feed = dip_feed()  # dips at 10:05
    run(store, feed, f"{DAY} 10:02")
    store.engage_paper_kill_switch("test")
    for m in range(3, 11):
        run(store, feed, f"{DAY} 10:{m:02d}:05")
    store.release_paper_kill_switch("test")
    run(store, feed, f"{DAY} 10:30")
    assert events(store, "triggered") == []
    feed.index["NIFTY"].loc[t(f"{DAY} 10:20"), "low"] = 21900
    run(store, feed, f"{DAY} 10:40")
    assert events(store, "triggered") == []  # 10:20 was already judged at 10:30
    feed.index["NIFTY"].loc[t(f"{DAY} 10:45"), "low"] = 21900
    run(store, feed, f"{DAY} 10:50")
    assert len(events(store, "triggered")) == 1


def test_cancelled_orders_never_trigger_and_triggered_ones_cannot_be_cancelled(store):
    a = save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    b = save(store, "If Nifty spot goes below 22000, buy 2 lots Nifty Futures", f"{DAY} 10:00:30")
    assert store.cancel_one_off_order(a) is True
    run(store, dip_feed(), f"{DAY} 10:30")
    assert {e["order_id"] for e in events(store, "triggered")} == {b}
    assert store.cancel_one_off_order(b) is False


def test_the_record_is_append_only(store):
    a = save(store, "If Nifty spot goes below 22000, buy 1 lot Nifty Futures", f"{DAY} 10:00:30")
    run(store, dip_feed(), f"{DAY} 10:30")
    conn = sqlite3.connect(store.path)
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("UPDATE one_off_event SET price = 1 WHERE order_id = ?", (a,))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        conn.execute("DELETE FROM one_off_event")


def test_charges_are_recorded_and_net_of_both_fills(store):
    oid = save(
        store, "If Nifty goes below 22000, buy 1 lot Nifty Futures, target 0.1%", f"{DAY} 10:00:30"
    )
    feed = dip_feed()
    feed.futures[OCT.tradingsymbol].loc[t(f"{DAY} 10:12"), "high"] = 22200
    run(store, feed, f"{DAY} 10:30", charges=runner.charges_for)
    (fill,) = events(store, "filled", oid)
    (ex,) = events(store, "exited", oid)
    assert fill["detail"]["charges"] > 0
    assert ex["detail"]["charges"] > fill["detail"]["charges"]
    assert ex["detail"]["net_pnl"] == pytest.approx(
        ex["detail"]["gross_pnl"] - ex["detail"]["charges"], abs=0.01
    )


def test_checking_hours_cover_the_mcx_evening():
    assert runner.is_due(t("2026-10-05 23:58"))  # Monday night
    assert runner.is_due(t("2026-10-06 00:20"))  # just after midnight, still Monday's session
    assert not runner.is_due(t("2026-10-06 01:00"))
    assert not runner.is_due(t("2026-10-10 10:00"))  # Saturday
    assert runner.is_due(t("2026-10-10 00:10"))  # Saturday 00:10 is Friday's session
    assert not runner.is_due(t("2026-10-11 00:10"))  # Sunday 00:10 is Saturday's
    assert not runner.is_due(t("2026-10-05 08:30"))
