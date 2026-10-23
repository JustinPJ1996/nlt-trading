"""One pass of the one-off order checker: fetch candles, run the engine, record events.

Run every minute by `scripts/paper_run.py`, alongside paper strategies.

The record in `one_off_event` is the only state. Each pass rebuilds the book
from it, judges every 1-minute candle that has finished since the cursor, and
records each event the first time it is seen -- before moving the cursor. A
crash between the two costs nothing: the next pass judges the same candles
again, and the events it finds are already recorded.

Nothing here can place an order. The only door to Zerodha is `nlt.data.kite`,
which can only read prices.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nlt.costs.slippage import LEGACY as LEGACY_SLIPPAGE
from nlt.costs.slippage import Slippage
from nlt.data import kite
from nlt.notify import notify
from nlt.orders.engine import (
    IST,
    MINUTE,
    Book,
    Contract,
    Market,
    SavedOrder,
    advance,
    book_from_record,
)
from nlt.orders.models import (
    AlertAction,
    ConditionalOrder,
    PriceTrigger,
    StreakTrigger,
    TradeAction,
)
from nlt.orders.readback import alert_fired_text
from nlt.store.db import Store

# How far back minute candles are fetched before the cursor, so the last price of
# every contract is known even after a weekend or a holiday.
_LOOKBACK = dt.timedelta(days=5)
# Enough daily candles to build 20 weekly ones with room to spare.
_DAILY_HISTORY = dt.timedelta(days=200)


@dataclass
class Feed:
    """Where prices come from. `kite_feed()` in real use; made-up candles in tests."""

    index_minutes: Callable[[str, dt.date, dt.date], pd.DataFrame]
    contract_minutes: Callable[[Contract, dt.date, dt.date], pd.DataFrame]
    daily: Callable[[str, dt.date, dt.date], pd.DataFrame]  # "NIFTY" or "NIFTY FUT"
    contracts: Callable[[str], list[Contract]]


def kite_feed() -> Feed:
    from nlt.data.futures import contract as terms

    source = kite.KiteSource()

    def contracts(symbol: str) -> list[Contract]:
        rows = kite.futures_contracts(symbol)
        return [
            Contract(symbol, r["tradingsymbol"], r["expiry"], int(r["instrument_token"]))
            for _, r in rows.iterrows()
        ]

    return Feed(
        index_minutes=lambda symbol, start, end: source.bars(symbol, "1m", start, end),
        contract_minutes=lambda c, start, end: source.contract_bars(
            c.token, "1m", start, end, terms(c.symbol).session
        ),
        daily=lambda name, start, end: source.bars(name, "1d", start, end),
        contracts=contracts,
    )


def charges_for(symbol: str):
    from nlt.costs.charges import McxFuturesCharges, NseFuturesCharges, charge_fn
    from nlt.data.futures import is_commodity

    return charge_fn(McxFuturesCharges() if is_commodity(symbol) else NseFuturesCharges())


def _ts(raw: str) -> pd.Timestamp:
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST)  # ty: ignore[invalid-return-type]


def _order_slippage(row: dict) -> Slippage:
    """What this order's fills use: what was recorded for it, or -- for an order
    saved before 2026-10-13, which recorded nothing -- the old hidden 0.02%.
    """
    if row.get("slippage_kind") is None:
        return LEGACY_SLIPPAGE
    return Slippage(kind=row["slippage_kind"], value=row["slippage_value"])


def saved_orders(store: Store) -> list[SavedOrder]:
    """Every order not cancelled, oldest first."""
    return [
        SavedOrder(
            r["id"],
            ConditionalOrder.model_validate_json(r["order_json"]),
            _ts(r["created_at"]),
            slippage=_order_slippage(r),
        )
        for r in store.one_off_orders()
        if r["cancelled_at"] is None
    ]


def current_book(store: Store) -> Book:
    return book_from_record(saved_orders(store), store.one_off_events())


def has_work(book: Book) -> bool:
    return bool(book.positions) or any(o.id not in book.finished for o in book.orders)


@dataclass
class StepResult:
    health: str
    message: str
    new_events: list[str] = field(default_factory=list)


def is_due(now: pd.Timestamp) -> bool:
    """Weekdays from just before NSE opens until just after MCX closes. Holidays find no candles.

    MCX closes at 23:55 in winter, so the last few checks of a trading day --
    an expiry-day close among them -- fall just after midnight.
    """
    now = now.tz_convert(IST)
    if now.time() >= dt.time(8, 55):
        return now.weekday() < 5
    return now.time() <= dt.time(0, 30) and (now - pd.Timedelta(days=1)).weekday() < 5


def step(
    store: Store,
    feed: Feed,
    *,
    now: pd.Timestamp | None = None,
    charges=charges_for,
) -> StepResult:
    now = (now or pd.Timestamp.now(tz=IST)).tz_convert(IST)
    book = current_book(store)
    if not has_work(book):
        store.set_one_off_book(
            health="idle", message="No orders waiting.", cursor=_iso(_floor(now))
        )
        return StepResult("idle", "no orders waiting")

    if store.paper_kill_switch_engaged():
        # Candles during a pause are skipped, never acted on later at stale prices.
        store.set_one_off_book(
            health="paused",
            message="Paused: the paper kill switch is on. Nothing is checked or filled.",
            cursor=_iso(_floor(now) - MINUTE),
        )
        return StepResult("paused", "paper kill switch engaged")

    state = store.one_off_book()
    if state.get("cursor"):
        cursor = _ts(state["cursor"])
    else:
        cursor = min(o.created_at for o in book.orders) - MINUTE
    start = (cursor - _LOOKBACK).date()
    end = now.date()

    try:
        market = _market(book, feed, start, end, cursor.date())  # ty: ignore[invalid-argument-type]
    except kite.KiteTokenExpired:
        return _problem(store, "token_expired", "Kite token has expired. Log in again to resume.")
    except kite.KiteNotConnected:
        return _problem(
            store, "not_connected", "Kite is not connected. Save your Kite login to start."
        )
    except kite.KiteLoginFailed as exc:
        return _problem(store, "login_failed", f"Could not log in to Kite. {exc}")
    except kite.KiteError as exc:
        return _problem(store, "feed_down", f"Could not get prices from Kite: {exc}")

    events, new_cursor = advance(book, market, cursor=cursor, now=now, charges=charges)  # ty: ignore[invalid-argument-type]
    orders_by_id = {o.id: o for o in book.orders}
    observed = _iso(now)
    new: list[str] = []
    for e in events:
        recorded = store.record_one_off_event(
            e.order_id,
            e.kind,
            e.key,
            observed_at=observed,
            symbol=e.symbol,
            bar_time=_iso(e.bar_time) if e.bar_time is not None else None,
            side=e.side,
            quantity=e.quantity,
            price=e.price,
            detail=e.detail,
        )
        if recorded:
            new.append(f"{e.kind} #{e.order_id}")
            # An alert's own "triggered" event IS the moment it fires -- see
            # `engine.advance`'s `trigger()`, which always follows it with
            # "finished" for an AlertAction and never with a fill. Delivering
            # the notification here, keyed off that same recorded event,
            # means it is sent exactly once, however many passes re-derive it.
            if e.kind == "triggered":
                saved = orders_by_id.get(e.order_id)
                if saved is not None and isinstance(saved.order.action, AlertAction):
                    notify(
                        store,
                        "one_off_alert",
                        e.order_id,
                        f"one_off_alert:{e.order_id}",
                        alert_fired_text(saved.order),
                    )

    prices = {}
    for key, df in market.minute.items():
        finished = df[df.index + MINUTE <= now]
        if not finished.empty:
            prices[key] = {
                "price": float(finished["close"].iloc[-1]),
                "at": _iso(finished.index[-1]),
            }
    message = (
        f"Checked up to the {new_cursor:%d %b %H:%M} candle."
        if new_cursor > cursor
        else "No new finished candles yet."
    )
    store.set_one_off_book(health="ok", message=message, cursor=_iso(new_cursor), prices=prices)
    return StepResult("ok", message, new)


def _problem(store: Store, health: str, message: str) -> StepResult:
    store.set_one_off_book(health=health, message=message)
    return StepResult(health, message)


def _market(book: Book, feed: Feed, start: dt.date, end: dt.date, from_day: dt.date) -> Market:
    minute: dict[str, pd.DataFrame] = {}
    daily: dict[str, pd.DataFrame] = {}
    contracts: dict[str, list[Contract]] = {}

    def live(symbol: str) -> list[Contract]:
        if symbol not in contracts:
            contracts[symbol] = sorted(feed.contracts(symbol), key=lambda c: c.expiry)
        return contracts[symbol]

    def follow(c: Contract) -> None:
        key = f"FUT:{c.tradingsymbol}"
        if key not in minute:
            minute[key] = feed.contract_minutes(c, start, end)

    for o in book.orders:
        if o.id in book.finished:
            continue
        t, a = o.order.trigger, o.order.action
        if o.id not in book.triggered:
            if isinstance(t, PriceTrigger):
                if t.watch.kind == "index":
                    key = f"IDX:{t.watch.symbol}"
                    if key not in minute:
                        minute[key] = feed.index_minutes(t.watch.symbol, start, end)
                else:
                    for c in [c for c in live(t.watch.symbol) if c.expiry >= from_day][:2]:
                        follow(c)
            elif isinstance(t, StreakTrigger):
                key = f"{'IDX' if t.watch.kind == 'index' else 'FUT'}:{t.watch.symbol}"
                if key not in daily:
                    name = t.watch.symbol if t.watch.kind == "index" else f"{t.watch.symbol} FUT"
                    daily[key] = feed.daily(name, end - _DAILY_HISTORY, end)
        if isinstance(a, TradeAction) and o.id not in book.positions and o.id not in book.closed:
            for c in [c for c in live(a.symbol) if c.expiry >= from_day][:2]:
                follow(c)

    for pos in book.positions.values():
        follow(Contract(pos.symbol, pos.contract, pos.expiry, pos.token))

    return Market(minute=minute, contracts=contracts, daily=daily)


def _floor(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.floor("min")


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()
