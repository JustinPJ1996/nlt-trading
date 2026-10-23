"""The one-off orders page, server side. The browser only lays out what this returns.

Reading a sentence adds what only today's prices can say -- how far away a
level is, whether it is already crossed, how long a run already is -- as
separate notes. The readback itself stays the order's own mechanical
description and never depends on the market.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable

import pandas as pd
from pydantic import ValidationError

from nlt.costs.slippage import ZERO as ZERO_SLIPPAGE
from nlt.costs.slippage import Slippage
from nlt.orders import runner
from nlt.orders.engine import IST, streak_status
from nlt.orders.models import (
    AlertAction,
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
from nlt.translate.readback import format_inr

Daily = Callable[[str], pd.DataFrame]
"""Kite's name ("NIFTY", "NIFTY FUT") -> recent daily candles, the last one today's so far."""


class OrderError(ValueError):
    """Something the person needs to fix, in words they can act on."""


def kite_daily(name: str) -> pd.DataFrame:
    from nlt.data.kite import KiteSource

    today = dt.date.today()
    return KiteSource().bars(name, "1d", today - dt.timedelta(days=200), today)


def _now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def _level(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _signed(x: float) -> str:
    return ("+" if x >= 0 else "-") + format_inr(abs(x))


def _when(raw) -> str:
    if raw is None:
        return ""
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST).strftime("%d %b %H:%M")


def _short(watch) -> str:
    """ "NIFTY 50", "Crude Oil futures" -- the name alone, for a sentence about its price."""
    from nlt.data.futures import contract

    if watch.kind == "index":
        return "NIFTY 50" if watch.symbol == "NIFTY" else "NIFTY BANK"
    return f"{contract(watch.symbol).title} futures"


# ------------------------------------------------------------------- reading


def read(store: Store, text: str, answers: dict[str, str], *, daily: Daily | None = None) -> dict:
    daily = daily or kite_daily
    r = parse_order(text, answers)
    out: dict = {
        "order": None,
        "readback": [],
        "warnings": [],
        "notes": [],
        "questions": [
            {"field": q.field, "text": q.text, "why": q.why, "suggestion": q.suggestion}
            for q in r.questions
        ],
        "answers": answers,
        "offer_stop": False,
        "needs_positions": False,
        "positions": [],
    }
    if r.order is None:
        return out
    order = r.order
    out["order"] = order.model_dump(mode="json")
    out["readback"] = describe(order)
    t, a = order.trigger, order.action
    out["offer_stop"] = isinstance(a, TradeAction) and a.stop_pct is None
    warnings, notes = out["warnings"], out["notes"]

    def last(name: str) -> tuple[float, pd.Timestamp] | None:
        try:
            df = daily(name)
        except Exception:  # noqa: BLE001 - a missing price is a note, never a failure
            return None
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[-1]), df.index[-1]

    if isinstance(t, PriceTrigger):
        name = t.watch.symbol if t.watch.kind == "index" else f"{t.watch.symbol} FUT"
        got = last(name)
        if got is None:
            notes.append(
                "I couldn't check today's price, so I can't say how far away that level is."
            )
        else:
            price, _ = got
            notes.append(f"The latest price of {_short(t.watch)} is {_level(price)}.")
            crossed = price < t.level if t.direction == "below" else price > t.level
            if crossed:
                warnings.append(
                    f"It is already {t.direction} {_level(t.level)}. If you save this, it will go "
                    "through on the first finished minute after you save it."
                )
            elif abs(t.level / price - 1) > 0.2:
                warnings.append(
                    f"{_level(t.level)} is {abs(t.level / price - 1) * 100:.0f}% away from the "
                    "latest price. Check that is the number you meant."
                )

    if isinstance(t, StreakTrigger):
        name = t.watch.symbol if t.watch.kind == "index" else f"{t.watch.symbol} FUT"
        try:
            run_now = streak_status(daily(name), t, _now())
        except Exception:  # noqa: BLE001
            run_now = None
        unit = "week" if t.candle == "week" else "day"
        if run_now is None:
            notes.append(
                "I couldn't check the recent candles, so I can't say how the run stands now."
            )
        else:
            what = {
                "red": f"red {unit}s",
                "green": f"green {unit}s",
                "lower_close": f"{unit}s closing lower than the one before",
                "higher_close": f"{unit}s closing higher than the one before",
            }[t.rule]
            notes.append(f"Right now {_short(t.watch)} has had {run_now} finished {what} in a row.")
            if run_now >= t.count:
                warnings.append(
                    f"That run is already there. Only {unit}s that end after you save this count, "
                    f"so it goes through only if the next {unit} does the same."
                )

    if isinstance(a, TradeAction):
        from nlt.data.futures import contract

        got = last(f"{a.symbol} FUT")
        if got is not None:
            c = contract(a.symbol)
            value = got[0] * a.lots * c.multiplier
            notes.append(
                f"At the latest price, {a.lots} lot{'s' if a.lots != 1 else ''} of {c.title} "
                f"futures {'is' if a.lots == 1 else 'are'} worth about {format_inr(value)}."
            )

    if isinstance(a, ExitAction) or (isinstance(a, AlertAction) and isinstance(t, PnlTrigger)):
        out["needs_positions"] = True
        out["positions"] = open_positions(store)
        if not out["positions"]:
            warnings.append(
                "You have no open paper positions yet. Positions opened by one-off orders show "
                "up here to choose from."
            )
    return out


# -------------------------------------------------------------------- saving


def create(
    store: Store,
    text: str,
    answers: dict[str, str],
    raw_order,
    positions,
    *,
    slippage: Slippage = ZERO_SLIPPAGE,
) -> int:
    """Saves exactly the order that was read back and confirmed.

    The sentence is read again here and must give the same order, so what is
    stored beside the sentence is always what that sentence means.

    `slippage` is a run setting, like a backtest's -- not part of the order
    itself -- and defaults to zero for every order saved from here on. Nothing
    reaches here for an order saved before 2026-10-13: those rows have no
    slippage recorded at all, which `nlt.orders.runner.saved_orders` treats as
    the old hidden 0.02%, never as zero.
    """
    try:
        order = ConditionalOrder.model_validate(raw_order)
    except ValidationError:
        raise OrderError(
            "That order could not be read back. Check it again from the start."
        ) from None
    if isinstance(order.trigger, NowTrigger):
        raise OrderError("Use the Close button on a position to close it now.")
    again = parse_order(text, answers).order
    if again is None or again != order:
        raise OrderError(
            "That order no longer matches its sentence. Check the sentence again before saving."
        )
    names = {}
    needs_positions = isinstance(order.action, ExitAction) or (
        isinstance(order.action, AlertAction) and isinstance(order.trigger, PnlTrigger)
    )
    if needs_positions:
        verb = "close" if isinstance(order.action, ExitAction) else "watch"
        try:
            chosen = tuple(sorted({int(p) for p in positions or ()}))
        except (TypeError, ValueError):
            raise OrderError(f"Choose which positions to {verb}.") from None
        open_now = {p["id"]: p for p in open_positions(store)}
        if not chosen:
            raise OrderError(f"Choose which positions to {verb}.")
        if any(p not in open_now for p in chosen):
            raise OrderError("One of the chosen positions is no longer open. Check again.")
        action_cls = type(order.action)
        order = ConditionalOrder(trigger=order.trigger, action=action_cls(positions=chosen))
        names = {p: open_now[p]["name"] for p in chosen}
    readback = "\n".join(describe(order, names))
    return store.save_one_off_order(
        text.strip(),
        order.model_dump_json(),
        readback,
        _now().isoformat(),
        slippage=slippage.to_dict(),
    )


def cancel(store: Store, order_id: int) -> None:
    if not store.cancel_one_off_order(order_id):
        raise OrderError(
            "That order has already gone through or was cancelled, so it can't be cancelled."
        )


def close_now(store: Store, position_id: int) -> int:
    open_now = {p["id"]: p for p in open_positions(store)}
    if position_id not in open_now:
        raise OrderError("That position is not open.")
    book = runner.current_book(store)
    for o in book.orders:
        if (
            isinstance(o.order.trigger, NowTrigger)
            # A "now" trigger is always an exit (ConditionalOrder validator), so it has `positions`.
            and position_id in o.order.action.positions  # ty: ignore[unresolved-attribute]
            and o.id not in book.finished
        ):
            raise OrderError("Closing that position is already under way.")
    order = ConditionalOrder(trigger=NowTrigger(), action=ExitAction(positions=(position_id,)))
    name = open_now[position_id]["name"]
    return store.save_one_off_order(
        f"Close {name} now",
        order.model_dump_json(),
        "\n".join(describe(order, {position_id: name})),
        _now().isoformat(),
        slippage=ZERO_SLIPPAGE.to_dict(),
    )


def check_now(store: Store) -> str:
    return runner.step(store, runner.kite_feed()).message


# ------------------------------------------------------------------- listing


def open_positions(store: Store) -> list[dict]:
    from nlt.data.futures import contract

    book = runner.current_book(store)
    prices = store.one_off_book().get("prices", {})
    out = []
    for oid, p in sorted(book.positions.items()):
        c = contract(p.symbol)
        mark = prices.get(f"FUT:{p.contract}")
        last = mark["price"] if mark else None
        pnl = p.gross(last) if last is not None else None
        lots = f"{p.lots} lot{'s' if p.lots != 1 else ''}"
        out.append(
            {
                "id": oid,
                "name": f"#{oid} ({'long' if p.direction == 'long' else 'short'} {lots} {c.title} futures)",
                "what": f"{'Bought' if p.direction == 'long' else 'Sold short'} {lots} of {p.contract}",
                "entered": _when(p.entry_time),
                "entry_price": _level(p.entry_price),
                "last_price": _level(last) if last is not None else "",
                "last_at": _when(mark["at"]) if mark else "",
                "stop": _level(p.stop_level) if p.stop_level is not None else "",
                "target": _level(p.target_level) if p.target_level is not None else "",
                "pnl": pnl,
                "pnl_text": _signed(pnl) if pnl is not None else "",
                "expiry": f"{p.expiry:%d %b %Y}",
            }
        )
    return out


_EXIT_REASON = {
    "stop": "Stop loss hit",
    "target": "Target hit",
    "expiry": "Contract expired, closed at the expiry-day close",
    "closed_by_order": "Closed by order",
}


def describe_event(
    e: dict,
    card_id: int | None = None,
    pressed: frozenset = frozenset(),
    *,
    is_alert: bool = False,
) -> str:
    """One recorded event in words. `card_id` is the order whose history this is shown in;
    `pressed` are the orders made by the Close now button; `is_alert` says this
    order's action is an AlertAction, so "finished" must never read as a trade."""
    d = e["detail"]
    kind = e["kind"]
    if kind == "finished" and is_alert:
        return "Alert sent. Nothing was bought or sold."
    if kind == "triggered":
        reason = d.get("reason")
        if reason == "price":
            if "low" in d:
                return f"Condition met: traded as low as {_level(d['low'])} in the {_when(d['candle'])} minute."
            return f"Condition met: traded as high as {_level(d['high'])} in the {_when(d['candle'])} minute."
        if reason == "pnl":
            return f"Condition met: P&L of the chosen positions reached {_signed(d['pnl'])} in the {_when(d['candle'])} minute."
        if reason == "streak":
            return f"Condition met when the candle ending {_when(e['bar_time'])} finished."
        return "Close requested."
    if kind == "filled":
        verb = "Bought" if e["side"] == "buy" else "Sold short"
        lots = int(d["lots"])
        raw_price = d["raw_price"]
        # Only claim slippage was applied when it actually moved the price --
        # at zero slippage the fill price and the opening price are the same,
        # and saying "plus slippage" there would be untrue.
        price_note = (
            f", {_level(raw_price)}, plus slippage)" if abs(e["price"] - raw_price) > 1e-9 else ")"
        )
        return (
            f"{verb} {lots} lot{'s' if lots != 1 else ''} of {d['contract']} at "
            f"{_level(e['price'])} (the {_when(e['bar_time'])} opening price{price_note}. "
            f"Charges {format_inr(d['charges'])}."
        )
    if kind == "exited":
        why = _EXIT_REASON.get(d.get("reason"), d.get("reason", "Closed"))
        by = d.get("by_order")
        if by in pressed:
            why = "You pressed Close now"
        elif by:
            why = f"Closed by order #{by}"
        if card_id is not None and e["order_id"] != card_id:
            why = f"Position #{e['order_id']} closed"
        both = (
            " The same minute also reached the target; the stop is assumed to have come first."
            if d.get("both_hit")
            else ""
        )
        at = " at" if why.endswith("closed") else ": closed at"
        return (
            f"{why}{at} {_level(e['price'])}. Profit or loss {_signed(d['gross_pnl'])}, "
            f"after charges {_signed(d['net_pnl'])}.{both}"
        )
    if kind == "finished":
        return "Order finished."
    return d.get("note", "")


def order_timeline(
    events: list[dict], order_id: int, pressed: frozenset = frozenset()
) -> list[str]:
    """One order's history, in words, from its recorded `one_off_event` rows alone.

    `events` are that order's own rows (and any 'exited' row belonging to a
    different order that closed a position this order opened), already in the
    order they were recorded. Text comes entirely from `describe_event`,
    never written fresh here, so the activity feed and this timeline can
    never disagree about what a given row means.
    """
    return [describe_event(e, order_id, pressed) for e in events if e["kind"] != "finished"]


def _status(row: dict, book, events: list[dict], is_alert: bool = False) -> tuple[str, str]:
    oid = row["id"]
    if row["cancelled_at"]:
        return "cancelled", "Cancelled"
    if oid in book.positions:
        return "open", "Position open"
    if oid in book.finished:
        if is_alert:
            return "alerted", "Alerted"
        return "done", "Done"
    if oid in book.triggered:
        return "filling", "Condition met, filling at the next price"
    return "waiting", "Waiting"


def page(store: Store) -> dict:
    from app import logic

    book = runner.current_book(store)
    events = store.one_off_events()
    by_order: dict[int, list[dict]] = {}
    for e in events:
        by_order.setdefault(e["order_id"], []).append(e)
        if e["kind"] == "exited" and e["detail"].get("by_order"):
            by_order.setdefault(e["detail"]["by_order"], []).append(e)
    state = store.one_off_book()
    pressed = frozenset(o.id for o in book.orders if isinstance(o.order.trigger, NowTrigger))
    orders = []
    for row in reversed(store.one_off_orders()):
        row_order = ConditionalOrder.model_validate_json(row["order_json"])
        is_alert = isinstance(row_order.action, AlertAction)
        key, label = _status(row, book, by_order.get(row["id"], []), is_alert)
        mine = sorted(by_order.get(row["id"], []), key=lambda e: e["id"])
        realised = next(
            (
                e["detail"]["net_pnl"]
                for e in mine
                if e["kind"] == "exited" and e["order_id"] == row["id"]
            ),
            None,
        )
        orders.append(
            {
                "id": row["id"],
                "sentence": row["sentence"],
                "readback": row["readback"].split("\n"),
                "saved": _when(row["created_at"]),
                "status": key,
                "status_label": label,
                "can_cancel": key == "waiting",
                "result_text": _signed(realised) if realised is not None else "",
                "result": realised,
                "events": [
                    {
                        "when": _when(e["bar_time"] or e["observed_at"]),
                        "text": describe_event(e, row["id"], pressed, is_alert=is_alert),
                    }
                    for e in mine
                    if e["kind"] != "finished"
                ],
            }
        )
    return {
        "kill_switch": store.paper_kill_switch_engaged(),
        "job_installed": logic.paper_job_installed(),
        "checker": {
            "health": state["health"],
            "message": state["message"],
            "checked": _when(state["updated_at"]),
        },
        "positions": open_positions(store),
        "orders": orders,
    }
