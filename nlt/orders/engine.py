"""What one-off orders did, worked out from candles. No clock, no network, no database.

`advance` takes the orders, what has already been recorded about them, and the
candles that have finished since the last pass, and returns what happened, in
time order. The runner records each event the first time it is seen.

The rules are the strategy engine's rules, so paper orders and backtests cannot
disagree about what a fill is:

  * **Only finished candles.** A condition is judged on a 1-minute candle once
    that minute is over, never on the one still forming. A candle that started
    before the order was saved never counts -- part of it happened before.
  * **Fills at the next candle's open,** plus the slippage the order was
    saved with -- zero by default, like a backtest -- the first candle of the
    traded contract that starts at or after the moment the condition was met.
    An order saved before slippage became a user setting (2026-10-13) still
    uses the old hidden 0.02%: see `SavedOrder.slippage` and `LEGACY_SLIPPAGE`.
  * **Stop first** when one candle could hit both stop and target, and a gap
    past the stop fills at the open, not at the stop.
  * **Futures close at expiry**, at the expiry day's last candle. Never rolled.
    A new position is never opened in a contract on its expiry day -- it would
    be closed within hours -- so it goes into the next month's instead.

What is different from a strategy, each a choice written down here:

  * **Stops and targets are measured from the fill price.** A strategy measures
    them from the signal candle's close, because there the levels belong to
    the setup. A one-off order has no setup: "1% stop" means 1% from what you
    paid. With a fill one minute after the condition, the two barely differ --
    except after a weekend, when a weekly condition is met on Friday and the
    fill is Monday's open.
  * **P&L is before charges,** as Kite's positions screen shows it, and counts
    only the chosen positions still open.
  * **Daily and weekly conditions are judged an hour after the close,** so a
    late-published closing price is not acted on. The fill is at the next
    session's open either way, so nothing is lost by waiting.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd

from nlt.costs.slippage import LEGACY as LEGACY_SLIPPAGE
from nlt.costs.slippage import Slippage
from nlt.data.session import NSE_EQUITY, Session, session_date
from nlt.orders.models import (
    AlertAction,
    ConditionalOrder,
    ExitAction,
    NowTrigger,
    PnlTrigger,
    PriceTrigger,
    StreakTrigger,
    TradeAction,
    Watched,
)

IST = "Asia/Kolkata"
MINUTE = pd.Timedelta(minutes=1)

# Kept for tests and callers that predate the user-set Slippage setting: what
# every order saved before 2026-10-13 is still treated as using. New code
# should use `LEGACY_SLIPPAGE` (from `nlt.costs.slippage`) directly rather
# than this float.
SLIPPAGE_PCT = LEGACY_SLIPPAGE.value

# How long after a daily or weekly candle's close it is judged.
DAILY_GRACE = pd.Timedelta(hours=1)

# A minute candle is judged one minute after it finishes, not the instant it
# does. Kite publishes each candle a few seconds late, and not for every
# contract at the same moment; judging the newest minute straight away would
# move past a candle that had not arrived yet and never come back to it. The
# fill is the next candle's open either way, so waiting changes when an event
# is noticed, never its price.
SETTLE = pd.Timedelta(minutes=1)

# The expiry-day close is taken this long after the session's official close,
# for the same reason.
EXPIRY_GRACE = pd.Timedelta(minutes=5)

ChargeFn = Callable[[float, int, str], float]


# ------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class Contract:
    """One live futures contract, as Kite lists it."""

    symbol: str  # "NIFTY", "CRUDEOIL" -- our name
    tradingsymbol: str  # "NIFTY26OCTFUT" -- Kite's
    expiry: dt.date
    token: int = 0


@dataclass(frozen=True)
class SavedOrder:
    id: int
    order: ConditionalOrder
    created_at: pd.Timestamp
    # What this order's fills use. Defaults to `LEGACY_SLIPPAGE` because that is
    # what every order saved before 2026-10-13 was actually filled at -- the
    # runner is responsible for passing the real value for orders that stored
    # one (see `nlt.orders.runner.saved_orders`), so this default only ever
    # matters for an order with nothing recorded.
    slippage: Slippage = LEGACY_SLIPPAGE


@dataclass
class Position:
    order_id: int
    symbol: str
    contract: str
    expiry: dt.date
    direction: str  # "long" / "short"
    lots: int
    quantity: int  # lots x the contract's multiplier: P&L is price change x quantity
    entry_time: pd.Timestamp
    entry_price: float  # after slippage
    entry_charges: float
    stop_level: float | None
    target_level: float | None
    token: int = 0
    # The slippage this position's own entry (and, later, its exit) used.
    # Carried on the position rather than looked up again, so an exit always
    # pays what the order was saved with, even if a later order of the same
    # kind used a different setting.
    slippage: Slippage = LEGACY_SLIPPAGE

    def gross(self, price: float) -> float:
        sign = 1 if self.direction == "long" else -1
        return (price - self.entry_price) * self.quantity * sign


@dataclass
class Market:
    """Candles for one pass. Keys: "IDX:NIFTY" for an index, "FUT:<tradingsymbol>" for a contract.

    `daily` is keyed by watch key: "IDX:NIFTY", or "FUT:NIFTY" for the
    continuous futures series.
    """

    minute: dict[str, pd.DataFrame]
    contracts: dict[str, list[Contract]]
    daily: dict[str, pd.DataFrame] = field(default_factory=dict)


@dataclass(frozen=True)
class Event:
    order_id: int  # the order this happened to; for an exit, the position's order
    kind: str  # "triggered" | "filled" | "exited" | "finished" | "note"
    key: str  # unique within (order_id, kind): recorded once, however many passes see it
    bar_time: pd.Timestamp | None = None
    symbol: str = ""
    side: str | None = None
    quantity: float | None = None
    price: float | None = None
    detail: dict = field(default_factory=dict)


@dataclass
class Book:
    """Everything recorded so far, as the engine needs it."""

    orders: list[SavedOrder]
    triggered: dict[int, pd.Timestamp] = field(default_factory=dict)
    positions: dict[int, Position] = field(default_factory=dict)  # open only
    finished: set[int] = field(default_factory=set)
    closed: set[int] = field(default_factory=set)  # orders whose position has been closed


# ------------------------------------------------------------------ helpers


def _session(symbol: str, kind: str = "future") -> Session:
    if kind == "index":
        return NSE_EQUITY
    from nlt.data.futures import contract

    return contract(symbol).session


def _close_of(day: dt.date, session: Session) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(day, session.close_time), tz=IST)  # ty: ignore[invalid-return-type]


def _slip(price: float, side: str, slippage: Slippage) -> float:
    return slippage.apply(price, side)


def near_contract(
    contracts: list[Contract], day: dt.date, *, skip_expiry_day: bool
) -> Contract | None:
    """The contract to watch or trade on `day`: the first not yet expired.

    To open a position, a contract expiring that very day is skipped.
    """
    for c in contracts:
        if c.expiry > day or (c.expiry == day and not skip_expiry_day):
            return c
    return None


def watch_key(watch: Watched, contracts: dict[str, list[Contract]], day: dt.date) -> str | None:
    if watch.kind == "index":
        return f"IDX:{watch.symbol}"
    c = near_contract(contracts.get(watch.symbol, []), day, skip_expiry_day=False)
    return f"FUT:{c.tradingsymbol}" if c else None


def _iso(ts) -> str:
    return pd.Timestamp(ts).isoformat()


def _money(x: float) -> str:
    return f"{x:,.2f}"


# ----------------------------------------------------- daily & weekly runs


def periods(daily: pd.DataFrame, candle: str, session: Session) -> pd.DataFrame:
    """Daily candles, or weekly ones built from them, each with the time it finished.

    A week is Monday to Friday, its open the first day's open and its close the
    last day's close -- the same candle TradingView and Kite draw. A week ends at
    Friday's close even when Friday is a holiday: nothing is filled before the
    next session opens either way.
    """
    if daily.empty:
        return pd.DataFrame(columns=["open", "close", "ends"])
    days = daily[["open", "close"]].copy()
    dates = [pd.Timestamp(ts).tz_convert(IST).date() for ts in days.index]
    if candle == "day":
        days["ends"] = [_close_of(d, session) for d in dates]  # ty: ignore[invalid-argument-type]
        return days.reset_index(drop=True)
    week_end = [d + dt.timedelta(days=4 - d.weekday()) for d in dates]
    days["week"] = week_end
    weeks = days.groupby("week", sort=True).agg(open=("open", "first"), close=("close", "last"))
    weeks["ends"] = [_close_of(d, session) for d in weeks.index]
    return weeks.reset_index(drop=True)


def _qualifies(rows: pd.DataFrame, i: int, trigger: StreakTrigger) -> bool:
    n = trigger.count
    if trigger.rule in ("red", "green"):
        if i + 1 < n:
            return False
        window = rows.iloc[i + 1 - n : i + 1]
        if trigger.rule == "red":
            return bool((window["close"] < window["open"]).all())
        return bool((window["close"] > window["open"]).all())
    if i < n:
        return False
    closes = rows["close"].iloc[i - n : i + 1].to_numpy()
    steps = closes[1:] - closes[:-1]
    return bool((steps < 0).all() if trigger.rule == "lower_close" else (steps > 0).all())


def streak_status(daily: pd.DataFrame, trigger: StreakTrigger, now: pd.Timestamp) -> int:
    """How many finished candles in a row, up to now, already meet the rule. For the readback."""
    session = _session(trigger.watch.symbol, trigger.watch.kind)
    rows = periods(daily, trigger.candle, session)
    rows = rows[rows["ends"] <= now].reset_index(drop=True)
    run = 0
    for i in range(len(rows) - 1, -1, -1):
        one = StreakTrigger(**{**trigger.model_dump(), "count": 1})  # ty: ignore[invalid-argument-type]
        if not _qualifies(rows, i, one):
            break
        run += 1
    return run


def streak_fires_at(
    daily: pd.DataFrame, trigger: StreakTrigger, created_at: pd.Timestamp, now: pd.Timestamp
) -> tuple[pd.Timestamp, dict] | None:
    """When the run was first complete, among candles that finished after the order was saved."""
    session = _session(trigger.watch.symbol, trigger.watch.kind)
    rows = periods(daily, trigger.candle, session)
    for i in range(len(rows)):
        ends = rows["ends"].iloc[i]
        if ends <= created_at or ends + DAILY_GRACE > now:
            continue
        if _qualifies(rows, i, trigger):
            return ends, {
                "closes": [
                    round(float(x), 2)
                    for x in rows["close"].iloc[max(0, i - trigger.count) : i + 1]
                ],
                "opens": [
                    round(float(x), 2) for x in rows["open"].iloc[max(0, i - trigger.count) : i + 1]
                ],
            }
    return None


# ------------------------------------------------------------------ the pass


def advance(
    book: Book,
    market: Market,
    *,
    cursor: pd.Timestamp,
    now: pd.Timestamp,
    charges: Callable[[str], ChargeFn],
) -> tuple[list[Event], pd.Timestamp]:
    """Everything that happened after `cursor`, up to `now`. Returns (events, new cursor).

    `charges(symbol)` gives the charge function for that contract. The book is
    updated in place as events happen, so later candles see earlier fills.
    """
    events: list[Event] = []
    waiting = [o for o in book.orders if o.id not in book.triggered and o.id not in book.finished]

    def emit(ev: Event) -> None:
        events.append(ev)

    def trigger(o: SavedOrder, at: pd.Timestamp, detail: dict) -> None:
        book.triggered[o.id] = at
        emit(Event(o.id, "triggered", _iso(at), bar_time=at, detail=detail))
        # An alert never fills and never opens a position: it is done the
        # instant its condition is met. No candle is needed after this one.
        if isinstance(o.order.action, AlertAction):
            book.finished.add(o.id)
            emit(Event(o.id, "finished", "done", bar_time=at, detail={"note": "alert sent"}))

    # Conditions that do not depend on minute candles.
    for o in list(waiting):
        t = o.order.trigger
        if isinstance(t, NowTrigger):
            trigger(o, o.created_at, {"reason": "now"})
            waiting.remove(o)
        elif isinstance(t, StreakTrigger):
            key = f"{'IDX' if t.watch.kind == 'index' else 'FUT'}:{t.watch.symbol}"
            daily = market.daily.get(key)
            if daily is None:
                continue
            found = streak_fires_at(daily, t, o.created_at, now)
            if found:
                trigger(o, found[0], {"reason": "streak", **found[1]})
                waiting.remove(o)

    # Every finished candle after the cursor, in time order.
    frames = {
        k: df[(df.index > cursor) & (df.index + MINUTE + SETTLE <= now)]
        for k, df in market.minute.items()
    }
    stamps = sorted({ts for df in frames.values() for ts in df.index})
    last_close: dict[str, float] = {}
    for k, df in market.minute.items():
        earlier = df[df.index <= cursor]
        if not earlier.empty:
            last_close[k] = float(earlier["close"].iloc[-1])

    def candle(key: str, ts: pd.Timestamp):
        df = frames.get(key)
        if df is None or ts not in df.index:
            return None
        return df.loc[ts]

    def close_position(pos: Position, ts, raw_price: float, reason: str, by: int | None) -> None:
        side = "sell" if pos.direction == "long" else "buy"
        price = _slip(raw_price, side, pos.slippage)
        exit_charges = charges(pos.symbol)(price, pos.quantity, side)
        gross = pos.gross(price)
        detail = {
            "reason": reason,
            "contract": pos.contract,
            "entry_price": pos.entry_price,
            "entry_time": _iso(pos.entry_time),
            "gross_pnl": round(gross, 2),
            "charges": round(pos.entry_charges + exit_charges, 2),
            "net_pnl": round(gross - pos.entry_charges - exit_charges, 2),
            "raw_price": raw_price,
        }
        if by is not None:
            detail["by_order"] = by
        emit(
            Event(
                pos.order_id,
                "exited",
                pos.contract,
                bar_time=ts,
                symbol=pos.contract,
                side=side,
                quantity=pos.quantity,
                price=price,
                detail=detail,
            )
        )
        del book.positions[pos.order_id]
        book.closed.add(pos.order_id)

    def finish(o: SavedOrder, ts, note: str) -> None:
        book.finished.add(o.id)
        emit(Event(o.id, "finished", "done", bar_time=ts, detail={"note": note}))

    for ts in stamps:
        day = session_date(ts)

        # 1. Fills for orders whose condition was met before this candle began.
        for o in book.orders:
            at = book.triggered.get(o.id)
            if at is None or at > ts or o.id in book.finished:
                continue
            action = o.order.action
            if isinstance(action, TradeAction):
                if o.id in book.positions or o.id in book.closed:
                    continue
                c = near_contract(
                    market.contracts.get(action.symbol, []), day, skip_expiry_day=True
                )
                if c is None:
                    continue
                bar = candle(f"FUT:{c.tradingsymbol}", ts)
                if bar is None:
                    continue
                _fill(book, o, action, c, ts, float(bar["open"]), charges, emit)
            elif isinstance(action, ExitAction):
                for pid in action.positions:
                    pos = book.positions.get(pid)
                    if pos is None:
                        continue
                    bar = candle(f"FUT:{pos.contract}", ts)
                    if bar is not None:
                        close_position(pos, ts, float(bar["open"]), "closed_by_order", o.id)
                if not any(pid in book.positions for pid in action.positions):
                    finish(o, ts, "positions closed")

        # 2. Stops and targets, on every open position with a candle now.
        for pos in list(book.positions.values()):
            bar = candle(f"FUT:{pos.contract}", ts)
            if bar is None:
                continue
            hit = _stop_or_target(pos, bar)
            if hit:
                reason, price, both = hit
                close_position(pos, ts, price, reason, None)
                if both:
                    events[-1].detail["both_hit"] = True

        for k, df in frames.items():
            if ts in df.index:
                last_close[k] = float(df.loc[ts, "close"])

        # 3. Conditions judged on this candle, now that it has finished.
        fired_at = ts + MINUTE
        for o in list(waiting):
            t = o.order.trigger
            if isinstance(t, PriceTrigger):
                if ts < o.created_at:
                    continue
                key = watch_key(t.watch, market.contracts, day)
                bar = candle(key, ts) if key else None
                if bar is None:
                    continue
                if t.direction == "below" and float(bar["low"]) < t.level:
                    trigger(
                        o,
                        fired_at,
                        {
                            "reason": "price",
                            "low": float(bar["low"]),
                            "candle": _iso(ts),
                            "watched": key,
                        },
                    )
                    waiting.remove(o)
                elif t.direction == "above" and float(bar["high"]) > t.level:
                    trigger(
                        o,
                        fired_at,
                        {
                            "reason": "price",
                            "high": float(bar["high"]),
                            "candle": _iso(ts),
                            "watched": key,
                        },
                    )
                    waiting.remove(o)
            elif isinstance(t, PnlTrigger):
                if ts < o.created_at:
                    continue
                chosen = [
                    book.positions[p]
                    for p in o.order.action.positions  # ty: ignore[unresolved-attribute]
                    if p in book.positions
                ]
                if not chosen:
                    finish(o, ts, "none of the chosen positions is still open")
                    waiting.remove(o)
                    continue
                if not any(candle(f"FUT:{p.contract}", ts) is not None for p in chosen):
                    continue
                marks = {
                    p.order_id: last_close.get(f"FUT:{p.contract}", p.entry_price) for p in chosen
                }
                pnl = sum(p.gross(marks[p.order_id]) for p in chosen)
                if pnl <= t.threshold:
                    trigger(
                        o, fired_at, {"reason": "pnl", "pnl": round(pnl, 2), "candle": _iso(ts)}
                    )
                    waiting.remove(o)

    new_cursor = stamps[-1] if stamps else cursor

    # 4. Expiry: once the expiry day's session is over, close at its last candle.
    for pos in list(book.positions.values()):
        session = _session(pos.symbol)
        if now < _close_of(pos.expiry, session) + EXPIRY_GRACE:
            continue
        df = market.minute.get(f"FUT:{pos.contract}")
        on_day = (
            df[[session_date(ts, session) == pos.expiry for ts in df.index]]
            if df is not None and not df.empty
            else None
        )
        if on_day is None or on_day.empty:
            emit(
                Event(
                    pos.order_id,
                    "note",
                    f"expiry-unpriced|{pos.contract}",
                    detail={
                        "note": f"{pos.contract} expired on {pos.expiry:%d %b %Y}, but its last "
                        "price that day could not be fetched. The paper position is still shown "
                        "as open; it needs closing by hand."
                    },
                )
            )
            continue
        last_ts = on_day.index[-1]
        price = float(on_day["close"].iloc[-1])
        side = "sell" if pos.direction == "long" else "buy"
        # Settled at the close: no slippage, as in the strategy engine's expiry close.
        exit_charges = charges(pos.symbol)(price, pos.quantity, side)
        gross = pos.gross(price)
        emit(
            Event(
                pos.order_id,
                "exited",
                pos.contract,
                bar_time=last_ts,
                symbol=pos.contract,
                side=side,
                quantity=pos.quantity,
                price=price,
                detail={
                    "reason": "expiry",
                    "contract": pos.contract,
                    "entry_price": pos.entry_price,
                    "entry_time": _iso(pos.entry_time),
                    "gross_pnl": round(gross, 2),
                    "charges": round(pos.entry_charges + exit_charges, 2),
                    "net_pnl": round(gross - pos.entry_charges - exit_charges, 2),
                    "raw_price": price,
                },
            )
        )
        del book.positions[pos.order_id]
        book.closed.add(pos.order_id)

    # A trade order is finished once its position is closed.
    for o in book.orders:
        if (
            o.id in book.closed
            and o.id not in book.finished
            and isinstance(o.order.action, TradeAction)
        ):
            book.finished.add(o.id)
            emit(Event(o.id, "finished", "done", detail={"note": "position closed"}))

    return events, new_cursor


def _fill(
    book, o: SavedOrder, action: TradeAction, c: Contract, ts, open_price: float, charges, emit
) -> None:
    from nlt.data.futures import contract as terms

    side = action.side
    price = _slip(open_price, side, o.slippage)
    quantity = action.lots * terms(action.symbol).multiplier
    entry_charges = charges(action.symbol)(price, quantity, side)
    long = side == "buy"
    stop = target = None
    if action.stop_pct is not None:
        stop = (
            open_price * (1 - action.stop_pct / 100)
            if long
            else open_price * (1 + action.stop_pct / 100)
        )
    if action.target_pct is not None:
        target = (
            open_price * (1 + action.target_pct / 100)
            if long
            else open_price * (1 - action.target_pct / 100)
        )
    pos = Position(
        order_id=o.id,
        symbol=action.symbol,
        contract=c.tradingsymbol,
        expiry=c.expiry,
        direction="long" if long else "short",
        lots=action.lots,
        quantity=quantity,
        entry_time=ts,
        entry_price=price,
        entry_charges=entry_charges,
        stop_level=stop,
        target_level=target,
        token=c.token,
        slippage=o.slippage,
    )
    book.positions[o.id] = pos
    emit(
        Event(
            o.id,
            "filled",
            c.tradingsymbol,
            bar_time=ts,
            symbol=c.tradingsymbol,
            side=side,
            quantity=quantity,
            price=price,
            detail={
                "contract": c.tradingsymbol,
                "symbol": action.symbol,
                "expiry": c.expiry.isoformat(),
                "token": c.token,
                "lots": action.lots,
                "raw_price": open_price,
                "charges": round(entry_charges, 2),
                "stop_level": stop,
                "target_level": target,
                "slippage": o.slippage.to_dict(),
            },
        )
    )


def _stop_or_target(pos: Position, bar) -> tuple[str, float, bool] | None:
    """(reason, raw price, both hit) -- the strategy engine's precedence: stop first."""
    long = pos.direction == "long"
    o, h, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
    target_hit = pos.target_level is not None and (
        h >= pos.target_level if long else low <= pos.target_level
    )
    if pos.stop_level is not None:
        stop_hit = low <= pos.stop_level if long else h >= pos.stop_level
        if stop_hit:
            gapped = o < pos.stop_level if long else o > pos.stop_level
            return "stop", (o if gapped else pos.stop_level), target_hit
    if target_hit:
        gapped = o > pos.target_level if long else o < pos.target_level
        return "target", (o if gapped else pos.target_level), False
    return None


# ------------------------------------------------------------ the record


def book_from_record(orders: list[SavedOrder], events: list[dict]) -> Book:
    """Rebuild the book from recorded events. The record is the only state there is."""
    book = Book(orders=orders)
    fills: dict[int, dict] = {}
    for e in events:
        oid, kind = e["order_id"], e["kind"]
        if kind == "triggered":
            book.triggered[oid] = pd.Timestamp(e["bar_time"]).tz_convert(IST)  # ty: ignore[invalid-assignment]
        elif kind == "filled":
            fills[oid] = e
        elif kind == "exited":
            book.closed.add(oid)
        elif kind == "finished":
            book.finished.add(oid)
    for oid, e in fills.items():
        if oid in book.closed:
            continue
        d = e["detail"]
        book.positions[oid] = Position(
            order_id=oid,
            symbol=d["symbol"],
            contract=d["contract"],
            expiry=dt.date.fromisoformat(d["expiry"]),
            direction="long" if e["side"] == "buy" else "short",
            lots=int(d["lots"]),
            quantity=int(e["quantity"]),
            entry_time=pd.Timestamp(e["bar_time"]).tz_convert(IST),  # ty: ignore[invalid-argument-type]
            entry_price=float(e["price"]),
            entry_charges=float(d["charges"]),
            stop_level=d.get("stop_level"),
            target_level=d.get("target_level"),
            token=int(d.get("token") or 0),
            # A fill recorded before 2026-10-13 has no "slippage" key at all --
            # it was filled at the old hidden 0.02%, so that is what re-pricing
            # it (e.g. for an exit) must use, not the new zero default.
            slippage=Slippage.from_dict(d["slippage"]) if "slippage" in d else LEGACY_SLIPPAGE,
        )
    return book
