"""A one-off order -> plain English. Mechanical, never written by a model.

The readback is the whole safety story for a person who cannot read the
recipe: they read "Buy 1 lot of NIFTY futures (65 units)" and can say "no,
that's wrong". So every line here is built from the recipe's fields and says
exactly what the engine will do -- including the things it will not do.
"""

from __future__ import annotations

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


def _level(x: float) -> str:
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def _rupees(x: float) -> str:
    return f"Rs {abs(x):,.0f}"


def watched_name(w: Watched) -> str:
    from nlt.data.futures import contract

    if w.kind == "index":
        return f"the {'NIFTY 50' if w.symbol == 'NIFTY' else 'NIFTY BANK'} index (spot)"
    return f"the {contract(w.symbol).title} futures price (nearest-month contract)"


def _streak_words(t: StreakTrigger) -> str:
    unit = "week" if t.candle == "week" else "day"
    plural = f"{unit}s" if t.count != 1 else unit
    candles = f"{'weekly' if t.candle == 'week' else 'daily'} candle{'s' if t.count != 1 else ''}"
    if t.rule in ("red", "green"):
        how = "below" if t.rule == "red" else "above"
        each = "each" if t.count != 1 else "the"
        return (
            f"{watched_name(t.watch)} has {t.count} {t.rule} {candles} in a row "
            f"({each} {unit} closed {how} where it opened)"
        )
    how = "lower" if t.rule == "lower_close" else "higher"
    return (
        f"{watched_name(t.watch)} closes {how} than the {unit} before, {t.count} {plural} in a row"
    )


def trigger_line(t) -> str:
    if isinstance(t, PriceTrigger):
        return f"When {watched_name(t.watch)} trades {t.direction} {_level(t.level)}"
    if isinstance(t, StreakTrigger):
        return f"When {_streak_words(t)}"
    if isinstance(t, PnlTrigger):
        if t.threshold < 0:
            what = f"a loss of {_rupees(t.threshold)} or more"
        else:
            what = f"{_rupees(t.threshold)} profit or less"
        return f"When the combined P&L of the chosen positions falls to {what}"
    if isinstance(t, NowTrigger):
        return "Now"
    raise TypeError(t)


def action_line(a, position_names: dict[int, str] | None = None) -> str:
    from nlt.data.futures import contract

    if isinstance(a, TradeAction):
        c = contract(a.symbol)
        verb = "Buy" if a.side == "buy" else "Sell (go short)"
        lots = f"{a.lots} lot{'s' if a.lots != 1 else ''}"
        size = a.lots * c.multiplier
        if c.exchange == "NFO":
            units = f"{size:,} units"
        else:
            units = f"1 lot = {c.lot_text}" if a.lots == 1 else f"each lot = {c.lot_text}"
        return f"{verb} {lots} of {c.title} futures ({units}), at the market price"
    if isinstance(a, ExitAction):
        if not a.positions:
            return "Close the positions you choose below, at the market price"
        names = position_names or {}
        listed = ", ".join(names.get(p, f"position #{p}") for p in a.positions)
        return f"Close {listed}, at the market price"
    if isinstance(a, AlertAction):
        return "Tell you once, on this website. Nothing is bought or sold."
    raise TypeError(a)


def describe(order: ConditionalOrder, position_names: dict[int, str] | None = None) -> list[str]:
    """The readback, one line each. Order matters: what, then what it won't do, then how."""
    t, a = order.trigger, order.action
    lines = [trigger_line(t), action_line(a, position_names)]

    if isinstance(a, TradeAction):
        direction_up = a.side == "buy"
        if a.stop_pct is None:
            lines.append(
                "No stop loss. This position stays open, however far the price moves "
                "against it, until you close it or the contract expires."
            )
        else:
            lines.append(
                f"Stop loss: {_level(a.stop_pct)}% below the price it is bought at"
                if direction_up
                else f"Stop loss: {_level(a.stop_pct)}% above the price it is sold at"
            )
        if a.target_pct is not None:
            lines.append(
                f"Target: {_level(a.target_pct)}% above the price it is bought at"
                if direction_up
                else f"Target: {_level(a.target_pct)}% below the price it is sold at"
            )
        lines.append(
            "The contract is the nearest month's, or the next month's if the nearest "
            "expires that day. It is closed at its expiry-day close, never rolled over."
        )

    if isinstance(t, PnlTrigger):
        lines.append(
            "P&L is counted before charges, as Kite's positions screen shows it, and only "
            "for the chosen positions that are still open."
        )

    is_alert = isinstance(a, AlertAction)
    lines.append(
        "Once only: after it tells you, the alert is finished."
        if is_alert
        else "Once only: when this goes through, the order is finished."
    )
    if isinstance(t, StreakTrigger):
        unit = "week" if t.candle == "week" else "day"
        when = (
            "You are told on this website about an hour after that candle ends."
            if is_alert
            else "It goes through at the next session's opening price."
        )
        lines.append(
            f"Judged when each {unit}'s candle has finished, counting only "
            f"{'weeks' if unit == 'week' else 'days'} that end after you save this. "
            f"{when}"
        )
    elif isinstance(t, (PriceTrigger, PnlTrigger)):
        when = (
            "You are told on this website about a minute after that candle ends, "
            "while the app is checking (weekdays, 8:30 AM to 12:30 AM)."
            if is_alert
            else "It goes through at the next minute's opening price."
        )
        lines.append(
            "Checked on each finished 1-minute candle, from the next full minute after you "
            f"save this. {when}"
        )

    lines.append("Paper only: nothing is sent to Zerodha.")
    return lines


def alert_fired_text(order: ConditionalOrder) -> str:
    """What the notification says once an alert's condition is met. Mechanical,
    built from the order's own recipe -- never from the candle that fired it,
    so it cannot say anything the order itself did not already promise.
    """
    t = order.trigger
    if isinstance(t, PriceTrigger):
        body = f"{watched_name(t.watch)} has traded {t.direction} {_level(t.level)}."
    elif isinstance(t, StreakTrigger):
        body = f"{_streak_words(t)}."
    elif isinstance(t, PnlTrigger):
        if t.threshold < 0:
            what = f"a loss of {_rupees(t.threshold)} or more"
        else:
            what = f"{_rupees(t.threshold)} profit or less"
        body = f"The combined P&L of the chosen positions has reached {what}."
    else:
        body = "The condition you set has been met."
    return f"{body} Nothing was bought or sold."
