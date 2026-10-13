"""The recipe for a one-off conditional order.

Like the strategy recipe, this is data, not code: every field is typed and
checked before anything is watched or filled. A trigger says what to wait for;
an action says what to do, once, when it happens.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

INDICES = ("NIFTY", "BANKNIFTY")


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Watched(Base):
    """What a trigger looks at: an index's own level, or a futures contract's price."""

    symbol: str
    kind: Literal["index", "future"]

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper().strip()

    @model_validator(mode="after")
    def _known(self) -> Watched:
        from nlt.data.futures import is_future

        if self.kind == "index" and self.symbol not in INDICES:
            raise ValueError(f"{self.symbol} is not an index this app can watch")
        if self.kind == "future" and not is_future(self.symbol):
            raise ValueError(f"{self.symbol} is not a futures contract this app supports")
        return self


class PriceTrigger(Base):
    """Fires on the first finished 1-minute candle that trades beyond `level`."""

    kind: Literal["price"] = "price"
    watch: Watched
    direction: Literal["below", "above"]
    level: float = Field(gt=0)


class StreakTrigger(Base):
    """Fires when the latest `count` finished daily or weekly candles all qualify.

    `red` / `green`: each candle closed below / above its own open.
    `lower_close` / `higher_close`: each closed below / above the one before.
    """

    kind: Literal["streak"] = "streak"
    watch: Watched
    candle: Literal["day", "week"]
    count: int = Field(ge=1, le=20)
    rule: Literal["red", "green", "lower_close", "higher_close"]


class PnlTrigger(Base):
    """Fires when the chosen paper positions' combined P&L is at or below `threshold`.

    Rupees, before charges -- the same figure Kite's positions screen shows. A
    loss of Rs 50,000 is `threshold=-50000`.
    """

    kind: Literal["pnl"] = "pnl"
    threshold: float

    @field_validator("threshold")
    @classmethod
    def _nonzero(cls, v: float) -> float:
        if v == 0:
            raise ValueError("a P&L limit of zero would fire on the first tick against you")
        return v


class NowTrigger(Base):
    """Fires immediately. Used by the "close this position now" button."""

    kind: Literal["now"] = "now"


Trigger = Annotated[
    PriceTrigger | StreakTrigger | PnlTrigger | NowTrigger, Field(discriminator="kind")
]


class TradeAction(Base):
    """Buy or sell futures, at the market, once."""

    kind: Literal["trade"] = "trade"
    side: Literal["buy", "sell"]
    symbol: str
    lots: int = Field(ge=1, le=100)
    stop_pct: float | None = Field(default=None, gt=0, le=50)
    target_pct: float | None = Field(default=None, gt=0, le=500)

    @field_validator("symbol")
    @classmethod
    def _future(cls, v: str) -> str:
        from nlt.data.futures import is_future

        v = v.upper().strip()
        if not is_future(v):
            raise ValueError(f"{v} is not a futures contract this app supports")
        return v


class ExitAction(Base):
    """Close open paper positions, at the market.

    `positions` holds the ids of the orders whose positions to close. Empty
    straight after reading a sentence: the person picks them on screen before
    the order can be saved (see `ConditionalOrder.ready`).
    """

    kind: Literal["exit"] = "exit"
    positions: tuple[int, ...] = ()


class AlertAction(Base):
    """Tell the person once, on this website. Never buys, sells or closes anything.

    `positions` is only used alongside a P&L trigger, to say whose combined
    P&L to watch -- the same choice an exit's P&L trigger needs. Empty for
    every other trigger.
    """

    kind: Literal["alert"] = "alert"
    positions: tuple[int, ...] = ()


Action = Annotated[TradeAction | ExitAction | AlertAction, Field(discriminator="kind")]


class ConditionalOrder(Base):
    trigger: Trigger
    action: Action

    @model_validator(mode="after")
    def _sensible(self) -> ConditionalOrder:
        if self.trigger.kind == "now" and self.action.kind != "exit":
            raise ValueError("'now' can only close positions, not open new ones or alert")
        if self.trigger.kind == "pnl" and self.action.kind not in ("exit", "alert"):
            raise ValueError("a P&L limit can only close positions or send an alert")
        return self

    def problem(self) -> str | None:
        """Why this order cannot be saved yet, in words, or None if it can."""
        if self.action.kind == "exit" and not self.action.positions:
            return "Choose which positions to close."
        if self.action.kind == "alert" and self.trigger.kind == "pnl" and not self.action.positions:
            return "Choose which positions to watch."
        return None
