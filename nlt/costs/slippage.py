"""What a fill really costs beyond the quoted price: slippage.

Every backtest, paper run and one-off order used to assume a hidden 0.02% on
every fill, baked into the engine's default argument. Justin's call
(2026-10-13): slippage becomes a user-set run setting, defaulting to ZERO
everywhere, so a result is never quietly pessimistic (or optimistic) by a
number nobody chose. Records made before this change keep the old 0.02% --
see `ZERO`, `LEGACY` and the note on `from_dict` below for how that split is
kept explicit rather than hidden in this module.

Two shapes, chosen at the point a run (or a one-off order) is started, never
part of the strategy spec itself -- it is a run setting, like starting capital
or which charges to deduct, not a rule the strategy is judged by:

  * "pct" -- a percentage of the price, same direction every time: a buy pays
    more, a sell receives less.
  * "rs" -- a fixed number of rupees per unit, added on a buy and subtracted
    on a sell.

A sell can never be modelled as settling below Rs 0.05 (one tick): no real
exchange prices anything lower, so an aggressive rupee or percentage slippage
on a cheap instrument must not produce a fill price that could never trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# No exchange prices anything below this; a slippage model that pushed a sell
# price under it would be modelling a fill that could not have happened.
MIN_SELL_PRICE = 0.05

_MAX_PCT = 5.0
_MAX_RS = 100.0


@dataclass(frozen=True)
class Slippage:
    """A run's slippage setting. Frozen: changing it means building a new one,
    never mutating a run's recorded setting in place.
    """

    kind: Literal["pct", "rs"] = "pct"
    value: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("pct", "rs"):
            raise ValueError(
                f"Slippage must be either a percentage or a rupee amount, not {self.kind!r}."
            )
        if self.value < 0:
            raise ValueError("Slippage cannot be negative.")
        if self.kind == "pct" and self.value > _MAX_PCT:
            raise ValueError(
                f"Slippage of {self.value:g}% is too high – the most this app will model is "
                f"{_MAX_PCT:g}%. A real order almost never slips anywhere near that much."
            )
        if self.kind == "rs" and self.value > _MAX_RS:
            raise ValueError(
                f"Slippage of Rs {self.value:g} per unit is too high – the most this app will "
                f"model is Rs {_MAX_RS:g}. A real order almost never slips anywhere near that much."
            )

    @property
    def is_zero(self) -> bool:
        return self.value == 0.0

    def apply(self, price: float, side: str) -> float:
        """The price actually paid (buy) or received (sell), after slippage.

        A buy always costs a little more; a sell always brings in a little
        less -- the same direction `_apply_slippage` in the engine always
        used. A sell can never be modelled as settling below `MIN_SELL_PRICE`.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', not {side!r}.")
        if self.kind == "pct":
            factor = 1.0 + self.value / 100.0 if side == "buy" else 1.0 - self.value / 100.0
            result = price * factor
        else:
            result = price + self.value if side == "buy" else price - self.value
        if side == "sell":
            result = max(result, MIN_SELL_PRICE)
        return result

    def describe(self) -> str:
        """What a non-technical user should be told, in one sentence."""
        if self.is_zero:
            return "No slippage assumed, so real fills would likely be worse."
        if self.kind == "pct":
            return f"{self.value:g}% slippage on every order."
        return f"Rs {self.value:.2f} per unit slippage on every order."

    def to_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value}

    @staticmethod
    def from_dict(d: dict | None) -> Slippage:
        """Rebuilds a `Slippage` from stored data.

        `d is None` returns `ZERO`, deliberately -- NOT `LEGACY`. A record made
        before this feature existed needs the 0.02% legacy behaviour, but that
        is a decision about *that record*, not something this function should
        guess at silently. The call site loading an old paper run or one-off
        order must decide for itself (typically: "no slippage was ever stored
        for this record" means LEGACY; "no slippage field in a fresh API
        request" means ZERO) and pick the right constant explicitly.
        """
        if d is None:
            return ZERO
        return Slippage(kind=d["kind"], value=float(d["value"]))


# The new default everywhere: no slippage assumed unless the user sets one.
ZERO = Slippage()

# What every record made before this feature shipped actually used -- the
# engines' old hidden default. Only ever applied explicitly, at the point a
# pre-existing paper run or one-off order is loaded with nothing stored for it.
LEGACY = Slippage("pct", 0.02)
