"""How much edge a trade has to have before charges stop eating it.

Charges only matter in the abstract until they are expressed against the
premium actually at risk. `round_trip_cost_pct` turns a buy+sell charge total
into a percentage of the premium paid, and `breakeven_move_pct` turns that into
"the premium must move this far just to get back to zero" -- the number that
should actually inform a strategy's stop/target sizing at small account size,
where a wide bid-ask plus statutory charges can be a double-digit percentage of
the premium on a single lot.
"""

from __future__ import annotations

from nlt.costs.charges import ChargeModel


def round_trip_cost_pct(model: ChargeModel, price: float, quantity: int) -> float:
    """Buy-then-sell charges as a percentage of the premium paid on entry.

    Returns 0.0 if there is no premium paid (price or quantity is zero) rather
    than raising, since a flat/zero-size trade has no meaningful cost ratio.
    """
    premium_paid = price * quantity
    if premium_paid <= 0:
        return 0.0

    buy = model.charges(price, quantity, "buy")
    sell = model.charges(price, quantity, "sell")
    total_cost = buy.total + sell.total
    return (total_cost / premium_paid) * 100.0


def breakeven_move_pct(model: ChargeModel, price: float, quantity: int) -> float:
    """How far the premium must rise (as a percent of entry premium) for a long
    position to break even after round-trip charges.

    This assumes the exit is a market sale at the moved price, not an expiry
    exercise -- exercise settlement has its own, generally worse, charge
    profile (see `NseOptionsCharges.charges(..., settled_at_expiry=True)`) and
    is not what this helper is measuring.
    """
    premium_paid = price * quantity
    if premium_paid <= 0:
        return 0.0

    buy = model.charges(price, quantity, "buy")
    # First-order approximation: charge the exit at the same price as entry.
    # Sell-side charges (STT in particular) are a fixed rate on premium, so
    # this is exact for rate-based components and negligibly off for the
    # flat brokerage component once the premium actually moves.
    sell = model.charges(price, quantity, "sell")
    total_cost = buy.total + sell.total
    return (total_cost / premium_paid) * 100.0
