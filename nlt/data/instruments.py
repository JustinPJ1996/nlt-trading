"""Contract terms for the instruments we trade.

Lot sizes are a decision of the exchange, not a property of the index, and they
change: NSE's circular of 28 November 2025 cut NIFTY from 75 to 65 and BANKNIFTY
from 35 to 30 with effect from the January 2026 series, to keep contract values
inside SEBI's Rs 10-15 lakh notional band as the indices rose.

**We deliberately use today's lot size for the whole backtest, including years
when a different one applied.** The question a backtest answers is "what would
this strategy do if I traded it now", and the answer should not have positions
from 2021 sized by a contract term nobody can trade today. Using the historical
size would also make two trades on identical signals differently sized for a
reason the user has no control over and would struggle to interpret.

The cost of that choice is that a backtest is not a faithful reconstruction of
what a trader would literally have executed at the time. That is the right
trade, but it is a trade, and `lot_size_note` says so out loud.
"""

from __future__ import annotations

# Source: NSE circular 28-Nov-2025, effective from the January 2026 series.
# Verified against broker references on 2026-09-25. These need reviewing
# whenever the exchange revises contract sizes, which it does roughly annually.
LOT_SIZES: dict[str, int] = {
    "NIFTY": 65,
    "BANKNIFTY": 30,
    "FINNIFTY": 60,
    "MIDCPNIFTY": 120,
}

LOT_SIZES_AS_OF = "January 2026 series (NSE circular 28-Nov-2025)"

# Used when an F&O symbol is not in the table. Refusing outright would block a
# strategy on any index we have not catalogued; a stated fallback is better than
# a silent guess, and `lot_size_note` names it.
FALLBACK_LOT_SIZE = 65


def lot_size(symbol: str, trade_as: str) -> int:
    """Units per lot for this instrument.

    Cash equities and cash indices trade in single units -- a "lot" is an F&O
    contract term, and applying one to a stock caps a position at that many
    shares, which is never what anyone means.
    """
    if trade_as != "option":
        return 1
    return LOT_SIZES.get(symbol.upper().strip(), FALLBACK_LOT_SIZE)


def lot_size_note(symbol: str, trade_as: str) -> str | None:
    """A plain-English caveat about the lot size used, or None if not applicable."""
    if trade_as != "option":
        return None

    key = symbol.upper().strip()
    size = lot_size(symbol, trade_as)

    if key not in LOT_SIZES:
        return (
            f"{key} is not in our lot-size table, so {size} units per lot was "
            f"assumed ({LOT_SIZES_AS_OF}). Check the real contract size before "
            "trading this."
        )
    return (
        f"Sized at today's lot of {size} units for the whole test "
        f"({LOT_SIZES_AS_OF}). The exchange has changed this before, so trades "
        "in earlier years would really have been a different size."
    )
