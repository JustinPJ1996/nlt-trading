"""How a backtest was done, in plain words: what it assumed, and where it is weak.

Two lists for the report (competitor review C2, C3). Every line is generated
mechanically from the spec and the run, and each states something the engine
really does -- the same rule as the readback. A line that merely sounds
responsible but is not true of this engine is worse than no line, so every
number here is read from the engine's own constants, never retyped.
"""

from __future__ import annotations

import datetime as dt

from nlt.costs.charges import (
    McxFuturesCharges,
    NseEquityDeliveryCharges,
    NseFuturesCharges,
    NseOptionsCharges,
    ZeroCharges,
)
from nlt.costs.slippage import Slippage
from nlt.spec.models import StrategySpec

_CHARGE_WORDS = {
    NseFuturesCharges: "NSE index futures charges",
    NseOptionsCharges: "NSE index options charges",
    McxFuturesCharges: "MCX commodity futures charges",
    NseEquityDeliveryCharges: "NSE share delivery charges",
}

SAME_RESULT = (
    "Same strategy, same prices, same result: nothing in the test is random and no AI "
    "is involved in it, so running it again on the same prices gives exactly this answer."
)


def _inr(value: float) -> str:
    from nlt.translate.readback import format_inr

    return format_inr(value)


def assumptions(
    spec: StrategySpec,
    *,
    capital: float,
    charge_model,
    first: dt.date,
    last: dt.date,
    slippage: Slippage,
) -> list[str]:
    """What this test took as given, in the order a trade happens."""
    from nlt.translate.readback import _sizing_desc

    if slippage.is_zero:
        fill_line = (
            "Decisions use only finished candles. A trade is placed at the next candle's open. "
            + slippage.describe()
        )
    else:
        fill_line = (
            "Decisions use only finished candles. A trade is placed at the next candle's open, "
            f"with {slippage.describe()} on the way in and on the way out."
        )
    out = [
        f"Tested from {first:%d %b %Y} to {last:%d %b %Y}, starting with {_inr(capital)}.",
        f"Size of each trade: {_sizing_desc(spec)[0].lower()}{_sizing_desc(spec)[1:]}, "
        "within the safety limits shown with the strategy.",
        fill_line,
    ]
    if spec.exit.stop_pct is not None or spec.exit.target_pct is not None:
        out.append(
            "Stop loss and target are measured from the close of the candle that gave the "
            "signal, not from the price paid."
        )
    out.append(
        "If the price opens past the stop loss, the trade is closed at that opening price, "
        "not at the stop."
    )
    if isinstance(charge_model, ZeroCharges):
        out.append("No charges were deducted: this was a comparison run.")
    else:
        words = _CHARGE_WORDS.get(type(charge_model), "Zerodha's charges")
        out.append(
            f"Charges: {words} as Zerodha bills them -- brokerage, transaction tax, exchange "
            "and SEBI fees, GST and stamp duty -- taken off every trade."
        )
    out.append(SAME_RESULT)
    return out


# Notes the engine and loader already write, which belong in the weak spots.
# Matched by their opening words, so the wording stays in one place.
_WEAK_SPOT_NOTE_STARTS = (
    "Sized at today's lot of",
    "FUTURES ON MARGIN",
)
_WEAK_SPOT_NOTE_CONTAINS = ("the backtest only sees survivors",)


def weak_spots(
    spec: StrategySpec, notes: list[str], *, slippage: Slippage
) -> tuple[list[str], list[str]]:
    """(weak spots, the notes left over), moving matching notes into the weak spots."""
    notes = list(dict.fromkeys(notes))
    moved = [
        n
        for n in notes
        if n.startswith(_WEAK_SPOT_NOTE_STARTS) or any(c in n for c in _WEAK_SPOT_NOTE_CONTAINS)
    ]
    remaining = [n for n in notes if n not in moved]
    # The loader and the basket engine each warn about survivors; say it once,
    # keeping the last (the engine's, dated from the first day actually traded).
    survivors = [n for n in moved if any(c in n for c in _WEAK_SPOT_NOTE_CONTAINS)]
    moved = [n for n in moved if n not in survivors[:-1]]

    if slippage.is_zero:
        slippage_spot = f"Fills are modelled, not real. {slippage.describe()}"
    else:
        slippage_spot = (
            f"Fills are modelled, not real: {slippage.describe()} Large "
            "orders, or thinly traded stocks, would usually fill worse."
        )
    spots = [
        "Candles cannot show what happened inside them. When one candle touched both the "
        "stop loss and the target, the stop was assumed to come first -- this can only make "
        "the result look worse, never better.",
        slippage_spot,
    ]
    trade_as = spec.instrument.trade_as
    if trade_as in ("stock", "index"):
        spots.append(
            "Dividends are not counted, for your strategy or for simply holding, because the "
            "prices used do not include them."
        )
    if trade_as == "stock":
        spots.append(
            "The small depository fee on each sale of shares (about Rs 15-16 at Zerodha) is "
            "not included."
        )
        if spec.instrument.timeframe != "1d" and spec.schedule.intraday:
            spots.append(
                "Shares squared off the same day are charged as delivery trades, which pay "
                "more transaction tax than intraday trades really do."
            )
    if spec.instrument.is_future and spec.instrument.timeframe != "1d":
        spots.append(
            "Intraday futures history exists only for the contract trading now; earlier "
            "months could not be tested on minute or hourly candles."
        )
    spots.extend(moved)
    spots.append("Past prices are not a promise: a strategy that worked before can stop working.")
    return spots, remaining
