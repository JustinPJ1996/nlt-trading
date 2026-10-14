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
    if spec.instrument.legs:
        out.extend(_legs_assumptions(spec))
    elif spec.instrument.is_option:
        out.append(
            f"The signal is judged on {spec.instrument.symbol}'s own candles. The option is "
            "chosen at that moment: the listed strike nearest the index's close on the signal "
            "candle, and the expiry the strategy names."
        )
        out.append(
            "Option prices come from real trades only, minute by minute. If the chosen option "
            "did not trade in the candle a trade would be placed in, that signal is skipped -- "
            "another strike is never used."
        )
        if spec.exit.stop_pct is not None or spec.exit.target_pct is not None:
            out.append(
                "Stop loss and target are percentages of the option's price at the close of "
                "the candle that gave the signal, not of the index and not of the price paid."
            )
        out.append(
            f"A same-day square-off sells at the {spec.schedule.square_off:%H:%M} price itself. "
            "A position still open on its expiry day is closed then too, so no option is "
            "ever held to settlement."
        )
    elif spec.exit.stop_pct is not None or spec.exit.target_pct is not None:
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


def _legs_assumptions(spec: StrategySpec) -> list[str]:
    """What `nlt.engine.legs_backtest` does, for sold and multi-leg option positions."""
    legs = spec.instrument.legs
    sym = spec.instrument.symbol
    sells = any(leg.side == "sell" for leg in legs)
    out = [
        f"The signal is judged on {sym}'s own candles. Every option in the position is chosen "
        "at that moment from the index's close on the signal candle: strikes counted from the "
        "nearest listed strike, or the listed strike nearest the number of points asked for. "
        "If nothing is listed close enough, the signal is skipped -- the strike is never moved.",
        "Option prices come from real trades only, minute by minute. Every option in the "
        "position must trade in the candle the trade is placed in, or none of them is opened; "
        "another strike is never used.",
    ]
    if any(leg.stop_pct or leg.target_pct for leg in legs):
        out.append(
            "Each option's stop loss and target are percentages of that option's own price at "
            "the close of the signal candle. A sold option's stop is above the price it was "
            "sold at. A stop on one option closes only that option; the others stay open."
        )
    if spec.exit.combined_target_pct or spec.exit.combined_stop_pct:
        out.append(
            "The whole position's profit or loss is compared with the premium at the signal. "
            "Candles cannot show when inside them each option was at its high or low, so the "
            "combined stop assumes every option was at its worst at the same moment, and the "
            "combined target counts only closing prices. When either is met, every option is "
            "closed at the next candle's opening price."
        )
    for leg in legs:
        if leg.reentry == "none":
            continue
        how = (
            "the same option is sold again once its price comes back to where it was first sold"
            if leg.reentry == "same_strike_at_cost"
            else "a strike is chosen afresh straight away and sold"
        )
        if leg.side == "buy":
            how = how.replace("sold again", "bought again").replace("and sold", "and bought")
        out.append(
            f"After the {leg.option_type} option's own stop loss, {how}, up to "
            f"{leg.reentry_max} time(s), never at or after "
            f"{spec.schedule.no_entry_after:%H:%M}, and only if the money for it fits. Its "
            "new stop is measured from its new price."
        )
        break
    out.append(
        f"A same-day square-off closes every option at the {spec.schedule.square_off:%H:%M} "
        "price itself. A position still open on its nearest option's expiry day is closed "
        "then, every option together, so no option is ever held to settlement."
    )
    if sells:
        pct = spec.risk.max_position_pct or 20
        out.append(
            "Selling is held on margin: the exchange margin for the position is worked out for "
            f"the day it opened, and as many sets are opened as keep it within {pct:g}% of the "
            "account and within the money left spare. Premium for bought options is paid in "
            "full on top of the margin."
        )
    out.append(
        f"The comparison is with simply holding {sym}. For a strategy that sells options that "
        "is only a reference point: the two make money in different ways."
    )
    return out


# Notes the engine and loader already write, which belong in the weak spots.
# Matched by their opening words, so the wording stays in one place.
_WEAK_SPOT_NOTE_STARTS = (
    "Sized at today's lot of",
    "FUTURES ON MARGIN",
    "MARGIN AT ENTRY",
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
    if spec.instrument.is_option:
        spots.append(
            "The gap between the price buyers offer and sellers ask is not charged on top: "
            "only the slippage set for this test stands in for it, and thinly traded strikes "
            "would usually cost more."
        )
        spots.append(
            "A stop or target can only fill in a candle where the option actually traded; "
            "between trades the position waits, and the next real trade may be well past "
            "the level."
        )
        spots.append(
            "Charges use today's tax rates for every year tested; the tax on selling an "
            "option was lower in the years tested than it is today."
        )
        legs = spec.instrument.legs
        if any(leg.side == "sell" for leg in legs):
            from nlt.spec.models import is_defined_risk

            if not is_defined_risk(legs):
                spots.append(
                    "A sold option with nothing bought to protect it can lose far more than its "
                    "stop loss: a sudden jump can blow straight past the stop, and the loss is "
                    "whatever the price jumped to."
                )
            spots.append(
                "The exchange's extra margin for sold options on their expiry day is counted "
                "only from 20 Nov 2024, when it began."
            )
        if spec.exit.combined_target_pct or spec.exit.combined_stop_pct:
            spots.append(
                "A whole-position exit is decided when a candle closes and placed at the next "
                "candle's open, so in a fast market it can fill well past its level."
            )
        if spec.instrument.symbol == "SENSEX":
            spots.append(
                "SENSEX trades on BSE, but is charged here at NSE's rates; BSE's exchange "
                "fees differ slightly."
            )
    if spec.instrument.is_future and spec.instrument.timeframe != "1d":
        spots.append(
            "Intraday futures history exists only for the contract trading now; earlier "
            "months could not be tested on minute or hourly candles."
        )
    spots.extend(moved)
    spots.append("Past prices are not a promise: a strategy that worked before can stop working.")
    return spots, remaining
