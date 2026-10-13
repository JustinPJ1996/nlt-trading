"""Choosing one contract: ATM strike, offset direction, and a sentence saying why.

`resolve()` is the single entry point Phase 2's engine will call at signal
time. It never looks at anything past the signal candle's close -- `listing`
is the caller's job to fetch for exactly `day = signal_time.date()`, which by
construction (see `nlt.options.data.listing`) cannot contain a later day's
contracts.

## ATM and offset, Justin's rules

- **ATM** = the listed strike nearest the signal candle's close. A tie (the
  close is exactly between two strikes) goes to the **lower** strike, and
  `why` says so, because "nearest" is ambiguous at the midpoint and a silent
  tiebreak is exactly the kind of thing the readback exists to surface.
- **`strike_offset` counts listed strikes from ATM, not points.** BANKNIFTY's
  uneven spacing (100 near the money, 500 further out) makes a points-based
  offset mean a different number of strikes depending on where the market is;
  counting listed strikes keeps "+1" meaning the same thing everywhere.
  `+1` is one strike **out of the money**: for a call that is the next
  **higher** listed strike (calls gain value as spot rises, so OTM is above);
  for a put it is the next **lower** listed strike. `-1` is one strike
  **in the money**, the opposite direction. Running off either end of the
  listed strikes is refused, never clamped -- a clamped "+5" that quietly
  becomes "+2" is a wrong contract reported as a right one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from nlt.data.instruments import lot_size
from nlt.options.expiry import pick_expiry

REASON_NO_LISTING = "no_listing"
REASON_OFFSET_OUT_OF_RANGE = "offset_out_of_range"

_EXPIRY_KIND_WORDS = {
    "nearest_weekly": "nearest weekly",
    "next_weekly": "next weekly",
    "monthly": "monthly",
}

_OPTION_TYPE_WORDS = {"CE": "call", "PE": "put"}


@dataclass(frozen=True)
class Contract:
    underlying: str
    tradingsymbol: str
    option_type: str  # "CE" or "PE"
    strike: float
    expiry: dt.date
    lot_size: int


@dataclass(frozen=True)
class Resolution:
    contract: Contract | None
    reason_code: str | None
    why: str


def _group_indian(digits: str) -> str:
    """Indian digit grouping on a bare digit string -- last three, then pairs.

    Factored out of `nlt.translate.readback.format_inr` so a plain index level
    (no "Rs" prefix, and with decimals) can use the same grouping.
    """
    if len(digits) <= 3:
        return digits
    last3, rest = digits[-3:], digits[:-3]
    parts: list[str] = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return ",".join(parts) + "," + last3


def format_level(value: float) -> str:
    """48,212.35 -- Indian grouping, two decimals, no currency symbol (an
    index level is a number, not a rupee amount of anything)."""
    sign = "-" if value < 0 else ""
    cents = round(abs(value) * 100)
    rupees, paise = divmod(cents, 100)
    return f"{sign}{_group_indian(str(rupees))}.{paise:02d}"


def format_strike(value: float) -> str:
    """48,200 -- strikes are whole numbers in this market; no decimals shown."""
    return _group_indian(str(int(round(value))))


def _nearest_strike(strikes: list[float], spot_close: float) -> float:
    """Nearest listed strike to `spot_close`; a tie goes to the lower one."""
    best = min(strikes, key=lambda s: (abs(s - spot_close), s))
    return best


def resolve(
    listing: pd.DataFrame,
    underlying: str,
    signal_time: dt.datetime,
    spot_close: float,
    option_type: str,
    offset: int,
    expiry_kind: str,
) -> Resolution:
    """Pick a contract at signal time, or explain in plain English why none exists.

    `listing` is `nlt.options.data.listing(underlying, signal_time.date())` (or
    `FrameSource`'s equivalent) -- the options listed on the signal day, and
    nothing later.
    """
    if option_type not in ("CE", "PE"):
        raise ValueError(f"option_type must be 'CE' or 'PE', not {option_type!r}")

    day = signal_time.date()
    candle_time = signal_time.strftime("%H:%M")
    level = format_level(spot_close)

    all_expiries = list(listing["expiry"]) if not listing.empty else []
    expiry, reason = pick_expiry(all_expiries, day, expiry_kind, underlying)

    if expiry is None:
        why = {
            "no_listing": (
                f"{underlying} had no listed options on {day:%d %b %Y}, so no "
                "contract could be chosen."
            ),
            "no_weekly": (
                f"{underlying} had no weekly options listed around {day:%d %b %Y} "
                "(it only lists monthly expiries now, or -- for 'next weekly' -- "
                "no second upcoming weekly expiry was listed), so no contract "
                "could be chosen."
            ),
        }.get(reason, f"{underlying}: no contract could be chosen ({reason}).")
        return Resolution(None, reason, why)

    chain = listing[(listing["expiry"] == expiry) & (listing["instrument_type"] == option_type)]
    strikes = sorted(chain["strike"].unique().tolist())
    if not strikes:
        why = (
            f"{underlying} had no {_OPTION_TYPE_WORDS[option_type]} options listed "
            f"for the {expiry:%d %b %Y} expiry on {day:%d %b %Y}, so no contract "
            "could be chosen."
        )
        return Resolution(None, REASON_NO_LISTING, why)

    atm = _nearest_strike(strikes, spot_close)
    atm_index = strikes.index(atm)
    tie = sum(1 for s in strikes if abs(s - spot_close) == abs(atm - spot_close)) > 1

    # +1 is one strike OTM: higher for a call, lower for a put. -1 is ITM, the
    # opposite direction.
    step = offset if option_type == "CE" else -offset
    target_index = atm_index + step

    if target_index < 0 or target_index >= len(strikes):
        why = (
            f"{underlying} closed at {level} on the {candle_time} candle; the "
            f"nearest listed strike was {format_strike(atm)}, but a strike "
            f"{offset:+d} from it is not listed for the {expiry:%d %b %Y} expiry, "
            "so no contract could be chosen."
        )
        return Resolution(None, REASON_OFFSET_OUT_OF_RANGE, why)

    strike = strikes[target_index]
    row = chain[chain["strike"] == strike].iloc[0]
    tradingsymbol = row["tradingsymbol"]

    contract = Contract(
        underlying=underlying,
        tradingsymbol=tradingsymbol,
        option_type=option_type,
        strike=strike,
        expiry=expiry,
        lot_size=lot_size(underlying, "option"),
    )

    kind_word = _EXPIRY_KIND_WORDS[expiry_kind]
    tie_note = " (a tie; the lower strike was chosen)" if tie and target_index == atm_index else ""
    if offset == 0:
        strike_note = f"the nearest listed strike was {format_strike(strike)}{tie_note}"
    else:
        direction = "OTM" if offset > 0 else "ITM"
        strike_note = (
            f"the nearest listed strike was {format_strike(atm)}{tie_note}, "
            f"and {offset:+d} strike{'s' if abs(offset) != 1 else ''} from there "
            f"({direction}) is {format_strike(strike)}"
        )
    why = (
        f"{underlying} closed at {level} on the {candle_time} candle; "
        f"{strike_note}, so the {expiry:%d %b %Y} {format_strike(strike)} "
        f"{option_type} ({kind_word} expiry) was chosen."
    )
    return Resolution(contract, None, why)
