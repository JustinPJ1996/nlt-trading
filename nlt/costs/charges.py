"""What it actually costs to put on and take off an NSE F&O trade.

At the account size this platform is built for -- under Rs 1 lakh, one lot of
NIFTY weekly options at a time -- brokerage is almost a rounding error next to
the statutory charges. STT alone on a single sell leg of an options trade can be
several times the flat brokerage. A backtest that prices trades at raw premium
only, or that doubles one side's charges to approximate a round trip, is not
approximately right -- it is wrong in the direction that makes every strategy
look better than it is.

The charges are asymmetric by design and the asymmetry is not an implementation
detail, it is the point:

- STT (options) falls on the *sell* leg only, and only on premium -- unless the
  option is bought and left to be exercised (settled ITM at expiry), in which
  case STT instead falls on the *buy/exercise* side, computed on intrinsic
  value rather than premium. Selling to close before expiry is almost always
  cheaper than letting an ITM option expire and get exercised, and that
  difference is real money, not a rounding choice.
- STT (futures) falls on the sell leg only, on notional (price * quantity),
  not premium -- there is no premium for a future.
- Stamp duty falls on the *buy* leg only, on notional/premium depending on
  instrument.
- Exchange transaction charges and SEBI turnover fees fall on *both* legs.
- Brokerage is flat per executed order at Zerodha for F&O, charged on both
  legs, and is deliberately not proportional to trade size -- it is the one
  component that does NOT scale with premium, which is exactly why it stops
  mattering at large size and dominates at the small size this platform
  targets.

Because of this asymmetry, "round trip cost" is never simply 2x a one-sided
number. Buy and sell legs must be priced separately and summed.

All rates below are Zerodha's own published rates, current as of the point in
this module docstring being written, with source URL and effective date given
next to each constant. Rates have changed twice in recent history -- once on
1 October 2024 (STT and exchange transaction charges revised) and again from
1 April 2026 (Budget 2026-27 revised STT further upward on both options and
futures) -- so any future change needs to be re-verified against
https://zerodha.com/charges rather than assumed stable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# --------------------------------------------------------------------------
# Shared statutory constants
# --------------------------------------------------------------------------

# GST on brokerage + exchange transaction charges + SEBI charges (not on STT,
# not on stamp duty -- those are themselves taxes/duties, not services).
# Source: https://zerodha.com/charges (all segments), fetched 2026-09-23.
GST_RATE = 0.18

# SEBI turnover fee: Rs 10 per crore = 0.0001%, applies to both buy and sell
# legs, across all segments (equity delivery, futures, options).
# Source: https://zerodha.com/charges , fetched 2026-09-23.
SEBI_TURNOVER_FEE_RATE = 10 / 1_00_00_000  # 0.0001%

# --------------------------------------------------------------------------
# NSE index OPTIONS (e.g. NIFTY weekly options) -- the primary case for this
# platform.
# Source: https://zerodha.com/charges , "F&O - Options" section,
# fetched 2026-09-23. Reflects the Budget 2026-27 STT revision effective
# 1 April 2026.
# --------------------------------------------------------------------------

# Flat brokerage per executed order (per leg), buy and sell each charged once.
OPTIONS_BROKERAGE_PER_ORDER = 20.0

# STT on options sold in the market (or bought and sold, never exercised):
# 0.15% of premium, sell side only.
OPTIONS_STT_SELL_RATE = 0.0015

# STT on options bought and exercised / physically settled ITM at expiry:
# 0.15% of *intrinsic value*, charged on the buy/exercise side instead of the
# sell-side premium STT above (the two are mutually exclusive for a given
# leg). Source: Zerodha support -- "If I let my option positions get
# exercised, are there extra charges?", fetched 2026-09-23.
OPTIONS_STT_EXERCISE_RATE = 0.0015

# NSE exchange transaction charge on options: 0.03553% of premium, both legs.
OPTIONS_TXN_CHARGE_RATE = 0.0003553

# Stamp duty on options: 0.003% of premium, buy side only.
OPTIONS_STAMP_DUTY_RATE = 0.00003

# --------------------------------------------------------------------------
# NSE index FUTURES.
# Source: https://zerodha.com/charges , "F&O - Futures" section,
# fetched 2026-09-23. Reflects the Budget 2026-27 STT revision effective
# 1 April 2026.
# --------------------------------------------------------------------------

# Brokerage: 0.03% of notional or Rs 20 per executed order, whichever is
# lower, both legs.
FUTURES_BROKERAGE_RATE = 0.0003
FUTURES_BROKERAGE_CAP = 20.0

# STT on futures: 0.05% of notional (price * quantity), sell side only.
FUTURES_STT_SELL_RATE = 0.0005

# NSE exchange transaction charge on futures: 0.00183% of notional, both legs.
FUTURES_TXN_CHARGE_RATE = 0.0000183

# Stamp duty on futures: 0.002% of notional, buy side only.
FUTURES_STAMP_DUTY_RATE = 0.00002

# --------------------------------------------------------------------------
# NSE cash equity DELIVERY -- needed for the eventual stock phase, not options
# trading, but priced the same way.
# Source: https://zerodha.com/charges , "Equity Delivery" section,
# fetched 2026-09-23.
# --------------------------------------------------------------------------

EQUITY_DELIVERY_BROKERAGE = 0.0  # Zero brokerage on delivery at Zerodha.

# STT on equity delivery: 0.1% of turnover, both buy and sell legs.
EQUITY_STT_RATE = 0.001

# NSE exchange transaction charge on equity delivery: 0.00307%, both legs.
EQUITY_TXN_CHARGE_RATE = 0.0000307

# Stamp duty on equity delivery: 0.015% of turnover, buy side only.
EQUITY_STAMP_DUTY_RATE = 0.00015

_VALID_SIDES = ("buy", "sell")


def _check_side(side: str) -> None:
    if side not in _VALID_SIDES:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")


@dataclass(frozen=True)
class ChargeBreakdown:
    """Every component of the cost of one order, already rounded to paise."""

    brokerage: float
    stt: float
    transaction_charges: float
    sebi_fees: float
    stamp_duty: float
    gst: float
    total: float

    def __str__(self) -> str:
        return (
            f"brokerage={self.brokerage:.2f} stt={self.stt:.2f} "
            f"txn={self.transaction_charges:.2f} sebi={self.sebi_fees:.2f} "
            f"stamp={self.stamp_duty:.2f} gst={self.gst:.2f} "
            f"total={self.total:.2f}"
        )


class ChargeModel(Protocol):
    def charges(
        self,
        price: float,
        quantity: int,
        side: str,
        *,
        settled_at_expiry: bool = False,
    ) -> ChargeBreakdown: ...


def _round2(value: float) -> float:
    return round(value, 2)


def _build_breakdown(
    brokerage: float,
    stt: float,
    transaction_charges: float,
    sebi_fees: float,
    stamp_duty: float,
) -> ChargeBreakdown:
    gst = GST_RATE * (brokerage + transaction_charges + sebi_fees)

    r_brokerage = _round2(brokerage)
    r_stt = _round2(stt)
    r_transaction_charges = _round2(transaction_charges)
    r_sebi_fees = _round2(sebi_fees)
    r_stamp_duty = _round2(stamp_duty)
    r_gst = _round2(gst)
    # Total is the sum of the *rounded* components, not a separately rounded
    # sum of raw values -- otherwise `total == sum(components)` would fail by
    # a paisa on some inputs, which is exactly the kind of discrepancy that
    # makes a contract note not match a backtest.
    total = r_brokerage + r_stt + r_transaction_charges + r_sebi_fees + r_stamp_duty + r_gst

    return ChargeBreakdown(
        brokerage=r_brokerage,
        stt=r_stt,
        transaction_charges=r_transaction_charges,
        sebi_fees=r_sebi_fees,
        stamp_duty=r_stamp_duty,
        gst=r_gst,
        total=_round2(total),
    )


@dataclass(frozen=True)
class NseOptionsCharges:
    """NSE index options (e.g. NIFTY weekly), one executed order at a time.

    `price` is the option premium per unit, `quantity` is the number of units
    (e.g. 75 for one NIFTY lot at the current lot size -- callers pass the
    actual lot size, this class does not hardcode it).
    """

    def charges(
        self,
        price: float,
        quantity: int,
        side: str,
        *,
        settled_at_expiry: bool = False,
    ) -> ChargeBreakdown:
        _check_side(side)
        turnover = price * quantity

        brokerage = OPTIONS_BROKERAGE_PER_ORDER if turnover > 0 else 0.0

        stt = 0.0
        if settled_at_expiry:
            # Exercise settlement: STT on intrinsic value, charged on the
            # buy/exercise side. `price` is expected to already be the
            # intrinsic value per unit in this case (the engine's job, not
            # this model's, to compute intrinsic value from spot and strike).
            if side == "buy":
                stt = OPTIONS_STT_EXERCISE_RATE * turnover
        else:
            if side == "sell":
                stt = OPTIONS_STT_SELL_RATE * turnover

        transaction_charges = OPTIONS_TXN_CHARGE_RATE * turnover
        sebi_fees = SEBI_TURNOVER_FEE_RATE * turnover
        stamp_duty = OPTIONS_STAMP_DUTY_RATE * turnover if side == "buy" else 0.0

        return _build_breakdown(brokerage, stt, transaction_charges, sebi_fees, stamp_duty)


@dataclass(frozen=True)
class NseFuturesCharges:
    """NSE index futures (e.g. NIFTY monthly futures), one executed order."""

    def charges(
        self,
        price: float,
        quantity: int,
        side: str,
        *,
        settled_at_expiry: bool = False,
    ) -> ChargeBreakdown:
        _check_side(side)
        turnover = price * quantity

        brokerage = min(FUTURES_BROKERAGE_RATE * turnover, FUTURES_BROKERAGE_CAP) if turnover > 0 else 0.0

        stt = FUTURES_STT_SELL_RATE * turnover if side == "sell" else 0.0
        transaction_charges = FUTURES_TXN_CHARGE_RATE * turnover
        sebi_fees = SEBI_TURNOVER_FEE_RATE * turnover
        stamp_duty = FUTURES_STAMP_DUTY_RATE * turnover if side == "buy" else 0.0

        return _build_breakdown(brokerage, stt, transaction_charges, sebi_fees, stamp_duty)


@dataclass(frozen=True)
class NseEquityDeliveryCharges:
    """NSE cash equity, delivery (not intraday). Zero brokerage at Zerodha."""

    def charges(
        self,
        price: float,
        quantity: int,
        side: str,
        *,
        settled_at_expiry: bool = False,
    ) -> ChargeBreakdown:
        _check_side(side)
        turnover = price * quantity

        brokerage = EQUITY_DELIVERY_BROKERAGE
        stt = EQUITY_STT_RATE * turnover
        transaction_charges = EQUITY_TXN_CHARGE_RATE * turnover
        sebi_fees = SEBI_TURNOVER_FEE_RATE * turnover
        stamp_duty = EQUITY_STAMP_DUTY_RATE * turnover if side == "buy" else 0.0

        return _build_breakdown(brokerage, stt, transaction_charges, sebi_fees, stamp_duty)


@dataclass(frozen=True)
class ZeroCharges:
    """No charges at all -- for comparison runs against the costed models."""

    def charges(
        self,
        price: float,
        quantity: int,
        side: str,
        *,
        settled_at_expiry: bool = False,
    ) -> ChargeBreakdown:
        _check_side(side)
        return ChargeBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def charge_fn(model: ChargeModel) -> Callable[[float, int, str], float]:
    """Adapt a `ChargeModel` to the `(price, quantity, side) -> total` signature
    the backtest engine calls on every fill."""

    def _fn(price: float, quantity: int, side: str) -> float:
        return model.charges(price, quantity, side).total

    return _fn
