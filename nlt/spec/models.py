"""The strategy recipe.

This is the only thing Claude produces and the only thing the engine consumes.
It is deliberately a *data* format with no executable content: every field is
typed, bounded and checked before anything runs.

Validation here is the product's main safety mechanism. A spec that reaches the
engine has already been proven to reference only declared indicators, to have an
exit path, and to carry risk limits.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nlt.indicators.registry import REGISTRY
from nlt.indicators.registry import get as get_indicator

PRICE_FIELDS = ("open", "high", "low", "close", "volume")


class Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------- value refs


class Ref(Base):
    """A reference to a computed value on some bar.

    `bars_ago=0` is the bar being decided on -- always a *closed* bar, never the
    one currently forming. `bars_ago=1` is the bar before it.
    """

    kind: Literal["ref"] = "ref"
    name: str = Field(description="A declared indicator id, or a price field like 'close'.")
    output: str | None = Field(
        default=None,
        description="For multi-output indicators, which output (e.g. 'signal' on macd).",
    )
    bars_ago: int = Field(default=0, ge=0, le=200)

    def key(self) -> str:
        return f"{self.name}.{self.output}" if self.output else self.name


class Const(Base):
    """A fixed number."""

    kind: Literal["const"] = "const"
    value: float


Operand = Annotated[Ref | Const, Field(discriminator="kind")]


# -------------------------------------------------------------- conditions


class Compare(Base):
    """A comparison between two values on the same bar."""

    kind: Literal["compare"] = "compare"
    op: Literal["lt", "lte", "gt", "gte", "eq", "crosses_above", "crosses_below"]
    left: Operand
    right: Operand

    @model_validator(mode="after")
    def _crossing_needs_history(self):
        # A cross compares this bar against the previous one, so referring to a
        # value 200 bars back would silently compare two ancient bars.
        if self.op.startswith("crosses") and isinstance(self.left, Ref):
            if self.left.bars_ago > 0:
                raise ValueError("crossing comparisons must use bars_ago=0 on the left side")
        return self


class IsTrue(Base):
    """Tests a boolean indicator, such as a candlestick pattern."""

    kind: Literal["is_true"] = "is_true"
    ref: Ref


class PercentChange(Base):
    """Percent change of a value over `lookback` bars.

    Expresses "NIFTY fell 1% in the last 3 bars" without needing an indicator.
    """

    kind: Literal["percent_change"] = "percent_change"
    ref: Ref
    lookback: int = Field(default=1, ge=1, le=200)
    op: Literal["lt", "lte", "gt", "gte"]
    value: float


class AtTime(Base):
    """True on the closed candle that ENDS at `time`, so the fill is at `time`'s price.

    "Sell the straddle at 09:20" is a signal on the candle that closes at 09:20
    and a fill at the next candle's open -- the 09:20 price itself -- which keeps
    the closed-candle / next-open rule intact. A candle size whose candles
    never end at `time` (09:20 on 15-minute candles) cannot express it; the
    engine raises rather than drifting to a nearby candle.
    """

    kind: Literal["at_time"] = "at_time"
    time: dt.time

    @field_validator("time")
    @classmethod
    def _inside_the_session(cls, v: dt.time) -> dt.time:
        if not dt.time(9, 16) <= v <= dt.time(15, 29):
            raise ValueError(f"an entry time of {v:%H:%M} is outside the 09:15-15:30 session")
        if v.second or v.microsecond:
            raise ValueError("an entry time is a whole minute")
        return v


class OnExpiryDay(Base):
    """True on candles of a day on which the strategy's option contract expires.

    Which expiry is the legs' own (`Instrument.expiry` / each leg's `expiry`);
    the engine supplies the calendar from the dataset's listed expiries.
    """

    kind: Literal["expiry_day"] = "expiry_day"


class IvPercentile(Base):
    """Compares the day's implied-volatility percentile against `value`.

    The IV is worked out from the at-the-money option's own price (never India
    VIX) and ranked against the previous `lookback_days` sessions. Only values
    known before the candle closes may be used -- the engine supplies the
    series and is responsible for that.
    """

    kind: Literal["iv_percentile"] = "iv_percentile"
    op: Literal["lt", "lte", "gt", "gte"]
    value: float = Field(ge=0, le=100)
    lookback_days: int = Field(default=252, ge=20, le=500)


class All(Base):
    kind: Literal["all"] = "all"
    conditions: list[Condition] = Field(min_length=1, max_length=10)


class Any_(Base):
    kind: Literal["any"] = "any"
    conditions: list[Condition] = Field(min_length=1, max_length=10)


class Not(Base):
    kind: Literal["not"] = "not"
    condition: Condition


Condition = Annotated[
    Compare | IsTrue | PercentChange | AtTime | OnExpiryDay | IvPercentile | All | Any_ | Not,
    Field(discriminator="kind"),
]

All.model_rebuild()
Any_.model_rebuild()
Not.model_rebuild()


# -------------------------------------------------------------- indicators


class IndicatorSpec(Base):
    """One indicator the strategy needs, with the id its conditions refer to."""

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,31}$")
    type: str
    params: dict[str, float | int | str] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        if v not in REGISTRY:
            raise ValueError(f"unknown indicator {v!r}; available: {', '.join(sorted(REGISTRY))}")
        return v

    @model_validator(mode="after")
    def _params_are_valid(self):
        d = get_indicator(self.type)
        unknown = set(self.params) - set(d.params)
        if unknown:
            raise ValueError(
                f"{self.type} has no parameter(s) {sorted(unknown)}; "
                f"accepts {sorted(d.params) or 'none'}"
            )
        for key, value in self.params.items():
            default = d.params[key]
            if isinstance(default, str):
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{self.type}.{key} must be a number, got {value!r}")
            if "length" in key or key in {"fast", "slow", "signal", "period"}:
                if value < 1 or value > 500:
                    raise ValueError(f"{self.type}.{key}={value} is outside 1..500")
        return self

    def resolved_params(self) -> dict:
        return {**get_indicator(self.type).params, **self.params}


# ------------------------------------------------------------------- exits


class ExitRules(Base):
    """How a position is closed. At least one exit must exist.

    A strategy with no way out is the most dangerous thing a beginner can
    describe, so the validator rejects it outright rather than defaulting.
    """

    target_pct: float | None = Field(default=None, gt=0, le=100)
    stop_pct: float | None = Field(default=None, gt=0, le=100)
    trailing_stop_pct: float | None = Field(default=None, gt=0, le=100)
    stop_atr_mult: float | None = Field(default=None, gt=0, le=20)
    atr_id: str | None = Field(
        default=None, description="Indicator id supplying ATR when stop_atr_mult is used."
    )
    condition: Condition | None = None
    max_bars_held: int | None = Field(default=None, ge=1, le=5000)

    # Multi-leg option positions only: exits on the whole position's P&L,
    # as a percentage of the net premium at the signal -- the credit received
    # for a selling position, the debit paid for a buying one. "Exit at 50% of
    # the credit received" is `combined_target_pct=50`; "stop at 30% of the
    # premium collected" is `combined_stop_pct=30` (a loss of 30% of it).
    combined_target_pct: float | None = Field(default=None, gt=0, le=100)
    combined_stop_pct: float | None = Field(default=None, gt=0, le=1000)

    # Set by a caller that knows every exit lives elsewhere -- a multi-leg
    # position whose only exits are its legs' own stops, or a hedged position
    # closed by the square-off. The spec-level check below then cannot see
    # them, so `StrategySpec._legs_are_consistent` re-checks it with the legs.
    exits_on_legs: bool = False

    @model_validator(mode="after")
    def _has_an_exit(self):
        if not any(
            [
                self.target_pct,
                self.stop_pct,
                self.trailing_stop_pct,
                self.stop_atr_mult,
                self.condition,
                self.max_bars_held,
                self.combined_target_pct,
                self.combined_stop_pct,
                self.exits_on_legs,
            ]
        ):
            raise ValueError(
                "the strategy has no exit rule -- it would hold forever. "
                "Add a target, a stop loss, an exit condition or a time limit."
            )
        if self.stop_atr_mult and not self.atr_id:
            raise ValueError("stop_atr_mult requires atr_id naming an ATR indicator")
        return self

    @property
    def has_stop(self) -> bool:
        return bool(
            self.stop_pct or self.trailing_stop_pct or self.stop_atr_mult or self.combined_stop_pct
        )


# ------------------------------------------------------------------ sizing


class Sizing(Base):
    """How large a position to take.

    `fixed_lots` is the default because it is what a small account can actually
    reason about; risk-based sizing needs a stop distance to divide by.
    """

    mode: Literal["fixed_lots", "fixed_value", "risk_based"] = "fixed_lots"
    lots: int = Field(default=1, ge=1, le=100)
    value: float | None = Field(default=None, gt=0, description="Rupees, for fixed_value.")
    risk_pct: float | None = Field(default=None, gt=0, le=10, description="Percent of capital.")

    @model_validator(mode="after")
    def _mode_has_its_field(self):
        if self.mode == "fixed_value" and self.value is None:
            raise ValueError("fixed_value sizing requires `value`")
        if self.mode == "risk_based" and self.risk_pct is None:
            raise ValueError("risk_based sizing requires `risk_pct`")
        return self


# -------------------------------------------------------------------- risk


class RiskLimits(Base):
    """Per-strategy limits. The risk manager enforces account-wide ones on top."""

    max_daily_loss: float = Field(default=2000.0, gt=0)
    max_loss_per_trade: float = Field(default=5000.0, gt=0)
    max_concurrent_positions: int = Field(default=1, ge=1, le=10)

    # A cap counted in LOTS only means something where lots exist. For a stock,
    # one lot is one share, so the old default of 2 silently capped every stock
    # position at two shares -- overriding whatever sizing the user asked for,
    # and no fixed number could work across a market priced from Rs 50 to
    # Rs 20,000 a share.
    #
    # None means no lot cap, which is the right answer for stocks: the limits
    # that bind there are the account balance and `max_loss_per_trade`, both of
    # which scale with price on their own.
    max_lots: int | None = Field(default=None, ge=1, le=100)

    # The concentration rail. Risk-based sizing with a tight stop produces a
    # large position for the same rupee risk -- a 1% stop on a 1% risk budget
    # puts nearly the whole account into one trade. The risk arithmetic is
    # correct and the outcome is still reckless: no room for anything else, and
    # full exposure to an overnight gap that jumps straight past the stop.
    #
    # This binds regardless of what the sizing mode computes, because a rail
    # that yields to the thing it is restraining is not a rail.
    max_position_pct: float | None = Field(default=None, gt=0, le=100)


# ---------------------------------------------------------------- schedule


class Schedule(Base):
    """When the strategy is allowed to act.

    `square_off` is mandatory for intraday and for options: an unclosed option
    position at expiry settles unpredictably and can lose far more than intended.
    """

    intraday: bool = True
    no_entry_after: dt.time = dt.time(15, 0)
    square_off: dt.time = dt.time(15, 15)
    weekdays: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])
    # The settings screen's "Entry cut-off time" box (Justin, 2026-10-15).
    # None -- every strategy saved before the box existed -- means what it
    # always did: the cut-off applies to an intraday strategy only.
    entry_cutoff: bool | None = None
    # "Once a day" (Justin, 2026-10-16): at most this many trades open per
    # session; a signal after that is skipped and said. None = no limit.
    max_entries_per_day: int | None = Field(default=None, ge=1, le=20)

    @property
    def cutoff_applies(self) -> bool:
        """Is `no_entry_after` in force? See `entry_cutoff`."""
        return self.intraday if self.entry_cutoff is None else self.entry_cutoff

    @field_validator("weekdays")
    @classmethod
    def _valid_weekdays(cls, v: list[int]) -> list[int]:
        if not v or any(d < 0 or d > 6 for d in v):
            raise ValueError("weekdays must be a non-empty list of 0 (Mon) to 6 (Sun)")
        return sorted(set(v))

    @model_validator(mode="after")
    def _square_off_after_cutoff(self):
        if self.intraday and self.cutoff_applies and self.square_off < self.no_entry_after:
            raise ValueError("square_off must not be earlier than no_entry_after")
        return self


# ------------------------------------------------------------- instrument

INDICES = ("NIFTY", "BANKNIFTY")

# Underlyings this platform can trade *as options*. SENSEX is here and nowhere
# else: it has no tradeable index or futures product here (the parser refuses
# a bare "SENSEX" or "SENSEX futures" -- see `nlt.translate.rules`), only
# options on it.
OPTION_UNDERLYINGS = (*INDICES, "SENSEX")


class OptionLeg(Base):
    """One option in a multi-leg (or single sold) position (Phase 2, batch 2).

    The strike is chosen at the signal, from the index's close on the signal
    candle, exactly as for a bought single option, in one of two ways:

    * `strike_offset` -- listed strikes away from at-the-money; +1 is one
      strike out of the money *for this leg's type* (higher for a call, lower
      for a put).
    * `strike_points` -- index points from the spot, signed by direction:
      -200 is "200 points below spot", +300 "300 points above". The listed
      strike nearest spot + points is taken; a target with no listed strike
      within one strike gap is refused, never clamped.

    `stop_pct` / `target_pct` are this leg's own exits, as a percentage of its
    price at the signal: a sold leg is stopped when its price RISES by
    `stop_pct`, a bought leg when it falls.
    """

    side: Literal["buy", "sell"]
    option_type: Literal["CE", "PE"]
    strike_offset: int = Field(default=0, ge=-10, le=10)
    strike_points: int | None = Field(default=None, ge=-5000, le=5000)
    expiry: Literal["nearest_weekly", "next_weekly", "monthly"] = "nearest_weekly"
    lots: int = Field(default=1, ge=1, le=10, description="Lots per set; a ratio, not a size.")
    stop_pct: float | None = Field(default=None, gt=0, le=1000)
    target_pct: float | None = Field(default=None, gt=0, le=100)
    # What happens after this leg's own stop closes it. "same_strike_at_cost"
    # re-sells the same contract when its price comes back to the first fill;
    # "fresh_strike_now" re-picks the strike at the next candle. Never guessed:
    # the reader asks which one when the words do not say.
    reentry: Literal["none", "same_strike_at_cost", "fresh_strike_now"] = "none"
    reentry_max: int = Field(default=0, ge=0, le=5)

    @model_validator(mode="after")
    def _one_strike_rule(self):
        if self.strike_points is not None and self.strike_offset != 0:
            raise ValueError(
                "a leg's strike is either strikes from ATM or points from spot, not both"
            )
        if (self.reentry == "none") != (self.reentry_max == 0):
            raise ValueError("re-entry needs both a kind and a count (or neither)")
        return self


class Instrument(Base):
    """What the signal is computed on, and what actually gets traded.

    `symbol` is one of three things, and the distinction drives everything
    downstream:

    * an index -- "NIFTY", "BANKNIFTY"
    * a single stock -- "TCS", "RELIANCE"
    * a universe -- "NIFTY 50", "NIFTY 100", "NIFTY 500"

    A universe means the same rules are evaluated independently on every member,
    and the portfolio holds whichever of them are signalling, up to the risk
    limits. That is what "buy Nifty 100 stocks when RSI drops below 40" actually
    asks for, and it is a materially different thing to backtest than one series:
    capital is shared, positions compete for it, and a day where forty names
    signal at once is the interesting case rather than an edge case.
    """

    symbol: str = "NIFTY"
    timeframe: Literal["1m", "3m", "5m", "15m", "30m", "1h", "1d"] = "1d"
    # "future" trades the nearest-month futures contract of `symbol`: an MCX
    # commodity (CRUDEOIL, GOLDM, ...) or an NSE index (NIFTY, BANKNIFTY). Any
    # position is closed when its contract expires -- see `nlt.data.futures`.
    trade_as: Literal["index", "option", "stock", "future"] = "index"

    # How to choose when more symbols signal than there is room to hold. There
    # is no neutral answer, so it is stated rather than left to whatever order
    # the data happened to arrive in.
    selection: Literal["first", "ranked"] = "first"

    # Only meaningful when trade_as == "option".
    option_type: Literal["auto", "CE", "PE"] = "auto"
    strike_offset: int = Field(
        default=0,
        ge=-10,
        le=10,
        description="Strikes away from at-the-money. 0 = ATM, +1 = one strike OTM.",
    )
    expiry: Literal["nearest_weekly", "next_weekly", "monthly"] = "nearest_weekly"

    # Batch 2: a sold option or several options opened together. Empty for
    # everything else, including batch 1's single bought option, which keeps
    # `option_type` / `strike_offset` above and its own engine.
    legs: list[OptionLeg] = Field(default_factory=list, max_length=4)

    @field_validator("symbol")
    @classmethod
    def _known_symbol(cls, v: str) -> str:
        """Accept an index, a universe name, or a plausible NSE ticker.

        Universe membership is checked against the bundled constituent lists, so
        a typo like "NIFTY 250" is rejected here rather than silently backtesting
        an empty basket. A bare ticker cannot be verified without a download, so
        it is only shape-checked; a symbol that does not exist surfaces as a load
        failure with its own reason attached.
        """
        cleaned = " ".join(str(v).upper().split())
        if cleaned in INDICES:
            return cleaned

        from nlt.data.universe import is_universe, normalise_universe_name

        if is_universe(cleaned):
            return normalise_universe_name(cleaned)

        if re.fullmatch(r"[A-Z][A-Z0-9&\-]{1,19}", cleaned):
            return cleaned

        raise ValueError(
            f"{v!r} is not an index ({', '.join(INDICES)}), a known universe, "
            "or a valid NSE ticker symbol"
        )

    @model_validator(mode="after")
    def _futures_name_a_contract(self):
        """A future must be a contract we have terms for; a commodity can only be a future.

        The terms -- above all how many units one lot is -- decide every rupee of
        profit and loss, so an unknown contract is refused rather than given a
        guessed size. And a commodity name read as a share would be backtested
        on whatever NSE ticker happened to share its letters.
        """
        from nlt.data.futures import FUTURES, is_commodity

        if self.trade_as == "future" and self.symbol not in FUTURES:
            raise ValueError(
                f"{self.symbol} futures are not supported; supported: {', '.join(FUTURES)}"
            )
        if is_commodity(self.symbol) and self.trade_as != "future":
            raise ValueError(f"{self.symbol} is an MCX commodity and can only be traded as futures")
        return self

    @model_validator(mode="after")
    def _option_fields_are_resolved(self):
        """A real option order needs a real strike and a real underlying.

        `option_type="auto"` is the field's own default, meant for callers that
        never touch options at all (most of the test suite exercises the lot
        and margin mechanics of "an F&O instrument" using `trade_as="option"`
        on an arbitrary symbol, not a real contract) -- so this only fires once
        a spec actually claims to be tradeable, i.e. `option_type` has been set
        to a real side (`CE` or `PE` -- the only other values the field's own
        `Literal` allows). At that point the underlying must be one this
        platform can actually resolve into a listed contract.
        """
        if self.trade_as == "option" and self.option_type != "auto":
            if self.symbol not in OPTION_UNDERLYINGS:
                raise ValueError(
                    f"{self.symbol} options are not supported; supported: "
                    f"{', '.join(OPTION_UNDERLYINGS)}"
                )
        return self

    @model_validator(mode="after")
    def _legs_are_options(self):
        if not self.legs:
            return self
        if self.trade_as != "option":
            raise ValueError("option legs need trade_as='option'")
        if self.option_type != "auto" or self.strike_offset != 0:
            raise ValueError("a multi-leg position describes its options in `legs` only")
        if self.symbol not in OPTION_UNDERLYINGS:
            raise ValueError(
                f"{self.symbol} options are not supported; supported: {', '.join(OPTION_UNDERLYINGS)}"
            )
        if len(set(self.legs)) != len(self.legs):
            raise ValueError("two legs are identical -- write them as one leg with more lots")
        return self

    @property
    def is_multi_leg(self) -> bool:
        return bool(self.legs)

    @property
    def is_future(self) -> bool:
        return self.trade_as == "future"

    @property
    def is_option(self) -> bool:
        return self.trade_as == "option"

    @property
    def is_universe(self) -> bool:
        from nlt.data.universe import is_universe as _is

        return _is(self.symbol)


# --------------------------------------------------------------- the spec


class StrategySpec(Base):
    """A complete, runnable strategy."""

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(description="The user's original words, kept verbatim.")
    instrument: Instrument = Field(default_factory=Instrument)
    indicators: list[IndicatorSpec] = Field(default_factory=list, max_length=20)
    direction: Literal["long", "short"] = "long"
    entry: Condition
    exit: ExitRules
    sizing: Sizing = Field(default_factory=Sizing)
    risk: RiskLimits = Field(default_factory=RiskLimits)
    schedule: Schedule = Field(default_factory=Schedule)

    @model_validator(mode="after")
    def _ids_unique(self):
        ids = [i.id for i in self.indicators]
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            raise ValueError(f"duplicate indicator ids: {sorted(dupes)}")
        return self

    @model_validator(mode="after")
    def _refs_resolve(self):
        """Every reference must point at a declared indicator output or a price field.

        This is what stops a hallucinated indicator name reaching the engine.
        """
        by_id = {i.id: i for i in self.indicators}

        for ref in _walk_refs(self.entry):
            _check_ref(ref, by_id, "entry")
        if self.exit.condition is not None:
            for ref in _walk_refs(self.exit.condition):
                _check_ref(ref, by_id, "exit condition")

        if self.exit.atr_id:
            spec = by_id.get(self.exit.atr_id)
            if spec is None:
                raise ValueError(f"exit.atr_id {self.exit.atr_id!r} is not a declared indicator")
            if spec.type != "atr":
                raise ValueError(f"exit.atr_id must name an 'atr' indicator, not {spec.type!r}")
        return self

    @model_validator(mode="after")
    def _volume_indicators_need_volume(self):
        """Index feeds often report zero volume, which makes these silently useless."""
        # Futures are traded contracts with real volume; only the index has none.
        if self.instrument.symbol in INDICES and not self.instrument.is_future:
            offenders = [i.id for i in self.indicators if get_indicator(i.type).needs_volume]
            if offenders:
                raise ValueError(
                    f"volume-based indicators {offenders} cannot be used on index data, "
                    "which has no reliable volume. Use a price-based indicator instead."
                )
        return self

    @model_validator(mode="after")
    def _options_long_only(self):
        """Batch 1 is buying one call or one put, never selling (writing) one.

        Selling an option is refused in words before a spec is ever built (see
        `nlt.translate.options_words`); this is the backstop for anything that
        reaches the model directly, e.g. a saved strategy edited by hand later.
        A naked short option has unbounded loss -- the one thing this platform
        refuses to let through silently.
        """
        if self.instrument.trade_as == "option" and self.instrument.option_type != "auto":
            if self.direction != "long":
                raise ValueError(
                    "a sold option is written as a leg (instrument.legs), not as direction='short'"
                )
        return self

    @model_validator(mode="after")
    def _legs_are_consistent(self):
        """The safety rules for selling (Justin, 2026-10-14).

        * Every position needs a way to lose no more than it can bear. A
          **defined-risk** position -- every sold option covered by a bought
          one of the same type and expiry, at least as many lots, further out
          -- is protected by its hedge, so it needs no stop loss. Anything
          else (a plain sell, a straddle, a strangle) must have a stop: on
          every sold leg, or on the combined position.
        * `direction` is "long" by convention: each leg carries its own side.
        * Per-leg exits make no sense on a single-option spec's own fields, and
          the stock-style exits measured from the index make no sense on legs.
        """
        legs = self.instrument.legs
        if not legs:
            if self.exit.combined_target_pct or self.exit.combined_stop_pct:
                raise ValueError("combined exits apply only to multi-leg option positions")
            if self.exit.exits_on_legs:
                raise ValueError("exits_on_legs applies only to multi-leg option positions")
            return self
        if self.direction != "long":
            raise ValueError("a multi-leg position keeps direction='long'; each leg has its side")
        if self.exit.stop_pct or self.exit.target_pct or self.exit.trailing_stop_pct:
            raise ValueError(
                "on a multi-leg position, a stop or target is per leg (legs[].stop_pct) "
                "or on the combined premium (combined_stop_pct)"
            )
        if self.exit.stop_atr_mult:
            raise ValueError("an ATR stop is in index points and cannot apply to option legs")
        if not is_defined_risk(legs):
            unstopped = [leg for leg in legs if leg.side == "sell" and leg.stop_pct is None]
            if unstopped and self.exit.combined_stop_pct is None:
                raise ValueError(
                    "a sold option without a bought one protecting it needs a stop loss -- "
                    "on each sold leg, or on the combined premium"
                )
        if self.exit.exits_on_legs and not (
            any(leg.stop_pct or leg.target_pct for leg in legs)
            or (self.schedule.intraday and self.instrument.timeframe != "1d")
        ):
            raise ValueError(
                "exits_on_legs is set, but no leg has a stop or target and there is no "
                "same-day square-off"
            )
        if any(leg.reentry != "none" and leg.stop_pct is None for leg in legs):
            raise ValueError("a leg can only re-enter after its own stop loss closes it")
        return self

    def indicator_ids(self) -> list[str]:
        return [i.id for i in self.indicators]


def is_defined_risk(legs: list[OptionLeg]) -> bool:
    """True when every sold option is covered by a bought one that caps its loss.

    Covered means: a bought leg of the same type and expiry, with at least as
    many lots in total, further out of the money than the sold one. Whether a
    strike is "further out" can only be known for strikes fixed relative to one
    another *in the same unit* -- both counted in listed strikes, or both in
    points -- so mixed units are treated as not covered (refuse, never assume).
    A ratio position (sell 2, buy 1) is not covered.
    """
    sells = [leg for leg in legs if leg.side == "sell"]
    buys = [leg for leg in legs if leg.side == "buy"]
    if not sells:
        return True
    for kind in ("CE", "PE"):
        for expiry in {leg.expiry for leg in sells}:
            s_legs = [x for x in sells if x.option_type == kind and x.expiry == expiry]
            b_legs = [x for x in buys if x.option_type == kind and x.expiry == expiry]
            if not s_legs:
                continue
            for s_leg in s_legs:
                if not any(_further_out(b, s_leg) for b in b_legs):
                    return False
            if sum(x.lots for x in b_legs) < sum(x.lots for x in s_legs):
                return False
    return True


def _further_out(bought: OptionLeg, sold: OptionLeg) -> bool:
    if (bought.strike_points is None) != (sold.strike_points is None):
        return False
    if bought.strike_points is not None:
        # Points are signed by direction: above spot is +, below is -.
        if sold.option_type == "CE":
            return bought.strike_points > sold.strike_points
        return bought.strike_points < sold.strike_points
    # Offsets count strikes out of the money for the leg's own type.
    return bought.strike_offset > sold.strike_offset


def _walk_refs(condition) -> list[Ref]:
    """Collect every Ref in a condition tree."""
    if isinstance(condition, Compare):
        return [o for o in (condition.left, condition.right) if isinstance(o, Ref)]
    if isinstance(condition, IsTrue):
        return [condition.ref]
    if isinstance(condition, PercentChange):
        return [condition.ref]
    if isinstance(condition, (All, Any_)):
        return [r for c in condition.conditions for r in _walk_refs(c)]
    if isinstance(condition, Not):
        return _walk_refs(condition.condition)
    return []


def _check_ref(ref: Ref, by_id: dict[str, IndicatorSpec], where: str) -> None:
    if ref.name in PRICE_FIELDS:
        if ref.output is not None:
            raise ValueError(f"price field {ref.name!r} has no outputs, got {ref.output!r}")
        return

    spec = by_id.get(ref.name)
    if spec is None:
        raise ValueError(
            f"{where} references {ref.name!r}, which is neither a declared indicator id "
            f"({sorted(by_id) or 'none declared'}) nor a price field {list(PRICE_FIELDS)}"
        )

    d = get_indicator(spec.type)
    if d.is_multi:
        if ref.output is None:
            raise ValueError(
                f"{where}: {ref.name!r} is a {spec.type} with several outputs "
                f"{list(d.outputs)} -- say which one"
            )
        if ref.output not in d.outputs:
            raise ValueError(
                f"{where}: {spec.type} has no output {ref.output!r}; available: {list(d.outputs)}"
            )
    elif ref.output is not None and ref.output not in d.outputs:
        raise ValueError(f"{where}: {spec.type} has no output {ref.output!r}")
