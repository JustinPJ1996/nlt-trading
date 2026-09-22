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
    Compare | IsTrue | PercentChange | All | Any_ | Not,
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
            raise ValueError(
                f"unknown indicator {v!r}; available: {', '.join(sorted(REGISTRY))}"
            )
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
        return bool(self.stop_pct or self.trailing_stop_pct or self.stop_atr_mult)


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
    max_lots: int = Field(default=2, ge=1, le=100)


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

    @field_validator("weekdays")
    @classmethod
    def _valid_weekdays(cls, v: list[int]) -> list[int]:
        if not v or any(d < 0 or d > 6 for d in v):
            raise ValueError("weekdays must be a non-empty list of 0 (Mon) to 6 (Sun)")
        return sorted(set(v))

    @model_validator(mode="after")
    def _square_off_after_cutoff(self):
        if self.intraday and self.square_off < self.no_entry_after:
            raise ValueError("square_off must not be earlier than no_entry_after")
        return self


# ------------------------------------------------------------- instrument

class Instrument(Base):
    """What the signal is computed on, and what actually gets traded."""

    symbol: Literal["NIFTY", "BANKNIFTY"] = "NIFTY"
    timeframe: Literal["1m", "3m", "5m", "15m", "30m", "1h", "1d"] = "1d"
    trade_as: Literal["index", "option"] = "index"

    # Only meaningful when trade_as == "option".
    option_type: Literal["auto", "CE", "PE"] = "auto"
    strike_offset: int = Field(
        default=0,
        ge=-10,
        le=10,
        description="Strikes away from at-the-money. 0 = ATM, +1 = one strike OTM.",
    )
    expiry: Literal["nearest_weekly", "next_weekly", "monthly"] = "nearest_weekly"


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
        if self.instrument.symbol in ("NIFTY", "BANKNIFTY"):
            offenders = [
                i.id for i in self.indicators if get_indicator(i.type).needs_volume
            ]
            if offenders:
                raise ValueError(
                    f"volume-based indicators {offenders} cannot be used on index data, "
                    "which has no reliable volume. Use a price-based indicator instead."
                )
        return self

    @model_validator(mode="after")
    def _options_must_square_off(self):
        if self.instrument.trade_as == "option" and not self.schedule.intraday:
            if self.exit.max_bars_held is None and self.exit.condition is None:
                raise ValueError(
                    "a positional option strategy needs an explicit exit condition or "
                    "max_bars_held, or it will be held into expiry"
                )
        return self

    def indicator_ids(self) -> list[str]:
        return [i.id for i in self.indicators]


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
                f"{where}: {spec.type} has no output {ref.output!r}; "
                f"available: {list(d.outputs)}"
            )
    elif ref.output is not None and ref.output not in d.outputs:
        raise ValueError(f"{where}: {spec.type} has no output {ref.output!r}")
