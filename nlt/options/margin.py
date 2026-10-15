"""The margin a broker blocks for an index options position, the way Zerodha's
F&O (SPAN) calculator works it out -- day by day, from that day's exchange file.

**What `total` is.** The money the broker *blocks* to let you hold the
position: SPAN + exposure, exactly Zerodha's "Total margin". It does **not**
include the premium you pay for bought legs -- Zerodha's calculator says so in
as many words ("Buying options require full premium and not margins ... premium
values to buy options don't show up"), and the backtest must charge that
premium as a cash payment of its own. Bought legs still *lower* the margin of
the legs they protect; that is where a spread's "margin benefit" comes from.
Premium received for sold legs is not deducted either (Zerodha shows it as
"Premium receivable", outside the total).

**SPAN** (per index, from the exchange's risk-parameter file -- see
`span_store`):

    risk  = max( worst of 16 scenarios of  sum(quantity x risk array)
                 + calendar-spread charge ,
                 short-option minimum )
    span  = max( 0, risk - net option value )

The net option value is (bought legs' price - sold legs' price) x units, so a
sold option's own premium is *inside* the SPAN figure, and a bought option on
its own needs nothing. This is CME SPAN's final step, and it is what makes our
figure match Zerodha's to the rupee (see `tests/test_span_margin.py`).

From **10 Feb 2025** (NSE Clearing circular 005/2025, carrying out SEBI's
1 Oct 2024 circular, para 5.2), on an expiry day the legs expiring that day are
margined as a separate portfolio from the rest: no offset across expiries and
no calendar-spread charge between them.

**Exposure** (extreme loss margin; not in the SPAN file -- NSE's
equity-derivatives margins page and SEBI's circular), on **sold** legs only:

- 2% of notional, notional = the index's previous close x units;
- 3% if the strike is more than 10% out of the money against that close;
- 5% if more than 9 months remain to expiry;
- plus 2% on the leg's own expiry day, from **20 Nov 2024** (SEBI circular
  SEBI/HO/MRD/TPD-1/P/CIR/2024/132, para 5.6; NSE Clearing 125/2024, whose
  worked example gives a deep out-of-the-money leg 3% + 2% = 5% that day).

Bought legs carry no exposure margin. BSE (SENSEX) applies the same rules
(ICCL notice 20241003-52).

**Which file applies on day D.** A broker blocks margin at entry, so we use the
exchange's *start-of-day* file for D -- NSE's ``i1`` (built ~21:30 the evening
before), BSE's ``00`` (built ~00:50 that night). Neither can know anything
that happened on D. Zerodha's calculator was using exactly that file on the
morning of 14 Oct 2026. Two fallbacks, each stated in ``note``:

- no start-of-day file for D (BSE skipped dozens of days in 2023): the most
  recent file published before D began -- normally the previous trading day's
  end-of-day file, built from the same closing prices (on days that have both,
  BSE's two files carry identical prices; the risk arrays differ by one day of
  time decay);
- a contract missing from D's file (listed during the day): the earliest later
  file that has it -- a small look-ahead, said out loud.

Outside the stored window the module refuses rather than guess.
"""

from __future__ import annotations

import datetime as dt
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np

from nlt.options.span_store import KIND_CALL, KIND_PUT, SpanDay, load_day

DEFAULT_ROOT = (
    Path(os.environ["NLT_SPAN_ROOT"])
    if os.environ.get("NLT_SPAN_ROOT")
    else Path(__file__).resolve().parents[2] / "data" / "span"
)

EXCHANGE = {"NIFTY": "NSE", "BANKNIFTY": "NSE", "SENSEX": "BSE"}
BOD_TAG = {"NSE": "i1", "BSE": "00"}
EOD_TAG = {"NSE": "s", "BSE": "FINAL"}

# The days `scripts/fetch_span.py` stores: the F&O dataset's coverage, and
# from paper trading (Phase 2 batch 3) every day's file as it is published --
# so the window stays open at the end.
WINDOW = {
    "NIFTY": (dt.date(2022, 12, 1), dt.date.max),
    "BANKNIFTY": (dt.date(2022, 12, 1), dt.date.max),
    "SENSEX": (dt.date(2023, 7, 20), dt.date.max),
}

# Rules by date. Each is the first day it applied.
EXPIRY_DAY_EXTRA_FROM = dt.date(2024, 11, 20)
NO_EXPIRY_DAY_CALENDAR_FROM = dt.date(2025, 2, 10)

ELM_BASE = 0.02
ELM_DEEP_OTM = 0.03
DEEP_OTM_BEYOND = 0.10
ELM_LONG_DATED = 0.05
LONG_DATED_MONTHS = 9
ELM_EXPIRY_DAY_EXTRA = 0.02

# How far to look for a fallback file, in calendar days.
_FALLBACK_DAYS = 10


class MarginUnavailable(ValueError):
    """No exchange file can honestly answer this question."""


@dataclass(frozen=True)
class MarginLeg:
    option_type: Literal["CE", "PE"]
    strike: float
    expiry: dt.date
    quantity: int  # units; + bought, - sold


@dataclass(frozen=True)
class MarginBreakdown:
    span: float  # rupees
    exposure: float  # rupees, incl. the expiry-day extra
    expiry_day_extra: float
    total: float
    source: str  # plain English, e.g. "NSE's margin file for 11 Mar 2025 (start of day)"
    note: str | None  # plain English when a fallback was used, else None


def _fmt(day: dt.date) -> str:
    return f"{day.day} {day:%b %Y}"


def _add_months(day: dt.date, months: int) -> dt.date:
    y, m = divmod(day.month - 1 + months, 12)
    year, month = day.year + y, m + 1
    for d in range(day.day, 27, -1):
        try:
            return dt.date(year, month, d)
        except ValueError:
            continue
    return dt.date(year, month, min(day.day, 28))


def exposure_rate(leg: MarginLeg, day: dt.date, index_prev_close: float) -> tuple[float, float]:
    """(rate, expiry-day extra rate) for a SOLD leg; bought legs pay none."""
    if leg.quantity >= 0:
        return 0.0, 0.0
    if leg.option_type == "CE":
        otm = (leg.strike - index_prev_close) / index_prev_close
    else:
        otm = (index_prev_close - leg.strike) / index_prev_close
    rate = ELM_BASE
    if otm > DEEP_OTM_BEYOND:
        rate = ELM_DEEP_OTM
    if leg.expiry > _add_months(day, LONG_DATED_MONTHS):
        rate = max(rate, ELM_LONG_DATED)
    extra = ELM_EXPIRY_DAY_EXTRA if (leg.expiry == day and day >= EXPIRY_DAY_EXTRA_FROM) else 0.0
    return rate, extra


def span_requirement(
    file: SpanDay,
    underlying: str,
    rows: list[int],
    quantities: list[int],
    separate_expiry: dt.date | None = None,
) -> float:
    """SPAN for one index's legs in one file. `rows` index the file's contracts.

    `separate_expiry`: margin the legs expiring that day as their own portfolio
    (the expiry-day rule from 10 Feb 2025)."""
    params = file.params[underlying]
    q = np.asarray(quantities, dtype=np.float64)
    idx = np.asarray(rows, dtype=np.int64)
    expiry = file.expiry[idx]
    if separate_expiry is not None:
        key = int(separate_expiry.strftime("%Y%m%d"))
        groups = [expiry == key, expiry != key]
    else:
        groups = [np.ones(len(idx), dtype=bool)]

    total = 0.0
    for mask in groups:
        if not mask.any():
            continue
        gq, gi = q[mask], idx[mask]
        scenarios = gq @ file.risk[gi]  # loss per scenario, rupees
        scan = max(0.0, float(scenarios.max()))
        calendar = _calendar_charge(params.spreads, file.expiry[gi], gq * file.delta[gi])
        short_units = float(-gq[gq < 0].sum())
        som = params.som_rate * short_units
        risk = max(scan + calendar, som)
        net_option_value = float(gq @ file.price[gi])
        total += max(0.0, risk - net_option_value)
    return total


def _calendar_charge(spreads, expiries: np.ndarray, deltas: np.ndarray) -> float:
    """The exchange's calendar-spread table, applied in priority order: a spread
    forms between two expiries whose net deltas have opposite signs."""
    net: dict[int, float] = {}
    for e, d in zip(expiries.tolist(), deltas.tolist(), strict=True):
        net[e] = net.get(e, 0.0) + d
    if len([v for v in net.values() if v != 0.0]) < 2:
        return 0.0
    charge = 0.0
    for s in spreads:
        da, db = net.get(s.leg_a, 0.0), net.get(s.leg_b, 0.0)
        if da == 0.0 or db == 0.0 or (da > 0) == (db > 0):
            continue
        n = min(abs(da) / s.ratio_a, abs(db) / s.ratio_b)
        charge += n * s.rate
        net[s.leg_a] = da - np.sign(da) * n * s.ratio_a
        net[s.leg_b] = db - np.sign(db) * n * s.ratio_b
    return charge


class SpanMargin:
    """Margin from the stored exchange files (`scripts/fetch_span.py`)."""

    def __init__(self, root: Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self._load = lru_cache(maxsize=64)(self._load_uncached)

    def _path(self, exchange: str, day: dt.date, tag: str) -> Path:
        return self.root / exchange / f"{day.year}" / f"{day:%Y%m%d}-{tag}.npz"

    def _load_uncached(self, exchange: str, day: dt.date, tag: str | None = None) -> SpanDay | None:
        path = self._path(exchange, day, tag or BOD_TAG[exchange])
        return load_day(path) if path.exists() else None

    def _file_for(self, underlying: str, day: dt.date) -> tuple[SpanDay, str | None]:
        exchange = EXCHANGE[underlying]
        exact = self._load(exchange, day)
        if exact is not None:
            return exact, None
        start, end = WINDOW[underlying]
        if not (start <= day <= end):
            raise MarginUnavailable(
                f"We have {underlying} margin files from {_fmt(start)} to {_fmt(end)} only, "
                f"so we cannot work out the margin for {_fmt(day)}."
            )
        # The most recent file published before the day began: an earlier day's
        # end-of-day file beats that same day's start-of-day file.
        for back in range(1, _FALLBACK_DAYS + 1):
            earlier = day - dt.timedelta(days=back)
            eod = self._load(exchange, earlier, EOD_TAG[exchange])
            if eod is not None:
                return eod, (
                    f"{exchange} did not publish a start-of-day margin file for {_fmt(day)}, "
                    f"so we used its end-of-day file for {_fmt(earlier)}, built from the "
                    f"same closing prices."
                )
            bod = self._load(exchange, earlier)
            if bod is not None:
                return bod, (
                    f"{exchange} did not publish a start-of-day margin file for {_fmt(day)}, "
                    f"so we used the one for {_fmt(earlier)}."
                )
        raise MarginUnavailable(
            f"No {exchange} margin file for {_fmt(day)} or the {_FALLBACK_DAYS} days before it."
        )

    @staticmethod
    def _row(file: SpanDay, underlying: str, leg: MarginLeg) -> int | None:
        kind = KIND_CALL if leg.option_type == "CE" else KIND_PUT
        mask = (
            (file.underlying == underlying)
            & (file.kind == kind)
            & (file.expiry == int(leg.expiry.strftime("%Y%m%d")))
            & (np.abs(file.strike - leg.strike) < 1e-6)
        )
        hits = np.flatnonzero(mask)
        return int(hits[0]) if len(hits) else None

    def margin(
        self,
        underlying: str,
        day: dt.date,
        legs: list[MarginLeg],
        *,
        index_prev_close: float,
    ) -> MarginBreakdown:
        underlying = underlying.upper()
        if underlying not in EXCHANGE:
            raise MarginUnavailable(f"We have no margin files for {underlying}.")
        if not legs:
            raise ValueError("a position needs at least one leg")
        if not index_prev_close > 0:
            raise ValueError("index_prev_close must be a positive price")
        for leg in legs:
            if leg.option_type not in ("CE", "PE"):
                raise ValueError(f"option_type must be CE or PE, not {leg.option_type!r}")
            if leg.expiry < day:
                raise ValueError(f"{leg} expired before {day}")

        file, note = self._file_for(underlying, day)
        notes = [note] if note else []
        exchange = EXCHANGE[underlying]

        # Every leg is looked up in the day's file. A leg it lacks (listed during
        # the day) comes from the earliest later file that has it; the scenario
        # arrays of different files cannot be summed, so the whole position is
        # then margined from that later file.
        rows = [self._row(file, underlying, leg) for leg in legs]
        if any(r is None for r in rows):
            missing = [leg for leg, r in zip(legs, rows, strict=True) if r is None]
            later = self._later_file_with(exchange, underlying, file.file_date, legs)
            if later is None:
                names = ", ".join(_leg_name(underlying, leg) for leg in missing)
                raise MarginUnavailable(
                    f"{names} is not in {exchange}'s margin file for {_fmt(file.file_date)} "
                    f"or any of the next {_FALLBACK_DAYS} days' files."
                )
            names = ", ".join(_leg_name(underlying, leg) for leg in missing)
            notes.append(
                f"{names} was not yet listed when {exchange}'s start-of-day file for "
                f"{_fmt(file.file_date)} was made, so the margin uses the file for "
                f"{_fmt(later.file_date)}, the first one that has it."
            )
            file = later
            rows = [self._row(file, underlying, leg) for leg in legs]

        separate = day if (day >= NO_EXPIRY_DAY_CALENDAR_FROM) else None
        span = span_requirement(
            file, underlying, rows, [leg.quantity for leg in legs], separate_expiry=separate
        )

        exposure = extra = 0.0
        for leg in legs:
            rate, extra_rate = exposure_rate(leg, day, index_prev_close)
            notional = index_prev_close * abs(leg.quantity)
            exposure += (rate + extra_rate) * notional
            extra += extra_rate * notional

        return MarginBreakdown(
            span=span,
            exposure=exposure,
            expiry_day_extra=extra,
            total=span + exposure,
            source=f"{exchange}'s margin file for {_fmt(file.file_date)} "
            f"({'end of day' if file.source_name.endswith(EOD_TAG[exchange]) else 'start of day'})",
            note=" ".join(notes) or None,
        )

    def _later_file_with(
        self, exchange: str, underlying: str, after: dt.date, legs: list[MarginLeg]
    ) -> SpanDay | None:
        for ahead in range(1, _FALLBACK_DAYS + 1):
            later = self._load(exchange, after + dt.timedelta(days=ahead))
            if later is None:
                continue
            if all(self._row(later, underlying, leg) is not None for leg in legs):
                return later
        return None


def _leg_name(underlying: str, leg: MarginLeg) -> str:
    strike = f"{leg.strike:g}"
    return f"{underlying} {_fmt(leg.expiry)} {strike} {leg.option_type}"


class FixedMargin:
    """For tests: a constant number of rupees per call, or a callable
    ``(underlying, day, legs, index_prev_close) -> rupees``. Everything is
    reported as SPAN; exposure is zero."""

    def __init__(
        self,
        rupees: float | Callable[[str, dt.date, list[MarginLeg], float], float],
    ) -> None:
        self.rupees = rupees

    def margin(
        self,
        underlying: str,
        day: dt.date,
        legs: list[MarginLeg],
        *,
        index_prev_close: float,
    ) -> MarginBreakdown:
        if callable(self.rupees):
            value = float(self.rupees(underlying, day, legs, index_prev_close))
        else:
            value = float(self.rupees)
        return MarginBreakdown(
            span=value,
            exposure=0.0,
            expiry_day_extra=0.0,
            total=value,
            source="a fixed margin set for testing",
            note=None,
        )
