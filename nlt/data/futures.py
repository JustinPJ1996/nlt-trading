"""Futures contracts: what one lot is, when each contract expires, which hours it trades.

Every rupee of futures profit and loss passes through `FuturesContract.multiplier`,
so it is the most dangerous number in this module. It is NOT Kite's `lot_size`:
Kite lists every MCX contract with a lot size of 1, because MCX orders are placed
in lots, while the price is quoted per barrel, per 10 grams, per kilogram. One
CRUDEOIL lot is 100 barrels, so a Rs 1 move in the quoted price is Rs 100 of
profit or loss. Trusting Kite's 1 would make every crude result 100 times too
small, and GOLD (quoted per 10 g, traded per kg) 100 times too small as well --
a wrong answer that looks exactly like a right one.

The multipliers below were checked on 2026-10-07 against Zerodha Varsity and
several brokers' contract pages, which all agree (MCX's own site refuses
automated requests, so its PDFs could not be read directly), and against Kite's
live prices: one CRUDEOIL lot at Rs 8,704 a barrel is Rs 8.7 lakh, which is the
contract value brokers quote.

**Expiry dates** are computed from each contract's rule, because Kite does not
keep a list of expired contracts and its continuous price series joins them
together without marking where one ends. The rules are not the folk versions
("crude expires on the 19th") -- those were wrong in four of the last twelve
months. MCX's energy contracts expire one US business day before the NYMEX
contract they settle against, so US exchange holidays move them. The rules are
held to two answer keys in `tests/test_futures.py`: every expiry Kite listed on
2026-10-07, and every contract switch visible in six years of Kite's own
continuous price history. A rule that drifts fails one of them.

Where a rule lands on a day the exchange was shut, the expiry moves to the
previous day the market traded. Which days it traded is read from the price data
itself (the same choice `nlt.data.session` makes for holidays), so no Indian
holiday table has to be kept up to date here.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

from nlt.data.session import MCX, NSE_EQUITY, Session

# --------------------------------------------------------------- the contracts


@dataclass(frozen=True)
class FuturesContract:
    symbol: str  # our name, and the `name` column in Kite's instrument list
    title: str  # what a person calls it
    exchange: str  # Kite's exchange code: "MCX" or "NFO"
    multiplier: int  # rupees of P&L for a Rs 1 move in the quoted price, per lot
    lot_text: str  # what one lot is, in words
    price_unit: str  # what the quoted price is per
    expiry_rule: str  # key into _EXPIRY_RULES
    months: tuple[int, ...]  # calendar months with a contract
    session: Session

    @property
    def market(self) -> str:
        """The exchange as people name it: Kite calls NSE's F&O segment "NFO"."""
        return "NSE" if self.exchange == "NFO" else self.exchange

    @property
    def describe_lot(self) -> str:
        return f"1 lot = {self.lot_text}, price quoted {self.price_unit}"


_EVERY_MONTH = tuple(range(1, 13))

FUTURES: dict[str, FuturesContract] = {
    c.symbol: c
    for c in (
        FuturesContract(
            "CRUDEOIL",
            "Crude Oil",
            "MCX",
            100,
            "100 barrels",
            "per barrel",
            "nymex_wti",
            _EVERY_MONTH,
            MCX,
        ),
        FuturesContract(
            "CRUDEOILM",
            "Crude Oil Mini",
            "MCX",
            10,
            "10 barrels",
            "per barrel",
            "nymex_wti",
            _EVERY_MONTH,
            MCX,
        ),
        FuturesContract(
            "NATURALGAS",
            "Natural Gas",
            "MCX",
            1250,
            "1,250 mmBtu",
            "per mmBtu",
            "nymex_henry_hub",
            _EVERY_MONTH,
            MCX,
        ),
        FuturesContract(
            "NATGASMINI",
            "Natural Gas Mini",
            "MCX",
            250,
            "250 mmBtu",
            "per mmBtu",
            "nymex_henry_hub",
            _EVERY_MONTH,
            MCX,
        ),
        FuturesContract(
            "GOLD",
            "Gold",
            "MCX",
            100,
            "1 kg",
            "per 10 grams",
            "fifth",
            (2, 4, 6, 8, 10, 12),
            MCX,
        ),
        FuturesContract(
            "GOLDM",
            "Gold Mini",
            "MCX",
            10,
            "100 grams",
            "per 10 grams",
            "fifth",
            _EVERY_MONTH,
            MCX,
        ),
        FuturesContract(
            "SILVER",
            "Silver",
            "MCX",
            30,
            "30 kg",
            "per kg",
            "fifth",
            (3, 5, 7, 9, 12),
            MCX,
        ),
        FuturesContract(
            "SILVERM",
            "Silver Mini",
            "MCX",
            5,
            "5 kg",
            "per kg",
            "last_day",
            (2, 4, 6, 8, 11),
            MCX,
        ),
        # NSE index futures. The lot is the exchange's current one, from the
        # same table options use -- see `nlt.data.instruments` for why today's
        # lot is used for the whole history.
        FuturesContract(
            "NIFTY",
            "NIFTY",
            "NFO",
            0,
            "",
            "per index point",
            "nse_index",
            _EVERY_MONTH,
            NSE_EQUITY,
        ),
        FuturesContract(
            "BANKNIFTY",
            "BANKNIFTY",
            "NFO",
            0,
            "",
            "per index point",
            "nse_index",
            _EVERY_MONTH,
            NSE_EQUITY,
        ),
    )
}

COMMODITIES = tuple(s for s, c in FUTURES.items() if c.exchange == "MCX")

# How a futures series is asked for from a price source, as opposed to the
# index of the same name: "NIFTY" is the index, "NIFTY FUT" its futures.
_DATA_SUFFIX = " FUT"


def data_key(symbol: str) -> str:
    return symbol.upper().strip() + _DATA_SUFFIX


def from_data_key(key: str) -> str | None:
    """The contract a data key names, or None if it is not a futures key."""
    return key[: -len(_DATA_SUFFIX)] if key.endswith(_DATA_SUFFIX) else None


def contract(symbol: str) -> FuturesContract:
    """The contract terms for `symbol`, with the NSE lot filled in from its one table."""
    key = symbol.upper().strip()
    try:
        c = FUTURES[key]
    except KeyError:
        raise ValueError(
            f"{symbol!r} is not a futures contract this platform supports ({', '.join(FUTURES)})"
        ) from None
    if c.exchange == "NFO":
        from nlt.data.instruments import LOT_SIZES

        lot = LOT_SIZES[key]
        return FuturesContract(
            c.symbol,
            c.title,
            c.exchange,
            lot,
            f"{lot} units of the index",
            c.price_unit,
            c.expiry_rule,
            c.months,
            c.session,
        )
    return c


def is_future(symbol: str) -> bool:
    return symbol.upper().strip() in FUTURES


def is_commodity(symbol: str) -> bool:
    return symbol.upper().strip() in COMMODITIES


# -------------------------------------------------------------- US holidays
#
# CME's holidays, which decide NYMEX's last trading days. Not the US federal
# calendar: CME trades on Columbus Day and Veterans Day, and closes on Good
# Friday, which is not a federal holiday.


class _CmeHolidays(AbstractHolidayCalendar):
    rules = [
        # A Saturday New Year is not moved to the Friday before -- CME, like NYSE,
        # trades on 31 December then. A Sunday one moves to Monday.
        Holiday("New Year", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_US_HOLIDAYS = frozenset(
    d.date() for d in _CmeHolidays().holidays(dt.datetime(2015, 1, 1), dt.datetime(2035, 12, 31))
)


def _us_business_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in _US_HOLIDAYS


def _us_business_days_before(d: dt.date, n: int) -> dt.date:
    """The n-th US business day strictly before `d`."""
    while n:
        d -= dt.timedelta(days=1)
        if _us_business_day(d):
            n -= 1
    return d


# ------------------------------------------------------------- expiry rules
#
# Each rule returns the NOMINAL expiry for the contract of (year, month): the
# day the rule names, before the Indian exchange's own holidays are applied.


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _nymex_wti(year: int, month: int) -> dt.date:
    """One US business day before NYMEX WTI's last trading day.

    MCX's crude contract for month M settles against NYMEX's contract for M+1,
    which stops trading three business days before the 25th of M (four if the
    25th is not a business day).
    """
    d25 = dt.date(year, month, 25)
    nymex = _us_business_days_before(d25, 3 if _us_business_day(d25) else 4)
    return _us_business_days_before(nymex, 1)


def _nymex_henry_hub(year: int, month: int) -> dt.date:
    """One US business day before NYMEX Henry Hub's last trading day.

    NYMEX's natural gas contract for M+1 stops three business days before the
    first calendar day of M+1.
    """
    ny, nm = _next_month(year, month)
    nymex = _us_business_days_before(dt.date(ny, nm, 1), 3)
    return _us_business_days_before(nymex, 1)


def _fifth(year: int, month: int) -> dt.date:
    return dt.date(year, month, 5)


def _last_day(year: int, month: int) -> dt.date:
    ny, nm = _next_month(year, month)
    return dt.date(ny, nm, 1) - dt.timedelta(days=1)


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    d = _last_day(year, month)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


# NSE moved index derivative expiries from Thursday to Tuesday for contracts
# expiring after 28 Aug 2025 (circular NSE/FAOP/68685, 23 Jun 2025). BANKNIFTY's
# monthly contracts expired on the last Wednesday from March to December 2024 --
# not in the circulars we found, but every one of those expiries in the F&O
# dataset is a Wednesday, and that dataset is the answer key in the tests.
_NSE_TUESDAY_FROM = (2025, 9)


def _nse_index(symbol: str) -> Callable[[int, int], dt.date]:
    def rule(year: int, month: int) -> dt.date:
        if (year, month) >= _NSE_TUESDAY_FROM:
            return _last_weekday(year, month, 1)
        if symbol == "BANKNIFTY" and (2024, 3) <= (year, month) <= (2024, 12):
            return _last_weekday(year, month, 2)
        return _last_weekday(year, month, 3)

    return rule


_EXPIRY_RULES: dict[str, Callable[[int, int], dt.date]] = {
    "nymex_wti": _nymex_wti,
    "nymex_henry_hub": _nymex_henry_hub,
    "fifth": _fifth,
    "last_day": _last_day,
}


# Expiries no rule predicts, read off Kite's continuous series: on each of these
# the open interest jumped 16-100x on the day after, so the switch to the next
# contract is unmistakable. All three fell in a Diwali or Gurpurab week, when MCX
# runs evening-only sessions that its expiry rules evidently do not count as
# business days. A rule cannot see that from daily candles, so they are listed.
_KNOWN_EXCEPTIONS: dict[tuple[str, int, int], dt.date] = {
    ("SILVERM", 2020, 11): dt.date(2020, 11, 27),
    ("GOLDM", 2021, 11): dt.date(2021, 11, 3),
    ("GOLDM", 2024, 11): dt.date(2024, 10, 31),
}


def nominal_expiry(symbol: str, year: int, month: int) -> dt.date:
    """The day the rule names for the (year, month) contract, before Indian holidays."""
    c = contract(symbol)
    if month not in c.months:
        raise ValueError(f"{c.symbol} has no contract for month {month}")
    if (c.symbol, year, month) in _KNOWN_EXCEPTIONS:
        return _KNOWN_EXCEPTIONS[(c.symbol, year, month)]
    if c.expiry_rule == "nse_index":
        return _nse_index(c.symbol)(year, month)
    return _EXPIRY_RULES[c.expiry_rule](year, month)


def expiry_dates(
    symbol: str,
    trading_days: Iterable[dt.date],
    listed: Iterable[dt.date] = (),
) -> list[dt.date]:
    """Every contract expiry that falls inside `trading_days`, as an actual trading day.

    `trading_days` are the days the market really traded, read from the price
    data. A rule that lands on a day missing from them -- a holiday, or a gap in
    the data -- moves back to the previous day present: closing a day early is
    the safe direction to be wrong in, holding across the switch to the next
    contract is not.

    `listed` are expiries the exchange has already published (Kite's instrument
    list, for contracts still trading). They replace the rule for their month,
    because a published date already accounts for holidays the price data
    cannot show yet -- on the morning of an expiry moved forward by tomorrow's
    holiday, the data does not know about tomorrow.
    """
    days = sorted(set(trading_days))
    if not days:
        return []
    c = contract(symbol)
    by_month = {(d.year, d.month): d for d in listed}
    first, last = days[0], days[-1]
    day_set = set(days)

    out: list[dt.date] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        if month in c.months:
            d = by_month.get((year, month)) or nominal_expiry(c.symbol, year, month)
            if d <= last:
                while d not in day_set and d >= first:
                    d -= dt.timedelta(days=1)
                if d >= first:
                    out.append(d)
        year, month = _next_month(year, month)
    return out


def expiry_sessions(
    symbol: str, index: pd.DatetimeIndex, listed: Iterable[dt.date] = ()
) -> set[dt.date]:
    """`expiry_dates` for a price series, judged by its own trading days."""
    from nlt.data.session import trading_days

    return set(expiry_dates(symbol, trading_days(index, contract(symbol).session), listed))


def expiry_close_mask(
    symbol: str,
    index: pd.DatetimeIndex,
    timeframe: str,
    listed: Iterable[dt.date] = (),
    last_bar_of_session=None,
):
    """True on each candle where a contract stops trading.

    The expiry day's candle on daily bars; on intraday bars, the last candle of
    the expiry day's session. `last_bar_of_session` may be passed in when the
    caller knows better than the data -- a paper run mid-session, where today's
    newest candle is not the day's last.
    """
    import numpy as np

    from nlt.data.session import is_last_bar_of_session, session_date

    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    session = contract(symbol).session
    days = expiry_sessions(symbol, index, listed)
    on_expiry_day = np.array([session_date(ts, session) in days for ts in index])
    if timeframe == "1d":
        return on_expiry_day
    if last_bar_of_session is None:
        last_bar_of_session = is_last_bar_of_session(index, session).to_numpy()
    return on_expiry_day & np.asarray(last_bar_of_session, dtype=bool)
