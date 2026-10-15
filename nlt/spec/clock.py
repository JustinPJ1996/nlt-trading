"""A strategy's settings in "minutes before market close", and back.

The spec stores clock times (`Schedule.no_entry_after`, `Schedule.square_off`),
which is what the engines compare candles against. The settings screen (Justin,
2026-10-15) speaks in minutes before the close of the market the strategy
trades -- each market's own close:

  * stocks and the index itself: 15:30;
  * NSE/BSE futures and options: 15:40, since 3 Aug 2026 (`NSE_FNO`);
  * MCX: 23:30, the EARLIER of its two closes (23:55 in winter), so a time
    set for it never falls after the market has shut. In winter that is 25
    minutes earlier than the minutes suggest; the screen says so.

Converting through one module keeps the screen, the defaults and the readback
agreeing about what "15 minutes before close" means for each market.
"""

from __future__ import annotations

import datetime as dt

from nlt.data.session import NSE_EQUITY, NSE_FNO

MCX_EARLIER_CLOSE = dt.time(23, 30)

# Justin's defaults (2026-10-15), when the sentence does not say.
DEFAULT_SQUARE_OFF_MINUTES = 15
DEFAULT_CUTOFF_MINUTES = 30

# The settings screen's limits (from the screen Justin supplied).
SQUARE_OFF_MINUTES = (10, 30)
CUTOFF_MINUTES = (10, 400)


def _is_mcx(trade_as: str, symbol: str) -> bool:
    if trade_as != "future":
        return False
    from nlt.data.futures import is_commodity

    return is_commodity(symbol)


def market_hours(trade_as: str, symbol: str) -> tuple[dt.time, dt.time]:
    """(open, close) the settings count from, for an instrument."""
    if _is_mcx(trade_as, symbol):
        from nlt.data.session import MCX

        return MCX.open_time, MCX_EARLIER_CLOSE
    if trade_as in ("option", "future"):
        return NSE_FNO.open_time, NSE_FNO.close_time
    return NSE_EQUITY.open_time, NSE_EQUITY.close_time


def market_close(trade_as: str, symbol: str) -> dt.time:
    return market_hours(trade_as, symbol)[1]


def before_close(trade_as: str, symbol: str, minutes: int) -> dt.time:
    """The clock time `minutes` before that market's close."""
    close = dt.datetime.combine(dt.date(2000, 1, 3), market_close(trade_as, symbol))
    return (close - dt.timedelta(minutes=minutes)).time()


def minutes_before_close(trade_as: str, symbol: str, clock: dt.time) -> int:
    """How many minutes before that market's close `clock` is."""
    day = dt.date(2000, 1, 3)
    close = dt.datetime.combine(day, market_close(trade_as, symbol))
    return int((close - dt.datetime.combine(day, clock)).total_seconds() // 60)


def default_times(trade_as: str, symbol: str) -> tuple[dt.time, dt.time]:
    """(no new entries after, square off at) when the sentence does not say."""
    return (
        before_close(trade_as, symbol, DEFAULT_CUTOFF_MINUTES),
        before_close(trade_as, symbol, DEFAULT_SQUARE_OFF_MINUTES),
    )
