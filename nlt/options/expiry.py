"""Choosing which expiry a signal means: `nearest_weekly`, `next_weekly`, `monthly`.

Everything here reasons about the expiries **actually listed in the data on
the signal day** -- never a fixed weekday rule -- because that is the only way
to get BANKNIFTY right. NSE stopped listing BANKNIFTY weekly options after the
13 November 2024 expiry (monthly-only from 27 November 2024 onwards); a rule
hard-coded to "Thursday" or "every 7 days" would happily keep inventing
BANKNIFTY weeklies that were never tradable. Verified directly from
`data/fno/1d`: every listed BANKNIFTY expiry from 27 Nov 2024 onwards is a
calendar month's last expiry, and the gap between consecutive listed expiries
jumps from ~7 days to ~28-35 days at that point.

**Detecting "does a weekly exist" from the data, not a hard-coded date.** The
primary signal is spacing: if the two nearest upcoming listed expiries sit
close together (within `WEEKLY_GAP_MAX_DAYS`), a weekly series is live --
this alone is enough for `next_weekly` everywhere, including on the very last
weekly day itself (13 Nov 2024's own next listed expiry, 27 Nov, is 14 days
out, correctly reading as "no next weekly").

`nearest_weekly` has one case the spacing signal cannot see on its own: on
the *last* weekly expiry day, the day's own nearest listed expiry (itself) is
a genuine weekly contract, but by construction nothing else short-dated is
listed any more, so the two-nearest-expiries gap looks exactly like a
monthly-only day's. `BANKNIFTY_WEEKLY_END` is a documented, stated fallback
used only to break that one ambiguity -- never as the primary rule, and never
for `next_weekly`, which the spacing signal already resolves correctly.
"""

from __future__ import annotations

import datetime as dt

# A deliberately generous threshold: real weekly spacing is 7 days (or less
# across a holiday-shortened week); real month-to-month spacing is never under
# three weeks. Anything comfortably between the two still reads as "weekly".
WEEKLY_GAP_MAX_DAYS = 10

# Stated fallback, used only to disambiguate `nearest_weekly` on the boundary
# day itself -- see module docstring. Verified directly from the data
# (`data/fno/1d/underlying=BANKNIFTY`): 13 Nov 2024 is the last day any
# BANKNIFTY expiry closer than a month out is listed.
BANKNIFTY_WEEKLY_END = dt.date(2024, 11, 13)

REASON_NO_LISTING = "no_listing"
REASON_NO_WEEKLY = "no_weekly"


def _spacing_says_weekly(upcoming: list[dt.date]) -> bool:
    """True if the two nearest upcoming listed expiries are weekly-spaced."""
    if len(upcoming) < 2:
        return False
    return (upcoming[1] - upcoming[0]).days <= WEEKLY_GAP_MAX_DAYS


def _fallback_weekly(day: dt.date, underlying: str) -> bool:
    """Stated fallback for the one case spacing cannot see -- see module docstring."""
    if underlying.upper() == "BANKNIFTY":
        return day <= BANKNIFTY_WEEKLY_END
    return True


def pick_expiry(
    listed_expiries: list[dt.date], day: dt.date, kind: str, underlying: str
) -> tuple[dt.date | None, str | None]:
    """The expiry `kind` means on `day`, or `(None, reason_code)` if there isn't one.

    `listed_expiries` must already be everything listed *as of `day`* --
    callers get this from `nlt.options.data.listing(underlying, day)['expiry']`,
    which by construction cannot see a later day's listing.
    """
    if kind not in ("nearest_weekly", "next_weekly", "monthly"):
        raise ValueError(f"unknown expiry kind {kind!r}")

    all_expiries = sorted(set(listed_expiries))
    upcoming = [e for e in all_expiries if e >= day]

    if kind == "nearest_weekly":
        if not upcoming:
            return None, REASON_NO_LISTING
        if _spacing_says_weekly(upcoming) or _fallback_weekly(day, underlying):
            return upcoming[0], None
        return None, REASON_NO_WEEKLY

    if kind == "next_weekly":
        if not upcoming:
            return None, REASON_NO_LISTING
        if not _spacing_says_weekly(upcoming):
            # Either genuinely monthly-only, or (the boundary day itself) no
            # second weekly is listed to be "next" -- both are "no weekly"
            # from `next_weekly`'s point of view, never silently the monthly.
            return None, REASON_NO_WEEKLY
        return upcoming[1], None

    # monthly: the last listed expiry in `day`'s calendar month, or -- if that
    # has already passed relative to `day` -- the next month that has any
    # listed expiry at all, and its last one.
    this_month = [e for e in all_expiries if (e.year, e.month) == (day.year, day.month)]
    if this_month and max(this_month) >= day:
        return max(this_month), None

    future_months = sorted({(e.year, e.month) for e in upcoming})
    for year, month in future_months:
        if (year, month) == (day.year, day.month):
            continue
        candidates = [e for e in upcoming if (e.year, e.month) == (year, month)]
        return max(candidates), None

    return None, REASON_NO_LISTING
