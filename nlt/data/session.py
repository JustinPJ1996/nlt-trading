"""Market-session calendar for intraday bars.

Daily bars carry no clock: a timestamp of `2026-09-22` is unambiguously one
trading day, wherever it falls. Intraday bars do carry a clock, and that clock
is the whole source of the failure modes this module exists to prevent --
a position held past the close, an indicator smoothing straight across the
overnight gap as though 15:15 and the next day's 09:15 were consecutive bars,
a signal acted on before the final bar of the session has actually printed.

Everything here answers one of those questions from the *data itself* rather
than from a hardcoded clock or holiday table:

- Trading hours come from `Session` (a small, explicit config -- there is no
  way to infer "market opens at 09:15" from a parquet file that only starts
  recording at 09:15 anyway).
- Trading *days* -- which days had a session at all -- are read off the bars
  that are actually present. A day is a holiday if it has no bars covering
  session hours. That is indistinguishable, on purpose, from a day whose data
  simply never arrived: no hardcoded table can be relied on not to rot the
  moment the exchange announces a change, so we accept that a genuine
  data gap looks exactly like a holiday and say so wherever it matters
  (`drop_incomplete_sessions`, `trading_days`).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

_MINUTES_PER_BAR = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
}


@dataclass(frozen=True)
class Session:
    """One market's trading hours."""

    name: str
    open_time: dt.time
    close_time: dt.time
    weekdays: tuple[int, ...]  # 0 = Monday
    tz: str = "Asia/Kolkata"


NSE_EQUITY = Session("NSE equity & F&O", dt.time(9, 15), dt.time(15, 30), (0, 1, 2, 3, 4))
MCX = Session("MCX commodities", dt.time(9, 0), dt.time(23, 30), (0, 1, 2, 3, 4))


def _session_minutes(session: Session) -> int:
    """Length of the session in minutes, handling a session that spans midnight."""
    open_m = session.open_time.hour * 60 + session.open_time.minute
    close_m = session.close_time.hour * 60 + session.close_time.minute
    if close_m <= open_m:
        close_m += 24 * 60
    return close_m - open_m


def bars_per_session(session: Session, timeframe: str) -> int:
    """How many bars of `timeframe` a full session contains.

    NSE is 09:15-15:30, 375 minutes. 15m bars are labelled by their *start*
    (yfinance convention -- the 15:15 bar covers 15:15-15:30, the close), so a
    full session is ceil(375/15) = 25, not 375/15 = 25 exactly by luck: the
    general case rounds up because a session length that is not an exact
    multiple of the bar size still owns one final, shorter bar.
    """
    if timeframe not in _MINUTES_PER_BAR:
        raise ValueError(f"unknown intraday timeframe {timeframe!r}, expected one of "
                          f"{sorted(_MINUTES_PER_BAR)}")
    minutes = _session_minutes(session)
    bar_len = _MINUTES_PER_BAR[timeframe]
    return -(-minutes // bar_len)  # ceil division


def is_session_time(session: Session, ts: pd.Timestamp) -> bool:
    """Is this timestamp inside trading hours on a trading weekday?

    A bar timestamped 15:15 is inside the session (it is the last bar, covering
    15:15-15:30); a bar timestamped exactly at close (15:30, if one existed)
    would not be, since the session has already ended by the time it starts.
    """
    ts = _to_session_tz(ts, session)
    if ts.weekday() not in session.weekdays:
        return False
    t = ts.time()
    if session.close_time <= session.open_time:
        # Overnight session (not used by NSE/MCX today, but keep correct).
        return t >= session.open_time or t < session.close_time
    return session.open_time <= t < session.close_time


def _to_session_tz(ts: pd.Timestamp, session: Session) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tz is None:
        ts = ts.tz_localize(session.tz)
    else:
        ts = ts.tz_convert(session.tz)
    return ts


def session_date(ts: pd.Timestamp, session: Session = NSE_EQUITY) -> dt.date:
    """The trading date a timestamp belongs to.

    Not simply `ts.date()`: a session that runs past midnight (none of ours do
    today, but MCX has in the past and could again) would put its late bars on
    the *next* calendar date while they still belong to the session that
    opened the day before. A bar belongs to the trading day on which its
    session opened.
    """
    ts = _to_session_tz(ts, session)
    if session.close_time <= session.open_time and ts.time() < session.open_time:
        # Past midnight, before today's open: still part of yesterday's session.
        return (ts - pd.Timedelta(days=1)).date()
    return ts.date()


def _session_date_series(index: pd.DatetimeIndex, session: Session) -> pd.Series:
    """`session_date` applied elementwise, as a Series aligned to `index`."""
    dates = [session_date(ts, session) for ts in index]
    return pd.Series(dates, index=index)


def is_first_bar_of_session(index: pd.DatetimeIndex, session: Session) -> pd.Series:
    """Boolean series aligned to `index`: True on each session's opening bar.

    Derived from the data's own session grouping (first bar seen per trading
    date) rather than by comparing against a hardcoded open time, so a session
    that opens late or is otherwise irregular is still handled correctly.
    """
    dates = _session_date_series(index, session)
    return pd.Series(~dates.duplicated(keep="first").values, index=index)


def is_last_bar_of_session(index: pd.DatetimeIndex, session: Session) -> pd.Series:
    """Boolean series aligned to `index`: True on each session's closing bar.

    Derived the same way as `is_first_bar_of_session`: the last bar seen for a
    trading date, whatever time that happens to be. This is what makes a
    muhurat/half-day session work without special-casing it -- if the data for
    that day stops at 12:15, 12:15 is flagged as the last bar, not 15:15.
    """
    dates = _session_date_series(index, session)
    return pd.Series(~dates.duplicated(keep="last").values, index=index)


def session_boundaries(index: pd.DatetimeIndex, session: Session) -> pd.Series:
    """True on every bar that begins a new session.

    Identical to `is_first_bar_of_session` today; kept as a separate name
    because it is the concept indicators and the engine actually reach for --
    "where must I reset a rolling calculation" -- distinct from "is this bar
    tradeable as the session's open", even though the two currently coincide.
    """
    return is_first_bar_of_session(index, session)


def trading_days(index: pd.DatetimeIndex, session: Session) -> list[dt.date]:
    """Distinct trading dates present in `index`, in order.

    "Present in the index" is the only signal available -- see the module
    docstring on why holidays are inferred rather than hardcoded. A weekday
    with no bars at all (holiday, or a data gap) simply does not appear here.
    """
    dates = _session_date_series(index, session)
    seen: list[dt.date] = []
    for d in dates:
        if not seen or seen[-1] != d:
            seen.append(d)
    # dates may not be contiguous if index itself isn't sorted; normalise via set+sort
    return sorted(set(seen))


def filter_to_session(bars: pd.DataFrame, session: Session) -> tuple[pd.DataFrame, list[str]]:
    """Drop bars outside trading hours. Returns the cleaned frame and notes.

    Never drops silently: every bar removed is accounted for in the returned
    notes so a caller (or a test) can see exactly what happened and why.
    """
    notes: list[str] = []
    if bars.empty:
        return bars, notes

    mask = pd.Series(
        [is_session_time(session, ts) for ts in bars.index], index=bars.index
    )
    dropped = bars.index[~mask]
    if len(dropped) > 0:
        sample = ", ".join(str(t) for t in dropped[:5])
        more = f" and {len(dropped) - 5} more" if len(dropped) > 5 else ""
        notes.append(
            f"dropped {len(dropped)} bar(s) outside {session.name} hours "
            f"({session.open_time}-{session.close_time}): {sample}{more}"
        )
    return bars[mask], notes


def drop_incomplete_sessions(
    bars: pd.DataFrame, session: Session, timeframe: str
) -> tuple[pd.DataFrame, list[str]]:
    """Remove any session with fewer bars than a full session should have.

    "Fewer than expected" is judged only against `bars_per_session`, i.e.
    against the *full* session length -- there is no table of known short
    days to consult (see the module docstring). That means a genuinely short
    trading day (muhurat session, a declared early close) is indistinguishable
    from an in-progress live session and both get dropped here.

    That is a deliberate, documented trade-off, not an oversight: the failure
    mode this function exists to prevent -- trading a day that has not
    finished happening yet -- is far more costly than the failure mode it
    causes -- refusing to backtest a handful of half-day sessions a year. A
    caller who specifically wants to keep known short sessions (e.g. because
    it has an authoritative holiday calendar) should filter the notes this
    function returns and decide from there; this function does not guess.
    """
    notes: list[str] = []
    if bars.empty:
        return bars, notes

    expected = bars_per_session(session, timeframe)
    dates = _session_date_series(bars.index, session)
    counts = dates.value_counts()
    incomplete = counts[counts < expected].index

    if len(incomplete) == 0:
        return bars, notes

    for d in sorted(incomplete):
        notes.append(
            f"dropped session {d}: {counts[d]}/{expected} bars present "
            "(short session or in-progress/live data)"
        )

    keep_mask = ~dates.isin(set(incomplete))
    return bars[keep_mask.values], notes


def is_partial_last_bar(
    bars: pd.DataFrame, session: Session, timeframe: str, now: pd.Timestamp | None = None
) -> bool:
    """Is the final bar of `bars` still forming, i.e. unsafe to signal on?

    True whenever the bar's own close time is still in the future relative to
    `now` (default: real wall-clock time). This does not depend on how many
    bars the session has had so far, so it works equally for a full session in
    progress and for a short session before it has actually finished.
    """
    if bars.empty:
        return False

    now = _to_session_tz(now, session) if now is not None else pd.Timestamp.now(tz=session.tz)
    last_ts = _to_session_tz(bars.index[-1], session)
    bar_len = _MINUTES_PER_BAR.get(timeframe)
    if bar_len is None:
        raise ValueError(f"unknown intraday timeframe {timeframe!r}")
    bar_close = last_ts + pd.Timedelta(minutes=bar_len)
    return now < bar_close
