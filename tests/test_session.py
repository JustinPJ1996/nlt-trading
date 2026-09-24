"""Session calendar tests -- all against constructed indexes, no network.

The point of most of these is to prove the logic reads the session boundary
off the *data*, not off a hardcoded clock: a short (muhurat-style) session
must still get a correctly-flagged last bar, and a holiday (a weekday with no
bars at all) must not break anything that comes after it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from nlt.data.session import (
    MCX,
    NSE_EQUITY,
    Session,
    bars_per_session,
    drop_incomplete_sessions,
    filter_to_session,
    is_first_bar_of_session,
    is_last_bar_of_session,
    is_partial_last_bar,
    session_boundaries,
    session_date,
    trading_days,
)

IST = "Asia/Kolkata"


def _session_index(date: dt.date, start: str, end: str, freq_minutes: int) -> pd.DatetimeIndex:
    """Bars from `start` (inclusive) to `end` (inclusive), every `freq_minutes`."""
    start_ts = pd.Timestamp(f"{date} {start}", tz=IST)
    end_ts = pd.Timestamp(f"{date} {end}", tz=IST)
    return pd.date_range(start_ts, end_ts, freq=f"{freq_minutes}min")


def _full_nse_day(date: dt.date) -> pd.DatetimeIndex:
    """A complete 25-bar 15m NSE session: 09:15 through 15:15."""
    return _session_index(date, "09:15", "15:15", 15)


def _bars_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000.0] * n,
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# bars_per_session
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "timeframe, expected",
    [
        ("1m", 375),   # 375 minutes / 1
        ("5m", 75),    # 375 / 5
        ("15m", 25),   # 375 / 15
        ("30m", 13),   # ceil(375 / 30) = 12.5 -> 13
        ("1h", 7),     # ceil(375 / 60) = 6.25 -> 7
    ],
)
def test_bars_per_session_matches_arithmetic(timeframe, expected):
    minutes = (dt.datetime.combine(dt.date.today(), NSE_EQUITY.close_time)
               - dt.datetime.combine(dt.date.today(), NSE_EQUITY.open_time)).seconds // 60
    bar_len = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[timeframe]
    assert expected == -(-minutes // bar_len)
    assert bars_per_session(NSE_EQUITY, timeframe) == expected


def test_bars_per_session_unknown_timeframe_raises():
    with pytest.raises(ValueError):
        bars_per_session(NSE_EQUITY, "3m")


# ---------------------------------------------------------------------------
# first/last bar of session, over multiple days including a weekend
# ---------------------------------------------------------------------------


def test_first_and_last_bar_across_three_days_with_weekend():
    # Friday, then Monday (skip the weekend entirely -- simulates a holiday
    # gap too, since Sat/Sun never appear in the index at all).
    friday = dt.date(2026, 9, 25)
    monday = dt.date(2026, 9, 28)
    tuesday = dt.date(2026, 9, 29)

    index = _full_nse_day(friday).append(_full_nse_day(monday)).append(_full_nse_day(tuesday))
    first = is_first_bar_of_session(index, NSE_EQUITY)
    last = is_last_bar_of_session(index, NSE_EQUITY)

    assert first.sum() == 3
    assert last.sum() == 3
    for day_start in (0, 25, 50):
        assert first.iloc[day_start]
        assert not first.iloc[day_start + 1]
    for day_end in (24, 49, 74):
        assert last.iloc[day_end]
        assert not last.iloc[day_end - 1]


def test_short_muhurat_session_last_bar_is_not_at_1515():
    """The critical test: a half-day session's last bar must be flagged last
    even though it never reaches the normal 15:15 close. Proves the logic is
    derived from the data's own grouping, not a hardcoded clock time."""
    normal_day = dt.date(2026, 9, 21)
    muhurat_day = dt.date(2026, 9, 22)  # e.g. Diwali muhurat, 09:15-12:15 only

    index = _full_nse_day(normal_day).append(_session_index(muhurat_day, "09:15", "12:15", 15))
    last = is_last_bar_of_session(index, NSE_EQUITY)

    n_normal = 25
    n_muhurat = 13  # 09:15..12:15 inclusive, every 15 min
    assert len(index) == n_normal + n_muhurat

    # The muhurat day's last bar is 12:15, not 15:15 -- but it must still be
    # the one flagged True.
    assert last.iloc[n_normal + n_muhurat - 1]
    assert not last.iloc[n_normal + n_muhurat - 2]
    # And a naive "== 15:15" check would have wrongly flagged zero bars that day.
    muhurat_last_ts = index[n_normal + n_muhurat - 1]
    assert muhurat_last_ts.time() != dt.time(15, 15)


def test_short_session_last_bar_still_first_if_only_one_bar():
    idx = _session_index(dt.date(2026, 9, 22), "09:15", "09:15", 15)
    assert len(idx) == 1
    assert is_first_bar_of_session(idx, NSE_EQUITY).iloc[0]
    assert is_last_bar_of_session(idx, NSE_EQUITY).iloc[0]


# ---------------------------------------------------------------------------
# session_boundaries
# ---------------------------------------------------------------------------


def test_session_boundaries_true_exactly_on_first_bars():
    day1, day2 = dt.date(2026, 9, 21), dt.date(2026, 9, 22)
    index = _full_nse_day(day1).append(_full_nse_day(day2))
    boundaries = session_boundaries(index, NSE_EQUITY)

    assert boundaries.sum() == 2
    assert boundaries.iloc[0] and boundaries.iloc[25]
    assert not boundaries.iloc[1] and not boundaries.iloc[24] and not boundaries.iloc[26]


# ---------------------------------------------------------------------------
# filter_to_session
# ---------------------------------------------------------------------------


def test_filter_to_session_drops_pre_open_and_after_hours_bars():
    day = dt.date(2026, 9, 22)
    idx = pd.DatetimeIndex(
        [
            pd.Timestamp(f"{day} 09:00", tz=IST),  # pre-open
            pd.Timestamp(f"{day} 09:15", tz=IST),  # in session
            pd.Timestamp(f"{day} 12:00", tz=IST),  # in session
            pd.Timestamp(f"{day} 16:00", tz=IST),  # after hours
        ]
    )
    bars = _bars_frame(idx)
    cleaned, notes = filter_to_session(bars, NSE_EQUITY)

    assert len(cleaned) == 2
    assert list(cleaned.index) == [idx[1], idx[2]]
    joined_notes = " ".join(notes)
    assert "09:00" in joined_notes or str(idx[0]) in joined_notes
    assert "16:00" in joined_notes or str(idx[3]) in joined_notes
    assert len(notes) >= 1


def test_filter_to_session_no_drops_means_no_notes():
    idx = _full_nse_day(dt.date(2026, 9, 22))
    cleaned, notes = filter_to_session(_bars_frame(idx), NSE_EQUITY)
    assert len(cleaned) == len(idx)
    assert notes == []


def test_filter_to_session_drops_weekend_bars():
    saturday = dt.date(2026, 9, 26)
    idx = _session_index(saturday, "09:15", "10:00", 15)
    cleaned, notes = filter_to_session(_bars_frame(idx), NSE_EQUITY)
    assert cleaned.empty
    assert notes


# ---------------------------------------------------------------------------
# drop_incomplete_sessions
# ---------------------------------------------------------------------------


def test_drop_incomplete_sessions_removes_short_day_keeps_full_ones():
    full_day = dt.date(2026, 9, 21)
    partial_day = dt.date(2026, 9, 22)

    full_idx = _full_nse_day(full_day)
    partial_idx = _session_index(partial_day, "09:15", "10:45", 15)  # 7 of 25 bars
    assert len(partial_idx) == 7

    index = full_idx.append(partial_idx)
    bars = _bars_frame(index)
    cleaned, notes = drop_incomplete_sessions(bars, NSE_EQUITY, "15m")

    assert len(cleaned) == 25
    assert all(session_date(ts, NSE_EQUITY) == full_day for ts in cleaned.index)
    assert any("7/25" in n for n in notes)


def test_drop_incomplete_sessions_keeps_all_when_all_full():
    day1, day2 = dt.date(2026, 9, 21), dt.date(2026, 9, 22)
    index = _full_nse_day(day1).append(_full_nse_day(day2))
    cleaned, notes = drop_incomplete_sessions(_bars_frame(index), NSE_EQUITY, "15m")
    assert len(cleaned) == 50
    assert notes == []


# ---------------------------------------------------------------------------
# is_partial_last_bar
# ---------------------------------------------------------------------------


def test_is_partial_last_bar_true_mid_session():
    day = dt.date(2026, 9, 22)
    idx = _session_index(day, "09:15", "10:00", 15)  # last bar 10:00-10:15
    bars = _bars_frame(idx)
    now = pd.Timestamp(f"{day} 10:05", tz=IST)
    assert is_partial_last_bar(bars, NSE_EQUITY, "15m", now=now) is True


def test_is_partial_last_bar_false_after_bar_closes():
    day = dt.date(2026, 9, 22)
    idx = _session_index(day, "09:15", "10:00", 15)
    bars = _bars_frame(idx)
    now = pd.Timestamp(f"{day} 10:16", tz=IST)  # bar 10:00 closed at 10:15
    assert is_partial_last_bar(bars, NSE_EQUITY, "15m", now=now) is False


def test_is_partial_last_bar_empty_frame_is_false():
    empty = _bars_frame(pd.DatetimeIndex([], tz=IST))
    assert is_partial_last_bar(empty, NSE_EQUITY, "15m") is False


# ---------------------------------------------------------------------------
# session_date
# ---------------------------------------------------------------------------


def test_session_date_normal_nse_bar():
    ts = pd.Timestamp("2026-09-22 12:00", tz=IST)
    assert session_date(ts, NSE_EQUITY) == dt.date(2026, 9, 22)


def test_session_date_mcx_evening_bar_same_calendar_day():
    ts = pd.Timestamp("2026-09-22 23:00", tz=IST)
    assert session_date(ts, MCX) == dt.date(2026, 9, 22)


def test_session_date_overnight_session_late_bar_belongs_to_prior_day():
    # A hypothetical overnight session (close < open, spans midnight): a bar
    # at 00:30 belongs to the session that opened the previous evening.
    overnight = Session("overnight", dt.time(18, 0), dt.time(2, 0), (0, 1, 2, 3, 4))
    ts = pd.Timestamp("2026-09-23 00:30", tz=IST)
    assert session_date(ts, overnight) == dt.date(2026, 9, 22)


# ---------------------------------------------------------------------------
# holidays inferred from absence in the data
# ---------------------------------------------------------------------------


def test_holiday_weekday_absent_from_index_does_not_appear_in_trading_days():
    monday = dt.date(2026, 9, 21)
    # Tuesday 2026-09-22 deliberately omitted -- simulates a holiday.
    wednesday = dt.date(2026, 9, 23)

    index = _full_nse_day(monday).append(_full_nse_day(wednesday))
    days = trading_days(index, NSE_EQUITY)

    assert days == [monday, wednesday]
    assert dt.date(2026, 9, 22) not in days


def test_holiday_gap_does_not_break_boundary_detection():
    monday = dt.date(2026, 9, 21)
    wednesday = dt.date(2026, 9, 23)  # Tuesday skipped, as if a holiday
    index = _full_nse_day(monday).append(_full_nse_day(wednesday))

    first = is_first_bar_of_session(index, NSE_EQUITY)
    last = is_last_bar_of_session(index, NSE_EQUITY)

    assert first.sum() == 2
    assert last.sum() == 2
    assert first.iloc[0] and first.iloc[25]
    assert last.iloc[24] and last.iloc[49]
