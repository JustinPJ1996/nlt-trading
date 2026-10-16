"""Implied volatility from the at-the-money options' own prices, and its percentile.

Justin's call: "IV" in a strategy such as "sell the NIFTY straddle only on days
when IV percentile is above 70" is worked out from the options themselves --
never India VIX. India VIX is used only as an outside check that the series
here is sane (see `scripts/build_iv.py`).

## One number per trading day: `daily_atm_iv`

For each session D, at that session's close:

- **Which expiry.** The nearest expiry listed on D whose last trading day is
  *after* D (a listed expiry falling on a market holiday really ends the
  session before; `_last_trading_day`, and time to expiry runs to 15:30 of
  that real last day). On an
  expiry day the contract expiring that afternoon is skipped for the next one:
  with minutes left to run, its price is almost all intrinsic value and
  bid/ask noise, and the volatility worked out from it swings wildly (tens of
  percent either way) for reasons that have nothing to do with the market's
  mood. For NIFTY and SENSEX that is the nearest weekly. For BANKNIFTY it is
  the nearest *monthly* for the whole history (Justin, 2026-10-16): its
  weeklies ended after 13 Nov 2024, and a series that switched from weekly
  to monthly options there would rank one against the other.
- **Which strike.** The listed strike nearest the index's close, a tie to the
  lower strike -- the same rule as `nlt.options.contracts`.
- **Which price.** The option's last *traded* 1-minute close at or before the
  session's last minute (the 15:29 candle, i.e. 15:30, on a normal day; a
  Saturday special session or a Muhurat session ends at its own time). Only
  rows passing `traded_mask` count: the vendor fills a minute with no trade
  with the previous price (volume 0), and such a minute is a price nobody
  paid at that time. The index level used is the index's close at that same
  minute, so price and level are a matched pair even when the option last
  traded a few minutes before the bell.
- **Which number.** The average of the call's and the put's IV. If only one
  of the two traded (or only one could be solved), that one is used and the
  row says so in `legs`. Averaging the two also makes the assumed interest
  rate (and the index's small dividend yield, which is ignored) all but
  irrelevant: they push the call's IV and the put's IV in opposite
  directions by nearly the same amount.

## The percentile: `iv_percentile`

The value for session D is the percentage of the `lookback_days` sessions'
IVs **before** D-1 that are strictly below D-1's IV, so it runs 0 to 100
(until 2026-10-16 D-1 was in its own window and the top was 99.6). D's own
IV is never used for D: a strategy deciding at 09:20 on D cannot know D's
15:30 close. Days with no IV are left out of the count (the window is the
last `lookback_days` days that *have* an IV). If D-1 has no IV, D uses the
most recent percentile from up to `STALE_SESSIONS` (5) sessions back -- a
real past value -- and the report counts those days; with none that recent,
or fewer than `lookback_days` earlier days of IV, D gets NaN.

With data from December 2022 and a 252-day window, the first percentile is
about a year in: December 2023 for NIFTY and BANKNIFTY; SENSEX options start
July 2024, so its first is August 2025.

## The cache

Working this out reads two contracts' minutes for every session -- a few
minutes per index. `scripts/build_iv.py` writes the result to
`data/fno/iv/underlying=U.parquet`; `daily_atm_iv` reads that file when it
exists (and was written by the current method, see `METHOD_VERSION`) and
the source is the real dataset. A test's in-memory `FrameSource` never reads
it.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

from nlt.data.source import IST
from nlt.options import data as option_data
from nlt.options.contracts import _nearest_strike
from nlt.options.data import OptionSource, ParquetSource

# The risk-free rate assumed in Black-Scholes. 6.5% was the RBI repo rate
# from February 2023 to February 2025, most of the dataset's span (it was
# 6.25% at the start, and cut to 5.5% during 2025). A constant is good
# enough here: for an at-the-money weekly the rate moves the call's IV and the
# put's IV by nearly equal and opposite amounts, and this module averages the
# two. Checked on three real days (NIFTY 7 Jun 2024, BANKNIFTY 12 Mar 2025,
# SENSEX 3 Oct 2024): moving the rate a whole percentage point either way
# changes the averaged IV by at most 0.011 of a volatility point.
RATE = 0.065

# Options on NSE and BSE indices expire at the 15:30 close of expiry day.
# Time to expiry is in calendar years (365 days), the Black-Scholes convention
# India VIX also uses (it counts minutes to expiry over a 365-day year).
EXPIRY_CLOSE = dt.time(15, 30)
_YEAR = pd.Timedelta(days=365)

# Bisection bounds for the volatility solve: 0.01% to 500% a year. An index
# option's IV has never been anywhere near either end; a price outside what
# these bounds can produce is reported as "no IV" rather than clamped.
_VOL_LOW = 1e-4
_VOL_HIGH = 5.0
_VOL_TOL = 1e-10

# Bumped whenever the way `daily_atm_iv` works changes, so an old cache file
# is ignored rather than silently trusted.
METHOD_VERSION = 3

IV_COLUMNS = [
    "iv",
    "legs",
    "reason",
    "expiry",
    "expires",
    "strike",
    "spot_close",
    "call_iv",
    "put_iv",
    "call_price",
    "put_price",
    "call_time",
    "put_time",
    "call_spot",
    "put_spot",
]

LEGS_BOTH = "both"
LEGS_CALL = "call only"
LEGS_PUT = "put only"
LEGS_NONE = "none"


# ------------------------------------------------------------- Black-Scholes


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    spot: float, strike: float, years: float, vol: float, option_type: str, rate: float = RATE
) -> float:
    """Black-Scholes price of a European call ("CE") or put ("PE"), no dividends."""
    if option_type not in ("CE", "PE"):
        raise ValueError(f"option_type must be 'CE' or 'PE', not {option_type!r}")
    discount = math.exp(-rate * years)
    root = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * years) / root
    d2 = d1 - root
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    days_to_expiry: float,
    option_type: str,
    rate: float = RATE,
) -> float | None:
    """The volatility (0.15 = 15% a year) at which Black-Scholes gives `price`.

    `days_to_expiry` is in calendar days (fractions allowed) to the 15:30
    close of expiry day. Returns None -- never a guess -- when there is no
    answer: the price is at or below what the option is worth with no
    volatility at all (its discounted intrinsic value), the price is more than
    500% volatility could explain, or an input is not a positive number.

    Solved by bisection: the Black-Scholes price rises strictly with
    volatility, so halving the bracket always converges and can never wander
    off the way a Newton step can when vega is tiny.
    """
    if option_type not in ("CE", "PE"):
        raise ValueError(f"option_type must be 'CE' or 'PE', not {option_type!r}")
    values = (price, spot, strike, days_to_expiry)
    if any(v is None or not math.isfinite(v) or v <= 0 for v in values):
        return None
    years = days_to_expiry / 365.0
    discount = math.exp(-rate * years)
    if option_type == "CE":
        floor = max(spot - strike * discount, 0.0)
    else:
        floor = max(strike * discount - spot, 0.0)
    if price <= floor:
        return None

    low, high = _VOL_LOW, _VOL_HIGH
    if price < bs_price(spot, strike, years, low, option_type, rate):
        return None
    if price > bs_price(spot, strike, years, high, option_type, rate):
        return None
    while high - low > _VOL_TOL:
        mid = 0.5 * (low + high)
        if bs_price(spot, strike, years, mid, option_type, rate) < price:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


# ------------------------------------------------------------ one IV per day


def _years_to_expiry(at: pd.Timestamp, expiry: dt.date) -> float:
    close = pd.Timestamp(dt.datetime.combine(expiry, EXPIRY_CLOSE), tz=IST)
    return (close - at) / _YEAR


def _last_traded(bars: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    """The last traded minute at or before `cutoff` (a candle's start time)
    and its close. Forward-filled minutes (volume 0) are never a price."""
    if bars.empty:
        return None
    traded = bars[bars["traded"].astype(bool) & (bars.index <= cutoff)]
    if traded.empty:
        return None
    stamp = traded.index[-1]
    return stamp, float(traded["close"].iloc[-1])


def _empty_row(reason: str) -> dict:
    row = dict.fromkeys(IV_COLUMNS, None)
    row.update(iv=np.nan, legs=LEGS_NONE, reason=reason)
    return row


# Justin, 2026-10-16: BANKNIFTY's IV comes from its monthly options for the
# whole history. Its weeklies ended after 13 Nov 2024, so a nearest-expiry
# series switched from weekly to monthly options there, and the percentile
# compared the two -- not like with like.
MONTHLY_ONLY = frozenset({"BANKNIFTY"})


def _monthly(expiries: set[dt.date]) -> set[dt.date]:
    """The last listed expiry of each calendar month: the monthly contracts."""
    last: dict[tuple[int, int], dt.date] = {}
    for e in expiries:
        key = (e.year, e.month)
        if key not in last or e > last[key]:
            last[key] = e
    return set(last.values())


def _last_trading_day(expiry: dt.date, sessions: list[dt.date]) -> dt.date:
    """The session a contract really expires on: its listed expiry, or the
    session before it when that date is a market holiday.

    The dataset records a few expiries at their *nominal* date even though
    the exchange moved them a day earlier for a holiday: NIFTY and BANKNIFTY
    30 Mar 2023 (traded until 29 Mar) and 29 Jun 2023 (until 28 Jun). Taken
    at face value, 29 Mar 2023 would look like "one day to go" when the
    contract was in its last hour (the put was quoted at Rs 0.05). The
    holiday list is published a year ahead, so knowing it is not hindsight.
    An expiry past the last session in the data is taken as listed.
    """
    if not sessions or expiry > sessions[-1]:
        return expiry
    earlier = [s for s in sessions if s <= expiry]
    return earlier[-1] if earlier else expiry


def _day_iv(
    underlying: str,
    day: dt.date,
    index_minutes: pd.DataFrame,
    source: OptionSource,
    sessions: list[dt.date] | None = None,
) -> dict:
    """The ATM IV row for one session -- see the module docstring for every rule.

    `sessions` is the trading calendar (every day with index minutes, ideally
    reaching past `day` by a few weeks), used only to find a holiday-shifted
    expiry's real last day.
    """
    if index_minutes.empty:
        return _empty_row("no index prices")
    cutoff = index_minutes.index[-1]  # the 15:29 candle, closing 15:30, on a normal day
    spot_close = float(index_minutes["close"].iloc[-1])

    listing = source.listing(underlying, day)
    if listing.empty:
        return _empty_row("no options listed")
    sessions = sessions or []
    expiries = set(listing["expiry"])
    if underlying in MONTHLY_ONLY:
        expiries = _monthly(expiries)
    later = sorted(
        (_last_trading_day(e, sessions), e)
        for e in expiries
        if _last_trading_day(e, sessions) > day
    )
    if not later:
        return _empty_row("no expiry after this day listed")
    expires, expiry = later[0]
    chain = listing[listing["expiry"] == expiry]
    strike = _nearest_strike(sorted(chain["strike"].unique().tolist()), spot_close)

    row = _empty_row("")
    row.update(expiry=expiry, expires=expires, strike=strike, spot_close=spot_close)

    ivs: dict[str, float] = {}
    missing: list[str] = []
    for option_type, name in (("CE", "call"), ("PE", "put")):
        match = chain[(chain["strike"] == strike) & (chain["instrument_type"] == option_type)]
        if match.empty:
            missing.append(f"{name} not listed")
            continue
        symbol = match["tradingsymbol"].iloc[0]
        last = _last_traded(source.contract_bars(underlying, symbol, "1m", day, day), cutoff)
        if last is None:
            missing.append(f"{name} did not trade")
            continue
        stamp, price = last
        # The index at the same minute, so price and level are a matched pair.
        spot = float(index_minutes["close"].asof(stamp))
        price_time = stamp + pd.Timedelta(minutes=1)  # a 1m candle's close
        iv = implied_vol(
            price, spot, strike, _years_to_expiry(price_time, expires) * 365.0, option_type
        )
        row.update(
            {
                f"{name}_price": price,
                f"{name}_time": price_time,
                f"{name}_spot": spot,
                f"{name}_iv": iv,
            }
        )
        if iv is None:
            missing.append(f"{name} price below what it is worth at zero volatility")
            continue
        ivs[name] = iv

    if "call" in ivs and "put" in ivs:
        row.update(iv=(ivs["call"] + ivs["put"]) / 2, legs=LEGS_BOTH)
    elif "call" in ivs:
        row.update(iv=ivs["call"], legs=LEGS_CALL)
    elif "put" in ivs:
        row.update(iv=ivs["put"], legs=LEGS_PUT)
    row["reason"] = "; ".join(missing)
    return row


def cache_path(underlying: str) -> Path:
    """`data/fno/iv/underlying=U.parquet`, under whatever `FNO_ROOT` is now."""
    return option_data.FNO_ROOT / "iv" / f"underlying={underlying}.parquet"


def _read_cache(underlying: str) -> pd.DataFrame | None:
    path = cache_path(underlying)
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    if "method_version" not in frame.columns or set(frame["method_version"]) != {METHOD_VERSION}:
        return None
    frame = frame.drop(columns=["method_version"])
    # Stored as a timestamp (Parquet has no plain date); handed back as the
    # same `datetime.date` a fresh calculation gives.
    for column in ("expiry", "expires"):
        frame[column] = [e.date() if pd.notna(e) else None for e in frame[column]]
    return frame


def write_cache(underlying: str, frame: pd.DataFrame) -> Path:
    path = cache_path(underlying)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["method_version"] = METHOD_VERSION
    for column in ("expiry", "expires"):
        out[column] = pd.to_datetime(out[column])
    out.to_parquet(path)
    return path


def daily_atm_iv(
    underlying: str,
    start: dt.date,
    end: dt.date,
    source: OptionSource | None = None,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """One row per session from `start` to `end`: the ATM IV at that session's close.

    Indexed by date (`pd.Timestamp` at midnight IST, like daily bars). `iv` is
    a fraction (0.12 = 12%), NaN when there is none; `legs` says whether it
    is the call-and-put average or only one side; `reason` says why a side
    or the whole day is missing. The other columns are the evidence: expiry,
    strike, the index close, each side's price, the minute it traded, the
    index at that minute, and each side's own IV.

    Expiry: the nearest listed after the day -- the nearest weekly for NIFTY
    and SENSEX, and for BANKNIFTY the nearest weekly until 13 Nov 2024 and the
    nearest *monthly* afterwards (its weeklies ended then). On an expiry day
    the next expiry is used; see the module docstring for why.

    A session is a day with index minutes. With `source` omitted, or the real
    `ParquetSource`, the cache from `scripts/build_iv.py` is used when present.
    """
    if source is None:
        source = ParquetSource()
    if use_cache and isinstance(source, ParquetSource):
        cached = _read_cache(underlying)
        if cached is not None:
            days = cached.index.date
            return cached[(days >= start) & (days <= end)]

    # Reaches a few weeks past `end` only to know the trading calendar (for a
    # holiday-shifted expiry); no price after `end` is used.
    minutes = source.underlying_bars(underlying, "1m", start, end + dt.timedelta(days=45))
    sessions = sorted(set(minutes.index.date))
    minutes = minutes[minutes.index.date <= end]
    rows: list[dict] = []
    stamps: list[pd.Timestamp] = []
    for day, day_minutes in minutes.groupby(minutes.index.date):
        rows.append(_day_iv(underlying, day, day_minutes, source, sessions))
        stamps.append(pd.Timestamp(day, tz=IST))
    frame = pd.DataFrame(rows, columns=IV_COLUMNS, index=pd.DatetimeIndex(stamps, name="date"))
    frame["iv"] = frame["iv"].astype(float)
    return frame


# ---------------------------------------------------------------- percentile


def iv_percentile(daily_iv: pd.Series, lookback_days: int) -> pd.Series:
    """The IV percentile each session may use, from the sessions before it only.

    `daily_iv` is one value per session in date order (NaN where none). The
    result for session D is `100 * (number of the lookback_days IVs before
    D-1 that are strictly below D-1's IV) / lookback_days`, where D-1 is the
    session before D in `daily_iv` and missing IVs are skipped when counting
    back. If D-1 has no IV, the latest ranking from up to `STALE_SESSIONS`
    sessions back is used; NaN when there is none, and on the first session.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be at least 1")
    valid = daily_iv.dropna()
    values = valid.to_numpy(dtype=float)
    ranks = np.full(len(values), np.nan)
    # Ranked against the `lookback_days` IVs BEFORE it, never itself: until
    # 2026-10-16 the window included the day being ranked, which can never be
    # below itself, so the top was 99.6 rather than 100.
    for k in range(lookback_days, len(values)):
        window = values[k - lookback_days : k]
        ranks[k] = 100.0 * np.count_nonzero(window < values[k]) / lookback_days
    known_at_close = pd.Series(ranks, index=valid.index).reindex(daily_iv.index)
    # A session with no IV of its own does not blank the next one: the most
    # recent ranking, at most STALE_SESSIONS sessions back, carries on (a real
    # past value, never an invented one). Then known at that close, so usable
    # from the next session's open: moved on one session.
    return known_at_close.ffill(limit=STALE_SESSIONS - 1).shift(1)


# How many sessions old the IV percentile a day uses may be, when the days
# just before it had no IV (2026-10-16). Older than that, no percentile.
STALE_SESSIONS = 5


def iv_stale_days(daily_iv: pd.Series, lookback_days: int) -> pd.Series:
    """Sessions whose percentile came from an older session than the one
    before -- the day before had no IV. For the report's note."""
    valid = daily_iv.notna()
    used = iv_percentile(daily_iv, lookback_days).notna()
    return used & ~valid.shift(1, fill_value=False)


def iv_feature(
    underlying: str,
    index_bars: pd.DataFrame,
    lookback_days: int,
    *,
    daily_iv: pd.DataFrame | pd.Series | None = None,
) -> pd.Series:
    """The IV percentile on every candle of `index_bars`: each candle of day D
    gets D's value (from D-1's close and before). For the feature table column
    `nlt.engine.conditions.iv_percentile_column(lookback_days)`.

    `daily_iv` defaults to the whole of `daily_atm_iv` for `underlying` (the
    cache). It must cover the sessions before the first candle; a candle on a
    day missing from it gets NaN, never a neighbouring day's value.
    """
    if daily_iv is None:
        first = dt.date(2000, 1, 1)
        last = index_bars.index.max().date() if len(index_bars) else first
        daily_iv = daily_atm_iv(underlying, first, last)
    series = daily_iv["iv"] if isinstance(daily_iv, pd.DataFrame) else daily_iv
    by_day = iv_percentile(series, lookback_days)
    lookup = pd.Series(by_day.to_numpy(), index=[d.date() for d in by_day.index])
    days = pd.Index(index_bars.index.date)
    out = pd.Series(
        lookup.reindex(days).to_numpy(dtype=float),
        index=index_bars.index,
        name=f"_iv_percentile_{lookback_days}",
    )
    # The candle days whose percentile was carried over from an older session
    # (the day before had no IV) -- for the report to count, never hidden.
    stale = iv_stale_days(series, lookback_days)
    stale_dates = {d.date() for d, flag in stale.items() if flag}
    out.attrs["stale_days"] = sorted(stale_dates & set(days))
    return out
