#!/usr/bin/env python
"""One pass over every running paper strategy and one-off order. Run by cron every minute.

    .venv/bin/python scripts/paper_run.py           # only runs that are due
    .venv/bin/python scripts/paper_run.py --force   # every running paper run, now

Installed by `scripts/install_paper_cron.sh`. Output goes to `logs/paper.log`.
A lock file stops two passes overlapping if one is slow.

Option runs (Phase 2 batch 3) need two more things, both done here:

  * Selling margin is worked out from the exchange's start-of-day file
    (`nlt.options.margin`). It is published the evening before; from 08:30 on a
    weekday, today's is fetched if missing (at most every 15 minutes).
  * Kite drops an option's prices once it expires, so each day's prices must be
    saved (`nlt.options.live`). Every pass keeps the minutes so far; a pass
    after 15:42 (when the day's last minute has closed) saves the whole day.
    The passes to 15:45 are the scheduled ones, and from 15:46 (`EVENING_FROM`)
    every option run gets one more pass each evening, retried until it
    succeeds, so a late or skipped minute does not lose an expiring contract's
    last day. If the box is off from the last pass until Kite drops the
    contract, the day is kept up to that pass and the run says so.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.logic import charge_model_for_spec, option_feed  # noqa: E402
from nlt.costs.charges import charge_fn  # noqa: E402
from nlt.data import kite  # noqa: E402
from nlt.data.kite import KiteSource  # noqa: E402
from nlt.orders import runner as orders_runner  # noqa: E402
from nlt.paper import runner  # noqa: E402
from nlt.store.db import Store  # noqa: E402

LOCK = ROOT / "logs" / "paper.lock"
STATE = ROOT / "logs" / "paper_state"
SPAN_FROM = pd.Timedelta(hours=8, minutes=30)
EVENING_FROM = pd.Timedelta(hours=15, minutes=46)
_SPAN_EXCHANGE = {"NIFTY": "NSE", "BANKNIFTY": "NSE", "SENSEX": "BSE"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore the market-hours check")
    args = parser.parse_args()

    LOCK.parent.mkdir(exist_ok=True)
    with open(LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(
                f"{pd.Timestamp.now(tz=runner.IST):%Y-%m-%d %H:%M:%S} previous pass still running"
            )
            return 0
        return _pass(force=args.force)


def _marker(name: str) -> Path:
    STATE.mkdir(parents=True, exist_ok=True)
    return STATE / name


def _fetch_todays_span(exchanges: set[str], now: pd.Timestamp) -> None:
    """Today's start-of-day margin file for each exchange a selling run needs."""
    import subprocess

    from nlt.options.margin import BOD_TAG

    today = now.date()
    for exchange in sorted(exchanges):
        stored = ROOT / "data" / "span" / exchange / f"{today.year}"
        if (stored / f"{today:%Y%m%d}-{BOD_TAG[exchange]}.npz").exists():
            continue
        tried = _marker(f"span-{exchange}")
        if tried.exists() and now.timestamp() - tried.stat().st_mtime < 15 * 60:
            continue
        tried.touch()
        done = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "fetch_span.py"), "--today"]
            + ["--exchange", exchange, "--retry-missing"],
            capture_output=True,
            text=True,
            timeout=600,
        )
        print(f"{now:%Y-%m-%d %H:%M:%S} margin file {exchange}: exit {done.returncode}")


def _pass(*, force: bool) -> int:
    store = Store()
    now = pd.Timestamp.now(tz=runner.IST)
    source = KiteSource()
    feed = None

    def fetch(symbol, timeframe, start, end):
        return source.bars(symbol, timeframe, start, end)

    runs = store.active_paper_runs()
    specs = {run["id"]: store.load_spec(run["strategy_id"]) for run in runs}
    weekday = now.weekday() < 5
    since_midnight = now - now.normalize()

    selling = {
        _SPAN_EXCHANGE[s.instrument.symbol]
        for s in specs.values()
        if s.instrument.is_option and s.instrument.legs
    }
    if selling and weekday and since_midnight >= SPAN_FROM:
        try:
            _fetch_todays_span(selling, now)
        except Exception as exc:  # the runs still go ahead; margin then says why
            print(f"{now:%Y-%m-%d %H:%M:%S} margin file ERROR {exc!r}")

    # Once each weekday evening: one more pass for every option run, to save
    # the day's option prices before an expiring contract disappears from Kite.
    evening = _marker(f"options-saved-{now.date()}")
    evening_pass = weekday and since_midnight >= EVENING_FROM and not evening.exists()
    evening_failed = False

    for run in runs:
        try:
            spec = specs[run["id"]]
            due = runner.is_due(run, store.paper_status(run["id"]), spec, now)
            if spec.instrument.is_option and evening_pass:
                due = True
            if not force and not due:
                continue
            if spec.instrument.is_option and feed is None:
                feed = option_feed()
            params = json.loads(run["params_json"] or "{}")
            model, _ = charge_model_for_spec(spec, params.get("cost_model_label", ""))
            result = runner.step(
                store,
                run,
                fetch=fetch,
                charge_fn=charge_fn(model),
                now=now,
                listed_expiries=kite.listed_expiries,
                option_feed=feed if spec.instrument.is_option else None,
            )
            if spec.instrument.is_option and result.health not in ("ok", "paused", "waiting"):
                evening_failed = True
            print(
                f"{now:%Y-%m-%d %H:%M:%S} run {run['id']} {result.health}: {result.message}"
                + (f" new: {', '.join(result.new_events)}" if result.new_events else "")
            )
        except Exception as exc:  # one broken run must not stop the others
            evening_failed = True
            store.set_paper_status(run["id"], "error", f"The paper runner hit an error: {exc}")
            print(f"{now:%Y-%m-%d %H:%M:%S} run {run['id']} ERROR {exc!r}")

    # A pass that failed (Kite down, say) is tried again next minute.
    if evening_pass and not evening_failed:
        evening.touch()

    # One-off conditional orders, all in one pass.
    if force or orders_runner.is_due(now):
        try:
            result = orders_runner.step(store, orders_runner.kite_feed(), now=now)
            if result.health != "idle":
                print(
                    f"{now:%Y-%m-%d %H:%M:%S} one-off orders {result.health}: {result.message}"
                    + (f" new: {', '.join(result.new_events)}" if result.new_events else "")
                )
        except Exception as exc:
            store.set_one_off_book(health="error", message=f"The order checker hit an error: {exc}")
            print(f"{now:%Y-%m-%d %H:%M:%S} one-off orders ERROR {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
