#!/usr/bin/env python
"""One pass over every running paper strategy. Run by cron every minute.

    .venv/bin/python scripts/paper_run.py           # only runs that are due
    .venv/bin/python scripts/paper_run.py --force   # every running paper run, now

Installed by `scripts/install_paper_cron.sh`. Output goes to `logs/paper.log`.
A lock file stops two passes overlapping if one is slow.
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

from app.logic import charge_model_for_spec  # noqa: E402
from nlt.costs.charges import charge_fn  # noqa: E402
from nlt.data import kite  # noqa: E402
from nlt.data.kite import KiteSource  # noqa: E402
from nlt.paper import runner  # noqa: E402
from nlt.store.db import Store  # noqa: E402

LOCK = ROOT / "logs" / "paper.lock"


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


def _pass(*, force: bool) -> int:
    store = Store()
    now = pd.Timestamp.now(tz=runner.IST)
    source = KiteSource()

    def fetch(symbol, timeframe, start, end):
        return source.bars(symbol, timeframe, start, end)

    for run in store.active_paper_runs():
        try:
            spec = store.load_spec(run["strategy_id"])
            if not force and not runner.is_due(run, store.paper_status(run["id"]), spec, now):
                continue
            params = json.loads(run["params_json"] or "{}")
            model, _ = charge_model_for_spec(spec, params.get("cost_model_label", ""))
            result = runner.step(
                store,
                run,
                fetch=fetch,
                charge_fn=charge_fn(model),
                now=now,
                listed_expiries=kite.listed_expiries,
            )
            print(
                f"{now:%Y-%m-%d %H:%M:%S} run {run['id']} {result.health}: {result.message}"
                + (f" new: {', '.join(result.new_events)}" if result.new_events else "")
            )
        except Exception as exc:  # one broken run must not stop the others
            store.set_paper_status(run["id"], "error", f"The paper runner hit an error: {exc}")
            print(f"{now:%Y-%m-%d %H:%M:%S} run {run['id']} ERROR {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
