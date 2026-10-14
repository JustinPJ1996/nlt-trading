#!/usr/bin/env python
"""Download the exchanges' daily SPAN margin files and keep a compact copy.

    .venv/bin/python scripts/fetch_span.py                 # the whole backtest window
    .venv/bin/python scripts/fetch_span.py --today         # today's file (calibration)
    .venv/bin/python scripts/fetch_span.py --exchange BSE --start 2024-01-01 --end 2024-01-31

For each trading day it downloads ONE file per exchange -- the **begin-of-day**
file, the one a broker blocks margin from at 09:15:

- NSE: ``nsccl.YYYYMMDD.i1.zip``. NSE has no "bod" tag; its first file for a
  day (``i1``, stamped ``<isSetl>0``) is built the evening before -- e.g. the
  file for 14 Oct 2026 says ``<created>202610132128</created>`` -- so it
  cannot know anything that happened on the day itself.
- BSE: ``BSERISKYYYYMMDD-00.zip`` (BSE calls it "begin of day"; built ~00:50).

When a start-of-day file is missing (BSE skipped dozens in 2023), the previous
trading day's end-of-day file is stored instead (NSE ``s``, BSE ``FINAL``):
published that evening and built from the same closing prices. BSE's
intraday ``01``-``04`` files are never used -- they are made during the day.

Each 45-75 MB XML is cut down to NIFTY, BANKNIFTY and SENSEX options and
futures (``nlt.options.span_store``) and written to
``data/span/<NSE|BSE>/<year>/<YYYYMMDD>-<tag>.npz`` (~150-270 KB each). The
XML is never kept.

Polite and resumable: one download at a time per exchange, a pause between
requests, days already stored are skipped, and a day the exchange does not
have (HTTP 404, or an HTML page served with status 200) is recorded in
``data/span/<exchange>/missing.json`` and reported at the end.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import io
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nlt.options.span_store import parse_span_xml, save_day  # noqa: E402

DEFAULT_ROOT = REPO / "data" / "span"
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

# The backtest window per exchange (see data/fno/1d/_coverage.json).
WINDOWS = {
    "NSE": (dt.date(2022, 12, 1), dt.date(2025, 12, 31)),
    "BSE": (dt.date(2023, 7, 20), dt.date(2025, 12, 31)),
}
BOD_TAG = {"NSE": "i1", "BSE": "00"}
# End of day. Used only when a day's start-of-day file is missing: the previous
# trading day's end-of-day file is built from the same closing prices and was
# published that evening, so it is the closest file with no look-ahead.
EOD_TAG = {"NSE": "s", "BSE": "FINAL"}


def url_for(exchange: str, day: dt.date, tag: str) -> tuple[str, str]:
    ymd = day.strftime("%Y%m%d")
    if exchange == "NSE":
        name = f"nsccl.{ymd}.{tag}"
        return f"https://nsearchives.nseindia.com/archives/nsccl/span/{name}.zip", name
    name = f"BSERISK{ymd}-{tag}"
    return f"https://www.bseindia.com/bsedata/Risk_Automate/SPN/{name}.zip", name


def store_path(root: Path, exchange: str, day: dt.date, tag: str) -> Path:
    return root / exchange / f"{day.year}" / f"{day:%Y%m%d}-{tag}.npz"


def trading_days(exchange: str, start: dt.date, end: dt.date) -> list[dt.date]:
    """Days the market traded, from our own F&O dataset where it reaches;
    plain weekdays elsewhere (a weekday holiday then shows as a 404)."""
    days: set[dt.date] = set()
    try:
        import pandas as pd

        und = "NIFTY" if exchange == "NSE" else "SENSEX"
        frame = pd.read_parquet(
            REPO / "data" / "fno" / "1d" / f"underlying={und}", columns=["timestamp"]
        )
        days = {t.date() for t in pd.to_datetime(frame["timestamp"]).unique()}
    except Exception as exc:  # noqa: BLE001 -- a fresh clone has no dataset
        print(f"[{exchange}] no F&O dataset to read trading days from ({exc}); using weekdays")
    covered_to = max(days) if days else start - dt.timedelta(days=1)
    out = [d for d in days if start <= d <= end]
    d = max(start, covered_to + dt.timedelta(days=1))
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return sorted(out)


NOT_PUBLISHED = "not published"


def download(url: str) -> bytes | str:
    """The zip's bytes, or a short reason the exchange has no usable file."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    err: object = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                ctype = resp.headers.get("Content-Type", "")
                expected = int(resp.headers.get("Content-Length") or 0)
                body = resp.read()
            if not body.startswith(b"PK") or "html" in ctype.lower():
                err = NOT_PUBLISHED  # BSE answers missing days with HTML and 200
            elif expected and len(body) != expected:
                err = f"short read {len(body)} of {expected} bytes"
            else:
                try:
                    zipfile.ZipFile(io.BytesIO(body)).testzip()
                    return body
                except zipfile.BadZipFile as exc:
                    err = f"corrupt zip: {exc}"
        except urllib.error.HTTPError as exc:
            err = NOT_PUBLISHED if exc.code in (403, 404) else exc
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
        ) as exc:
            err = exc
        # Asked again even after a 404: under load BSE answers files it has
        # with "not found" (dozens of days in a first full run), so one refusal
        # is not evidence. Three, spaced out, are taken as the answer.
        time.sleep(15 * (attempt + 1))
    if err == NOT_PUBLISHED:
        return NOT_PUBLISHED
    # NSE serves a few archive files truncated, the same bytes every time.
    return f"unusable ({err})"


def extract_and_save(blob: bytes, exchange: str, name: str, out: str) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".spn")]
        if len(members) != 1:
            raise ValueError(f"{name}: expected one .spn in the zip, found {members}")
        xml = zf.read(members[0])
    day = parse_span_xml(io.BytesIO(xml), exchange, name)
    save_day(day, Path(out))
    return f"{name} -> {Path(out).name} ({Path(out).stat().st_size // 1024} KB, {len(day.kind)} contracts)"


def _report(exchange: str, item: tuple, missing: dict[str, str]) -> None:
    key, url, future = item
    try:
        print(f"[{exchange}] {future.result()}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- one bad file must not stop the run
        missing[key] = f"unreadable ({exc}): {url}"
        print(f"[{exchange}] {key}: unreadable ({exc})", flush=True)


def run(
    exchange: str, days: list[dt.date], root: Path, tag: str, pause: float, retry: bool
) -> list:
    missing_file = root / exchange / "missing.json"
    missing: dict[str, str] = json.loads(missing_file.read_text()) if missing_file.exists() else {}
    pending: list[tuple[str, str, Future]] = []
    with ProcessPoolExecutor(max_workers=2) as pool:
        for day in days:
            out = store_path(root, exchange, day, tag)
            if out.exists():
                continue
            key = f"{day.isoformat()}-{tag}"
            if key in missing and not retry:
                continue
            url, name = url_for(exchange, day, tag)
            blob = download(url)
            if isinstance(blob, str):
                missing[key] = f"{blob}: {url}"
                missing_file.parent.mkdir(parents=True, exist_ok=True)
                missing_file.write_text(json.dumps(missing, indent=1, sort_keys=True))
                print(f"[{exchange}] {day} {tag}: {blob}", flush=True)
            else:
                missing.pop(key, None)
                future = pool.submit(extract_and_save, blob, exchange, name, str(out))
                pending.append((key, url, future))
            for item in [i for i in pending if i[2].done()]:
                _report(exchange, item, missing)
                pending.remove(item)
            time.sleep(pause)
        for item in pending:
            _report(exchange, item, missing)
    missing_file.parent.mkdir(parents=True, exist_ok=True)
    missing_file.write_text(json.dumps(missing, indent=1, sort_keys=True))
    wanted = {f"{d.isoformat()}-{tag}" for d in days}
    return sorted(k for k in missing if k in wanted)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--exchange", choices=["NSE", "BSE", "both"], default="both")
    parser.add_argument("--start", type=dt.date.fromisoformat)
    parser.add_argument("--end", type=dt.date.fromisoformat)
    parser.add_argument("--today", action="store_true", help="only today's begin-of-day file")
    parser.add_argument(
        "--tag", help="a file other than begin-of-day (NSE s/i1..i5, BSE 00/02/03/04/FINAL)"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    parser.add_argument("--retry-missing", action="store_true")
    args = parser.parse_args()

    exchanges = ["NSE", "BSE"] if args.exchange == "both" else [args.exchange]
    for exchange in exchanges:
        tag = args.tag or BOD_TAG[exchange]
        if args.today:
            days = [dt.date.today()]
        else:
            start = args.start or WINDOWS[exchange][0]
            end = args.end or WINDOWS[exchange][1]
            days = trading_days(exchange, start, end)
        print(f"[{exchange}] {len(days)} days, tag {tag}", flush=True)
        missing = run(exchange, days, args.root, tag, args.pause, args.retry_missing)
        print(f"[{exchange}] done. Not published by the exchange: {len(missing)}")
        for key in missing:
            print(f"  {key}")
        if tag == BOD_TAG[exchange] and missing and not args.today:
            calendar = trading_days(exchange, days[0] - dt.timedelta(days=10), days[-1])
            before = sorted(
                {max(d for d in calendar if d < dt.date.fromisoformat(k[:10])) for k in missing}
            )
            eod = EOD_TAG[exchange]
            print(f"[{exchange}] fetching {len(before)} end-of-day files to stand in for them")
            gaps = run(exchange, before, args.root, eod, args.pause, args.retry_missing)
            for key in gaps:
                print(f"  also not published: {key}")


if __name__ == "__main__":
    main()
