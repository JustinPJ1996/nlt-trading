#!/usr/bin/env python
"""Check `nlt.options.margin` against Zerodha's public F&O (SPAN) calculator.

    .venv/bin/python scripts/fetch_span.py --today
    .venv/bin/python scripts/calibrate_span_zerodha.py positions.json

Zerodha's calculator (https://zerodha.com/margin-calculator/SPAN/) has no date
picker: it answers for *today*, from today's start-of-day exchange file. So
this enters each position into the real page with a headless browser, reads
Zerodha's SPAN / exposure / total, and works out the same position with our
module on today's stored file. It writes
``tests/fixtures/zerodha_span_calibration.json`` and a cut-down copy of
today's files (only the contracts the positions use, plus every expiry's
risk parameters) so `tests/test_span_margin.py` can replay it offline forever.

``positions.json`` is a list of ``[name, underlying, legs]`` where a leg is
``[zerodha_scrip, expiry_iso, "CE"|"PE", strike, lots, "buy"|"sell"]``.

An optional second argument names the fixture to write, so a later capture
can sit beside the first instead of replacing it (debit spreads and ratios
were added on 16 Oct 2026 as ``zerodha_span_calibration_2026-10-16.json``).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nlt.options.margin import EXCHANGE, MarginLeg, SpanMargin  # noqa: E402
from nlt.options.span_store import save_day, subset  # noqa: E402

LOT = {"NIFTY": 65, "BANKNIFTY": 30, "SENSEX": 20}
ZERODHA_EXCHANGE = {"NSE": "NFO", "BSE": "BFO"}
FIXTURES = REPO / "tests" / "fixtures"


def zerodha(page, exchange: str, legs: list[list]) -> dict:
    page.goto("https://zerodha.com/margin-calculator/SPAN/", wait_until="domcontentloaded")
    page.wait_for_function("typeof FUTURES !== 'undefined' && $('#scrip option').length > 0")
    last = None
    for scrip, _expiry, ot, strike, qty, trade in legs:
        page.select_option("#exchange", exchange)
        page.select_option("#product", "OPT")
        if not page.evaluate(f"$('#scrip option[value=\"{scrip}\"]').length"):
            raise SystemExit(f"Zerodha does not list {scrip}")
        page.evaluate(f"$('#scrip').val('{scrip}').change()")
        page.select_option("#option_type", ot)
        page.fill("#strike_price", str(strike))
        page.fill("#qty", str(qty))
        page.check(f"#form-span .trade-{trade}")
        with page.expect_response(
            lambda r: r.request.method == "POST" and "margin-calculator/SPAN" in r.url
        ) as info:
            page.click("#form-span [type=submit]")
        last = info.value.json()
        if "total" not in last:
            raise SystemExit(f"Zerodha rejected {scrip} {strike} {ot}: {last}")
    page.wait_for_timeout(300)
    screen = page.inner_text("#tally")
    return {
        **last["total"],
        "screen": " | ".join(x.strip() for x in screen.splitlines() if x.strip()),
    }


def main() -> None:
    from playwright.sync_api import sync_playwright

    positions = json.loads(Path(sys.argv[1]).read_text())
    today = dt.date.today()
    margin = SpanMargin()
    files = {ex: margin._file_for(u, today)[0] for u, ex in EXCHANGE.items()}
    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        for name, und, legs in positions:
            exchange = EXCHANGE[und]
            entered = [
                [scrip, exp, ot, strike, lots * LOT[und], trade]
                for scrip, exp, ot, strike, lots, trade in legs
            ]
            z = zerodha(page, ZERODHA_EXCHANGE[exchange], entered)
            margin_legs = [
                MarginLeg(ot, float(strike), dt.date.fromisoformat(exp), q if t == "buy" else -q)
                for _s, exp, ot, strike, q, t in entered
            ]
            spot = files[exchange].params[und].spot
            ours = margin.margin(und, today, margin_legs, index_prev_close=spot)
            out.append(
                {
                    "name": name,
                    "underlying": und,
                    "legs": [
                        {
                            "option_type": leg.option_type,
                            "strike": leg.strike,
                            "expiry": leg.expiry.isoformat(),
                            "quantity": leg.quantity,
                            "zerodha_scrip": e[0],
                        }
                        for leg, e in zip(margin_legs, entered, strict=True)
                    ],
                    "zerodha": z,
                    "ours_at_capture": {
                        "span": round(ours.span, 2),
                        "exposure": round(ours.exposure, 2),
                        "total": round(ours.total, 2),
                    },
                }
            )
            print(
                f"{name:55s} Zerodha span {z['span']:>11,.2f} ours {ours.span:>11,.2f} | "
                f"exposure {z['exposure']:>10,.2f} ours {ours.exposure:>10,.2f}",
                flush=True,
            )
        browser.close()

    # A small copy of today's files: only the contracts used, all parameters.
    for exchange, file in files.items():
        keep = np.zeros(len(file.kind), dtype=bool)
        for pos in out:
            if EXCHANGE[pos["underlying"]] != exchange:
                continue
            for leg in pos["legs"]:
                keep |= (
                    (file.underlying == pos["underlying"])
                    & (file.kind == (1 if leg["option_type"] == "CE" else 2))
                    & (file.expiry == int(leg["expiry"].replace("-", "")))
                    & (np.abs(file.strike - leg["strike"]) < 1e-6)
                )
        tag = "i1" if exchange == "NSE" else "00"
        save_day(
            subset(file, keep),
            FIXTURES / "span" / exchange / f"{today.year}" / f"{today:%Y%m%d}-{tag}.npz",
        )

    fixture = {
        "captured_on": today.isoformat(),
        "captured_at_ist": dt.datetime.now(
            dt.timezone(dt.timedelta(hours=5, minutes=30))
        ).isoformat(timespec="minutes"),
        "zerodha_page": "https://zerodha.com/margin-calculator/SPAN/",
        "files": {
            ex: {
                "source_name": f.source_name,
                "created": f.created,
                "index_close": {u: p.spot for u, p in f.params.items()},
            }
            for ex, f in files.items()
        },
        "positions": out,
    }
    path = FIXTURES / (sys.argv[2] if len(sys.argv) > 2 else "zerodha_span_calibration.json")
    path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
