"""The exchanges' daily SPAN margin files, cut down to the three indices we trade.

NSE (NIFTY, BANKNIFTY) and BSE (SENSEX) each publish a SPAN "risk parameter"
file several times a day: a CME SPAN XML 4.00 document of 45-75 MB covering
every F&O contract on the exchange. For each contract it lists a 16-number
*risk array* -- what one unit of that contract, held long, loses under sixteen
price/volatility scenarios (a loss is positive, a gain negative; the numbers
are in rupees **per unit**, not per lot -- `cvf` is 1 throughout) -- plus a
"composite delta" used to charge calendar spreads.

This module keeps only what margining NIFTY, BANKNIFTY and SENSEX needs, so a
year of files fits in a few hundred megabytes instead of tens of gigabytes:

- every option and future of those three: kind, expiry, strike, price,
  composite delta and the risk array (stored exactly, as whole paise);
- per index: the index level the file was built on, the price scan range, the
  short-option-minimum rate, and the calendar-spread table (`dSpread`).

`parse_span_xml` reads an exchange file; `save_day` / `load_day` write and
read the compact form (a compressed NumPy `.npz`; Parquet is gitignored
repo-wide, and the calibration test needs one day of this tracked).
"""

from __future__ import annotations

import datetime as dt
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np

# Which SPAN codes belong to which index. `cc` is SPAN's "combined commodity":
# everything margined together. On NSE the codes are the plain symbols; BSE
# names SENSEX's portfolios BSX (the index), BSXFUT and BSXOPT, combined as BSX.
EXCHANGE_CODES: dict[str, dict[str, str]] = {
    "NSE": {"NIFTY": "NIFTY", "BANKNIFTY": "BANKNIFTY"},
    "BSE": {"BSX": "SENSEX", "BSXFUT": "SENSEX", "BSXOPT": "SENSEX"},
}

KIND_FUT, KIND_CALL, KIND_PUT = 0, 1, 2


@dataclass(frozen=True)
class CalendarSpread:
    """One row of the exchange's calendar-spread table, in priority order."""

    priority: int
    method: str  # 'F' flat rate per spread; NSE and BSE use only this
    rate: float  # rupees per matched unit of delta
    leg_a: int  # expiry as YYYYMMDD
    leg_b: int
    ratio_a: float = 1.0
    ratio_b: float = 1.0


@dataclass
class IndexParams:
    underlying: str
    spot: float  # the index level the file's scenarios were built around
    price_scan: float  # rupees: the "1/1 move" of the scenarios
    som_rate: float  # short option minimum, rupees per short unit (0 on NSE/BSE)
    spreads: list[CalendarSpread] = field(default_factory=list)


@dataclass
class SpanDay:
    """One exchange file, cut down to the three indices."""

    exchange: str  # "NSE" or "BSE"
    file_date: dt.date  # the business date the file is FOR (its <date>)
    created: str  # the file's own <created> stamp, YYYYMMDDHHMM (IST)
    is_settlement: bool  # True for an end-of-day file
    source_name: str  # the exchange's file name, e.g. "nsccl.20250311.i1"
    params: dict[str, IndexParams]
    # Contract table: one row per contract, columns aligned.
    underlying: np.ndarray  # str
    kind: np.ndarray  # int8: KIND_FUT / KIND_CALL / KIND_PUT
    expiry: np.ndarray  # int32 YYYYMMDD
    strike: np.ndarray  # float64 (0 for futures)
    price: np.ndarray  # float64, rupees per unit
    delta: np.ndarray  # float64, composite delta
    risk: np.ndarray  # float64 (n, 16), rupees per unit held long; loss > 0


# --------------------------------------------------------------------- parse


def _text(elem, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else default


def _risk_array(elem) -> tuple[list[float], float]:
    ra = elem.find("ra")
    if ra is None:
        raise ValueError("contract without a risk array")
    values = [float(a.text) for a in ra.findall("a")]
    if len(values) != 16:
        raise ValueError(f"risk array has {len(values)} values, expected 16")
    return values, float(_text(ra, "d", "0"))


def parse_span_xml(source: str | Path | IO[bytes], exchange: str, source_name: str) -> SpanDay:
    """Stream an exchange SPAN XML file and keep only our three indices."""
    from lxml import etree

    codes = EXCHANGE_CODES[exchange]
    wanted_cc = set(codes.values())
    rows: dict[str, list] = {k: [] for k in ("u", "k", "e", "s", "p", "d", "r")}
    spot: dict[str, float] = {}
    scan: dict[str, float] = {}
    params: dict[str, IndexParams] = {}
    file_date = created = None
    is_setl = False

    tags = ("created", "date", "isSetl", "phyPf", "futPf", "oopPf", "ccDef")
    for _event, elem in etree.iterparse(source, events=("end",), tag=tags, huge_tree=True):
        tag = elem.tag
        if tag == "created" and created is None:
            created = (elem.text or "").strip()
        elif tag == "date" and file_date is None and elem.getparent().tag == "pointInTime":
            file_date = dt.datetime.strptime(elem.text.strip(), "%Y%m%d").date()
        elif tag == "isSetl" and elem.getparent().tag == "pointInTime":
            is_setl = (elem.text or "").strip() == "1"
        elif tag in ("phyPf", "futPf", "oopPf"):
            pf_code = _text(elem, "pfCode")
            und = codes.get(pf_code)
            if und is not None:
                if tag == "phyPf":
                    phy = elem.find("phy")
                    spot[und] = float(_text(phy, "p"))
                    scan[und] = float(_text(phy.find("scanRate"), "priceScan"))
                elif tag == "futPf":
                    for fut in elem.findall("fut"):
                        values, delta = _risk_array(fut)
                        rows["u"].append(und)
                        rows["k"].append(KIND_FUT)
                        rows["e"].append(int(_text(fut, "pe")))
                        rows["s"].append(0.0)
                        rows["p"].append(float(_text(fut, "p")))
                        rows["d"].append(delta)
                        rows["r"].append(values)
                else:
                    for series in elem.findall("series"):
                        pe = int(_text(series, "pe"))
                        for opt in series.findall("opt"):
                            values, delta = _risk_array(opt)
                            o = _text(opt, "o")
                            rows["u"].append(und)
                            rows["k"].append(KIND_CALL if o == "C" else KIND_PUT)
                            rows["e"].append(pe)
                            rows["s"].append(float(_text(opt, "k")))
                            rows["p"].append(float(_text(opt, "p")))
                            rows["d"].append(delta)
                            rows["r"].append(values)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        elif tag == "ccDef":
            cc = _text(elem, "cc")
            und = codes.get(cc)
            if und is not None and und in wanted_cc:
                som = 0.0
                for tier in elem.findall("somTiers/tier"):
                    rate = tier.find("rate")
                    if rate is not None:
                        som = max(som, float(_text(rate, "val", "0")))
                spreads = []
                for ds in elem.findall("dSpread"):
                    legs = ds.findall("pLeg")
                    if len(legs) != 2:
                        raise ValueError(f"{cc}: calendar spread with {len(legs)} legs")
                    by_side = {_text(leg, "rs"): leg for leg in legs}
                    a, b = by_side.get("A", legs[0]), by_side.get("B", legs[1])
                    for leg in (a, b):
                        if leg.find("pe") is None or len(_text(leg, "pe")) != 8:
                            raise ValueError(f"{cc}: calendar spread leg not by expiry date")
                    rate_elem = ds.find("rate")
                    spreads.append(
                        CalendarSpread(
                            priority=int(_text(ds, "spread")),
                            method=_text(ds, "chargeMeth"),
                            rate=float(_text(rate_elem, "val")),
                            leg_a=int(_text(a, "pe")),
                            leg_b=int(_text(b, "pe")),
                            ratio_a=float(_text(a, "i", "1")),
                            ratio_b=float(_text(b, "i", "1")),
                        )
                    )
                spreads.sort(key=lambda s: s.priority)
                params[und] = IndexParams(
                    underlying=und,
                    spot=spot.get(und, float("nan")),
                    price_scan=scan.get(und, float("nan")),
                    som_rate=som,
                    spreads=spreads,
                )
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    if file_date is None or created is None:
        raise ValueError("not a SPAN file: no <date>/<created>")
    missing = wanted_cc - set(params)
    if missing:
        raise ValueError(f"{source_name}: no SPAN definition for {sorted(missing)}")
    return SpanDay(
        exchange=exchange,
        file_date=file_date,
        created=created,
        is_settlement=is_setl,
        source_name=source_name,
        params=params,
        underlying=np.array(rows["u"], dtype="U9"),
        kind=np.array(rows["k"], dtype=np.int8),
        expiry=np.array(rows["e"], dtype=np.int32),
        strike=np.array(rows["s"], dtype=np.float64),
        price=np.array(rows["p"], dtype=np.float64),
        delta=np.array(rows["d"], dtype=np.float64),
        risk=np.array(rows["r"], dtype=np.float64).reshape(-1, 16),
    )


# ------------------------------------------------------------ compact store

_UNDERLYINGS = ("NIFTY", "BANKNIFTY", "SENSEX")


def _paise(values: np.ndarray) -> np.ndarray:
    out = np.rint(values * 100.0)
    if np.abs(out).max(initial=0) >= 2**31:
        raise ValueError("value too large to store as int32 paise")
    return out.astype(np.int32)


def save_day(day: SpanDay, path: Path) -> None:
    """Write the compact form. Prices, strikes and risk arrays are stored as
    whole paise (the files carry two decimals), so nothing is rounded away."""
    meta = {
        "exchange": day.exchange,
        "file_date": day.file_date.isoformat(),
        "created": day.created,
        "is_settlement": day.is_settlement,
        "source_name": day.source_name,
        "params": {
            u: {
                "spot": p.spot,
                "price_scan": p.price_scan,
                "som_rate": p.som_rate,
                "spreads": [
                    [s.priority, s.method, s.rate, s.leg_a, s.leg_b, s.ratio_a, s.ratio_b]
                    for s in p.spreads
                ],
            }
            for u, p in day.params.items()
        },
    }
    und_code = np.array([_UNDERLYINGS.index(u) for u in day.underlying], dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
        underlying=und_code,
        kind=day.kind,
        expiry=day.expiry,
        strike=_paise(day.strike),
        price=_paise(day.price),
        delta=np.rint(day.delta * 1e6).astype(np.int32),
        risk=_paise(day.risk),
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(path)


def load_day(path: Path) -> SpanDay:
    with np.load(path) as z:
        meta = json.loads(bytes(z["meta"]).decode())
        params = {
            u: IndexParams(
                underlying=u,
                spot=p["spot"],
                price_scan=p["price_scan"],
                som_rate=p["som_rate"],
                spreads=[CalendarSpread(*row) for row in p["spreads"]],
            )
            for u, p in meta["params"].items()
        }
        return SpanDay(
            exchange=meta["exchange"],
            file_date=dt.date.fromisoformat(meta["file_date"]),
            created=meta["created"],
            is_settlement=meta["is_settlement"],
            source_name=meta["source_name"],
            params=params,
            underlying=np.array(_UNDERLYINGS, dtype="U9")[z["underlying"]],
            kind=z["kind"].copy(),
            expiry=z["expiry"].copy(),
            strike=z["strike"] / 100.0,
            price=z["price"] / 100.0,
            delta=z["delta"] / 1e6,
            risk=z["risk"] / 100.0,
        )


def subset(day: SpanDay, mask: np.ndarray) -> SpanDay:
    """A copy keeping only the contracts in `mask` (used to make small fixtures)."""
    return SpanDay(
        exchange=day.exchange,
        file_date=day.file_date,
        created=day.created,
        is_settlement=day.is_settlement,
        source_name=day.source_name,
        params=day.params,
        underlying=day.underlying[mask],
        kind=day.kind[mask],
        expiry=day.expiry[mask],
        strike=day.strike[mask],
        price=day.price[mask],
        delta=day.delta[mask],
        risk=day.risk[mask],
    )
