"""Everything the dashboard does, minus the widgets.

Streamlit widgets are awkward to unit test -- they need a running script
context -- so every decision the app makes (parse text, run a backtest, judge
the result, format a number for a screen) lives here as a plain function.
`app/main.py` (and the page modules it drives) should do nothing but call
these functions and hand the results to `st.*` calls; if a test wants to
check *behaviour* rather than *layout*, it imports from here.

The other governing rule, inherited from `nlt.translate.rules`: nothing in
this module may turn an ambiguous or broken result into a silent best guess.
A backtest that cannot run becomes a `PipelineResult.error` string a
non-technical user can read, never a raised traceback on their screen.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
from pydantic import ValidationError

from nlt.costs.charges import (
    ChargeModel,
    McxFuturesCharges,
    NseEquityDeliveryCharges,
    NseFuturesCharges,
    NseOptionsCharges,
    ZeroCharges,
)
from nlt.costs.charges import charge_fn as _adapt_charge_fn
from nlt.costs.slippage import ZERO as ZERO_SLIPPAGE
from nlt.costs.slippage import Slippage
from nlt.data.source import DATA_FLOOR
from nlt.data.yahoo import YahooSource
from nlt.engine import warmup
from nlt.engine.backtest import BacktestResult, run_backtest
from nlt.engine.basket import BasketResult, run_basket_backtest
from nlt.engine.loader import load_for_spec
from nlt.spec.models import StrategySpec
from nlt.store.db import Store
from nlt.translate.llm import translate
from nlt.translate.readback import describe, format_inr
from nlt.translate.rules import BacktestPeriod, Question, TranslationResult

try:  # pragma: no cover -- exercised once nlt.report exists
    from nlt.report.benchmark import Benchmark, Comparison, buy_and_hold, compare
    from nlt.report.verdict import Flag, Verdict, assess

    REPORT_SOURCE = "nlt.report"
except ImportError:  # nlt.report is being built by another agent concurrently
    from app.report_fallback import (
        Benchmark,
        Comparison,
        Flag,
        Verdict,
        assess,
        buy_and_hold,
        compare,
    )

    REPORT_SOURCE = "app.report_fallback"

logger = logging.getLogger(__name__)

__all__ = [
    "COST_MODEL_LABELS",
    "DEFAULT_CAPITAL",
    "EXAMPLE_STRATEGIES",
    "PipelineResult",
    "Benchmark",
    "Comparison",
    "Flag",
    "Verdict",
    "answers_from_questions",
    "backtest_spec",
    "charge_model_for_label",
    "charge_model_for_spec",
    "color_for_flag",
    "color_for_pnl",
    "describe_spec",
    "flags_by_severity",
    "format_inr",
    "format_pct",
    "format_signed_pct",
    "list_runs_view",
    "list_strategies_view",
    "load_bars",
    "run_pipeline",
    "safe_call",
    "save_strategy",
    "translate_text",
]

# --------------------------------------------------------------------- misc

# Justin's call (2026-10-12). At Rs 1,00,000 the 20% limit is Rs 20,000, less
# than one unit of NIFTY, so the index examples could take no trades at all.
DEFAULT_CAPITAL = 2_500_000.0

# Every example is verified (in tests/test_app.py) to parse into a spec with
# no follow-up questions -- a "try this" button that itself needs clarifying
# would be a bad first impression -- and to take at least one trade at the
# default capital over its own test period. Each names a stop loss on its own,
# the habit the prompt guide teaches.
EXAMPLE_STRATEGIES = [
    "Buy NIFTY when RSI(14) crosses below 30. Target 2%, stop loss 1%. Backtest the last 3 years.",
    "Buy NIFTY when 20 SMA crosses above 50 SMA. Target 3%, stop loss 1.5%. "
    "Backtest the last 5 years.",
    "Buy RELIANCE when RSI(14) crosses below 30. Target 5%, stop loss 2%. "
    "Backtest the last 5 years.",
    "Buy INFY when 9 EMA crosses above 21 EMA. Target 4%, stop loss 2%. Backtest the last 3 years.",
    "Buy TCS when close is above 200 EMA. Target 5%, trailing stop 2%. Backtest the last 5 years.",
]

# The first entry is the default. Index futures, not options: "buy NIFTY" is
# tested on the index itself, and futures are the way to hold that; option
# charges (STT on premium) describe a different trade (changed 2026-10-12).
COST_MODEL_LABELS: dict[str, ChargeModel | None] = {
    "Index futures": NseFuturesCharges(),
    "Index options (NIFTY/BANKNIFTY weekly)": NseOptionsCharges(),
    "MCX commodity futures": McxFuturesCharges(),
    "Equity delivery": NseEquityDeliveryCharges(),
    "No costs (for comparison only)": ZeroCharges(),
}


def applied_charges_label(spec: StrategySpec, label: str) -> str:
    """The name of the charges a run really used, which for shares and futures is
    not the dropdown's (see `charge_model_for_spec`)."""
    model, _note = charge_model_for_spec(spec, label)
    for name, candidate in COST_MODEL_LABELS.items():
        if type(candidate) is type(model):
            return name
    return {McxFuturesCharges: "MCX commodity futures"}.get(type(model), label)


def charge_model_for_label(label: str) -> ChargeModel:
    """Looks up a cost model by its dropdown label, defaulting to index futures costs."""
    return COST_MODEL_LABELS.get(label, NseFuturesCharges())


def charge_model_for_spec(spec: StrategySpec, label: str) -> tuple[ChargeModel, str | None]:
    """The cost model to actually use, and a note if it overrode the dropdown.

    The dropdown is a real choice for an index strategy -- "buy NIFTY" is not
    directly tradable, so whether you would express it as options or futures
    changes the costs and only the user knows which they meant.

    It is not a real choice for shares. A strategy that buys RELIANCE pays
    equity delivery charges, and the dropdown defaults to *options* costs, so
    the untouched default silently applied option STT to a share trade. The two
    differ enough to move a marginal strategy from profitable to not, which is
    exactly the kind of wrong answer this project treats as serious.

    So: shares are reconciled to equity charges and the override is said out
    loud rather than done quietly. "No costs" is never overridden -- it is a
    deliberate comparison baseline, and silently adding charges to it would
    destroy the one thing it is for.
    """
    chosen = charge_model_for_label(label)

    if isinstance(chosen, ZeroCharges):
        return chosen, None

    if spec.instrument.is_future:
        # A futures strategy names its own contract, so its costs are not a
        # choice either: MCX charges commodity transaction tax at 0.01%, NSE
        # index futures STT at 0.05%, and options figures fit neither.
        from nlt.data.futures import is_commodity

        if is_commodity(spec.instrument.symbol):
            right, name = McxFuturesCharges(), "MCX commodity futures"
        else:
            right, name = NseFuturesCharges(), "NSE index futures"
        if isinstance(chosen, type(right)):
            return chosen, None
        return right, (
            f"Costs: this strategy trades {name}, so those charges were applied "
            f"instead of '{label}'."
        )

    if spec.instrument.is_option and not isinstance(chosen, NseOptionsCharges):
        sells = any(leg.side == "sell" for leg in spec.instrument.legs)
        verb = "buys and sells" if sells else "buys"
        return NseOptionsCharges(), (
            f"Costs: this strategy {verb} index options, so option charges were applied "
            f"instead of '{label}'."
        )

    if spec.instrument.trade_as == "stock" and not isinstance(chosen, NseEquityDeliveryCharges):
        return NseEquityDeliveryCharges(), (
            f"Costs: this strategy buys and sells shares, so equity delivery charges "
            f"were applied instead of '{label}'. Shares are taxed and charged differently, "
            f"and those figures would have misstated what this strategy costs to run."
        )

    return chosen, None


# ---------------------------------------------------------------- formatting


def format_pct(value: float | None, *, decimals: int = 1) -> str:
    """`12.345` -> `'12.3%'`. `None` reads as 'n/a', never as a crash or 'None%'."""
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}%"


def format_signed_pct(value: float | None, *, decimals: int = 1) -> str:
    """Like `format_pct`, but always shows the sign: `+2.0%` / `-1.0%`."""
    if value is None:
        return "n/a"
    return f"{value:+.{decimals}f}%"


def color_for_pnl(value: float) -> str:
    """The delta-good / delta-bad ink from the palette -- never a raw guess.

    Positive is the success-text green, negative the status-critical red,
    zero the muted/secondary ink. These are *text* colors (a P&L number in a
    table), which is the one place the palette allows a status hue on text.
    """
    if value > 0:
        return "#006300"
    if value < 0:
        return "#d03b3b"
    return "#52514e"


_SEVERITY_COLORS = {
    "critical": "#d03b3b",
    "warning": "#fab219",
    "info": "#52514e",
}
_SEVERITY_ICONS = {"critical": "🔴", "warning": "⚠️", "info": "ℹ️"}


def color_for_flag(severity: str) -> str:
    """The fixed status hex for a verdict flag's severity, never eyeballed."""
    return _SEVERITY_COLORS.get(severity, _SEVERITY_COLORS["info"])


def icon_for_flag(severity: str) -> str:
    return _SEVERITY_ICONS.get(severity, _SEVERITY_ICONS["info"])


def flags_by_severity(flags: list[Flag]) -> list[Flag]:
    """Most severe first -- critical, then warning, then info."""
    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(flags, key=lambda f: order.get(f.severity, 3))


def describe_spec(spec: StrategySpec) -> str:
    return describe(spec)


# ---------------------------------------------------------------- translate


def translate_text(text: str, answers: dict[str, str] | None = None) -> TranslationResult:
    """Wraps `nlt.translate.llm.translate` so a bug in a reader cannot crash the page.

    The rules read the sentence first; the AI reader is only asked when they
    could not understand it. Both are written to never raise on bad input -- unparseable text comes
    back as `unparsed` fragments and `Question`s -- but this dashboard is the
    only thing standing between that guarantee and a stack trace on a
    non-technical user's screen, so it is wrapped anyway.
    """
    try:
        return translate(text, answers=answers or {})
    except Exception:  # pragma: no cover -- defensive; translate() is designed not to raise
        logger.exception("translate_text: translate() raised on %r", text)
        return TranslationResult(
            spec=None,
            questions=[
                Question(
                    text="I couldn't read that description. Try rephrasing it more simply.",
                    why="Something in the parser broke; the detail is in the app log.",
                    suggestion=None,
                    field="description",
                )
            ],
            source="rules",
        )


def period_dates(
    period: BacktestPeriod, today: dt.date, *, data_end: dt.date | None = None
) -> dict | None:
    """The dates a test period from the sentence means, as of `today`.

    None when there is nothing to test yet (a year still in the future).
    Never earlier than the history we trust (`DATA_FLOOR`), and it says so.

    `data_end` is the last day prices exist for (option prices stop at the
    end of the dataset): "the last year" then counts back from it, and the
    note says so (Justin, 2026-10-13: test the latest year we have).
    """
    shifted = None
    if data_end is not None and data_end < today:
        today, shifted = data_end, data_end
    if period.year is not None:
        start, end = dt.date(period.year, 1, 1), min(dt.date(period.year, 12, 31), today)
    else:
        offset = pd.DateOffset(**{period.unit: period.last})
        start, end = (pd.Timestamp(today) - offset).date(), today
    if start >= today:
        return None
    note = None
    if start < DATA_FLOOR:
        start = DATA_FLOOR
        note = (
            f"History here starts on {DATA_FLOOR:%d %b %Y}, so the test starts there "
            f'rather than at the start of "{period.words}".'
        )
    if shifted is not None:
        moved = (
            f'Option prices are only available up to {shifted:%d %b %Y}, so "{period.words}" '
            f"is counted back from then."
        )
        note = f"{moved} {note}" if note else moved
    return {"words": period.words, "start": start.isoformat(), "end": end.isoformat(), "note": note}


def option_data_end(spec: StrategySpec | None) -> dt.date | None:
    """The last day with option prices, for an option strategy; else None."""
    if spec is None or not spec.instrument.is_option:
        return None
    from nlt.options.data import ParquetSource

    cover = ParquetSource().coverage(spec.instrument.symbol, spec.instrument.timeframe)
    return cover[1] if cover else None


def answers_from_questions(questions: list[Question], values: dict[str, str]) -> dict[str, str]:
    """Keeps only non-empty answers, keyed by `Question.field`, for re-parsing."""
    return {q.field: values[q.field] for q in questions if values.get(q.field)}


# --------------------------------------------------------------------- data


def load_bars(
    symbol: str,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Daily OHLCV bars for `symbol`, from the local Yahoo cache (or a fresh download).

    Plain function, no `st.cache_data` here -- `app/main.py` wraps this with
    the Streamlit cache decorator so this module stays importable and testable
    without a Streamlit runtime.
    """
    return YahooSource().bars(symbol, start=start, end=end)


def staleness_days(symbol: str) -> int | None:
    return YahooSource().staleness_days(symbol)


# ------------------------------------------------------------------ pipeline


@dataclass
class PipelineResult:
    """The full text-to-verdict path, or as much of it as completed.

    `error` is set (and everything after `translation` is `None`) whenever a
    step downstream of a valid spec failed -- a data problem, an engine bug,
    anything unexpected. `translation.spec is None` (no `error`) means the
    text itself was not yet understood, which is a normal, expected outcome,
    not a failure.
    """

    translation: TranslationResult
    spec: StrategySpec | None = None
    bars: pd.DataFrame | None = None
    backtest: BacktestResult | None = None
    benchmark: Benchmark | None = None
    comparison: Comparison | None = None
    verdict: Verdict | None = None
    # Populated only for a basket run. `backtest` still carries the metrics and
    # trades so every downstream consumer (charts, verdict, tables) works on one
    # shape regardless of whether one symbol or a hundred were traded.
    basket: BasketResult | None = None
    # The same run again with charges switched off, for the before/after
    # comparison. Exact: a second run, not charges added back to one number.
    before_charges: BacktestResult | None = None
    # How the test was done, in plain words (nlt.report.method).
    assumptions: list[str] = field(default_factory=list)
    weak_spots: list[str] = field(default_factory=list)
    data_notes: list[str] = field(default_factory=list)
    # Said at the very top of the result: the test covered different dates
    # from the ones asked for, because option prices exist only for some.
    window_notes: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.backtest is not None


def _as_backtest_result(basket: BasketResult, spec: StrategySpec) -> BacktestResult:
    """Present a basket run in the shape the rest of the UI already understands.

    Charts, the metrics table, the trade list and the verdict were all written
    against `BacktestResult`. A basket carries the same information -- one
    portfolio equity curve, one set of metrics, one list of trades -- so it is
    adapted rather than special-cased everywhere downstream. The alternative is
    a second rendering path that drifts from the first.
    """
    return BacktestResult(
        spec=spec,
        trades=basket.trades,
        equity=basket.equity,
        metrics=basket.metrics,
        features=pd.DataFrame(index=basket.equity.index),
        warnings=list(basket.warnings),
    )


def _equal_weight_index(bars_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """An equal-weight basket of the same names, to benchmark a basket against.

    Comparing a hundred-stock strategy to NIFTY would conflate two questions:
    whether the timing rules add anything, and whether those hundred stocks beat
    the index. This isolates the first -- "would picking your moments have beaten
    simply owning all of them?" -- which is what the user is actually asking.

    Each symbol is normalised to its own first close so no single high-priced
    stock dominates, then averaged across whichever symbols have a bar that day.
    """
    normalised = {}
    for symbol, df in bars_by_symbol.items():
        if df.empty:
            continue
        base = df["close"].iloc[0]
        if base > 0:
            normalised[symbol] = df["close"] / base

    if not normalised:
        raise ValueError("no usable price history in the basket")

    combined = pd.DataFrame(normalised).sort_index()
    level = 100.0 * combined.mean(axis=1, skipna=True)

    # An OHLCV frame is what the benchmark and charts expect. The synthetic
    # index has no meaningful intraday range, so open/high/low all equal close;
    # `buy_and_hold` only reads the first open and the last close.
    return pd.DataFrame(
        {"open": level, "high": level, "low": level, "close": level, "volume": 0.0},
        index=level.index,
    ).dropna()


def _listed_expiries(spec: StrategySpec) -> list[dt.date]:
    """Expiries the exchange has published for a futures strategy's live contracts.

    Best effort: without them the computed rule decides every expiry, which is
    only wrong for a live contract whose expiry a holiday the price data cannot
    see yet has moved -- see `nlt.data.futures.expiry_dates`.
    """
    if not spec.instrument.is_future:
        return []
    from nlt.data import kite

    try:
        return kite.listed_expiries(spec.instrument.symbol)
    except kite.KiteError:
        return []


def run_pipeline(
    text: str,
    *,
    capital: float = DEFAULT_CAPITAL,
    cost_model_label: str = next(iter(COST_MODEL_LABELS)),
    symbol: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    answers: dict[str, str] | None = None,
    slippage: Slippage = ZERO_SLIPPAGE,
) -> PipelineResult:
    """Plain English in, verdict out -- the function `run_backtest_flow` in the brief.

    Every downstream step (data load, engine, benchmark, verdict) is wrapped
    so a real bug never reaches the UI as a traceback; it comes back as
    `PipelineResult.error` instead.
    """
    translation = translate_text(text, answers)
    if translation.spec is None:
        return PipelineResult(translation=translation)
    return backtest_spec(
        translation.spec,
        capital=capital,
        cost_model_label=cost_model_label,
        symbol=symbol,
        start=start,
        end=end,
        translation=translation,
        slippage=slippage,
    )


def backtest_spec(
    spec: StrategySpec,
    *,
    capital: float = DEFAULT_CAPITAL,
    cost_model_label: str = next(iter(COST_MODEL_LABELS)),
    symbol: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    translation: TranslationResult | None = None,
    slippage: Slippage = ZERO_SLIPPAGE,
) -> PipelineResult:
    """The backtest half of `run_pipeline`, for a spec the user has already confirmed.

    Re-reading the sentence to run it is not the same as running what the user
    approved: the AI reader is not guaranteed to answer identically twice, so
    the strategy tested could differ from the readback they said yes to. The web
    front end therefore hands back the confirmed spec and this runs exactly that.

    `slippage` is a run setting -- like `capital` or `cost_model_label` -- never
    part of the spec. It applies identically to the real run and the
    before-charges comparison run below, so the gap between them is only ever
    the charges, never a slippage difference nobody asked for.
    """
    if translation is None:
        translation = TranslationResult(spec=spec)
    if spec.instrument.legs:
        return _backtest_legs_spec(
            spec,
            capital=capital,
            cost_model_label=cost_model_label,
            start=start,
            end=end,
            translation=translation,
            slippage=slippage,
        )
    if spec.instrument.is_option:
        return _backtest_option_spec(
            spec,
            capital=capital,
            cost_model_label=cost_model_label,
            start=start,
            end=end,
            translation=translation,
            slippage=slippage,
        )
    try:
        model, cost_note = charge_model_for_spec(spec, cost_model_label)
        cf = _adapt_charge_fn(model)

        # `load_for_spec` knows how to turn a symbol into bars whether it names
        # an index, a single stock or a whole universe, and hands back every
        # data-quality note it collected on the way -- artefact cleaning, load
        # failures, survivorship. Those notes reach the user; they are the
        # reason a basket that quietly lost fifteen constituents is visible.
        # An explicit `symbol` override replaces the one the spec carries. It has
        # to be pushed into the spec rather than passed alongside it, because
        # `load_for_spec` decides index-vs-stock-vs-universe from the spec --
        # honouring the override anywhere else would load one symbol and then
        # backtest against another's rules.
        load_spec = spec
        if symbol and symbol != spec.instrument.symbol:
            load_spec = spec.model_copy(
                update={"instrument": spec.instrument.model_copy(update={"symbol": symbol})}
            )

        # Load extra history before `start` so indicators are ready on it; the
        # runs below may not trade before it, and are cut back to it after.
        load_from = warmup.padded_start(load_spec, start) if start else None
        bars_by_symbol, data_notes = load_for_spec(
            load_spec, start=load_from, end=end, tested_from=start
        )
        if cost_note:
            data_notes = [cost_note, *data_notes]
        if not bars_by_symbol or all(df.empty for df in bars_by_symbol.values()):
            return PipelineResult(
                translation=translation,
                spec=spec,
                data_notes=data_notes,
                error="There is no price history for that symbol and date range to test against.",
            )

        no_charges = _adapt_charge_fn(ZeroCharges())
        all_index = pd.DatetimeIndex(
            sorted(set().union(*(df.index for df in bars_by_symbol.values() if not df.empty)))
        )
        start_bar = warmup.first_bar_on_or_after(all_index, start) if start else None
        if start and start_bar is None:
            return PipelineResult(
                translation=translation,
                spec=spec,
                data_notes=data_notes,
                error="There is no price history for that symbol and date range to test against.",
            )

        def trimmed(result: BacktestResult) -> BacktestResult:
            return warmup.trim(result, start_bar, capital) if start_bar is not None else result

        if len(bars_by_symbol) > 1:
            basket = run_basket_backtest(
                load_spec,
                bars_by_symbol,
                capital=capital,
                charge_fn=cf,
                trade_from=start_bar,
                slippage=slippage,
            )
            backtest = trimmed(_as_backtest_result(basket, load_spec))
            before_charges = trimmed(
                _as_backtest_result(
                    run_basket_backtest(
                        load_spec,
                        bars_by_symbol,
                        capital=capital,
                        charge_fn=no_charges,
                        trade_from=start_bar,
                        slippage=slippage,
                    ),
                    load_spec,
                )
            )
            # The benchmark for a basket is an equal-weight buy-and-hold of the
            # same names, which is the honest comparison: "would picking your
            # moments have beaten simply owning all of them?"
            bars = _equal_weight_index(bars_by_symbol)
            if start_bar is not None:
                bars = bars.loc[start_bar:]
            benchmark_name = f"holding all {len(bars_by_symbol)} of these stocks equally"
            data_notes = data_notes + basket.warnings
        else:
            all_bars = next(iter(bars_by_symbol.values()))
            benchmark_name = f"holding {load_spec.instrument.symbol}"
            basket = None
            listed = _listed_expiries(load_spec)
            backtest = trimmed(
                run_backtest(
                    load_spec,
                    all_bars,
                    capital=capital,
                    charge_fn=cf,
                    listed_expiries=listed,
                    trade_from=start_bar,
                    slippage=slippage,
                )
            )
            before_charges = trimmed(
                run_backtest(
                    load_spec,
                    all_bars,
                    capital=capital,
                    charge_fn=no_charges,
                    listed_expiries=listed,
                    trade_from=start_bar,
                    slippage=slippage,
                )
            )
            bars = all_bars.loc[start_bar:] if start_bar is not None else all_bars
            data_notes = data_notes + list(backtest.warnings)
            if start_bar is not None:
                note = warmup.warmup_note(load_spec, all_bars, start_bar)
                if note:
                    data_notes.append(note)

        if load_spec.instrument.is_future:
            from nlt.data.futures import contract, expiry_close_mask
            from nlt.report.benchmark import hold_futures

            c = contract(load_spec.instrument.symbol)
            benchmark = hold_futures(
                bars,
                capital,
                multiplier=c.multiplier,
                expiry_close=expiry_close_mask(
                    c.symbol, bars.index, load_spec.instrument.timeframe, listed
                ),
                name=f"holding {c.title} futures, closed at each expiry",
                charge_fn=cf,
            )
        else:
            benchmark = buy_and_hold(bars, capital, charge_fn=cf, name=benchmark_name)
        comparison = compare(backtest, benchmark, bars)
        verdict = assess(backtest, comparison)
        from nlt.report import method

        equity_index = backtest.equity.index
        stated = (
            method.assumptions(
                load_spec,
                capital=capital,
                charge_model=model,
                first=equity_index[0].date(),
                last=equity_index[-1].date(),
                slippage=slippage,
            )
            if len(equity_index)
            else []
        )
        spots, data_notes = method.weak_spots(load_spec, data_notes, slippage=slippage)

        return PipelineResult(
            translation=translation,
            spec=spec,
            bars=bars,
            backtest=backtest,
            benchmark=benchmark,
            comparison=comparison,
            verdict=verdict,
            basket=basket,
            before_charges=before_charges,
            assumptions=stated,
            weak_spots=spots,
            data_notes=data_notes,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad; this is the UI's last line of defence
        logger.exception("run_pipeline failed for spec %r", getattr(spec, "name", None))
        return PipelineResult(
            translation=translation,
            spec=spec,
            error=f"The backtest could not be run: {exc}",
        )


def _backtest_option_spec(
    spec: StrategySpec,
    *,
    capital: float,
    cost_model_label: str,
    start: dt.date | None,
    end: dt.date | None,
    translation: TranslationResult,
    slippage: Slippage,
) -> PipelineResult:
    """`backtest_spec` for buying a call or a put (`nlt.engine.options_backtest`).

    Signals come from the index's candles in the F&O dataset, trades from the
    contract chosen at each signal. The test window is clamped to the days
    with real option prices, and the result says so at the top.
    """
    from nlt.engine.loader import load_option_bars, option_window
    from nlt.engine.options_backtest import run_option_backtest
    from nlt.options.data import ParquetSource
    from nlt.options.execution import OptionExecution
    from nlt.report import method

    try:
        source = ParquetSource()
        start, end, window_notes = option_window(spec, start, end, source)
        model, cost_note = charge_model_for_spec(spec, cost_model_label)
        cf = _adapt_charge_fn(model)
        data_notes = [cost_note] if cost_note else []

        load_from = max(
            warmup.padded_start(spec, start),
            source.coverage(spec.instrument.symbol, spec.instrument.timeframe)[0],
        )
        all_bars = load_option_bars(spec, load_from, end, source)
        if all_bars.empty:
            return PipelineResult(
                translation=translation,
                spec=spec,
                data_notes=data_notes,
                window_notes=window_notes,
                error="There is no price history for that index and date range to test against.",
            )
        start_bar = warmup.first_bar_on_or_after(all_bars.index, start)
        if start_bar is None:
            return PipelineResult(
                translation=translation,
                spec=spec,
                window_notes=window_notes,
                error="There is no price history for that index and date range to test against.",
            )

        def run(charge_fn) -> BacktestResult:
            execution = OptionExecution(
                source,
                spec.instrument.symbol,
                spec.instrument.timeframe,
                spec.instrument.option_type,
                spec.instrument.strike_offset,
                spec.instrument.expiry,
            )
            result = run_option_backtest(
                spec,
                all_bars,
                execution,
                capital=capital,
                charge_fn=charge_fn,
                slippage=slippage,
                trade_from=start_bar,
            )
            return warmup.trim(result, start_bar, capital)

        backtest = run(cf)
        before_charges = run(_adapt_charge_fn(ZeroCharges()))
        bars = all_bars.loc[start_bar:]
        data_notes = data_notes + list(backtest.warnings)
        note = warmup.warmup_note(spec, all_bars, start_bar)
        if note:
            data_notes.append(note)

        benchmark = buy_and_hold(
            bars, capital, charge_fn=cf, name=f"holding {spec.instrument.symbol}"
        )
        comparison = compare(backtest, benchmark, bars)
        verdict = assess(backtest, comparison)
        equity_index = backtest.equity.index
        stated = (
            method.assumptions(
                spec,
                capital=capital,
                charge_model=model,
                first=equity_index[0].date(),
                last=equity_index[-1].date(),
                slippage=slippage,
            )
            if len(equity_index)
            else []
        )
        spots, data_notes = method.weak_spots(spec, data_notes, slippage=slippage)
        return PipelineResult(
            translation=translation,
            spec=spec,
            bars=bars,
            backtest=backtest,
            benchmark=benchmark,
            comparison=comparison,
            verdict=verdict,
            before_charges=before_charges,
            assumptions=stated,
            weak_spots=spots,
            data_notes=data_notes,
            window_notes=window_notes,
        )
    except Exception as exc:  # noqa: BLE001 - the UI's last line of defence, as above
        logger.exception("option backtest failed for spec %r", getattr(spec, "name", None))
        return PipelineResult(
            translation=translation,
            spec=spec,
            error=f"The backtest could not be run: {exc}",
        )


def span_margin_fn():
    """The exchange-margin calculator (`nlt.options.margin`), shaped for the engine.

    Imported here, not by the engine, so the engine can be tested with a fixed
    margin and does not depend on the calculator's data files.
    """
    from nlt.engine.legs_backtest import MarginNotAvailable
    from nlt.options.margin import MarginLeg, MarginUnavailable, SpanMargin

    span = SpanMargin()

    def margin(underlying, day, legs, index_prev_close):
        try:
            return _span_call(underlying, day, legs, index_prev_close)
        except MarginUnavailable as exc:
            raise MarginNotAvailable(str(exc)) from exc

    def _span_call(underlying, day, legs, index_prev_close):
        return span.margin(
            underlying,
            day,
            [
                MarginLeg(
                    option_type=leg.option_type,
                    strike=leg.strike,
                    expiry=leg.expiry,
                    quantity=leg.quantity,
                )
                for leg in legs
            ],
            index_prev_close=index_prev_close,
        )

    return margin


def _backtest_legs_spec(
    spec: StrategySpec,
    *,
    capital: float,
    cost_model_label: str,
    start: dt.date | None,
    end: dt.date | None,
    translation: TranslationResult,
    slippage: Slippage,
    margin_fn=None,
) -> PipelineResult:
    """`backtest_spec` for a sold option or several options opened together
    (`nlt.engine.legs_backtest`).

    The same shape as `_backtest_option_spec`: signals from the index's candles
    in the F&O dataset (1-minute candles for a 1-minute strategy), the window
    clamped to the days with option prices and said so at the top, and a
    second run without charges. `margin_fn` defaults to the exchange-margin
    calculator.
    """
    from nlt.engine.legs_backtest import run_legs_backtest
    from nlt.engine.loader import load_option_bars, option_window
    from nlt.options.data import ParquetSource
    from nlt.options.execution import OptionExecution
    from nlt.report import method

    try:
        if margin_fn is None:
            try:
                margin_fn = span_margin_fn()
            except ImportError:
                return PipelineResult(
                    translation=translation,
                    spec=spec,
                    error="Selling options needs the exchange-margin calculator, which is not "
                    "installed yet.",
                )
        source = ParquetSource()
        start, end, window_notes = option_window(spec, start, end, source)
        model, cost_note = charge_model_for_spec(spec, cost_model_label)
        cf = _adapt_charge_fn(model)
        data_notes = [cost_note] if cost_note else []

        load_from = max(
            warmup.padded_start(spec, start),
            source.coverage(spec.instrument.symbol, spec.instrument.timeframe)[0],
        )
        all_bars = load_option_bars(spec, load_from, end, source)
        if all_bars.empty:
            return PipelineResult(
                translation=translation,
                spec=spec,
                data_notes=data_notes,
                window_notes=window_notes,
                error="There is no price history for that index and date range to test against.",
            )
        start_bar = warmup.first_bar_on_or_after(all_bars.index, start)
        if start_bar is None:
            return PipelineResult(
                translation=translation,
                spec=spec,
                window_notes=window_notes,
                error="There is no price history for that index and date range to test against.",
            )

        def run(charge_fn) -> BacktestResult:
            execution = OptionExecution(source, spec.instrument.symbol, spec.instrument.timeframe)
            result = run_legs_backtest(
                spec,
                all_bars,
                execution,
                margin_fn,
                capital=capital,
                charge_fn=charge_fn,
                slippage=slippage,
                trade_from=start_bar,
            )
            return warmup.trim(result, start_bar, capital)

        backtest = run(cf)
        before_charges = run(_adapt_charge_fn(ZeroCharges()))
        bars = all_bars.loc[start_bar:]
        data_notes = data_notes + list(backtest.warnings)
        note = warmup.warmup_note(spec, all_bars, start_bar)
        if note:
            data_notes.append(note)

        benchmark = buy_and_hold(
            bars, capital, charge_fn=cf, name=f"holding {spec.instrument.symbol}"
        )
        comparison = compare(backtest, benchmark, bars)
        verdict = assess(backtest, comparison)
        equity_index = backtest.equity.index
        stated = (
            method.assumptions(
                spec,
                capital=capital,
                charge_model=model,
                first=equity_index[0].date(),
                last=equity_index[-1].date(),
                slippage=slippage,
            )
            if len(equity_index)
            else []
        )
        spots, data_notes = method.weak_spots(spec, data_notes, slippage=slippage)
        return PipelineResult(
            translation=translation,
            spec=spec,
            bars=bars,
            backtest=backtest,
            benchmark=benchmark,
            comparison=comparison,
            verdict=verdict,
            before_charges=before_charges,
            assumptions=stated,
            weak_spots=spots,
            data_notes=data_notes,
            window_notes=window_notes,
        )
    except Exception as exc:  # noqa: BLE001 - the UI's last line of defence, as above
        logger.exception("option legs backtest failed for spec %r", getattr(spec, "name", None))
        return PipelineResult(
            translation=translation,
            spec=spec,
            error=f"The backtest could not be run: {exc}",
        )


def safe_call(fn: Callable[[], object], *, on_error: str) -> tuple[object | None, str | None]:
    """Runs `fn`, turning any exception into a plain-English message instead of a crash.

    Returns `(result, None)` on success or `(None, message)` on failure. Every
    place the dashboard touches the store, the data source, or the engine
    outside `run_pipeline` (re-runs, kill switch, etc.) should go through this
    so a raw traceback never reaches the browser.
    """
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001
        logger.error("safe_call failed: %s\n%s", exc, traceback.format_exc())
        return None, f"{on_error}: {exc}"


# ----------------------------------------------------------------- storage


def save_strategy(
    store: Store,
    spec: StrategySpec,
    result: PipelineResult | None = None,
    *,
    capital: float | None = None,
    cost_model_label: str | None = None,
    slippage: dict | None = None,
) -> int:
    """Saves the strategy and, when given, the backtest that was just shown.

    Recording the backtest as a completed run is what lets the strategy earn
    "Proven" later: the proving gate wants both a backtest and a paper run.
    Before this, nothing ever recorded a backtest, so no strategy could pass.

    `slippage` (a `Slippage.to_dict()`) is recorded alongside `capital` purely
    as a record of what was tested -- a backtest run is never re-run later the
    way a paper run is, so nothing downstream reads it back.
    """
    readback = describe_spec(spec)
    strategy_id = store.save_strategy(spec, readback)
    if result is not None and result.ok:
        run_id = store.start_run(
            strategy_id,
            "backtest",
            {"capital": capital, "cost_model_label": cost_model_label, "slippage": slippage},
        )
        store.save_trades(run_id, result.backtest.trades, spec.instrument.symbol)
        store.finish_run(run_id, result.backtest.metrics)
        if store.get_strategy(strategy_id)["state"] == "draft":
            store.set_state(strategy_id, "backtested")
    return strategy_id


def list_strategies_view(store: Store) -> list[dict]:
    """Empty database -> empty list, never an exception."""
    return store.list_strategies()


def list_runs_view(store: Store, strategy_id: int | None = None) -> list[dict]:
    return store.list_runs(strategy_id=strategy_id)


# ------------------------------------------------------------ paper trading


def kite_status() -> tuple[str, str]:
    """("connected" | "expired" | "not_connected" | "unreachable", plain-English line)."""
    from nlt.data import kite, kite_login

    if not kite.is_connected():
        return "not_connected", "Kite is not connected. Save your Kite login below."
    try:
        user_id = kite.check_connection()
    except kite.KiteTokenExpired:
        return "expired", "The saved Kite token has expired. Save your Kite login below."
    except kite.KiteLoginFailed as exc:
        return "expired", f"Could not log in to Kite. {exc}"
    except kite.KiteError as exc:
        return "unreachable", f"Could not reach Kite just now: {exc}"
    if kite_login.has_credentials():
        return "connected", f"Connected to Kite as {user_id}. It logs in again by itself."
    return "connected", f"Connected to Kite as {user_id}."


def save_kite_token(token: str) -> None:
    from nlt.data import kite

    kite.save_token(token)


def has_kite_login() -> bool:
    from nlt.data import kite_login

    return kite_login.has_credentials()


def save_kite_login(user_id: str, password: str, authenticator_secret: str) -> None:
    from nlt.data import kite_login

    kite_login.save_credentials(user_id, password, authenticator_secret)


def forget_kite_login() -> None:
    from nlt.data import kite_login

    kite_login.forget_credentials()


def paper_job_installed() -> bool:
    """Is the every-minute background job in this machine's crontab?"""
    import subprocess

    try:
        out = subprocess.run(["crontab", "-l"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return "# nlt-paper" in out.stdout


def _uses_iv_percentile(spec: StrategySpec) -> bool:
    from nlt.engine.legs_backtest import _walk
    from nlt.spec.models import IvPercentile

    conditions = [spec.entry] + ([spec.exit.condition] if spec.exit.condition else [])
    return any(isinstance(c, IvPercentile) for cond in conditions for c in _walk(cond))


def paper_problem(spec: StrategySpec) -> str | None:
    """Why this strategy cannot be paper traded, in plain English, or None."""
    if spec.instrument.is_option and _uses_iv_percentile(spec):
        if spec.instrument.symbol != "NIFTY":
            return (
                f"Strategies using IV percentile can only be paper traded on NIFTY for now. "
                "We have no option prices from January to mid-October 2026 to rank "
                f"{spec.instrument.symbol}'s volatility against, and India VIX, which stands "
                "in for it, measures NIFTY only."
            )
    if spec.instrument.is_universe and spec.instrument.timeframe != "1d":
        return "Stock baskets can only run on daily candles for now."
    return None


def paper_notes(spec: StrategySpec) -> list[str]:
    """What is different when this strategy is paper traded rather than backtested,
    said before Start is pressed."""
    if not spec.instrument.is_option:
        return []
    notes = ["Option prices come from Zerodha as they happen."]
    if not (spec.schedule.intraday and spec.instrument.timeframe != "1d"):
        # A strategy that squares off each day at 15:15 does so on expiry day
        # too, in paper as in a backtest; only one holding overnight differs.
        notes.append(
            "One difference from the description above: in paper trading, an option still "
            "open on its expiry day is NOT closed at the square-off time. It is held to the "
            "3:40 PM close and valued at its last traded price, then shown as Expired. "
            "Backtests keep the square-off rule."
        )
    if _uses_iv_percentile(spec):
        notes.append(
            "IV percentile is read from India VIX here, not from NIFTY's own option prices as "
            "in the backtest: we have no option prices from January to mid-October 2026 to "
            "rank against. The two usually move together, but not always."
        )
    return notes


def option_feed():
    """Live option prices, exchange margin and India VIX for an option paper run."""
    from nlt.options.live import KiteClient, LiveOptionSource, vix_iv_fn
    from nlt.paper.runner import OptionFeed

    client = KiteClient()
    return OptionFeed(
        source=lambda now: LiveOptionSource(client, now),
        margin_fn=span_margin_fn(),
        iv_fn=lambda now: vix_iv_fn(client, now),
    )


def start_paper(
    store: Store,
    strategy_id: int,
    *,
    capital: float,
    cost_model_label: str,
    alert_only: bool = False,
    slippage: Slippage = ZERO_SLIPPAGE,
) -> int:
    """Starts a paper run, or -- with `alert_only` -- an alert-only run: the same
    engine and runner, but one that only ever tells you what it would have
    done. It records no paper money and never counts toward the proving gate
    (`Store.is_proven` excludes it at the database level, not just here).

    `slippage` is a run setting, stored in `params_json` (like `capital`) so
    every later pass of the runner uses exactly what this run started with.
    """
    spec = store.load_spec(strategy_id)
    problem = paper_problem(spec)
    if problem:
        raise ValueError(problem)
    if any(r["strategy_id"] == strategy_id for r in store.active_paper_runs()):
        raise ValueError("This strategy is already being paper traded.")
    started_at = pd.Timestamp.now(tz="Asia/Kolkata")
    run_id = store.start_run(
        strategy_id,
        "paper",
        {
            "capital": capital,
            "cost_model_label": cost_model_label,
            "started_at": started_at.isoformat(),
            "alert_only": alert_only,
            "slippage": slippage.to_dict(),
        },
        alert_only=alert_only,
    )
    if not alert_only:
        store.set_state(strategy_id, "paper")
    store.set_paper_status(run_id, "waiting", "Started. Waiting for the next finished candle.")
    return run_id


def check_paper_now(store: Store, run_id: int) -> str:
    """One pass for one run, right now, regardless of the schedule."""
    from nlt.data.kite import KiteSource
    from nlt.paper import runner

    run = store.get_run(run_id)
    spec = store.load_spec(run["strategy_id"])
    params = json.loads(run["params_json"] or "{}")
    model, _ = charge_model_for_spec(spec, params.get("cost_model_label", ""))
    source = KiteSource()
    from nlt.data import kite

    result = runner.step(
        store,
        run,
        fetch=lambda sym, tf, a, b: source.bars(sym, tf, a, b),
        charge_fn=_adapt_charge_fn(model),
        listed_expiries=kite.listed_expiries,
        option_feed=option_feed() if spec.instrument.is_option else None,
    )
    return result.message


def pause_paper(store: Store, run_id: int, reason: str = "paused from web dashboard") -> None:
    store.pause_paper_run(run_id, reason)


def resume_paper(store: Store, run_id: int, reason: str = "resumed from web dashboard") -> None:
    store.resume_paper_run(run_id, reason)


def paper_pause_summary(store: Store, run_id: int, run: dict, as_of: pd.Timestamp) -> dict:
    """How long this run has been paused in total (its own pause/resume history
    plus any overlapping global paper-kill-switch windows -- both produce the
    same `paused` skip decision, so both count as "paused" here) and how many
    signals that skipped. Plain numbers for `stop_paper`'s final result; the
    wording lives in `paused_summary_text`.
    """
    from nlt.paper.runner import pause_windows_for_run

    params = json.loads(run["params_json"] or "{}")
    started_at = pd.Timestamp(params["started_at"]).tz_convert("Asia/Kolkata")
    windows = pause_windows_for_run(store, run_id, params, started_at, as_of)
    seconds = sum(max((end - start).total_seconds(), 0) for start, end in windows)
    skipped = sum(
        1
        for d in store.paper_decisions(run_id)
        if d["kind"] == "signal_skipped" and d["reason_code"] == "paused"
    )
    return {"paused_seconds": seconds, "paused_signal_count": skipped}


def paused_summary_text(paused_seconds: float, paused_signal_count: int) -> str:
    if paused_seconds <= 0 and paused_signal_count == 0:
        return ""
    hours = paused_seconds / 3600
    if hours < 1:
        duration = f"{int(round(paused_seconds / 60))} minute(s)"
    else:
        duration = f"{hours:.1f} hour(s)"
    plural = "signal" if paused_signal_count == 1 else "signals"
    return (
        f"Paused for a total of {duration}. {paused_signal_count} entry {plural} "
        "became known while paused and were skipped, never queued for later."
    )


def stop_paper(store: Store, run_id: int) -> None:
    """Ends a paper run. A completed paper run is what the proving gate counts,
    even one that spent time paused -- but the result says so, including how
    many entries it missed because of it (see `paper_pause_summary`).

    An alert-only run never traded, so its stored metrics must not read as if
    it had: no equity, no P&L, no positions -- only how many alerts it sent.
    It is also never allowed to satisfy the proving gate (enforced again, at
    the database level, in `Store.is_proven`).
    """
    status = store.paper_status(run_id) or {}
    events = store.paper_events(run_id)
    run = store.get_run(run_id)
    alert_only = store.is_alert_only(run_id)
    if alert_only:
        metrics = {
            "alert_only": True,
            "signals_notified": sum(1 for e in events if e["kind"] == "signal"),
            "exits_notified": sum(1 for e in events if e["kind"] == "exit"),
        }
    else:
        metrics = json.loads(status.get("snapshot_json") or "{}")
    metrics["drift_events"] = sum(e["kind"] == "drift" for e in events)
    stopped_at = pd.Timestamp.now(tz="Asia/Kolkata")
    metrics["stopped_at"] = stopped_at.isoformat()
    pause_info = paper_pause_summary(store, run_id, run, stopped_at)
    metrics.update(pause_info)
    metrics["paused_summary"] = paused_summary_text(**pause_info)
    store.finish_run(run_id, metrics, status="complete")


_HEALTH_LABELS = {
    "ok": ("success", "Running"),
    "waiting": ("info", "Waiting for the next candle"),
    "paused": ("warning", "Paused"),
    "token_expired": ("error", "Kite token expired"),
    "not_connected": ("error", "Kite not connected"),
    "login_failed": ("error", "Kite login failed"),
    "feed_down": ("error", "Price feed problem"),
    "error": ("error", "Error"),
}


def paper_health_label(health: str) -> tuple[str, str]:
    """(streamlit message kind, short label) for a run's health."""
    return _HEALTH_LABELS.get(health, ("info", health))


def when(iso: str | None) -> str:
    if not iso:
        return ""
    return pd.Timestamp(iso).tz_convert("Asia/Kolkata").strftime("%d %b %H:%M")


def quantity_words(spec: StrategySpec | None, quantity: float | None) -> str:
    """How much was traded, the way the trader would say it.

    The engine counts futures in units of the quoted price -- 100 barrels for one
    crude lot, 100 lots of 10 grams for one gold lot -- because that is what makes
    price times quantity the contract value. Nobody orders "100 GOLD" though;
    Kite's order window and the contract note both say 1 lot.
    """
    if not quantity:
        return ""
    if spec is not None and spec.instrument.is_future:
        from nlt.data.futures import contract

        c = contract(spec.instrument.symbol)
        lots = quantity / c.multiplier
        each = f", {c.lot_text} each" if lots != 1 else f", {c.lot_text}"
        return f"{lots:g} lot{'s' if lots != 1 else ''} of {c.title} futures ({each[2:]}) "
    return f"{quantity:g} "


def describe_paper_event(
    event: dict, spec: StrategySpec | None = None, *, alert_only: bool = False
) -> str:
    """One line of the activity log, written from the recorded event alone.

    `alert_only` is for a strategy run that only ever tells you what it would
    have done (see `app.logic.start_paper`): the wording moves to "would buy"
    / "would sell" and drops every money figure, so this line can never be
    read as a fill or a result that happened.
    """
    detail = json.loads(event.get("detail_json") or "{}")
    kind = event["kind"]
    qty_text = quantity_words(spec, event.get("quantity"))
    if spec is not None and spec.instrument.is_future:
        event = {**event, "symbol": ""}  # the quantity words already name it
    if kind == "signal":
        return f"Signal: the entry rule was met on the {when(event['bar_time'])} candle. " + (
            "You were told it would buy at the next candle's open."
            if alert_only
            else "It is queued for the next candle's open, where the money and "
            "position-size checks are made."
        )
    if kind == "entry":
        if alert_only:
            verb = "Would buy" if event["direction"] == "long" else "Would sell short"
            return (
                f"{verb} {qty_text}{event['symbol']}".rstrip()
                + f" at the open of the {when(event['bar_time'])} candle "
                "(a simulated position -- nothing is bought or sold)."
            )
        verb = "Bought" if event["direction"] == "long" else "Sold short"
        return (
            f"{verb} {qty_text}{event['symbol']}".rstrip() + f" at {format_inr(event['price'])} "
            f"(the open of the {when(event['bar_time'])} candle)."
        )
    if kind == "exit":
        reason = _EXIT_WORDS.get(detail.get("reason"), detail.get("reason", "exit"))
        if alert_only:
            verb = "Would sell" if event["direction"] == "long" else "Would buy back"
            return (
                f"{verb} {qty_text}{event['symbol']}".rstrip()
                + f" on the {when(event['bar_time'])} candle -- {reason} "
                "(the simulated position it would have held)."
            )
        pnl = detail.get("net_pnl")
        pnl_text = (
            f" Result: {'+' if pnl >= 0 else '-'}{format_inr(abs(pnl))}." if pnl is not None else ""
        )
        verb = "Sold" if event["direction"] == "long" else "Bought back"
        return (
            f"{verb} {qty_text}{event['symbol']}".rstrip() + f" at {format_inr(event['price'])} "
            f"on the {when(event['bar_time'])} candle -- {reason}.{pnl_text}"
        )
    if kind == "drift":
        return (
            f"Kite's prices changed after this was recorded ({detail.get('event')} "
            f"{detail.get('key')}): recorded {detail.get('recorded')}, now {detail.get('now')}. "
            "The original record stands."
        )
    if kind == "note":
        # Written once per health change (see `nlt.paper.runner._set_status`),
        # not every pass, so this line marks a transition, not a status quo.
        return detail.get("message", "Status changed.")
    return detail.get("note", kind)


_EXIT_WORDS = {
    "stop": "stop loss hit",
    "trailing_stop": "trailing stop hit",
    "target": "target reached",
    "condition": "exit rule met",
    "max_bars": "time limit reached",
    "square_off": "end-of-day square-off",
    "expiry": "it expired",
    "end_of_data": "still open when the test ended",
    "combined_target": "whole-position target reached",
    "combined_stop": "whole-position stop loss hit",
}


# A skipped signal's reason, in words a non-technical reader can act on: not
# just "what rule fired" but "why nothing happened", which is the entire
# point of showing it at all (see the module docstring's E1 live activity feed).
_SKIP_REASON_WORDS = {
    "concurrency": (
        "the maximum number of positions this strategy may hold at once was already open"
    ),
    "concurrency_at_signal": (
        "the maximum number of positions this strategy may hold at once was "
        "already open the moment the signal appeared"
    ),
    "capital": "there was not enough spare money to buy even one lot or share",
    "position_cap": (
        "one position may not be more than the account's per-position limit, and a "
        "single lot or share already costs more than that"
    ),
    "expiry": (
        "it fired on the last candle of an expiring contract -- it would have filled "
        "in the next contract at the old contract's levels"
    ),
    "boundary": (
        "it fired on the last candle of the trading session -- filling it would have "
        "meant holding overnight, which this intraday strategy never asked for"
    ),
    "weekday": "it fired on a day of the week this strategy is not allowed to trade on",
    "no_entry_after": "it fired after the time of day this strategy stops opening new trades",
    "paused": "the strategy was paused for this stretch of time",
    "slots": ("the basket's maximum number of positions open at once was already reached"),
}


def describe_decision(decision: dict, spec: StrategySpec | None = None) -> str:
    """One line of the activity feed for a `paper_decision` row.

    `describe_paper_event`'s "signal" row only ever appears when the engine
    actually queued an entry (see `nlt.paper.runner.step`). This covers the
    other half: every signal the engine saw and chose NOT to act on, each
    with its own stated reason -- so "it looked like it should have traded"
    never has to be guessed at from silence, which is the mistake this
    feature exists to stop making. Text is entirely mechanical, keyed off
    `reason_code`, never generated by a model.
    """
    symbol = decision.get("symbol") or (spec.instrument.symbol if spec is not None else "")
    when_text = when(decision.get("bar_time"))
    prefix = f"{symbol}: " if symbol else ""
    if decision["kind"] == "signal_taken":
        return (
            f"{prefix}entry rule met on the {when_text} candle -- queued for the next "
            "candle's open, where the money and position-size checks are made."
        )
    reason = _SKIP_REASON_WORDS.get(
        decision.get("reason_code"), decision.get("reason_code") or "skipped"
    )
    return f"{prefix}entry rule met on the {when_text} candle, but skipped: {reason}."


def trade_timeline(
    events: list[dict],
    spec: StrategySpec | None = None,
    *,
    open_pnl: float | None = None,
    open_pnl_as_of: str | None = None,
) -> list[str]:
    """The step-by-step story of one paper trade, built only from recorded rows.

    `events` is the subset of one run's `paper_event` rows that belong to a
    single trade: the matching 'signal' row (if any -- an older paper run
    recorded before this feature may not have one), the 'entry' row keyed
    `event_key`, and the 'exit' row sharing that same key if the trade has
    closed. Nothing here is re-derived from the engine; it is only ever the
    words for what was already written down, which is the same safety rule
    `describe_paper_event` and `describe_decision` already follow.

    `open_pnl` is the trade's current mark-to-market, read from the run's
    `paper_status` snapshot by the caller -- it is never computed here, so a
    stale or missing snapshot shows as "not available" rather than a guess.
    """
    by_kind = {e["kind"]: e for e in events}
    entry = by_kind.get("entry")
    if entry is None:
        return []
    detail = json.loads(entry.get("detail_json") or "{}")
    steps: list[str] = []

    signal_time = detail.get("signal_time")
    reason = detail.get("reason") or "the entry rule"
    steps.append(f"Signal candle closed at {when(signal_time)}: {reason}.")

    steps.append(
        "Queued for the next candle's open, where the money and position-size checks are made."
    )

    qty_text = quantity_words(spec, entry.get("quantity"))
    symbol = "" if (spec is not None and spec.instrument.is_future) else (entry.get("symbol") or "")
    raw_entry = detail.get("raw_entry_price")
    entry_price = entry.get("price")
    # Only claim slippage was applied when it actually moved the price -- at
    # zero slippage the fill and the raw price are the same, and saying "plus
    # slippage" there would be untrue.
    slippage_bit = (
        f" (raw price {format_inr(raw_entry)} plus slippage)"
        if raw_entry is not None and entry_price is not None and abs(raw_entry - entry_price) > 1e-9
        else ""
    )
    steps.append(
        f"Filled {qty_text}{symbol}".rstrip() + f" at {format_inr(entry.get('price'))}"
        f"{slippage_bit} on the {when(entry.get('bar_time'))} candle."
    )

    stop = detail.get("stop_level")
    target = detail.get("target_level")
    stop_text = format_inr(stop) if stop is not None else "no stop"
    target_text = format_inr(target) if target is not None else "no target"
    steps.append(f"Stop at {stop_text}, target at {target_text}.")

    exit_ = by_kind.get("exit")
    if exit_ is not None:
        exit_detail = json.loads(exit_.get("detail_json") or "{}")
        reason_text = _EXIT_WORDS.get(
            exit_detail.get("reason"), exit_detail.get("reason") or "exit"
        )
        both_bit = (
            " The stop and the target were both touched on this candle; the stop is "
            "assumed to have come first."
            if exit_detail.get("both_hit")
            else ""
        )
        steps.append(
            f"Exited at {format_inr(exit_.get('price'))} on the {when(exit_.get('bar_time'))} "
            f"candle because {reason_text}.{both_bit}"
        )
        net_pnl = exit_detail.get("net_pnl")
        charges = exit_detail.get("charges")
        if net_pnl is not None:
            sign = "+" if net_pnl >= 0 else "-"
            charges_bit = f" Charges came to {format_inr(charges)}." if charges is not None else ""
            steps.append(
                f"Net result: {sign}{format_inr(abs(net_pnl))} after charges.{charges_bit}"
            )
    else:
        open_bit = ""
        if open_pnl is not None:
            sign = "+" if open_pnl >= 0 else "-"
            # The snapshot is only as fresh as the last successful check; a
            # paused run or a dead feed leaves it hours old. Say which candle.
            as_of = f" as of the {when(open_pnl_as_of)} candle" if open_pnl_as_of else ""
            open_bit = f" Open profit or loss{as_of}: {sign}{format_inr(abs(open_pnl))}."
        steps.append(f"Still open.{open_bit}")

    return steps


def activity_feed(store: Store) -> list[dict]:
    """One merged, newest-first feed across every paper run and every one-off
    order. Built only from recorded rows (`paper_event`, `paper_decision`,
    `one_off_event`) and this module's own describe functions -- no line here
    is written fresh, so the feed can never say anything those records do not
    already back up.

    Each item: time (the bar's own time, falling back to when it was
    observed), source ("strategy"/"order"), source_name, source_id (the run id
    or order id), kind, text, skip (a signal the engine chose not to act on),
    and trade_key (set only on a strategy's entry/exit row, for opening its
    step-by-step timeline).
    """
    from app import orders_view  # deferred: orders_view imports this module too

    items: list[dict] = []
    for run in store.list_runs(mode="paper"):
        strategy = store.get_strategy(run["strategy_id"])
        name = strategy["name"] if strategy else ""
        spec = store.load_spec(run["strategy_id"])
        alert_only = store.is_alert_only(run["id"])
        for e in store.paper_events(run["id"]):
            items.append(
                {
                    "time": e["bar_time"] or e["observed_at"],
                    "source": "strategy",
                    "source_name": name,
                    "source_id": run["id"],
                    "kind": e["kind"],
                    "text": describe_paper_event(e, spec, alert_only=alert_only),
                    "skip": False,
                    "trade_key": e["event_key"] if e["kind"] in ("entry", "exit") else None,
                }
            )
        for d in store.paper_decisions(run["id"]):
            items.append(
                {
                    "time": d["bar_time"] or d["observed_at"],
                    "source": "strategy",
                    "source_name": name,
                    "source_id": run["id"],
                    "kind": d["kind"],
                    "text": describe_decision(d, spec),
                    "skip": d["kind"] == "signal_skipped",
                    "trade_key": None,
                }
            )

    try:
        book = orders_view.runner.current_book(store)
    except Exception:
        # One unreadable order record must not blank the whole feed: say so,
        # and still show every strategy's activity.
        items.append(
            {
                "time": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
                "source": "order",
                "source_name": "One-off orders",
                "source_id": 0,
                "kind": "note",
                "text": "One-off orders could not be read just now, so they are missing "
                "from this list. Open the One-off orders page for details.",
                "skip": False,
                "trade_key": None,
            }
        )
        items.sort(key=lambda it: pd.Timestamp(it["time"]), reverse=True)
        return items
    pressed = frozenset(
        o.id for o in book.orders if isinstance(o.order.trigger, orders_view.NowTrigger)
    )
    by_order: dict[int, list[dict]] = {}
    for e in store.one_off_events():
        by_order.setdefault(e["order_id"], []).append(e)
    for row in store.one_off_orders():
        oid = row["id"]
        for e in by_order.get(oid, []):
            if e["kind"] == "finished":
                continue
            items.append(
                {
                    "time": e["bar_time"] or e["observed_at"],
                    "source": "order",
                    "source_name": row["sentence"],
                    "source_id": oid,
                    "kind": e["kind"],
                    "text": orders_view.describe_event(e, oid, pressed),
                    "skip": False,
                    "trade_key": None,
                }
            )

    items.sort(key=lambda it: pd.Timestamp(it["time"]), reverse=True)
    return items


# ------------------------------------------------------------------ CSV


TRADE_CSV_COLUMNS = [
    "symbol",
    "direction",
    "entered",
    "entry_price",
    "exited",
    "exit_price",
    "quantity",
    "why_it_closed",
    "profit_before_charges",
    "charges",
    "profit_after_charges",
]


def trades_csv(result: PipelineResult) -> str:
    """Every trade of a run as CSV, for a spreadsheet (competitor review C4).

    Plain numbers (no "Rs", no commas) so a spreadsheet can add them up; the
    columns are the same trades the results screen lists, and the last column
    adds up to the run's profit after charges.
    """
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(TRADE_CSV_COLUMNS)
    spec = result.backtest.spec
    # A daily candle has no time of day; printing 00:00 would suggest one.
    when = "%Y-%m-%d" if spec.instrument.timeframe == "1d" else "%Y-%m-%d %H:%M"
    for t in result.backtest.trades:
        writer.writerow(
            [
                t.symbol or spec.instrument.symbol,
                "buy" if t.direction == "long" else "short",
                t.entry_time.strftime(when),
                round(float(t.entry_price), 2),
                t.exit_time.strftime(when),
                round(float(t.exit_price), 2),
                int(t.quantity),
                _EXIT_WORDS.get(t.exit_reason, t.exit_reason),
                round(float(t.gross_pnl), 2),
                round(float(t.charges), 2),
                round(float(t.net_pnl), 2),
            ]
        )
    return buf.getvalue()


# ------------------------------------------------------- adjust before running


TIMEFRAMES = ["1m", "3m", "5m", "15m", "30m", "1h", "1d"]

# Which readback part each box sets, so the labels can say "You said".
_EDIT_PARTS = {
    "stop_pct": "exit.stop",
    "trailing_stop_pct": "exit.trailing_stop",
    "target_pct": "exit.target",
    "timeframe": "timeframe",
    # Option legs. The combined exits have their own readback parts so that
    # changing one never relabels the per-leg line beside it.
    "leg_stop_pct": "exit.stop",
    "combined_target_pct": "exit.combined_target",
    "combined_stop_pct": "exit.combined_stop",
}

_CANDLE_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60}


def _has_at_time(condition) -> bool:
    kind = getattr(condition, "kind", None)
    if kind == "at_time":
        return True
    if kind in ("all", "any"):
        return any(_has_at_time(c) for c in condition.conditions)
    if kind == "not":
        return _has_at_time(condition.condition)
    return False


def allowed_timeframes(spec: StrategySpec) -> list[str] | None:
    """The candle sizes an option-legs strategy can use; None for any other strategy.

    An entry "at 09:20" is the candle that ends at 09:20, so the candle length
    must divide the minutes since 09:15 (the engine refuses otherwise), and
    daily candles have no time of day at all.
    """
    if not spec.instrument.legs:
        return None
    if not _has_at_time(spec.entry):
        return list(TIMEFRAMES)
    times = []

    def collect(c) -> None:
        if getattr(c, "kind", None) == "at_time":
            times.append(c.time)
        for child in getattr(c, "conditions", None) or []:
            collect(child)
        if getattr(c, "kind", None) == "not":
            collect(c.condition)

    collect(spec.entry)
    minutes = [t.hour * 60 + t.minute - (9 * 60 + 15) for t in times]
    return [tf for tf, m in _CANDLE_MINUTES.items() if all(x % m == 0 for x in minutes)]


_CANDLE_NAMES = [
    ("1m", "1-minute"),
    ("3m", "3-minute"),
    ("5m", "5-minute"),
    ("15m", "15-minute"),
    ("30m", "30-minute"),
    ("1h", "hourly"),
    ("1d", "daily"),
]


class AdjustError(ValueError):
    """A box holds something the strategy cannot run with; the text says why."""


def _percent(raw, label: str, *, optional: bool, top: float = 50) -> float | None:
    text = str(raw if raw is not None else "").strip().removesuffix("%").strip()
    if not text:
        if optional:
            return None
        raise AdjustError(f"The {label} cannot be empty: a strategy always needs one.")
    try:
        value = float(text)
    except ValueError:
        raise AdjustError(f"The {label} must be a number, like 1.5.") from None
    if not 0 < value <= top:
        raise AdjustError(f"The {label} must be more than 0% and at most {top:g}%.")
    return value


def adjust_spec(
    spec: StrategySpec, edits: dict, sources: dict[str, str] | None
) -> tuple[StrategySpec, dict[str, str] | None, dict[str, str]]:
    """The confirmed strategy with the form's boxes applied (competitor review A4).

    Returns (spec, sources, the boxes as typed). Only the boxes the strategy
    already has can change: a stop loss can be moved but never removed, and a
    trailing stop stays a trailing stop -- the form edits the strategy, it does
    not rewrite it. Every change is checked by the same rules the reader uses,
    and the readback is regenerated from the result, so it always says what
    will be tested. A box the user set is labelled "You said".
    """
    from nlt.translate.readback import SAID
    from nlt.translate.rules import _vwap_refusal

    exit_updates: dict = {}
    typed: dict[str, str] = {}
    legs = list(spec.instrument.legs)
    new_legs = None
    if legs:
        sold = [leg for leg in legs if leg.side == "sell"]
        same_stop = len({leg.stop_pct for leg in sold}) == 1 and sold[0].stop_pct is not None
        if "leg_stop_pct" in edits and sold and same_stop:
            value = _percent(
                edits["leg_stop_pct"], "stop loss on each sold option", optional=False, top=1000
            )
            new_legs = [
                leg.model_copy(update={"stop_pct": value}) if leg.side == "sell" else leg
                for leg in legs
            ]
        if "combined_target_pct" in edits and sold:
            exit_updates["combined_target_pct"] = _percent(
                edits["combined_target_pct"],
                "take profit",
                optional=True,
                top=100,
            )
        if "combined_stop_pct" in edits and spec.exit.combined_stop_pct is not None:
            exit_updates["combined_stop_pct"] = _percent(
                edits["combined_stop_pct"], "combined stop", optional=False, top=1000
            )
    elif "stop_pct" in edits and spec.exit.stop_pct is not None:
        exit_updates["stop_pct"] = _percent(edits["stop_pct"], "stop loss", optional=False)
    if "trailing_stop_pct" in edits and spec.exit.trailing_stop_pct is not None and not legs:
        exit_updates["trailing_stop_pct"] = _percent(
            edits["trailing_stop_pct"], "trailing stop", optional=False
        )
    if "target_pct" in edits and not legs:
        exit_updates["target_pct"] = _percent(edits["target_pct"], "target", optional=True)

    instrument = spec.instrument
    if new_legs is not None:
        instrument = instrument.model_copy(update={"legs": new_legs})
    if "timeframe" in edits:
        timeframe = str(edits["timeframe"])
        if timeframe not in TIMEFRAMES:
            raise AdjustError("That candle size is not one this platform runs.")
        allowed = allowed_timeframes(spec)
        if allowed is not None and timeframe not in allowed:
            raise AdjustError(
                "This strategy enters at a set time of day, and "
                f"{dict(_CANDLE_NAMES).get(timeframe, timeframe)} candles do not end at that "
                "time. Pick a candle size that does: "
                + ", ".join(dict(_CANDLE_NAMES)[t] for t in allowed)
                + "."
            )
        if timeframe != "1d" and instrument.is_universe:
            raise AdjustError("A basket of stocks can only be tested on daily candles.")
        instrument = instrument.model_copy(update={"timeframe": timeframe})
        indicators = [{"type": i.type, "params": dict(i.params)} for i in spec.indicators]
        refusal = _vwap_refusal(
            indicators,
            {"symbol": instrument.symbol, "trade_as": instrument.trade_as, "timeframe": timeframe},
        )
        if refusal is not None:
            raise AdjustError(refusal.text)

    for key in _EDIT_PARTS:
        in_effect = (
            key in exit_updates
            or key == "timeframe"
            or (key == "leg_stop_pct" and new_legs is not None)
        )
        if key in edits and in_effect:
            typed[f"form.{key}"] = str(edits[key])

    try:
        adjusted = StrategySpec.model_validate(
            {
                **spec.model_dump(),
                "instrument": instrument.model_dump(),
                "exit": {**spec.exit.model_dump(), **exit_updates},
            }
        )
    except ValidationError as exc:  # the rule broken, without pydantic's wrapping
        reason = exc.errors()[0]["msg"].removeprefix("Value error, ")
        raise AdjustError(f"That does not make a valid strategy: {reason}") from None
    except Exception as exc:
        raise AdjustError(f"That does not make a valid strategy: {exc}") from None

    if sources is not None:
        sources = dict(sources)
        for key, part in _EDIT_PARTS.items():
            if f"form.{key}" in typed:
                if key in ("target_pct", "combined_target_pct") and exit_updates.get(key) is None:
                    sources.pop(part, None)
                else:
                    sources[part] = SAID
    return adjusted, sources, typed


# ------------------------------------------------------- ready-made strategies


# The library (competitor review B4): ready-made strategies shown as cards with
# their own tested numbers. Each must read with no questions and take trades at
# the default capital (tests/test_library.py). Tapping one puts its sentence in
# the box, where it can be changed like anything the user typed.
LIBRARY: list[dict[str, str]] = [
    {"title": "NIFTY oversold bounce", "sentence": EXAMPLE_STRATEGIES[0]},
    {"title": "NIFTY moving-average crossover", "sentence": EXAMPLE_STRATEGIES[1]},
    {"title": "RELIANCE oversold bounce", "sentence": EXAMPLE_STRATEGIES[2]},
    {"title": "INFY EMA crossover", "sentence": EXAMPLE_STRATEGIES[3]},
    {"title": "TCS trend follower", "sentence": EXAMPLE_STRATEGIES[4]},
    {
        "title": "MARUTI 20-day breakout",
        "sentence": "Buy MARUTI when the close is above the highest high of the last 20 days. "
        "Sell when the close falls below the lowest low of the last 10 days. Stop loss 3%. "
        "Backtest the last 5 years.",
    },
]


def library_card(entry: dict[str, str], tested_to: dt.date) -> dict:
    """One card: the strategy run as the screen would run it, up to `tested_to`.

    The holding return sits beside the strategy's, and serious warnings are
    flagged on the card itself, so a card cannot look better than its report.
    """
    translation = translate_text(entry["sentence"])
    card: dict = {"title": entry["title"], "sentence": entry["sentence"]}
    if translation.spec is None or translation.period is None:
        return {**card, "error": "This strategy could not be read."}
    dates = period_dates(translation.period, tested_to)
    start = dt.date.fromisoformat(dates["start"])
    result = backtest_spec(translation.spec, start=start, end=tested_to, translation=translation)
    if result.error:
        return {**card, "error": result.error}
    m = result.backtest.metrics
    equity = result.backtest.equity
    return {
        **card,
        "name": translation.spec.name,
        "return": format_signed_pct(result.comparison.strategy_return_pct),
        "holding_return": format_signed_pct(result.comparison.benchmark_return_pct),
        "benchmark_name": result.benchmark.name,
        "win_rate": format_pct(m.get("win_rate_pct")),
        "worst_drop": format_pct(m.get("max_drawdown_pct")),
        "trades": int(m.get("total_trades") or 0),
        "first": f"{equity.index[0]:%d %b %Y}",
        "last": f"{equity.index[-1]:%d %b %Y}",
        "serious_warning": any(f.severity == "critical" for f in result.verdict.flags),
    }


def library_cards(
    cache_path,
    today: dt.date,
    compute: Callable[[dict, dt.date], dict] | None = None,
) -> dict:
    """The cards, recomputed at most once a day and kept on disk between restarts.

    Tested to the day before `today`: today's daily candle is still forming.
    """
    from pathlib import Path

    compute = compute or library_card
    path = Path(cache_path)
    tested_to = today - dt.timedelta(days=1)
    try:
        cached = json.loads(path.read_text())
        if cached.get("tested_to") == tested_to.isoformat() and len(cached["cards"]) == len(
            LIBRARY
        ):
            return cached
    except (OSError, ValueError, KeyError):
        pass
    cards = [compute(entry, tested_to) for entry in LIBRARY]
    out = {
        "tested_to": tested_to.isoformat(),
        "capital_text": format_inr(DEFAULT_CAPITAL),
        "cards": cards,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------ readable option names

_MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_WEEKLY_MONTH_CODES = {**{str(i): i for i in range(1, 10)}, "O": 10, "N": 11, "D": 12}
_OPTION_SYMBOL_BODY = r"([A-Z][A-Z&-]*?)(\d{2})(?:([1-9OND])(\d{2})|([A-Z]{3}))(\d{1,6})(CE|PE)"
_OPTION_SYMBOL = re.compile(rf"^{_OPTION_SYMBOL_BODY}$")
_OPTION_SYMBOL_IN_TEXT = re.compile(rf"(?<![A-Z0-9]){_OPTION_SYMBOL_BODY}(?![A-Z0-9])")


def _readable_match(m: re.Match) -> str:
    name, _yy, code, day, mon3, strike, side = m.groups()
    if code is not None:
        month = _WEEKLY_MONTH_CODES[code]
        if not 1 <= int(day) <= 31:
            return m.group(0)
        when_text = f"{int(day)} {_MONTH_NAMES[month - 1]}"
    else:
        if mon3.capitalize() not in _MONTH_NAMES:
            return m.group(0)
        when_text = f"{mon3.capitalize()} monthly"
    return f"{name} {int(strike):,} {side} \u00b7 {when_text}"


def readable_symbol(symbol: str) -> str:
    """An exchange option symbol in words ("NIFTY26O1923000CE" -> "NIFTY 23,000 CE
    \u00b7 19 Oct"). A " + " joined list is converted part by part. Anything not
    recognised (stocks, futures, a plain index name) comes back unchanged."""
    if not symbol:
        return symbol
    parts = symbol.split(" + ")
    out = []
    for part in parts:
        m = _OPTION_SYMBOL.match(part.strip())
        out.append(_readable_match(m) if m else part)
    return " + ".join(out)


def readable_text(text: str) -> str:
    """Every option symbol inside a sentence, put into words."""
    return _OPTION_SYMBOL_IN_TEXT.sub(_readable_match, text)
