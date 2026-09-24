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
import logging
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from nlt.costs.charges import (
    ChargeModel,
    NseEquityDeliveryCharges,
    NseFuturesCharges,
    NseOptionsCharges,
    ZeroCharges,
)
from nlt.costs.charges import charge_fn as _adapt_charge_fn
from nlt.data.yahoo import YahooSource
from nlt.engine.backtest import BacktestResult, run_backtest
from nlt.spec.models import StrategySpec
from nlt.store.db import Store
from nlt.translate.readback import describe, format_inr
from nlt.translate.rules import Question, TranslationResult, parse

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
    "charge_model_for_label",
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

DEFAULT_CAPITAL = 100_000.0

# Every example is verified (in tests/test_app.py) to parse into a spec with
# no follow-up questions -- a "try this" button that itself needs clarifying
# would be a bad first impression.
EXAMPLE_STRATEGIES = [
    "buy nifty when rsi cracks 30, target 2%, stop loss 1%",
    "buy nifty when 20 sma crosses above 50 sma, target 3%, stop loss 1.5%",
    "short banknifty on death cross, target 2%, stop 1%",
    "buy nifty when close is above 200 ema, target 5%, trailing stop 2%",
]

COST_MODEL_LABELS: dict[str, ChargeModel | None] = {
    "Index options (NIFTY/BANKNIFTY weekly)": NseOptionsCharges(),
    "Index futures": NseFuturesCharges(),
    "Equity delivery": NseEquityDeliveryCharges(),
    "No costs (for comparison only)": ZeroCharges(),
}


def charge_model_for_label(label: str) -> ChargeModel:
    """Looks up a cost model by its dropdown label, defaulting to options costs."""
    return COST_MODEL_LABELS.get(label, NseOptionsCharges())


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
    """Wraps `nlt.translate.rules.parse` so a bug in the parser cannot crash the page.

    `parse` is written to never raise on bad input -- unparseable text comes
    back as `unparsed` fragments and `Question`s -- but this dashboard is the
    only thing standing between that guarantee and a stack trace on a
    non-technical user's screen, so it is wrapped anyway.
    """
    try:
        return parse(text, answers=answers or {})
    except Exception:  # pragma: no cover -- defensive; parse() is designed not to raise
        logger.exception("translate_text: parse() raised on %r", text)
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
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.backtest is not None


def run_pipeline(
    text: str,
    *,
    capital: float = DEFAULT_CAPITAL,
    cost_model_label: str = next(iter(COST_MODEL_LABELS)),
    symbol: str | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
    answers: dict[str, str] | None = None,
) -> PipelineResult:
    """Plain English in, verdict out -- the function `run_backtest_flow` in the brief.

    Every downstream step (data load, engine, benchmark, verdict) is wrapped
    so a real bug never reaches the UI as a traceback; it comes back as
    `PipelineResult.error` instead.
    """
    translation = translate_text(text, answers)
    if translation.spec is None:
        return PipelineResult(translation=translation)

    spec = translation.spec
    try:
        bars = load_bars(symbol or spec.instrument.symbol, start, end)
        if bars.empty:
            return PipelineResult(
                translation=translation,
                spec=spec,
                error="There is no price history for that symbol and date range to test against.",
            )

        model = charge_model_for_label(cost_model_label)
        cf = _adapt_charge_fn(model)

        backtest = run_backtest(spec, bars, capital=capital, charge_fn=cf)
        benchmark = buy_and_hold(bars, capital, charge_fn=cf)
        comparison = compare(backtest, benchmark, bars)
        verdict = assess(backtest, comparison)

        return PipelineResult(
            translation=translation,
            spec=spec,
            bars=bars,
            backtest=backtest,
            benchmark=benchmark,
            comparison=comparison,
            verdict=verdict,
        )
    except Exception as exc:  # noqa: BLE001 - deliberately broad; this is the UI's last line of defence
        logger.exception("run_pipeline failed for spec %r", getattr(spec, "name", None))
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


def save_strategy(store: Store, spec: StrategySpec) -> int:
    readback = describe_spec(spec)
    return store.save_strategy(spec, readback)


def list_strategies_view(store: Store) -> list[dict]:
    """Empty database -> empty list, never an exception."""
    return store.list_strategies()


def list_runs_view(store: Store, strategy_id: int | None = None) -> list[dict]:
    return store.list_runs(strategy_id=strategy_id)
