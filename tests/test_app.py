"""Tests for `app/logic.py` -- the dashboard's decisions, not its widgets.

Streamlit widgets need a live script-run context, so nothing here drives
`app/main.py`'s UI directly. What matters for a non-technical user is that
the *logic* behind every widget is right: text becomes a spec or a clear
question, a backtest becomes a verdict, a number becomes a readable string,
an empty database renders instead of crashing. That is what is tested here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app import logic
from nlt.store.db import Store


@pytest.fixture
def store(tmp_path) -> Store:
    return Store(tmp_path / "test.sqlite")


# --------------------------------------------------------------- end-to-end


def test_end_to_end_produces_a_populated_result():
    """Text in -> spec -> backtest -> comparison -> verdict, all populated."""
    result = logic.run_pipeline("buy nifty when rsi cracks 30, target 2%, stop loss 1%")

    assert result.error is None
    assert result.translation.spec is not None
    assert result.spec is not None
    assert result.bars is not None and not result.bars.empty
    assert result.backtest is not None
    assert result.benchmark is not None
    assert result.comparison is not None
    assert result.verdict is not None
    assert isinstance(result.verdict.summary, str) and result.verdict.summary
    assert isinstance(result.verdict.passed, bool)
    assert result.ok is True


@pytest.mark.parametrize("example", logic.EXAMPLE_STRATEGIES)
def test_every_example_strategy_parses_cleanly(example: str) -> None:
    """The 'try this' buttons on page 1 must never themselves need clarifying."""
    translation = logic.translate_text(example)
    assert translation.spec is not None, f"{example!r} produced questions: {translation.questions}"
    assert not translation.unparsed


# ------------------------------------------------------------------ questions


def test_missing_stop_loss_returns_questions_not_a_spec() -> None:
    translation = logic.translate_text("buy nifty when rsi cracks 30")
    assert translation.spec is None
    assert translation.questions
    fields = {q.field for q in translation.questions}
    assert "exit.stop_pct" in fields


def test_answering_questions_yields_a_spec() -> None:
    first_pass = logic.translate_text("buy nifty when rsi cracks 30")
    assert first_pass.spec is None

    answers = logic.answers_from_questions(
        first_pass.questions, {"exit.stop_pct": "1", "exit.rule": ""}
    )
    second_pass = logic.translate_text("buy nifty when rsi cracks 30", answers)
    assert second_pass.spec is not None
    assert second_pass.spec.exit.stop_pct == 1.0


def test_answers_from_questions_drops_empty_values() -> None:
    from nlt.translate.rules import Question

    questions = [
        Question(text="a", why="w", suggestion=None, field="exit.stop_pct"),
        Question(text="b", why="w", suggestion=None, field="exit.target_pct"),
    ]
    answers = logic.answers_from_questions(questions, {"exit.stop_pct": "1", "exit.target_pct": ""})
    assert answers == {"exit.stop_pct": "1"}


# ------------------------------------------------------------------- errors


def test_unparseable_description_returns_friendly_result_never_raises() -> None:
    translation = logic.translate_text("do something clever with the moon phases and unicorns")
    assert translation.spec is None
    assert translation.unparsed or translation.questions


def test_empty_description_returns_a_question_not_an_exception() -> None:
    translation = logic.translate_text("")
    assert translation.spec is None
    assert translation.questions


def test_pipeline_never_raises_on_a_bad_symbol() -> None:
    """A spec pointing at a symbol with no cached/downloadable data must come
    back as `PipelineResult.error`, never propagate an exception to the page."""
    result = logic.run_pipeline(
        "buy nifty when rsi cracks 30, target 2%, stop loss 1%",
        symbol="NOTASYMBOL",
    )
    assert result.spec is not None
    assert result.error is not None
    assert result.backtest is None


def test_safe_call_turns_an_exception_into_a_message() -> None:
    def boom():
        raise RuntimeError("kaboom")

    value, error = logic.safe_call(boom, on_error="Could not do the thing")
    assert value is None
    assert error == "Could not do the thing: kaboom"


def test_safe_call_passes_through_on_success() -> None:
    value, error = logic.safe_call(lambda: 42, on_error="unused")
    assert value == 42
    assert error is None


# --------------------------------------------------------------- empty db


def test_list_strategies_view_empty_db_returns_empty_list(store: Store) -> None:
    assert logic.list_strategies_view(store) == []


def test_list_runs_view_empty_db_returns_empty_list(store: Store) -> None:
    assert logic.list_runs_view(store) == []


def test_audit_trail_empty_db_returns_empty_list(store: Store) -> None:
    assert store.audit_trail() == []


def test_kill_switch_starts_disengaged(store: Store) -> None:
    assert store.kill_switch_engaged() is False


def test_save_strategy_then_list_round_trips(store: Store) -> None:
    result = logic.run_pipeline("buy nifty when rsi cracks 30, target 2%, stop loss 1%")
    assert result.spec is not None

    strategy_id = logic.save_strategy(store, result.spec)
    strategies = logic.list_strategies_view(store)

    assert len(strategies) == 1
    assert strategies[0]["id"] == strategy_id
    assert strategies[0]["name"] == result.spec.name


# ------------------------------------------------------------- formatting


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (100000, "Rs 1,00,000"),
        (0, "Rs 0"),
        (-2500, "-Rs 2,500"),
    ],
)
def test_format_inr(value: float, expected: str) -> None:
    assert logic.format_inr(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (12.345, "12.3%"),
        (0, "0.0%"),
        (None, "n/a"),
    ],
)
def test_format_pct(value, expected: str) -> None:
    assert logic.format_pct(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (5.0, "+5.0%"),
        (-5.0, "-5.0%"),
        (None, "n/a"),
    ],
)
def test_format_signed_pct(value, expected: str) -> None:
    assert logic.format_signed_pct(value) == expected


def test_color_for_pnl() -> None:
    assert logic.color_for_pnl(100.0) == "#006300"
    assert logic.color_for_pnl(-100.0) == "#d03b3b"
    assert logic.color_for_pnl(0.0) == "#52514e"


def test_color_for_flag_matches_status_palette() -> None:
    assert logic.color_for_flag("critical") == "#d03b3b"
    assert logic.color_for_flag("warning") == "#fab219"
    assert logic.color_for_flag("info") == "#52514e"
    assert logic.color_for_flag("nonsense") == logic.color_for_flag("info")


def test_flags_by_severity_orders_critical_first() -> None:
    flags = [
        logic.Flag(severity="info", code="a", headline="a", detail="a"),
        logic.Flag(severity="critical", code="b", headline="b", detail="b"),
        logic.Flag(severity="warning", code="c", headline="c", detail="c"),
    ]
    ordered = logic.flags_by_severity(flags)
    assert [f.severity for f in ordered] == ["critical", "warning", "info"]


def test_charge_model_for_label_defaults_to_options() -> None:
    from nlt.costs.charges import NseOptionsCharges

    assert isinstance(logic.charge_model_for_label("not a real label"), NseOptionsCharges)


# ------------------------------------------------------------------- charts


def test_equity_and_drawdown_figure_has_three_traces() -> None:
    from app.charts import equity_and_drawdown_figure

    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    strategy = pd.Series([100_000, 101_000, 99_000, 102_000, 103_000], index=idx)
    benchmark = pd.Series([100_000, 100_500, 100_200, 101_000, 101_500], index=idx)

    fig = equity_and_drawdown_figure(strategy, benchmark)
    assert len(fig.data) == 3
    names = [trace.name for trace in fig.data]
    assert "Your strategy" in names
    assert "Buy and hold" in names


# ---------------------------------------------------------------- imports


def test_import_app_main_does_not_raise() -> None:
    import app.main  # noqa: F401
