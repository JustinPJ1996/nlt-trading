"""The dashboard's entry point -- the only product surface this platform has today.

Run with:
    .venv/bin/streamlit run app/main.py --server.port 8501 --server.address 0.0.0.0

Four pages, chosen from the sidebar, sharing state through `st.session_state`:

1. New strategy   -- describe it in English, check the readback, run a backtest.
2. Backtest results -- the verdict, the equity curve, the trades, the warnings.
3. My strategies  -- what has been saved, and whether it has earned trust yet.
4. Safety         -- the kill switch, data freshness, the audit trail.

Every widget here does layout and nothing else. Decisions (parse this text,
judge this backtest, format this number) live in `app/logic.py` and
`app/charts.py`, which is what `tests/test_app.py` actually exercises --
Streamlit widgets themselves are not unit tested.

This module is guarded by `if __name__ == "__main__"` so `import app.main`
(what the test suite does) never touches `st.*` calls that need a live
Streamlit script-run context.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from app import logic
from app.charts import equity_and_drawdown_figure
from nlt.store.db import Store
from nlt.translate.rules import Question

PAGE_NEW = "New strategy"
PAGE_RESULTS = "Backtest results"
PAGE_MINE = "My strategies"
PAGE_SAFETY = "Safety"


# ----------------------------------------------------------------- plumbing


@st.cache_resource
def get_store() -> Store:
    return Store()


@st.cache_data(ttl=3600, show_spinner="Fetching price history...")
def cached_load_bars(
    symbol: str, start: dt.date | None, end: dt.date | None
) -> pd.DataFrame:
    return logic.load_bars(symbol, start, end)


def _init_state() -> None:
    defaults = {
        "text_input": "",
        "answers": {},
        "translation": None,
        "confirmed_spec": None,
        "pipeline_result": None,
        "capital": logic.DEFAULT_CAPITAL,
        "cost_model_label": next(iter(logic.COST_MODEL_LABELS)),
        "page": PAGE_NEW,
        "nav_request": None,
        "selected_strategy_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _apply_pending_navigation() -> None:
    """Moves a page change requested mid-run into effect on the *next* run.

    The sidebar radio below is bound to `st.session_state["page"]` via its
    `key`; once that widget has been instantiated in a run, Streamlit refuses
    any further write to `"page"` in that same run
    (`StreamlitWidgetAlreadyInstantiatedError`). A button elsewhere on the
    page that wants to navigate (e.g. "Run backtest" jumping to the results
    page) therefore stashes the target in `"nav_request"` and calls
    `st.rerun()`; this function, called before the radio is created, is the
    only place allowed to write `"page"` directly.
    """
    target = st.session_state.pop("nav_request", None)
    if target is not None:
        st.session_state["page"] = target


# ------------------------------------------------------------------ page 1


def _render_readback(spec) -> None:
    st.markdown("### Here is what I understood. Please read it carefully before going further.")
    st.markdown(
        """
        <style>
        .nlt-readback {
            background-color: rgba(42, 120, 214, 0.08);
            border: 2px solid #2a78d6;
            border-radius: 8px;
            padding: 1rem 1.25rem;
            font-family: ui-monospace, "SFMono-Regular", Menlo, monospace;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="nlt-readback">{logic.describe_spec(spec)}</div>', unsafe_allow_html=True)


def _render_questions(questions: list[Question]) -> None:
    st.warning("I need a bit more detail before I can build this strategy.")
    values: dict[str, str] = {}
    for q in questions:
        if q.field == "unparsed":
            continue
        values[q.field] = st.text_input(q.text, value=q.suggestion or "", help=q.why, key=f"q_{q.field}")
    if st.button("Use these answers", type="primary"):
        st.session_state["answers"] = logic.answers_from_questions(questions, values)
        st.session_state["translation"] = logic.translate_text(
            st.session_state["text_input"], st.session_state["answers"]
        )
        st.rerun()


def page_new_strategy() -> None:
    st.title("Build a strategy")
    st.write("Describe your strategy in plain English. No code, no jargon required.")

    st.write("Or start from an example:")
    cols = st.columns(len(logic.EXAMPLE_STRATEGIES))
    for col, example in zip(cols, logic.EXAMPLE_STRATEGIES, strict=True):
        if col.button(example, use_container_width=True):
            st.session_state["text_input"] = example
            st.session_state["translation"] = None
            st.session_state["confirmed_spec"] = None
            st.session_state["answers"] = {}

    text = st.text_area(
        "Describe your strategy in plain English",
        key="text_input",
        height=120,
        placeholder="e.g. buy nifty when rsi cracks 30, target 2%, stop loss 1%",
    )

    if st.button("Check my strategy", type="primary"):
        st.session_state["translation"] = logic.translate_text(text, st.session_state["answers"])
        st.session_state["confirmed_spec"] = None

    translation = st.session_state["translation"]
    if translation is None:
        return

    if translation.unparsed:
        st.error("I did not understand: " + "; ".join(translation.unparsed))

    if translation.notes:
        for note in translation.notes:
            st.info(note)

    if translation.spec is not None:
        _render_readback(translation.spec)
        if st.button("Yes, this is right -- continue", type="primary"):
            st.session_state["confirmed_spec"] = translation.spec
    elif translation.questions:
        _render_questions(translation.questions)

    if st.session_state["confirmed_spec"] is not None:
        _render_backtest_panel(st.session_state["confirmed_spec"])


def _render_backtest_panel(spec) -> None:
    st.markdown("---")
    st.markdown("### Run a backtest")
    st.caption("This checks how the strategy would have done on real past prices. It is not a guarantee.")

    c1, c2 = st.columns(2)
    with c1:
        capital = st.number_input(
            "Starting money",
            min_value=1_000.0,
            value=float(st.session_state["capital"]),
            step=5_000.0,
            format="%.0f",
            help="How much money to pretend you started with.",
        )
        cost_label = st.selectbox(
            "What kind of trading costs should we charge?",
            list(logic.COST_MODEL_LABELS),
            index=list(logic.COST_MODEL_LABELS).index(st.session_state["cost_model_label"]),
            help="Real trades cost brokerage, taxes and exchange fees. Pick 'No costs' only to see the idealised result.",
        )
    with c2:
        today = dt.date.today()
        date_range = st.date_input(
            "Date range to test over",
            value=(today - dt.timedelta(days=365 * 3), today),
            help="Leave generous room -- a longer history gives a more trustworthy answer.",
        )

    if st.button("Run backtest", type="primary"):
        start, end = (date_range if isinstance(date_range, tuple) else (None, None))
        # No pre-flight load here. `run_pipeline` resolves the symbol itself --
        # an index, a ticker or a whole universe -- and reports failures as
        # `result.error`. Loading separately first meant asking Yahoo for a
        # ticker called "NIFTY 100" and rejecting every basket strategy before
        # the pipeline that knows how to handle one was ever called.
        st.session_state["capital"] = capital
        st.session_state["cost_model_label"] = cost_label
        with st.spinner("Running the backtest..."):
            result = logic.run_pipeline(
                spec.description,
                capital=capital,
                cost_model_label=cost_label,
                symbol=spec.instrument.symbol,
                start=start,
                end=end,
                answers=st.session_state["answers"],
            )
        st.session_state["pipeline_result"] = result
        if result.error:
            st.error(f"I couldn't finish that backtest: {result.error}")
        else:
            st.success("Backtest complete. Open 'Backtest results' in the sidebar to see it.")
            st.session_state["nav_request"] = PAGE_RESULTS
            st.rerun()


# ------------------------------------------------------------------ page 2


def page_results() -> None:
    st.title("Backtest results")
    result = st.session_state.get("pipeline_result")
    if result is None or not result.ok:
        st.info("Run a backtest from 'New strategy' first.")
        return

    verdict = result.verdict
    bt = result.backtest
    bench = result.benchmark
    comp = result.comparison

    # -------------------------------------------------------- the verdict
    verdict_color = "#0ca30c" if verdict.passed else "#d03b3b"
    st.markdown(
        f"""
        <div style="border-left: 6px solid {verdict_color}; padding: 0.75rem 1rem;
                    background: rgba({'12,163,12' if verdict.passed else '208,59,59'}, 0.08);
                    border-radius: 4px; margin-bottom: 1rem;">
            <span style="font-size: 1.4rem; font-weight: 600; color: {verdict_color};">
                {verdict.summary}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for flag in logic.flags_by_severity(verdict.flags):
        color = logic.color_for_flag(flag.severity)
        icon = logic.icon_for_flag(flag.severity)
        st.markdown(
            f"""
            <div style="border: 1px solid {color}; border-radius: 6px; padding: 0.6rem 0.9rem;
                        margin-bottom: 0.5rem;">
                <b>{icon} {flag.headline}</b><br/>
                <span style="color: #52514e;">{flag.detail}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ------------------------------------------------------------ charts
    st.markdown("### Your strategy vs. just buying and holding")
    fig = equity_and_drawdown_figure(bt.equity, bench.equity)
    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------------------------------------- metrics
    st.markdown("### The numbers, side by side")
    m = bt.metrics
    rows = [
        ("Total return", logic.format_signed_pct(m.get("total_return_pct")), logic.format_signed_pct(bench.total_return_pct)),
        ("Biggest drop from a peak (max drawdown)", logic.format_pct(m.get("max_drawdown_pct")), logic.format_pct(bench.max_drawdown_pct)),
        ("Win rate", logic.format_pct(m.get("win_rate_pct")), "n/a"),
        ("Number of trades", str(int(m.get("total_trades") or 0)), "1 (buy and hold)"),
        ("Time spent in the market", logic.format_pct(comp.time_in_market_pct), "100.0%"),
        ("Total charges paid", logic.format_inr(m.get("total_charges") or 0), "n/a"),
    ]
    metrics_df = pd.DataFrame(rows, columns=["Metric", "Your strategy", "Buy and hold"])
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    with st.expander("Show every technical metric"):
        st.dataframe(pd.DataFrame([m]).T.rename(columns={0: "value"}), use_container_width=True)

    # ------------------------------------------------------------ trades
    st.markdown("### Every trade")
    if bt.trades:
        trades_df = pd.DataFrame(
            [
                {
                    "Entered": t.entry_time,
                    "Entry price": t.entry_price,
                    "Exited": t.exit_time,
                    "Exit price": t.exit_price,
                    "Reason": t.exit_reason,
                    "Net P&L (Rs)": round(t.net_pnl, 2),
                }
                for t in bt.trades
            ]
        )

        def _pnl_style(v: float) -> str:
            return f"color: {logic.color_for_pnl(v)}"

        st.dataframe(
            trades_df.style.map(_pnl_style, subset=["Net P&L (Rs)"]),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("This strategy never took a trade in the tested period.")

    # ---------------------------------------------------------- warnings
    # `data_notes` carries everything the loaders and the engine wanted to say:
    # corporate-action history dropped, symbols that failed to load, the
    # survivorship caveat on a universe, signals skipped for want of capital.
    # De-duplicated because the engine's own warnings are folded in upstream,
    # but never filtered -- a note the user does not see is a note not made.
    notes = list(dict.fromkeys([*getattr(result, "data_notes", []), *bt.warnings]))
    if notes:
        st.markdown("### Things you should know about this test")
        for note in notes:
            st.warning(note)

    # -------------------------------------------------------------- save
    st.markdown("---")
    if st.button("Save this strategy", type="primary"):
        store = get_store()
        _, err = logic.safe_call(
            lambda: logic.save_strategy(store, result.spec),
            on_error="Could not save this strategy",
        )
        if err:
            st.error(err)
        else:
            st.success(f"Saved '{result.spec.name}'. Find it under 'My strategies'.")


# ------------------------------------------------------------------ page 3


def page_my_strategies() -> None:
    st.title("My strategies")
    store = get_store()
    strategies, err = logic.safe_call(
        lambda: logic.list_strategies_view(store), on_error="Could not load your strategies"
    )
    if err:
        st.error(err)
        return
    if not strategies:
        st.info("You haven't saved any strategies yet. Build one under 'New strategy'.")
        return

    for s in strategies:
        proven, _ = logic.safe_call(lambda sid=s["id"]: store.is_proven(sid), on_error="")
        with st.expander(f"{s['name']} (v{s['version']}) -- {s['state']}"):
            st.write(f"Created: {s['created_at']}")
            st.write(f"Proven (backtested and paper traded): {'Yes' if proven else 'No'}")
            st.text(s["readback"])
            runs, _ = logic.safe_call(
                lambda sid=s["id"]: logic.list_runs_view(store, sid), on_error=""
            )
            if runs:
                st.write("Past runs:")
                st.dataframe(pd.DataFrame(runs), hide_index=True, use_container_width=True)
            if st.button("Re-run this strategy", key=f"rerun_{s['id']}"):
                st.session_state["text_input"] = s["description"]
                st.session_state["translation"] = logic.translate_text(s["description"])
                st.session_state["nav_request"] = PAGE_NEW
                st.rerun()


# ------------------------------------------------------------------ page 4


def page_safety() -> None:
    st.title("Safety")
    store = get_store()

    st.markdown("### Kill switch")
    engaged, err = logic.safe_call(store.kill_switch_engaged, on_error="Could not read kill switch status")
    if err:
        st.error(err)
    elif engaged:
        st.error("The kill switch is ENGAGED. Nothing live can trade right now.")
        if st.button("Release kill switch"):
            _, err2 = logic.safe_call(
                lambda: store.release_kill_switch("released from dashboard"),
                on_error="Could not release the kill switch",
            )
            if err2:
                st.error(err2)
            else:
                st.rerun()
    else:
        st.success("The kill switch is OFF. Trading is not blocked.")
        confirm = st.checkbox("I understand this will halt every live strategy immediately.")
        if st.button("Engage kill switch", disabled=not confirm, type="primary"):
            _, err2 = logic.safe_call(
                lambda: store.engage_kill_switch("engaged from dashboard"),
                on_error="Could not engage the kill switch",
            )
            if err2:
                st.error(err2)
            else:
                st.rerun()

    st.markdown("### Data freshness")
    for symbol in ("NIFTY", "BANKNIFTY"):
        days, err = logic.safe_call(lambda s=symbol: logic.staleness_days(s), on_error="")
        if err or days is None:
            st.info(f"{symbol}: no data cached yet.")
        elif days > 3:
            st.error(f"{symbol}: price data is {days} trading day(s) old. Refresh before trusting a new backtest.")
        else:
            st.success(f"{symbol}: price data is up to date ({days} trading day(s) old).")

    st.markdown("### Recent activity")
    trail, err = logic.safe_call(lambda: store.audit_trail(limit=50), on_error="Could not load the audit log")
    if err:
        st.error(err)
    elif not trail:
        st.info("Nothing has happened yet.")
    else:
        st.dataframe(pd.DataFrame(trail), hide_index=True, use_container_width=True)


# ------------------------------------------------------------------- shell


def main() -> None:
    st.set_page_config(page_title="Trading Strategy Builder", page_icon="chart", layout="wide")
    _init_state()
    _apply_pending_navigation()

    st.sidebar.title("Trading Strategy Builder")
    page = st.sidebar.radio(
        "Go to", [PAGE_NEW, PAGE_RESULTS, PAGE_MINE, PAGE_SAFETY], key="page"
    )

    pages = {
        PAGE_NEW: page_new_strategy,
        PAGE_RESULTS: page_results,
        PAGE_MINE: page_my_strategies,
        PAGE_SAFETY: page_safety,
    }
    pages[page]()


if __name__ == "__main__":
    main()
