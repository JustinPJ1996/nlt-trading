"""Plotly figures for the dashboard, built per the `dataviz` skill.

Two charts matter more than anything else in this product: the equity curve
(strategy vs. buy-and-hold, one axis, both lines labelled) and the drawdown
chart beneath it. Everything here follows the skill's mark specs and color
rules rather than picking colors by eye:

* Colors are the two adjacent, validated categorical slots from the skill's
  reference palette (`references/palette.md`) -- slot 1 blue for "your
  strategy", slot 2 orange for "buy and hold" -- never an ad hoc hex. That
  pair clears the CVD/contrast checks in both light and dark mode
  (worst adjacent CVD delta-E 9.1 light / 8.4 dark), so it survives
  colorblind simulation, not just a normal-vision eyeball.
* Lines are 2px, markers (where used) carry an 8px+ surface ring, gridlines
  are hairline and recessive, and a legend is always shown because there are
  two or more series on every multi-line chart here (a single-series chart
  needs no legend box -- the title already names it).
* `paper_bgcolor`/`plot_bgcolor` are transparent so the chart follows
  Streamlit's own light/dark theme rather than fighting it with a fixed
  chart-surface color -- the one deliberate departure from the skill's
  literal CSS recipe (written for a hand-built HTML surface), adapted for a
  host that already owns the surface color.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Categorical slots 1 and 2 from references/palette.md, light-mode steps.
_STRATEGY_COLOR = "#2a78d6"  # slot 1, blue
_BENCHMARK_COLOR = "#eb6834"  # slot 2, orange
_DRAWDOWN_COLOR = "#d03b3b"  # status: critical red -- drawdown is a "how bad" magnitude
_GRIDLINE_COLOR = "rgba(137, 135, 129, 0.25)"  # muted ink, low opacity, works on light & dark
_AXIS_COLOR = "rgba(137, 135, 129, 0.55)"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=30, b=10),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    font=dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
)


def equity_and_drawdown_figure(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    *,
    strategy_name: str = "Your strategy",
    benchmark_name: str = "Buy and hold",
) -> go.Figure:
    """Strategy vs. benchmark equity, with a shared-x drawdown panel beneath.

    This is the single most important visual in the product: it is how a
    user sees that +5% is a bad result when the index alone did +419% over
    the same stretch. Sharing the x-axis means a spike in the drawdown panel
    always lines up with the equity dip that caused it.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("Account value over time", "Drop from the peak so far"),
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_equity.index,
            y=strategy_equity.values,
            name=strategy_name,
            mode="lines",
            line=dict(color=_STRATEGY_COLOR, width=2),
            hovertemplate="%{y:,.0f}<extra>" + strategy_name + "</extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark_equity.index,
            y=benchmark_equity.values,
            name=benchmark_name,
            mode="lines",
            line=dict(color=_BENCHMARK_COLOR, width=2),
            hovertemplate="%{y:,.0f}<extra>" + benchmark_name + "</extra>",
        ),
        row=1,
        col=1,
    )

    running_max = strategy_equity.cummax()
    drawdown_pct = 100.0 * (strategy_equity - running_max) / running_max.replace(0, pd.NA)
    fig.add_trace(
        go.Scatter(
            x=drawdown_pct.index,
            y=drawdown_pct.values,
            name="Drop from peak",
            mode="lines",
            line=dict(color=_DRAWDOWN_COLOR, width=2),
            fill="tozeroy",
            fillcolor="rgba(208, 59, 59, 0.10)",
            showlegend=False,
            hovertemplate="%{y:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(**_LAYOUT_DEFAULTS, height=520)
    fig.update_xaxes(showgrid=False, showline=True, linecolor=_AXIS_COLOR, row=2, col=1)
    fig.update_xaxes(showgrid=False, showline=False, row=1, col=1)
    fig.update_yaxes(
        title_text="Rupees",
        gridcolor=_GRIDLINE_COLOR,
        gridwidth=1,
        zeroline=False,
        tickformat=",.0f",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="% from peak",
        gridcolor=_GRIDLINE_COLOR,
        gridwidth=1,
        zeroline=False,
        ticksuffix="%",
        row=2,
        col=1,
    )
    return fig
