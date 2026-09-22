"""Vectorised evaluation of a spec's condition tree.

Everything here operates on whole columns at once -- never a Python loop over
bars -- because the backtest runs this once per bar_ago/indicator combination
and a strategy can nest conditions ten deep. The one rule that matters more
than speed: a NaN anywhere in a comparison must resolve to False. Indicators
have warm-up periods (an SMA-200 has no value for its first 199 bars), and a
NaN accidentally treated as truthy would open trades before the indicator
exists to justify them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nlt.spec.models import (
    All,
    Any_,
    Compare,
    Condition,
    Const,
    IsTrue,
    Not,
    Operand,
    PercentChange,
    Ref,
)


def evaluate(condition: Condition, features: pd.DataFrame) -> pd.Series:
    """Evaluate a condition tree over every bar in `features`, returning a bool Series."""
    if isinstance(condition, Compare):
        result = _evaluate_compare(condition, features)
    elif isinstance(condition, IsTrue):
        result = _resolve(condition.ref, features).astype(bool)
    elif isinstance(condition, PercentChange):
        result = _evaluate_percent_change(condition, features)
    elif isinstance(condition, All):
        result = pd.concat([evaluate(c, features) for c in condition.conditions], axis=1).all(axis=1)
    elif isinstance(condition, Any_):
        result = pd.concat([evaluate(c, features) for c in condition.conditions], axis=1).any(axis=1)
    elif isinstance(condition, Not):
        result = ~evaluate(condition.condition, features)
    else:
        raise TypeError(f"unknown condition type {type(condition).__name__}")

    # Any stray NaN surviving to here (e.g. from an `All` mixing a warmed-up
    # child with a still-NaN one via bool coercion) is treated as "no signal".
    return result.fillna(False).astype(bool)


def _resolve(ref: Ref, features: pd.DataFrame) -> pd.Series:
    key = ref.key()
    if key not in features.columns:
        raise KeyError(
            f"condition references {key!r}, which is not a column of the feature table "
            f"(available: {sorted(features.columns)})"
        )
    return features[key].shift(ref.bars_ago)


def _resolve_operand(operand: Operand, features: pd.DataFrame) -> pd.Series | float:
    if isinstance(operand, Const):
        return operand.value
    return _resolve(operand, features)


def _evaluate_compare(condition: Compare, features: pd.DataFrame) -> pd.Series:
    left = _resolve_operand(condition.left, features)
    right = _resolve_operand(condition.right, features)
    op = condition.op

    if op == "crosses_above":
        return _crosses(left, right, features.index, above=True)
    if op == "crosses_below":
        return _crosses(left, right, features.index, above=False)

    # Broadcast a bare scalar operand against the other side's index so the
    # comparison always yields a Series, not a single bool.
    left, right = _align(left, right, features.index)

    if op == "lt":
        return left < right
    if op == "lte":
        return left <= right
    if op == "gt":
        return left > right
    if op == "gte":
        return left >= right
    if op == "eq":
        # Exact float equality on indicator values is essentially never true;
        # np.isclose gives the "same price" the user meant.
        result = np.isclose(left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64"), equal_nan=False)
        return pd.Series(result, index=features.index)
    raise ValueError(f"unknown comparison op {op!r}")


def _align(
    left: pd.Series | float, right: pd.Series | float, index: pd.Index
) -> tuple[pd.Series, pd.Series]:
    if not isinstance(left, pd.Series):
        left = pd.Series(left, index=index)
    if not isinstance(right, pd.Series):
        right = pd.Series(right, index=index)
    return left, right


def _crosses(
    left: pd.Series | float, right: pd.Series | float, index: pd.Index, *, above: bool
) -> pd.Series:
    left, right = _align(left, right, index)
    left_prev, right_prev = left.shift(1), right.shift(1)
    if above:
        return (left > right) & (left_prev <= right_prev)
    return (left < right) & (left_prev >= right_prev)


def _evaluate_percent_change(condition: PercentChange, features: pd.DataFrame) -> pd.Series:
    value = _resolve(condition.ref, features)
    prior = value.shift(condition.lookback)
    pct = 100.0 * (value - prior) / prior

    op = condition.op
    if op == "lt":
        return pct < condition.value
    if op == "lte":
        return pct <= condition.value
    if op == "gt":
        return pct > condition.value
    if op == "gte":
        return pct >= condition.value
    raise ValueError(f"unknown percent_change op {op!r}")
