"""Turns a spec's indicator declarations into one aligned feature table.

Every indicator is computed exactly once, on the full bar history, and keyed by
the same strings `Ref.key()` produces -- so `conditions.py` never touches the
registry itself, it just looks a column up. Computing here instead of lazily
inside condition evaluation also means a strategy that uses the same indicator
in both its entry and its exit only pays for it once.
"""

from __future__ import annotations

import pandas as pd

from nlt.indicators.registry import get as get_indicator
from nlt.spec.models import PRICE_FIELDS, StrategySpec


def build_features(spec: StrategySpec, bars: pd.DataFrame) -> pd.DataFrame:
    """Compute every declared indicator, aligned to `bars.index`.

    Columns: the five price fields, plus one column per indicator output ---
    `<id>` for single-output indicators, `<id>.<output>` for multi-output ones,
    matching `Ref.key()` exactly.
    """
    features = pd.DataFrame(index=bars.index)
    for field in PRICE_FIELDS:
        features[field] = bars[field]

    for ind in spec.indicators:
        d = get_indicator(ind.type)
        result = d.fn(bars, **ind.resolved_params())

        if d.is_multi:
            for output in d.outputs:
                features[f"{ind.id}.{output}"] = result[output]
        else:
            # Single-output indicators may return either a bare Series or a
            # one-column DataFrame (patterns do the latter); handle both.
            series = result[d.outputs[0]] if isinstance(result, pd.DataFrame) else result
            features[ind.id] = series

    return features
