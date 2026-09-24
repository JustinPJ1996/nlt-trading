"""The Streamlit dashboard -- the only product surface for the platform.

`logic.py` holds every function that makes a decision (parse, backtest,
format, judge); `charts.py` builds the Plotly figures; `main.py` wires both
into the actual pages. Nothing outside this package is allowed to change --
see the module docstrings for why.
"""

from __future__ import annotations
