"""
Uses Bokeh to plot the cummulative profit of a trading strategy.
"""

from __future__ import annotations
from __future__ import annotations

import pandas as pd
from bokeh.models import (
    ColumnDataSource,
    DatetimeTickFormatter,
    LinearAxis,
    Range1d,
)
from bokeh.plotting import figure, output_file, save, show

_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
_X_FMT = DatetimeTickFormatter(hours="%H:%M", days="%d/%m")