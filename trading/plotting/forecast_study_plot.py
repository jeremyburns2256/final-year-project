"""
forecast_study_plot.py

Bokeh plots for the forecast-driven MPC study (FORECAST_NOTES.md):

  1. Cumulative grid profit (excluding degradation) over the test month, one line per forecaster.
  2. Final net profit including rainflow degradation, and price MAE, per forecaster (bars).
  3. For one chosen day: the 24 h price forecast each forecaster issued at `issue_hour`
     against the actual RRP, and the SoC trajectory each controller followed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, HoverTool, Span
from bokeh.plotting import figure, output_file, save
from bokeh.transform import dodge

_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"
_X_FMT = DatetimeTickFormatter(hours="%H:%M", days="%d/%m")
COLOURS = {"perfect": "#212121", "naive": "#9E9E9E", "aemo": "#1565C0", "lstm": "#C62828"}


def plot_forecast_study(
    summary: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    frame,
    forecasters: dict,
    *,
    title: str,
    output_path: str,
    day: str = "2025-01-15",
    issue_hour: float = 4.0 + 5 / 60,
    price_ylim: float | None = 600.0,
) -> None:
    output_file(output_path, title=title)
    panels = []

    # 1. cumulative profit
    p1 = figure(height=320, sizing_mode="stretch_width", title=f"{title}: cumulative grid profit (excl. degradation)",
                tools=_TOOLS, x_axis_type="datetime")
    for name, df in results.items():
        t = pd.to_datetime(df["time"], dayfirst=True)
        p1.line(t, df["cumulative_profit"], color=COLOURS.get(name, "#555"), line_width=2, legend_label=name)
    p1.xaxis.formatter = _X_FMT
    p1.yaxis.axis_label = "$"
    p1.legend.location = "top_left"
    p1.legend.click_policy = "hide"
    panels.append([p1])

    # 2. bars: net profit incl. degradation and price MAE
    names = list(summary["forecaster"])
    src = ColumnDataSource(summary.assign(colour=[COLOURS.get(n, "#555") for n in names]))
    p2 = figure(height=300, x_range=names, sizing_mode="stretch_width", tools=_TOOLS,
                title="Net profit incl. rainflow degradation (bars) and gap to perfect foresight")
    p2.vbar(x="forecaster", top="net_profit_incl_degradation", width=0.6, source=src, color="colour")
    p2.yaxis.axis_label = "$"
    p2.add_tools(HoverTool(tooltips=[("forecaster", "@forecaster"), ("net profit incl. deg.", "@net_profit_incl_degradation{0.00}"),
                                     ("gap to perfect", "@profit_gap_to_perfect{0.00}"), ("EFC", "@equivalent_full_cycles{0.0}")]))
    p3 = figure(height=300, x_range=names, sizing_mode="stretch_width", tools=_TOOLS,
                title="Price forecast error over the 24 h horizon ($/MWh)")
    p3.vbar(x=dodge("forecaster", -0.2, range=p3.x_range), top="price_mae", width=0.18, source=src, color="colour", legend_label="MAE")
    p3.vbar(x=dodge("forecaster", 0.0, range=p3.x_range), top="price_median_ae", width=0.18, source=src, color="colour", alpha=0.6, legend_label="median AE")
    p3.vbar(x=dodge("forecaster", 0.2, range=p3.x_range), top="price_mae_clip1000", width=0.18, source=src, color="colour", alpha=0.3, legend_label="MAE, forecast clipped at 1000")
    p3.legend.location = "top_left"
    p3.yaxis.axis_label = "$/MWh"
    panels.append([p2, p3])

    # 3. one day: forecasts issued at issue_hour vs actual, and SoC
    day_start = pd.Timestamp(day)
    issue_time = day_start + pd.Timedelta(hours=issue_hour)
    t = int(np.searchsorted(frame.start_times.values, np.datetime64(issue_time)))
    h = min(288, frame.n - t)
    times = frame.start_times[t : t + h]
    p4 = figure(height=340, sizing_mode="stretch_width", tools=_TOOLS, x_axis_type="datetime",
                title=f"24 h price forecasts issued {issue_time:%d/%m %H:%M} vs actual RRP")
    p4.line(times, frame.rrp[t : t + h], color="#212121", line_width=2.5, legend_label="actual")
    for name, fc in forecasters.items():
        if name == "perfect":
            continue
        p4.line(times, fc.price(t, h), color=COLOURS.get(name, "#555"), line_width=2, legend_label=name)
    if price_ylim is not None:
        lo = min(-50.0, float(frame.rrp[t : t + h].min()) - 10)
        p4.y_range.start, p4.y_range.end = lo, price_ylim
    p4.xaxis.formatter = _X_FMT
    p4.yaxis.axis_label = "$/MWh (axis clipped; hover for spikes)"
    p4.legend.location = "top_left"
    p4.legend.click_policy = "hide"
    p4.add_layout(Span(location=issue_time.timestamp() * 1000, dimension="height", line_dash="dashed", line_color="grey"))

    p5 = figure(height=300, sizing_mode="stretch_width", tools=_TOOLS, x_axis_type="datetime", x_range=p4.x_range,
                title="Battery SoC under each controller")
    for name, df in results.items():
        tt = pd.to_datetime(df["time"], dayfirst=True)
        m = (tt >= issue_time) & (tt <= issue_time + pd.Timedelta(hours=24))
        p5.line(tt[m], df.loc[m, "battery_state"], color=COLOURS.get(name, "#555"), line_width=2, legend_label=name)
    p5.xaxis.formatter = _X_FMT
    p5.yaxis.axis_label = "kWh"
    p5.legend.location = "top_left"
    p5.legend.click_policy = "hide"
    panels += [[p4], [p5]]

    save(gridplot(panels, sizing_mode="stretch_width"))
