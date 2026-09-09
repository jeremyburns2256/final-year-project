"""
milp_sweep_plot.py

Bokeh plot of MILP results against the number of cycle-depth segments J:
profit (excluding and including degradation), model vs rainflow aging cost,
and simulated life loss.
"""

from __future__ import annotations

import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def plot_j_sweep(summary: pd.DataFrame, *, title: str, output_path: str) -> None:
    """summary: one row per run with columns scenario, n_segments, r_cell and the metrics from rolling.summarise."""
    output_file(output_path, title=title)
    df = summary[summary["r_cell"] > 0].sort_values("n_segments")
    panels = []
    colours = {"bess_only": "#1565C0", "household": "#2E7D32"}

    p1 = figure(height=350, sizing_mode="stretch_width", title=f"{title}: net profit vs J", tools=_TOOLS)
    p2 = figure(height=350, sizing_mode="stretch_width", title="Aging cost: model vs ex-post rainflow", tools=_TOOLS, x_range=p1.x_range)
    p3 = figure(height=350, sizing_mode="stretch_width", title="Cycle life consumed over the period", tools=_TOOLS, x_range=p1.x_range)
    for scenario, grp in df.groupby("scenario"):
        src = ColumnDataSource(grp)
        col = colours.get(scenario, "#555")
        p1.line("n_segments", "net_profit_ex_degradation", source=src, color=col, line_dash="dashed", line_width=2, legend_label=f"{scenario} excl. degradation")
        p1.line("n_segments", "net_profit_incl_degradation", source=src, color=col, line_width=2, legend_label=f"{scenario} incl. degradation")
        p1.scatter("n_segments", "net_profit_incl_degradation", source=src, color=col, size=7)
        p2.line("n_segments", "degradation_cost_model", source=src, color=col, line_width=2, legend_label=f"{scenario} model")
        p2.line("n_segments", "degradation_cost_rainflow", source=src, color=col, line_dash="dotted", line_width=2, legend_label=f"{scenario} rainflow")
        p2.scatter("n_segments", "degradation_cost_model", source=src, color=col, size=7)
        p3.line("n_segments", "life_loss_pct", source=src, color=col, line_width=2, legend_label=scenario)
        p3.scatter("n_segments", "life_loss_pct", source=src, color=col, size=7)
    for p, ylab in ((p1, "$"), (p2, "$"), (p3, "% of cell life")):
        p.xaxis.axis_label = "J (cycle depth segments)"
        p.yaxis.axis_label = ylab
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.add_tools(HoverTool(tooltips=[("J", "@n_segments"), ("scenario", "@scenario")]))
        panels.append([p])
    save(gridplot(panels, sizing_mode="stretch_width"))
