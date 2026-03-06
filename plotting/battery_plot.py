"""
battery_plot.py
Reusable Bokeh plotting module for BESS trading visualisations.

Multiple vertically stacked panels with a shared, linked x-axis:

  Top panel
    Left primary   — SoC (kWh)
    Left secondary — Solar / Load power (kW)   [only when data present]
    Right          — RRP ($/MWh)

  Middle panel
    Center         — Local Net (Load - Solar) and Grid Action (Import/Export)

  Bottom panel
    Left           — Cumulative $ (profit, revenue, cost)

Price y-axis is clipped to a percentile range to suppress outlier spikes.
"""

from __future__ import annotations

import pandas as pd
from bokeh.layouts import column, row
from bokeh.models import (
    ColumnDataSource,
    DatetimeTickFormatter,
    LabelSet,
    LinearAxis,
    Range1d,
    Span,
)
from bokeh.plotting import figure, output_file, save, show

_TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, dayfirst=True)
    except Exception:
        return pd.to_datetime(series)


def plot_battery_trading(
    results_df: pd.DataFrame,
    *,
    title: str = "BESS Trading Results",
    output_path: str = "battery_plot.html",
    bess_size: float | None = None,
    price_percentile_clip: tuple[float, float] = (2, 98),
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    show_plot: bool = True,
) -> None:
    """
    Plot multiple linked panels: trading overview, energy flows, and cumulative profit.

    Required columns in results_df:
        time, battery_state, rrp,
        cumulative_profit, cumulative_revenue, cumulative_cost,
        grid_import_kwh, grid_export_kwh
    Optional columns (plotted when present and non-zero):
        export_kw (net export = solar - load), import_kw (load)

    Parameters
    ----------
    bess_size             : Battery capacity in kWh — sets SoC y-axis ceiling.
    price_percentile_clip : (low, high) percentiles used to clip the price
                            y-axis range. Default (2, 98).
    buy_threshold         : Price in $/MWh below which the strategy buys.
    sell_threshold        : Price in $/MWh above which the strategy sells.
    """
    df = results_df.copy()
    df["time"] = _to_datetime(df["time"])

    # Pre-compute positive / negative profit bands for the fill
    df["profit_pos"] = df["cumulative_profit"].clip(lower=0)
    df["profit_neg"] = df["cumulative_profit"].clip(upper=0)

    # Compute net grid action (positive = import, negative = export)
    INTERVAL_HOURS = 5 / 60
    df["grid_net_kw"] = (df["grid_import_kwh"] - df["grid_export_kwh"]) / INTERVAL_HOURS

    # Compute local energy balance (positive = deficit, negative = surplus)
    df["local_net_kw"] = df["import_kw"] - df["export_kw"]

    src = ColumnDataSource(df)

    output_file(output_path, title=title)

    # ── Detect optional export / load columns ──────────────────────────────────
    has_export = "export_kw" in df.columns and df["export_kw"].max() > 0
    has_load   = "load_kw"   in df.columns and df["load_kw"].max()  > 0
    has_power  = has_export or has_load

    # ── Shared x-axis range: default view = first day ─────────────────────────
    x_start   = df["time"].iloc[0]
    x_end     = x_start + pd.Timedelta(days=1)
    x_range   = Range1d(start=x_start, end=x_end)

    # ═══════════════════════════════════════════════════════════════════════════
    # TOP PANEL — SoC / Solar / Load / RRP
    # ═══════════════════════════════════════════════════════════════════════════

    # ── Y-axis ranges ─────────────────────────────────────────────────────────
    soc_max     = (bess_size * 1.05) if bess_size is not None else (df["battery_state"].max() * 1.1)
    soc_range   = Range1d(start=0, end=soc_max)

    lo, hi      = price_percentile_clip
    price_lo    = df["rrp"].quantile(lo / 100)
    price_hi    = df["rrp"].quantile(hi / 100)
    padding     = (price_hi - price_lo) * 0.05
    price_range = Range1d(start=price_lo - padding, end=price_hi + padding)

    if has_power:
        power_max   = max(
            df["export_kw"].max() if has_export else 0,
            df["load_kw"].max()   if has_load   else 0,
        )
        power_range = Range1d(start=0, end=power_max * 1.15)

    # ── Figure ────────────────────────────────────────────────────────────────
    extra_ranges = {"rrp": price_range}
    if has_power:
        extra_ranges["power"] = power_range

    p = figure(
        height=500,
        sizing_mode="stretch_width",
        x_axis_type="datetime",
        x_range=x_range,
        y_range=soc_range,
        title=title,
        tools=_TOOLS,
        toolbar_location="right",
    )
    p.extra_y_ranges = extra_ranges
    p.xaxis.formatter = DatetimeTickFormatter(hours="%H:%M", days="%d/%m")
    p.xaxis.ticker.desired_num_ticks = 24
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "SoC (kWh)"

    if has_power:
        power_axis = LinearAxis(y_range_name="power", axis_label="Power (kW)")
        power_axis.ticker.desired_num_ticks = 8
        p.add_layout(power_axis, "left")

    price_axis = LinearAxis(y_range_name="rrp", axis_label="Price ($/MWh)")
    price_axis.ticker.desired_num_ticks = 20
    p.add_layout(price_axis, "right")

    if has_export:
        p.varea(x="time", y1=0, y2="export_kw", source=src,
                fill_alpha=0.30, fill_color="#FFB300", y_range_name="power")
        p.line(x="time", y="export_kw", source=src,
               line_width=1.5, color="#FF8F00",
               y_range_name="power", legend_label="Excess Solar (kW)")

    if has_load:
        p.line(x="time", y="load_kw", source=src,
               line_width=1.5, color="#E53935", line_dash="dashed",
               y_range_name="power", legend_label="Load (kW)")

    p.varea(x="time", y1=0, y2="battery_state", source=src,
            fill_alpha=0.25, fill_color="#4CAF50")
    p.line(x="time", y="battery_state", source=src,
           line_width=1.5, color="#2E7D32", legend_label="SoC (kWh)")

    p.line(x="time", y="rrp", source=src,
           line_width=1, color="#1565C0", alpha=0.8,
           y_range_name="rrp", legend_label="RRP ($/MWh)")

    x_span = [df["time"].iloc[0], df["time"].iloc[-1]]
    if buy_threshold is not None:
        p.line(x=x_span, y=[buy_threshold, buy_threshold],
               y_range_name="rrp", color="#2196F3",
               line_dash="dashed", line_width=1.5,
               legend_label=f"Buy < ${buy_threshold:.0f}")
    if sell_threshold is not None:
        p.line(x=x_span, y=[sell_threshold, sell_threshold],
               y_range_name="rrp", color="#F44336",
               line_dash="dashed", line_width=1.5,
               legend_label=f"Sell >= ${sell_threshold:.0f}")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    # ═══════════════════════════════════════════════════════════════════════════
    # MIDDLE PANEL — Local Energy Balance & Grid Action
    # ═══════════════════════════════════════════════════════════════════════════

    # Calculate range for energy flow panel
    flow_max = max(
        df["local_net_kw"].abs().max() if "local_net_kw" in df.columns else 0,
        df["grid_net_kw"].abs().max() if "grid_net_kw" in df.columns else 0,
    )
    flow_padding = flow_max * 0.15
    flow_range = Range1d(start=-flow_max - flow_padding, end=flow_max + flow_padding)

    p_flow = figure(
        height=300,
        sizing_mode="stretch_width",
        x_axis_type="datetime",
        x_range=x_range,  # shared → linked pan/zoom
        y_range=flow_range,
        title="Energy Balance & Grid Action (kW)",
        tools=_TOOLS,
        toolbar_location="right",
    )
    p_flow.xaxis.formatter = DatetimeTickFormatter(hours="%H:%M", days="%d/%m")
    p_flow.xaxis.ticker.desired_num_ticks = 24
    p_flow.xaxis.axis_label = "Time"
    p_flow.yaxis.axis_label = "Power (kW)"

    # Zero reference line
    p_flow.add_layout(Span(location=0, dimension="width",
                           line_color="black", line_dash="dotted", line_width=1))

    # Local net (load - solar): positive = deficit, negative = surplus
    if "local_net_kw" in df.columns:
        p_flow.line(x="time", y="local_net_kw", source=src,
                    line_width=1.5, color="#9C27B0", alpha=0.6, line_dash="dashed",
                    legend_label="Local Net (Load - Solar)")

    # Grid action: positive = importing, negative = exporting
    if "grid_net_kw" in df.columns:
        # Color-code by direction: red for import, green for export
        df["grid_import_pos"] = df["grid_net_kw"].clip(lower=0)
        df["grid_export_neg"] = df["grid_net_kw"].clip(upper=0)

        src_updated = ColumnDataSource(df)

        # Filled areas for visual clarity
        p_flow.varea(x="time", y1=0, y2="grid_import_pos", source=src_updated,
                     fill_color="#E65100", fill_alpha=0.2)
        p_flow.varea(x="time", y1="grid_export_neg", y2=0, source=src_updated,
                     fill_color="#2E7D32", fill_alpha=0.2)

        # Main line
        p_flow.line(x="time", y="grid_net_kw", source=src_updated,
                    line_width=2.5, color="#1565C0",
                    legend_label="Grid Action (+ Import / - Export)")

    p_flow.legend.location = "top_left"
    p_flow.legend.click_policy = "hide"

    # ═══════════════════════════════════════════════════════════════════════════
    # BOTTOM PANEL — Cumulative profit / revenue / cost
    # ═══════════════════════════════════════════════════════════════════════════

    profit_min  = df["cumulative_profit"].min()
    profit_max  = df[["cumulative_profit", "cumulative_revenue"]].max().max()
    profit_pad  = max(abs(profit_max - profit_min) * 0.08, 1.0)
    profit_range = Range1d(
        start=min(profit_min - profit_pad, -profit_pad),
        end=profit_max + profit_pad,
    )

    p2 = figure(
        height=250,
        sizing_mode="stretch_width",
        x_axis_type="datetime",
        x_range=x_range,          # shared → linked pan/zoom
        y_range=profit_range,
        title="Cumulative Profit / Revenue / Cost",
        tools=_TOOLS,
        toolbar_location="right",
    )
    p2.xaxis.formatter = DatetimeTickFormatter(hours="%H:%M", days="%d/%m")
    p2.xaxis.ticker.desired_num_ticks = 24
    p2.xaxis.axis_label = "Time"
    p2.yaxis.axis_label = "Cumulative ($)"

    # Zero reference line
    p2.add_layout(Span(location=0, dimension="width",
                       line_color="black", line_dash="dotted", line_width=1))

    # Filled profit bands (green above zero, red below)
    p2.varea(x="time", y1=0, y2="profit_pos", source=src,
             fill_color="#4CAF50", fill_alpha=0.25)
    p2.varea(x="time", y1="profit_neg", y2=0, source=src,
             fill_color="#F44336", fill_alpha=0.25)

    # Revenue and cost as lighter background lines
    p2.line(x="time", y="cumulative_revenue", source=src,
            line_width=1.2, color="#1565C0", alpha=0.6,
            legend_label="Revenue ($)")
    p2.line(x="time", y="cumulative_cost", source=src,
            line_width=1.2, color="#E53935", alpha=0.6, line_dash="dashed",
            legend_label="Cost ($)")

    # Net profit as the prominent line
    p2.line(x="time", y="cumulative_profit", source=src,
            line_width=2.5, color="#2E7D32",
            legend_label="Net Profit ($)")

    p2.legend.location = "top_left"
    p2.legend.click_policy = "hide"

    # ── Save / show ───────────────────────────────────────────────────────────
    layout = column(p, p_flow, p2, sizing_mode="stretch_width")

    if show_plot:
        show(layout)
    else:
        save(layout)
        print(f"Plot saved to {output_path}")


# ── Summary panel helper ──────────────────────────────────────────────────────

def _build_summary_panel(df: pd.DataFrame, has_export: bool, has_load: bool):
    """
    Build a bar chart summarising total energy flows for the simulation period.

    Always shown  : Grid Export, Grid Import, Net Import/Export
    Shown if present: Household Load, Export (net generation = solar - load)
    """
    INTERVAL_HOURS = 5 / 60

    # ── Compute totals ────────────────────────────────────────────────────────
    total_export = df["grid_export_kwh"].sum()
    total_import = df["grid_import_kwh"].sum()
    net          = total_import - total_export   # >0 net importer, <0 net exporter

    categories, values, colors = [], [], []

    if has_load:
        categories.append("Household\nLoad")
        values.append(df["load_kw"].sum() * INTERVAL_HOURS)
        colors.append("#E53935")

    if has_export:
        categories.append("Local\nExport")
        values.append(df["export_kw"].sum() * INTERVAL_HOURS)
        colors.append("#FFB300")

    categories += ["Grid\nExport", "Grid\nImport", "Net Grid\nImport" if net >= 0 else "Net Grid\nExport"]
    values     += [total_export,   total_import,   net]
    colors     += ["#1565C0",      "#E65100",      "#BF360C" if net > 0 else "#2E7D32"]

    # ── Y range (accommodate negative net bar) ────────────────────────────────
    y_max = max(values) * 1.25
    y_min = min(min(values) * 1.25, -y_max * 0.05)

    # ── Value labels (above positive bars, below negative bars) ───────────────
    label_offset = (y_max - y_min) * 0.03
    label_y    = [v + label_offset if v >= 0 else v - label_offset for v in values]
    label_text = [f"{abs(v):.0f} kWh" for v in values]

    bar_src   = ColumnDataSource(dict(x=categories, top=values, color=colors))
    label_src = ColumnDataSource(dict(x=categories, y=label_y, text=label_text))

    # ── Figure ────────────────────────────────────────────────────────────────
    p3 = figure(
        height=280,
        sizing_mode="stretch_width",
        x_range=categories,
        y_range=Range1d(start=y_min, end=y_max),
        title="Period Energy Summary",
        tools="",
        toolbar_location=None,
    )
    p3.vbar(x="x", top="top", bottom=0, width=0.55,
            color="color", alpha=0.85, source=bar_src)

    # Zero baseline
    p3.add_layout(Span(location=0, dimension="width",
                       line_color="black", line_width=1))

    # Value labels
    labels = LabelSet(x="x", y="y", text="text", source=label_src,
                      text_align="center", text_font_size="12px",
                      text_font_style="bold")
    p3.add_layout(labels)

    p3.yaxis.axis_label  = "Energy (kWh)"
    p3.xgrid.grid_line_color = None
    p3.xaxis.major_label_text_font_size = "12px"
    p3.outline_line_color = None

    return p3


def _build_price_metrics_panel(df: pd.DataFrame):
    """
    Bar chart: Avg Import Price, Avg Export Price, Effective Cost per kWh.
    All values in cents per kWh (c/kWh).

    The effective-cost bar is capped at ±5× the larger of the two price bars so that
    an extreme value (common when net import is near zero) does not compress the
    other bars into invisibility.  When the bar is capped the label shows the
    true value with a '▲' / '▼' indicator so the user knows it is clipped.
    """
    total_export  = df["grid_export_kwh"].sum()
    total_import  = df["grid_import_kwh"].sum()
    total_cost    = df["cumulative_cost"].iloc[-1]
    total_revenue = df["cumulative_revenue"].iloc[-1]
    net_import    = total_import - total_export

    # Convert from $/kWh to c/kWh: multiply by 100
    avg_import = (total_cost    / total_import  * 100) if total_import  > 1e-6 else 0.0
    avg_export = (total_revenue / total_export  * 100) if total_export  > 1e-6 else 0.0

    if abs(net_import) > 1e-6:
        net_cost_true = (total_cost - total_revenue) / net_import * 100
    else:
        net_cost_true = 0.0

    # Cap for display so extreme values don't squash the other bars
    price_scale   = max(abs(avg_import), abs(avg_export), 1.0)
    cap           = price_scale * 5.0
    net_cost_disp = max(-cap, min(cap, net_cost_true))
    clipped       = abs(net_cost_true) > cap

    if clipped:
        arrow        = "▲" if net_cost_true > 0 else "▼"
        net_lbl      = f"{arrow} {net_cost_true:.0f}"
    else:
        net_lbl      = f"{net_cost_true:.1f}"

    categories = ["Avg Import\nPrice", "Avg Export\nPrice", "Effective Cost\nper kWh"]
    values     = [avg_import,           avg_export,           net_cost_disp]
    raw_labels = [f"{avg_import:.1f}",  f"{avg_export:.1f}",  net_lbl]
    colors     = ["#E65100", "#1565C0", "#2E7D32" if net_cost_true <= 0 else "#BF360C"]

    y_all  = [avg_import, avg_export, net_cost_disp]
    y_max  = max(v for v in y_all if v is not None) * 1.35 if any(v > 0 for v in y_all) else  cap * 0.1
    y_min  = min(v for v in y_all if v is not None) * 1.35 if any(v < 0 for v in y_all) else -cap * 0.1
    y_min  = min(y_min, -y_max * 0.05)    # always show a little below zero

    label_offset = (y_max - y_min) * 0.04
    label_y = [v + label_offset if v >= 0 else v - label_offset for v in values]

    bar_src   = ColumnDataSource(dict(x=categories, top=values, color=colors))
    label_src = ColumnDataSource(dict(x=categories, y=label_y, text=raw_labels))

    p4 = figure(
        height=280,
        sizing_mode="stretch_width",
        x_range=categories,
        y_range=Range1d(start=y_min, end=y_max),
        title="Price Metrics (c/kWh)",
        tools="",
        toolbar_location=None,
    )
    p4.vbar(x="x", top="top", bottom=0, width=0.55,
            color="color", alpha=0.85, source=bar_src)
    p4.add_layout(Span(location=0, dimension="width",
                       line_color="black", line_width=1))
    p4.add_layout(LabelSet(x="x", y="y", text="text", source=label_src,
                           text_align="center", text_font_size="12px",
                           text_font_style="bold"))

    p4.yaxis.axis_label             = "c/kWh"
    p4.xgrid.grid_line_color        = None
    p4.xaxis.major_label_text_font_size = "12px"
    p4.outline_line_color           = None
    return p4


def _build_load_breakdown_panel(df: pd.DataFrame):
    """
    Bar chart showing how household load was served: Export / Battery / Grid.

    NOTE: With export data (B1 = solar - load), we can't accurately break down
    how load was served without actual solar generation data. This function
    provides a rough approximation.

    Battery contribution is inferred from the change in battery_state each
    interval (discharge = negative delta → covers load before the grid does).
    Labels show both kWh totals and percentage of total load.
    """
    IH = 5 / 60

    export_kwh  = df["export_kw"] * IH if "export_kw" in df.columns else 0
    load_kwh    = df["load_kw"]  * IH
    bess_delta  = df["battery_state"].diff().fillna(df["battery_state"].iloc[0])

    # Approximate: when export > 0, some local generation is available
    # This is a rough estimate since we don't have actual solar data
    local_to_load   = (-export_kwh).clip(lower=0, upper=load_kwh)
    remaining       = (load_kwh - local_to_load).clip(lower=0)
    batt_discharge  = (-bess_delta).clip(lower=0)
    battery_to_load = batt_discharge.clip(upper=remaining)
    grid_to_load    = (remaining - battery_to_load).clip(lower=0)

    s = local_to_load.sum() if isinstance(local_to_load, pd.Series) else 0
    b = battery_to_load.sum()
    g = grid_to_load.sum()
    total = s + b + g

    if total < 1e-6:
        return None

    categories = ["Local Gen",  "Battery", "Grid"]
    values     = [s,         b,          g]
    colors     = ["#FFB300", "#4CAF50",  "#E65100"]
    pcts       = [v / total * 100 for v in values]
    raw_labels = [f"{v:.0f} kWh\n({p:.0f}%)" for v, p in zip(values, pcts)]

    y_max        = max(values) * 1.35
    label_offset = y_max * 0.04
    label_y      = [v + label_offset for v in values]

    bar_src   = ColumnDataSource(dict(x=categories, top=values, color=colors))
    label_src = ColumnDataSource(dict(x=categories, y=label_y, text=raw_labels))

    p5 = figure(
        height=280,
        sizing_mode="stretch_width",
        x_range=categories,
        y_range=Range1d(start=0, end=y_max),
        title="Load Source Breakdown",
        tools="",
        toolbar_location=None,
    )
    p5.vbar(x="x", top="top", bottom=0, width=0.55,
            color="color", alpha=0.85, source=bar_src)
    p5.add_layout(LabelSet(x="x", y="y", text="text", source=label_src,
                           text_align="center", text_font_size="12px",
                           text_font_style="bold"))

    p5.yaxis.axis_label             = "Energy (kWh)"
    p5.xgrid.grid_line_color        = None
    p5.xaxis.major_label_text_font_size = "12px"
    p5.outline_line_color           = None
    return p5
