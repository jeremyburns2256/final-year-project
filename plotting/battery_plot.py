"""
battery_plot.py
Reusable Bokeh plotting module for BESS trading visualisations.

Plots RRP and battery SoC on a single graph with dual y-axes.
Price y-axis is clipped to a percentile range to suppress outlier spikes.
"""

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
    Plot RRP and battery SoC on a single graph with dual y-axes.

    Required columns in results_df: time, battery_state, rrp

    Parameters
    ----------
    bess_size : float, optional
        Battery capacity in kWh — sets the SoC y-axis ceiling.
    price_percentile_clip : tuple, optional
        (low, high) percentiles used to clip the price y-axis range,
        suppressing extreme outlier spikes. Default (2, 98).
    buy_threshold : float, optional
        Price in $/MWh below which the strategy buys — draws a dashed line.
    sell_threshold : float, optional
        Price in $/MWh above which the strategy sells — draws a dashed line.
    """
    df = results_df.copy()
    df["time"] = _to_datetime(df["time"])
    src = ColumnDataSource(df)

    output_file(output_path, title=title)

    # ── Y-axis ranges ─────────────────────────────────────────────────────────
    soc_max = (bess_size * 1.05) if bess_size is not None else (df["battery_state"].max() * 1.1)
    soc_range = Range1d(start=0, end=soc_max)

    lo, hi = price_percentile_clip
    price_lo = df["rrp"].quantile(lo / 100)
    price_hi = df["rrp"].quantile(hi / 100)
    padding = (price_hi - price_lo) * 0.05
    price_range = Range1d(start=price_lo - padding, end=price_hi + padding)

    # ── X-axis range: default view = first day ────────────────────────────────
    x_start = df["time"].iloc[0]
    x_end   = x_start + pd.Timedelta(days=1)

    # ── Figure ────────────────────────────────────────────────────────────────
    p = figure(
        height=600,
        sizing_mode="stretch_width",
        x_axis_type="datetime",
        x_range=(x_start, x_end),
        y_range=soc_range,
        title=title,
        tools=_TOOLS,
        toolbar_location="right",
    )
    p.xaxis.formatter = _X_FMT
    p.xaxis.ticker.desired_num_ticks = 24
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "SoC (kWh)"

    # Secondary y-axis for price (right side)
    p.extra_y_ranges = {"rrp": price_range}
    price_axis = LinearAxis(y_range_name="rrp", axis_label="Price ($/MWh)")
    price_axis.ticker.desired_num_ticks = 20
    p.add_layout(price_axis, "right")

    # ── SoC (left axis) ───────────────────────────────────────────────────────
    p.varea(x="time", y1=0, y2="battery_state", source=src,
            fill_alpha=0.25, fill_color="#4CAF50")
    p.line(x="time", y="battery_state", source=src,
           line_width=1.5, color="#2E7D32", legend_label="SoC (kWh)")

    # ── RRP (right axis) ──────────────────────────────────────────────────────
    p.line(x="time", y="rrp", source=src,
           line_width=1, color="#1565C0", alpha=0.8,
           y_range_name="rrp", legend_label="RRP ($/MWh)")

    # ── Threshold lines (optional) ────────────────────────────────────────────
    x_span = [df["time"].iloc[0], df["time"].iloc[-1]]
    if buy_threshold is not None:
        p.line(x=x_span, y=[buy_threshold, buy_threshold],
               y_range_name="rrp", color="#2196F3",
               line_dash="dashed", line_width=1.5, legend_label=f"Buy < ${buy_threshold}")
    if sell_threshold is not None:
        p.line(x=x_span, y=[sell_threshold, sell_threshold],
               y_range_name="rrp", color="#F44336",
               line_dash="dashed", line_width=1.5, legend_label=f"Sell ≥ ${sell_threshold}")

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"

    if show_plot:
        show(p)
    else:
        save(p)
        print(f"Plot saved to {output_path}")
