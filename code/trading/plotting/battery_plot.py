"""
battery_plot.py

One page per simulation run (state machine, perfect-foresight MILP, or MPC):
a KPI row, then four linked panels that share the x axis, each with one y axis:

  1. Dispatch price ($/MWh) on an asinh scale, with buy/sell thresholds when given
  2. Battery state of charge (kWh) against the usable capacity
  3. Power (kW): battery charge/discharge, household net-local, grid import/export
  4. Cumulative $: net grid profit, revenue, cost and (when present) the optimiser's
     degradation charge

followed by the rainflow cycle-depth histogram of the SoC trajectory. Everything
shown in the panels is also in the hover readout.

Required columns: time, battery_state, rrp, cumulative_profit, cumulative_revenue,
cumulative_cost, grid_import_kwh, grid_export_kwh. Optional: export_kw, import_kw,
charge_kw, discharge_kw, cumulative_degradation, degradation_cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Label, Range1d, RangeTool, Span

from plotting import theme as T

INTERVAL_HOURS = 5 / 60


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    try:
        return pd.to_datetime(series, dayfirst=True)
    except Exception:
        return pd.to_datetime(series)


def _prepare(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    df["time"] = _to_datetime(df["time"])
    df["grid_kw"] = (df["grid_import_kwh"] - df["grid_export_kwh"]) / INTERVAL_HOURS     # + import, - export
    if "export_kw" in df and "import_kw" in df:
        df["net_local_kw"] = df["export_kw"] - df["import_kw"]                          # + surplus, - deficit
    else:
        df["net_local_kw"] = 0.0
    if "charge_kw" in df and "discharge_kw" in df:
        df["battery_kw"] = df["charge_kw"] - df["discharge_kw"]                          # + charging
    else:
        delta = df["battery_state"].diff().fillna(0.0)
        df["battery_kw"] = delta / INTERVAL_HOURS
    if "cumulative_degradation" not in df:
        df["cumulative_degradation"] = np.nan
    df["profit_after_model_deg"] = df["cumulative_profit"] - df["cumulative_degradation"].fillna(0.0)
    df["rrp_y"] = T.price_to_axis(df["rrp"])
    return df


def _rainflow_panel(soc: np.ndarray, e_max: float):
    from milp.degradation import rainflow_life_loss

    loss, cycles = rainflow_life_loss(soc, e_max)
    depths = np.array([c[0] for c in cycles]) if cycles else np.array([0.0])
    counts = np.array([c[2] for c in cycles]) if cycles else np.array([0.0])
    edges = np.linspace(0, 1, 11)
    hist, _ = np.histogram(depths, bins=edges, weights=counts)
    src = ColumnDataSource(dict(left=edges[:-1], right=edges[1:], top=hist, mid=(edges[:-1] + edges[1:]) / 2))
    p = T.make_figure(height=260, title="Rainflow cycle count by depth of discharge", tools="save,hover")
    p.quad(left="left", right="right", bottom=0, top="top", source=src, fill_color=T.BLUE, line_color=T.SURFACE, line_width=2)
    p.hover.tooltips = [("depth", "@left{0.0}–@right{0.0}"), ("cycles", "@top{0.0}")]
    p.xaxis.axis_label = "Cycle depth (fraction of usable capacity)"
    p.yaxis.axis_label = "Cycles (half cycles count 0.5)"
    p.x_range = Range1d(0, 1)
    p.y_range.start = 0
    T.style(p, legend=False)
    stats = {"life_loss": loss, "cycles": float(counts.sum()),
             "mean_depth": float((depths * counts).sum() / counts.sum()) if counts.sum() > 0 else 0.0,
             "max_depth": float(depths.max()) if len(depths) else 0.0}
    return p, stats


def plot_battery_trading(
    results_df: pd.DataFrame,
    *,
    title: str = "BESS trading results",
    output_path: str = "battery_plot.html",
    bess_size: float | None = None,
    price_percentile_clip=None,          # kept for call compatibility; the price axis is asinh-scaled instead
    buy_threshold: float | None = None,
    sell_threshold: float | None = None,
    show_plot: bool = True,
    subtitle: str = "",
    window_days: float = 3.0,
) -> None:
    """window_days: initial width of the detail window; the overview strip's handles move it over the full run."""
    df = _prepare(results_df)
    src = ColumnDataSource(df)
    t0, t1 = df["time"].iloc[0], df["time"].iloc[-1]
    x_range = Range1d(start=t0, end=min(t1, t0 + pd.Timedelta(days=window_days)))
    hover_time = [("time", "@time{%d %b %H:%M}")]
    fmt = {"@time": "datetime"}

    # -- KPIs -----------------------------------------------------------------
    profit = float(df["cumulative_profit"].iloc[-1])
    revenue = float(df["cumulative_revenue"].iloc[-1])
    cost = float(df["cumulative_cost"].iloc[-1])
    discharged = float((-df["battery_kw"]).clip(lower=0).sum() * INTERVAL_HOURS)
    e_max = bess_size or float(df["battery_state"].max())
    rf_panel, rf = _rainflow_panel(np.concatenate([[df["battery_state"].iloc[0]], df["battery_state"].to_numpy()]), e_max)
    tiles = [
        {"label": "Net grid profit", "value": (f"−${abs(profit):,.2f}" if profit < 0 else f"${profit:,.2f}"), "sub": "revenue − cost, before degradation", "tone": "good" if profit >= 0 else "bad"},
        {"label": "Grid revenue", "value": f"${revenue:,.2f}", "sub": f"{df['grid_export_kwh'].sum():,.0f} kWh exported"},
        {"label": "Grid cost", "value": f"${cost:,.2f}", "sub": f"{df['grid_import_kwh'].sum():,.0f} kWh imported, incl. network tariff"},
        {"label": "Equivalent full cycles", "value": f"{discharged / e_max:,.1f}", "sub": f"{discharged:,.0f} kWh discharged"},
        {"label": "Cycle life consumed", "value": f"{100 * rf['life_loss']:.3f}%", "sub": f"{rf['cycles']:.0f} rainflow cycles, mean depth {rf['mean_depth']:.2f}"},
    ]
    if df["cumulative_degradation"].notna().any():
        deg = float(df["cumulative_degradation"].iloc[-1])
        tiles.insert(1, {"label": "Optimiser's degradation charge", "value": f"${deg:,.2f}", "sub": f"net after charge ${profit - deg:,.2f}"})

    # -- 0. overview strip: whole run, drag the window to move the detail panels --
    p0 = T.make_figure(height=120, title="Whole run: drag or resize the shaded window to choose what the detail panels show",
                       x_axis_type="datetime", x_range=Range1d(start=t0, end=t1), tools="")
    p0.line("time", "rrp_y", source=src, color=T.MUTED, line_width=1)
    T.asinh_price_axis(p0, ticks=(0, 300, 17500), axis_label="")
    p0.y_range = Range1d(float(T.price_to_axis(min(-50.0, float(df["rrp"].min()) - 10))), float(T.price_to_axis(20000)))
    rt = RangeTool(x_range=x_range)
    rt.overlay.fill_color = T.BLUE
    rt.overlay.fill_alpha = 0.12
    p0.add_tools(rt)
    p0.toolbar_location = None
    T.style(p0, legend=False)

    # -- 1. price -------------------------------------------------------------
    p1 = T.make_figure(height=260, title="Dispatch price", x_axis_type="datetime", x_range=x_range)
    T.asinh_price_axis(p1)
    r1 = p1.line("time", "rrp_y", source=src, color=T.INK, line_width=1.5)
    T.line_hover(p1, [r1], hover_time + [("price", "@rrp{0.0} $/MWh")], formatters=fmt)
    lo = min(-50.0, float(df["rrp"].min()) - 10)
    p1.y_range = Range1d(float(T.price_to_axis(lo)), float(T.price_to_axis(max(1000.0, float(df["rrp"].max()) * 1.3))))
    for thr, name, base, off in ((buy_threshold, "buy below", "top", -3), (sell_threshold, "sell above", "bottom", 3)):
        if thr is not None:
            y = float(T.price_to_axis(thr))
            p1.add_layout(Span(location=y, dimension="width", line_color=T.MUTED, line_dash="dashed", line_width=1))
            p1.add_layout(Label(x=t0, y=y, text=f"{name} {thr:.0f} $/MWh", x_offset=4, y_offset=off, text_baseline=base,
                                text_font=T.FONT, text_font_size="11px", text_color=T.MUTED))
    T.style(p1, legend=False)

    # -- 2. SoC ---------------------------------------------------------------
    p2 = T.make_figure(height=240, title="Battery state of charge", x_axis_type="datetime", x_range=x_range)
    p2.varea(x="time", y1=0, y2="battery_state", source=src, fill_color=T.BLUE, fill_alpha=0.10)
    r2 = p2.line("time", "battery_state", source=src, color=T.BLUE, line_width=1.5)
    T.line_hover(p2, [r2], hover_time + [("SoC", "@battery_state{0.00} kWh")], formatters=fmt)
    if bess_size:
        p2.add_layout(Span(location=bess_size, dimension="width", line_color=T.AXIS, line_width=1))
        p2.add_layout(Label(x=t0, y=bess_size, text=f"usable capacity {bess_size:g} kWh", x_offset=4, y_offset=-3, text_align="left",
                            text_baseline="top", text_font=T.FONT, text_font_size="11px", text_color=T.MUTED))
        p2.y_range = Range1d(0, bess_size * 1.08)
    else:
        p2.y_range.start = 0
    p2.yaxis.axis_label = "kWh"
    T.style(p2, legend=False)

    # -- 3. power -------------------------------------------------------------
    p3 = T.make_figure(height=280, title="Power flows", x_axis_type="datetime", x_range=x_range)
    T.zero_line(p3)
    has_local = bool(np.any(df["net_local_kw"] != 0))
    rends = [p3.line("time", "battery_kw", source=src, color=T.BLUE, line_width=1.5, legend_label="Battery (+ charge / − discharge)")]
    rends.append(p3.line("time", "grid_kw", source=src, color=T.ORANGE, line_width=1.5, legend_label="Grid (+ import / − export)"))
    if has_local:
        rends.append(p3.line("time", "net_local_kw", source=src, color=T.AQUA, line_width=1.5, legend_label="Household net (+ surplus / − deficit)"))
    tips = hover_time + [("battery", "@battery_kw{0.00} kW"), ("grid", "@grid_kw{0.00} kW")] + ([("household net", "@net_local_kw{0.00} kW")] if has_local else [])
    T.line_hover(p3, rends[:1], tips, formatters=fmt)
    m = float(max(df["battery_kw"].abs().max(), df["grid_kw"].abs().max(), df["net_local_kw"].abs().max(), 1.0))
    p3.y_range = Range1d(-1.15 * m, 1.45 * m)
    p3.yaxis.axis_label = "kW"
    p3.legend.location = "top_left"
    p3.legend.orientation = "horizontal"
    T.style(p3)

    # -- 4. cumulative $ ------------------------------------------------------
    p4 = T.make_figure(height=280, title="Cumulative grid revenue, cost and profit over the whole run", x_axis_type="datetime",
                       x_range=Range1d(start=t0, end=t1))
    T.zero_line(p4)
    rends = [p4.line("time", "cumulative_profit", source=src, color=T.INK, line_width=2.5, legend_label="Net profit")]
    rends.append(p4.line("time", "cumulative_revenue", source=src, color=T.BLUE, line_width=1.5, legend_label="Revenue"))
    rends.append(p4.line("time", "cumulative_cost", source=src, color=T.ORANGE, line_width=1.5, legend_label="Cost"))
    tips = hover_time + [("net profit", "$@cumulative_profit{0.00}"), ("revenue", "$@cumulative_revenue{0.00}"), ("cost", "$@cumulative_cost{0.00}")]
    if df["cumulative_degradation"].notna().any():
        rends.append(p4.line("time", "cumulative_degradation", source=src, color=T.AQUA, line_width=1.5, legend_label="Optimiser's degradation charge"))
        tips.append(("degradation charge", "$@cumulative_degradation{0.00}"))
    T.line_hover(p4, rends[:1], tips, formatters=fmt)
    ys = df[["cumulative_profit", "cumulative_revenue", "cumulative_cost"]].to_numpy()
    lo, hi = min(0.0, float(ys.min())), float(ys.max())
    span = max(hi - lo, 1.0)
    p4.y_range = Range1d(lo - 0.05 * span, hi + 0.25 * span)
    p4.yaxis.axis_label = "$"
    p4.legend.location = "top_left"
    p4.legend.orientation = "horizontal"
    T.style(p4)

    # -- page -----------------------------------------------------------------
    children = [
        T.heading(title, subtitle or f"{t0:%-d %b %Y} to {t1:%-d %b %Y}, 5-minute intervals. The three detail panels share the window chosen "
                                     "on the strip below; drag to pan, scroll to zoom, hover for values. The cumulative panel always shows the whole run."),
        T.stat_tiles(tiles),
        p0, p1, p2, p3, p4,
        T.section("Cycling"),
        row(rf_panel, T.note(
            "Rainflow counting on the SoC trajectory, normalised by usable capacity. Life loss per cycle is Φ(δ) = a·δ^k "
            f"(NMC fit, thesis Eq. 3.8). Deepest cycle {rf['max_depth']:.2f}, mean depth {rf['mean_depth']:.2f}."),
            sizing_mode="stretch_width"),
    ]
    T.save_page(children, title=title, output_path=output_path)
    if show_plot:
        import webbrowser
        webbrowser.open(output_path)
