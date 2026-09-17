"""
forecast_study_plot.py

One page for the forecast-driven MPC study (FORECAST_NOTES.md), built to be
read top-down:

  1. Headline: net profit incl. rainflow degradation per forecaster, and the
     accuracy-versus-value scatter that carries the study's main finding.
  2. Forecast accuracy: MAE, median AE and spike-clipped MAE, and error by lead time.
  3. Through the month: cumulative grid profit and the cumulative gap to perfect
     foresight, with the price spike that made the month marked.
  4. Case-study day: the 24 h forecasts each forecaster issued, on an asinh price
     axis so the cap-price spikes and the normal range both read, and the SoC each
     controller followed.
  5. Table view of every number on the page.

Colour is fixed per forecaster (plotting/theme.py) so the same hue means the same
forecaster in every panel; perfect foresight is the ink reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, FactorRange, Label, LabelSet, Span

from plotting import theme as T

ORDER = ("perfect", "perfect_price", "naive", "aemo", "lstm")
INTERVALS_PER_HOUR = 12
HORIZON = 288


def _ms(ts) -> float:
    return pd.Timestamp(ts).timestamp() * 1000.0


def _present(summary: pd.DataFrame) -> list[str]:
    have = set(summary["forecaster"])
    return [f for f in ORDER if f in have]


def _real(names) -> list[str]:
    """Forecasters that actually forecast the price."""
    return [n for n in names if n not in T.PERFECT_PRICE]


def _times(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["time"], dayfirst=True)


# ---------------------------------------------------------------------------
# 1. Headline
# ---------------------------------------------------------------------------

def _profit_bars(summary: pd.DataFrame, names: list[str]):
    s = summary.set_index("forecaster").loc[names]
    labels = [T.FORECASTER_LABEL[n] for n in names]
    perfect = float(s.loc["perfect", "net_profit_incl_degradation"]) if "perfect" in s.index else np.nan
    ref = float(s.loc["perfect_price", "net_profit_incl_degradation"]) if "perfect_price" in s.index else np.nan

    def caption(n, v):
        if n == "perfect" or np.isnan(perfect):
            return f"${v:.2f}"
        if n == "perfect_price":
            return f"${v:.2f}   (load forecast costs ${perfect - v:.2f})"
        parts = [f"{100 * v / perfect:.0f}% of perfect"]
        if ref == ref:
            parts.append(f"price forecast costs ${ref - v:.2f}")
        return f"${v:.2f}   ({', '.join(parts)})"

    src = ColumnDataSource(dict(
        label=labels,
        value=s["net_profit_incl_degradation"].to_numpy(),
        colour=[T.FORECASTER_COLOUR[n] for n in names],
        text=[caption(n, v) for n, v in zip(names, s["net_profit_incl_degradation"])],
    ))
    p = T.make_figure(height=60 + 46 * len(names), title="Net profit over the month, after rainflow degradation cost",
                      y_range=FactorRange(factors=labels[::-1]), tools="save")
    p.hbar(y="label", right="value", height=0.5, source=src, color="colour")
    p.add_layout(LabelSet(x="value", y="label", text="text", source=src, x_offset=8, text_baseline="middle",
                          text_font=T.FONT, text_font_size="11px", text_color=T.INK2))
    vmax = float(np.nanmax(src.data["value"]))
    p.x_range.start, p.x_range.end = 0, vmax * 2.2
    p.xaxis.axis_label = "$ over JAN25"
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = T.GRID
    p.yaxis.major_label_text_color = T.INK
    T.style(p, legend=False)
    return p


def _value_vs_accuracy(summary: pd.DataFrame, names: list[str]):
    s = summary.set_index("forecaster").loc[names]
    src = ColumnDataSource(dict(
        x=s["price_mae"].to_numpy(), y=s["net_profit_incl_degradation"].to_numpy(),
        label=[T.FORECASTER_LABEL[n] for n in names], colour=[T.FORECASTER_COLOUR[n] for n in names],
        med=s["price_median_ae"].to_numpy(),
    ))
    p = T.make_figure(height=60 + 46 * len(names), title="Forecast accuracy against dispatch value", tools="save,hover")
    p.scatter("x", "y", source=src, size=12, color="colour", line_color=T.SURFACE, line_width=2)
    xs, ys = np.asarray(src.data["x"], dtype=float), np.asarray(src.data["y"], dtype=float)
    x_span = max(float(np.nanmax(xs)) * 1.35 + 18, 1.0)
    y_span = max(float(np.nanmax(ys)) * 1.15, 1.0)
    y_off = [0.0] * len(xs)
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    for a, i in enumerate(order):                       # stack labels of points that sit close together
        for j in order[:a]:
            if abs(xs[i] - xs[j]) < 0.22 * x_span and abs(ys[i] - ys[j]) < 0.08 * y_span:
                y_off[i] = min(y_off[i], y_off[j] - 15)
    for i in range(len(xs)):
        p.add_layout(Label(x=float(xs[i]), y=float(ys[i]), text=src.data["label"][i], x_offset=10, y_offset=int(y_off[i]),
                           text_baseline="middle", text_font=T.FONT, text_font_size="11px", text_color=T.INK2))
    p.hover.tooltips = [("forecaster", "@label"), ("net profit", "$@y{0.00}"), ("price MAE", "@x{0.0} $/MWh"), ("median AE", "@med{0.0} $/MWh")]
    p.xaxis.axis_label = "Price forecast MAE over the 24 h horizon ($/MWh)"
    p.yaxis.axis_label = "Net profit incl. degradation ($)"
    p.x_range.start = -8
    p.x_range.end = float(np.nanmax(src.data["x"])) * 1.35 + 10
    p.y_range.start = 0
    p.y_range.end = float(np.nanmax(src.data["y"])) * 1.15
    p.xgrid.grid_line_color = T.GRID
    T.style(p, legend=False)
    return p


# ---------------------------------------------------------------------------
# 2. Accuracy
# ---------------------------------------------------------------------------

def _accuracy_bars(summary: pd.DataFrame, names: list[str]):
    real = _real(names)
    s = summary.set_index("forecaster").loc[real]
    labels = [T.FORECASTER_LABEL[n] for n in real]
    panels = []
    specs = [
        ("price_mae", "Mean absolute error", "$/MWh"),
        ("price_median_ae", "Median absolute error", "$/MWh"),
        ("price_mae_clip1000", "MAE with forecasts clipped at 1,000 $/MWh", "$/MWh"),
    ]
    for field, title, unit in specs:
        if field not in s:
            continue
        vals = s[field].to_numpy(dtype=float)
        src = ColumnDataSource(dict(label=labels, value=vals, colour=[T.FORECASTER_COLOUR[n] for n in real],
                                    text=[f"{v:.1f}" for v in vals]))
        p = T.make_figure(height=50 + 40 * len(real), title=title, y_range=FactorRange(factors=labels[::-1]), tools="save")
        p.hbar(y="label", right="value", height=0.5, source=src, color="colour")
        p.add_layout(LabelSet(x="value", y="label", text="text", source=src, x_offset=6, text_baseline="middle",
                              text_font=T.FONT, text_font_size="11px", text_color=T.INK2))
        p.x_range.start, p.x_range.end = 0, float(np.nanmax(vals)) * 1.25
        p.xaxis.axis_label = unit
        p.ygrid.grid_line_color = None
        p.xgrid.grid_line_color = T.GRID
        p.yaxis.major_label_text_color = T.INK
        T.style(p, legend=False)
        panels.append(p)
    return panels


def _error_by_lead_time(frame, forecasters: dict, names: list[str]):
    """MAE and median AE at each 5-min lead time over the test month (forecasts issued every half hour)."""
    real = [n for n in _real(names) if n in forecasters]
    if not real or frame is None:
        return []
    t0, T_ = frame.test_start, frame.n
    issues = range(t0, T_ - HORIZON + 1, 6)
    actual = np.stack([frame.rrp[t : t + HORIZON] for t in issues])
    lead_h = (np.arange(HORIZON // 6) * 6 + 3) / INTERVALS_PER_HOUR      # centre of each half-hour bin
    data = {"lead_h": lead_h}
    for n in real:
        fc = np.stack([np.asarray(forecasters[n].price(t, HORIZON), dtype=float) for t in issues])
        fc[:, 0] = actual[:, 0]                          # step-0 actual, as in the MPC loop
        ae = np.abs(fc - actual).reshape(len(actual), HORIZON // 6, 6)   # (issue, half-hour bin, step)
        data[f"mae_{n}"] = ae.mean(axis=(0, 2))
        data[f"med_{n}"] = np.median(ae.reshape(len(actual), HORIZON // 6, 6).transpose(1, 0, 2).reshape(HORIZON // 6, -1), axis=1)
    src = ColumnDataSource(data)
    panels = []
    for stat, title in (("mae", "Mean absolute error by lead time (half-hour bins)"), ("med", "Median absolute error by lead time (half-hour bins)")):
        p = T.make_figure(height=280, title=title)
        rends = []
        for n in real:
            rends.append(p.line("lead_h", f"{stat}_{n}", source=src, line_width=2, legend_label=T.FORECASTER_LABEL[n], **T.line_style(n)))
        T.line_hover(p, rends[:1], [("lead", "@lead_h{0.0} h")] + [(T.FORECASTER_LABEL[n], f"@{stat}_{n}{{0.0}} $/MWh") for n in real])
        p.xaxis.axis_label = "Lead time (hours ahead)"
        p.yaxis.axis_label = "$/MWh"
        p.x_range.start, p.x_range.end = 0, 24
        p.y_range.start = 0
        p.xaxis.ticker = [0, 4, 8, 12, 16, 20, 24]
        p.legend.location = "top_left" if stat == "mae" else "bottom_right"
        T.style(p)
        panels.append(p)
    return panels


# ---------------------------------------------------------------------------
# 3. Through the month
# ---------------------------------------------------------------------------

def _month_source(results: dict, names: list[str]) -> tuple[ColumnDataSource, pd.Series]:
    base = results[names[0]]
    t = _times(base)
    data = {"time": t}
    for n in names:
        data[f"profit_{n}"] = results[n]["cumulative_profit"].to_numpy()
        data[f"soc_{n}"] = results[n]["battery_state"].to_numpy()
        data[f"bat_{n}"] = (results[n]["discharge_kw"] - results[n]["charge_kw"]).to_numpy()
    if "perfect" in names:
        for n in names:
            data[f"gap_{n}"] = data["profit_perfect"] - data[f"profit_{n}"]
    data["rrp"] = base["rrp"].to_numpy()
    return ColumnDataSource(data), t


def _spike_marker(p, results: dict, names: list[str], y_text: float):
    base = results[names[0]]
    i = int(base["rrp"].to_numpy().argmax())
    ts = _times(base).iloc[i]
    p.add_layout(Span(location=_ms(ts), dimension="height", line_color=T.MUTED, line_width=1, line_dash="dashed"))
    p.add_layout(Label(x=_ms(ts), y=y_text, text=f"{base['rrp'].iloc[i]:,.0f} $/MWh spike, {ts:%-d %b %H:%M}", x_offset=6,
                       text_font=T.FONT, text_font_size="11px", text_color=T.MUTED))


def _cumulative_profit(src: ColumnDataSource, results: dict, names: list[str]):
    p = T.make_figure(height=340, title="Cumulative grid profit through the month (before degradation cost)", x_axis_type="datetime")
    T.zero_line(p)
    rends = []
    for n in names:
        rends.append(p.line("time", f"profit_{n}", source=src, line_width=2.5 if n == "perfect" else 2,
                            legend_label=T.FORECASTER_LABEL[n], **T.line_style(n)))
    T.line_hover(p, rends[:1], [("time", "@time{%d %b %H:%M}")] + [(T.FORECASTER_LABEL[n], f"$@profit_{n}{{0.00}}") for n in names],
                 formatters={"@time": "datetime"})
    ys = np.concatenate([src.data[f"profit_{n}"] for n in names])
    lo, hi = float(ys.min()), float(ys.max())
    span = hi - lo
    p.y_range.start, p.y_range.end = lo - 0.05 * span, hi + 0.12 * span
    last_x = _ms(src.data["time"].iloc[-1])
    T.end_labels(p, [(last_x, float(src.data[f"profit_{n}"][-1]), f"${src.data[f'profit_{n}'][-1]:.2f}") for n in names],
                 y_span=1.17 * span, inner_height_px=340 - 80)
    p.x_range.end = last_x + 0.06 * (last_x - _ms(src.data["time"].iloc[0]))
    _spike_marker(p, results, names, y_text=hi + 0.04 * span)
    p.yaxis.axis_label = "$"
    p.legend.location = "top_left"
    T.style(p)
    return p


def _cumulative_gap(src: ColumnDataSource, results: dict, names: list[str]):
    real = [n for n in names if n != "perfect"]   # perfect_price stays: its gap is the cost of the load forecast
    p = T.make_figure(height=280, title="Cumulative shortfall against perfect foresight (where the gap opens)", x_axis_type="datetime")
    T.zero_line(p)
    rends = []
    for n in real:
        rends.append(p.line("time", f"gap_{n}", source=src, line_width=2, legend_label=T.FORECASTER_LABEL[n], **T.line_style(n)))
    T.line_hover(p, rends[:1], [("time", "@time{%d %b %H:%M}")] + [(T.FORECASTER_LABEL[n], f"$@gap_{n}{{0.00}}") for n in real],
                 formatters={"@time": "datetime"})
    ys = np.concatenate([src.data[f"gap_{n}"] for n in real])
    lo, hi = min(0.0, float(ys.min())), float(ys.max())
    span = hi - lo
    p.y_range.start, p.y_range.end = lo - 0.05 * span, hi + 0.15 * span
    last_x = _ms(src.data["time"].iloc[-1])
    T.end_labels(p, [(last_x, float(src.data[f"gap_{n}"][-1]), f"${src.data[f'gap_{n}'][-1]:.2f}") for n in real],
                 y_span=1.2 * span, inner_height_px=280 - 80)
    p.x_range.end = last_x + 0.06 * (last_x - _ms(src.data["time"].iloc[0]))
    _spike_marker(p, results, names, y_text=hi + 0.06 * span)
    p.yaxis.axis_label = "$ behind perfect foresight"
    p.legend.location = "top_left"
    T.style(p)
    return p


# ---------------------------------------------------------------------------
# 4. Case-study day
# ---------------------------------------------------------------------------

def _forecast_panel(frame, forecasters: dict, names: list[str], issue_time: pd.Timestamp, *, x_range=None):
    real = [n for n in _real(names) if n in forecasters]
    t = int(np.searchsorted(frame.start_times.values, np.datetime64(issue_time)))
    h = min(HORIZON, frame.n - t)
    times = frame.start_times[t : t + h]
    data = {"time": times, "actual": frame.rrp[t : t + h], "actual_y": T.price_to_axis(frame.rrp[t : t + h])}
    for n in real:
        fc = np.asarray(forecasters[n].price(t, h), dtype=float)
        data[n] = fc
        data[f"{n}_y"] = T.price_to_axis(fc)
    src = ColumnDataSource(data)
    p = T.make_figure(height=330, title=f"24 h price forecasts issued {issue_time:%-d %b %H:%M}", x_axis_type="datetime",
                      **({"x_range": x_range} if x_range is not None else {}))
    T.asinh_price_axis(p)
    rends = [p.line("time", "actual_y", source=src, color=T.INK, line_width=2.5, legend_label="Actual")]
    for n in real:
        p.line("time", f"{n}_y", source=src, line_width=2, legend_label=T.FORECASTER_LABEL[n], **T.line_style(n))
    T.line_hover(p, rends, [("time", "@time{%d %b %H:%M}"), ("actual", "@actual{0} $/MWh")]
                 + [(T.FORECASTER_LABEL[n], f"@{n}{{0}} $/MWh") for n in real], formatters={"@time": "datetime"})
    p.add_layout(Span(location=_ms(issue_time), dimension="height", line_color=T.MUTED, line_dash="dashed", line_width=1))
    lo = min(-50.0, float(frame.rrp[t : t + h].min()) - 10)
    p.y_range.start, p.y_range.end = float(T.price_to_axis(lo)), float(T.price_to_axis(20000))
    p.legend.location = "top_right"
    T.style(p)
    return p


def _day_soc_panel(src_full: ColumnDataSource, names: list[str], day_start: pd.Timestamp, bess_size: float | None, spike_ts):
    t = src_full.data["time"]
    m = (t >= day_start) & (t < day_start + pd.Timedelta(days=1))
    data = {k: (np.asarray(v)[m.to_numpy()] if k != "time" else t[m]) for k, v in src_full.data.items()}
    src = ColumnDataSource(data)
    p = T.make_figure(height=280, title=f"Battery state of charge under each controller, {day_start:%-d %b}", x_axis_type="datetime")
    rends = []
    for n in names:
        rends.append(p.line("time", f"soc_{n}", source=src, line_width=2.5 if n == "perfect" else 2,
                            legend_label=T.FORECASTER_LABEL[n], **T.line_style(n)))
    T.line_hover(p, rends[:1], [("time", "@time{%H:%M}"), ("price", "@rrp{0} $/MWh")] + [(T.FORECASTER_LABEL[n], f"@soc_{n}{{0.0}} kWh") for n in names],
                 formatters={"@time": "datetime"})
    if bess_size:
        p.add_layout(Span(location=bess_size, dimension="width", line_color=T.AXIS, line_width=1))
        p.y_range.start, p.y_range.end = 0, bess_size * 1.08
    if spike_ts is not None and day_start <= spike_ts < day_start + pd.Timedelta(days=1):
        p.add_layout(Span(location=_ms(spike_ts), dimension="height", line_color=T.MUTED, line_dash="dashed", line_width=1))
    p.yaxis.axis_label = "kWh"
    p.legend.location = "bottom_right"
    T.style(p)
    return p


def _day_price_panel(src_full: ColumnDataSource, day_start: pd.Timestamp, x_range):
    t = src_full.data["time"]
    m = (t >= day_start) & (t < day_start + pd.Timedelta(days=1))
    rrp = np.asarray(src_full.data["rrp"])[m.to_numpy()]
    src = ColumnDataSource({"time": t[m], "rrp": rrp, "y": T.price_to_axis(rrp)})
    p = T.make_figure(height=220, title=f"Actual dispatch price, {day_start:%-d %b}", x_axis_type="datetime", x_range=x_range)
    T.asinh_price_axis(p)
    r = p.line("time", "y", source=src, color=T.INK, line_width=2)
    T.line_hover(p, [r], [("time", "@time{%H:%M}"), ("price", "@rrp{0} $/MWh")], formatters={"@time": "datetime"})
    p.y_range.start, p.y_range.end = float(T.price_to_axis(min(-50, rrp.min() - 10))), float(T.price_to_axis(20000))
    T.style(p, legend=False)
    return p


# ---------------------------------------------------------------------------
# 5. Table
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    ("label", "Forecaster", None),
    ("net_profit_incl_degradation", "Net profit ($)", "0.00"),
    ("profit_gap_to_perfect", "Gap ($)", "0.00"),
    ("profit_pct_of_perfect", "% of perfect", "0.0"),
    ("net_profit_ex_degradation", "Grid profit ($)", "0.00"),
    ("degradation_cost_rainflow", "Rainflow ($)", "0.00"),
    ("life_loss_pct", "Life loss (%)", "0.000"),
    ("price_mae", "MAE", "0.0"),
    ("price_median_ae", "Median AE", "0.0"),
    ("price_mae_clip1000", "MAE clipped", "0.0"),
    ("price_rmae_vs_naive", "rMAE vs naive", "0.00"),
    ("price_fc_spike_frac", "Forecast > 1000", "0.0%"),
    ("net_local_mae_kw", "Net-local (kW)", "0.00"),
]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def plot_forecast_study(
    summary: pd.DataFrame,
    results: dict[str, pd.DataFrame],
    frame=None,
    forecasters: dict | None = None,
    *,
    title: str,
    output_path: str,
    day: str = "2025-01-15",
    issue_hours=(4.0 + 5 / 60, 12.0 + 5 / 60),
    bess_size: float | None = 13.5,
    price_ylim=None,   # kept for call compatibility; the price axis is asinh-scaled instead of clipped
) -> None:
    forecasters = forecasters or {}
    names = [n for n in _present(summary) if n in results]
    s = summary.set_index("forecaster")
    perfect = float(s.loc["perfect", "net_profit_incl_degradation"]) if "perfect" in s.index else np.nan
    real = _real(names)
    if isinstance(issue_hours, (int, float)):
        issue_hours = (float(issue_hours),)

    # KPI row
    tiles = []
    if "perfect" in s.index:
        tiles.append({"label": "Perfect foresight, net of degradation", "value": f"${perfect:.2f}", "sub": "upper bound, same 24 h MPC loop"})
    if "perfect_price" in s.index and perfect == perfect:
        v = float(s.loc["perfect_price", "net_profit_incl_degradation"])
        tiles.append({"label": "Perfect price, forecast household load", "value": f"${v:.2f}",
                      "sub": f"load forecast costs ${perfect - v:.2f}; the rest of any gap is the price forecast"})
    if real:
        best = max(real, key=lambda n: s.loc[n, "net_profit_incl_degradation"])
        v = float(s.loc[best, "net_profit_incl_degradation"])
        ref = float(s.loc["perfect_price", "net_profit_incl_degradation"]) if "perfect_price" in s.index else np.nan
        sub = f"{100 * v / perfect:.0f}% of perfect" if perfect == perfect else ""
        if ref == ref:
            sub += f", price forecast costs ${ref - v:.2f}"
        tiles.append({"label": f"Best real forecaster: {T.FORECASTER_LABEL[best]}", "value": f"${v:.2f}", "sub": sub})
    if "lstm" in s.index and "naive" in s.index:
        r = float(s.loc["lstm", "price_mae"] / s.loc["naive", "price_mae"])
        tiles.append({"label": "LSTM price MAE relative to naive", "value": f"{100 * (1 - r):.0f}% lower", "sub": f"rMAE {r:.2f}"})
    if "aemo" in s.index and "price_fc_spike_frac" in s:
        tiles.append({"label": "AEMO intervals forecast above 1,000 $/MWh", "value": f"{100 * float(s.loc['aemo', 'price_fc_spike_frac']):.1f}%",
                      "sub": f"actual: {100 * float(s.loc['aemo', 'price_actual_spike_frac']):.1f}%"})

    children = [
        T.heading(title, "Household MILP re-planned every 5 minutes on a 24 h horizon under each price forecaster. "
                         "The current interval always uses the actual dispatch price. Net-local power uses the same 7-day profile "
                         "for every forecaster except full perfect foresight, so 'perfect price, forecast load' is the fair reference "
                         "for the real forecasters: the only thing that changes from it is the price forecast."),
        T.stat_tiles(tiles),
        T.section("Headline: what each forecast is worth in dispatch"),
        row(_profit_bars(summary, names), _value_vs_accuracy(summary, names), sizing_mode="stretch_width"),
    ]

    acc = _accuracy_bars(summary, names)
    if acc:
        children += [T.section("Price forecast accuracy over the 24 h horizon",
                               "Errors on the price path the MILP actually planned on (step 0 = actual). The clipped MAE caps each forecast at 1,000 $/MWh "
                               "before scoring, which shows how much of the AEMO error comes from forecast cap-price spikes that did not eventuate."),
                     row(*acc, sizing_mode="stretch_width")]
    lead = _error_by_lead_time(frame, forecasters, names)
    if lead:
        children.append(row(*lead, sizing_mode="stretch_width"))

    src_month, _ = _month_source(results, names)
    children += [T.section("Through the month"), _cumulative_profit(src_month, results, names)]
    if "perfect" in names and real:
        children.append(_cumulative_gap(src_month, results, names))

    if frame is not None and forecasters:
        day_start = pd.Timestamp(day)
        base = results[names[0]]
        i = int(base["rrp"].to_numpy().argmax())
        spike_ts = _times(base).iloc[i]
        panels = []
        for hrs in issue_hours:
            panels.append(_forecast_panel(frame, forecasters, names, day_start + pd.Timedelta(hours=hrs)))
        children += [T.section(f"Case study: {day_start:%-d %B %Y}",
                               "Each panel shows the 24 h forecasts issued at the dashed line against what happened. "
                               "The price axis is asinh-scaled: linear near zero, logarithmic in the spikes, so a 17,500 $/MWh cap price and a 40 $/MWh afternoon both read."),
                     row(*panels, sizing_mode="stretch_width")]
        soc = _day_soc_panel(src_month, names, day_start, bess_size, spike_ts)
        price = _day_price_panel(src_month, day_start, soc.x_range)
        children += [price, soc]

    tbl = summary.copy()
    tbl["label"] = [T.FORECASTER_LABEL.get(n, n) for n in tbl["forecaster"]]
    tbl = tbl.set_index("forecaster").loc[names].reset_index()
    children += [T.section("Table view"), T.summary_table(tbl, TABLE_COLUMNS, height=48 + 28 * len(names)),
                 T.note("Net profit incl. degradation = grid revenue − grid cost (incl. network tariff on imports) − rainflow life loss × R_cell. "
                        "MAE and median AE are over all 288 steps of every 24 h forecast issued at half-hour boundaries in the test month. "
                        "Use the toolbar's save button on any panel to export it as PNG.")]
    T.save_page(children, title=title, output_path=output_path)
