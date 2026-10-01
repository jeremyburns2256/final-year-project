"""
noise_study_plot.py

One page on how forecast error affects trading value (FORECAST_NOTES: noise
sensitivity), built from the noise study and the real-forecaster study:

  -  The kinds of forecast compared: synthetic noise and the real forecasters.
  -  Value lost by every run, synthetic and real, as one bar chart.
  0. Example: the 24 h forecasts each run was given at one issue time on the
     spike day, synthetic and real, against what happened.
  1. Error by lead time: the synthetic error grows as sqrt(lead) while the real
     forecasters are roughly flat, which decides how the two can be compared.
  2. Value lost against error at three lead times (the committed interval,
     1 h ahead, the 24 h average), one row per forecast input.
  3. Mechanism: trading profit and degradation against price error, and where
     the rainflow cycles and their cost fall by cycle depth.
  4. Value lost by price band, and a table of every run.

Synthetic runs are greys (darker = less noise, ink = perfect foresight); colour
always means a real forecaster (plotting/theme.py). Each run is measured against
the reference that isolates its input: noise runs against perfect foresight, the
real price forecasters against perfect price with the 7-day load profile, and the
load profile against perfect foresight.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, Div, FactorRange, HoverTool, LabelSet, Span

from milp.degradation import rainflow_life_loss, stress_function
from milp.model import INTERVAL_HOURS, BatteryParams
from plotting import theme as T

PROFIT = "net_profit_incl_degradation"
PROFILE = "perfect_price"            # its net-local errors are the 7-day profile's
PROFILE_COLOUR = T.YELLOW
PROFILE_LABEL = "7-day load profile"
REAL_PRICE = ("naive", "aemo", "lstm")
GREYS = ["#3a3936", "#5f5e59", "#85847d", "#aaa9a1"]   # sigma levels, low -> high noise
BANDS = [(-np.inf, 0, "< 0"), (0, 50, "0 – 50"), (50, 100, "50 – 100"), (100, 300, "100 – 300"),
         (300, 1000, "300 – 1000"), (1000, np.inf, "> 1000")]
STALE_NOTE = ("Real-forecaster runs (naive, AEMO, LSTM, perfect price) are read from results/forecast_summary.csv and "
              "results/mpc_household_J4_<name>.csv; until forecast_trading.py is re-run they predate the 2026-09-21 meter re-stamp, "
              "so their value-lost figures are on the old join. Their forecast errors are recomputed here and are current.")


# ---------------------------------------------------------------------------
# Run bookkeeping
# ---------------------------------------------------------------------------

def _sigma(name: str) -> float:
    return 0.0 if name == "perfect" else float(name.rsplit("_", 1)[1])


def _noise_names(noise: pd.DataFrame, prefix: str) -> list[str]:
    names = [f for f in noise["forecaster"] if f.startswith(prefix)]
    return sorted(names, key=_sigma)


def _grey(names: list[str], name: str) -> str:
    return T.INK if name == "perfect" else GREYS[min(names.index(name), len(GREYS) - 1)]


def _label(name: str) -> str:
    if name == "perfect":
        return "Perfect foresight"
    if name.startswith("noise_price_"):
        return f"Price noise σ_max = {_sigma(name):g}"
    if name.startswith("noise_net_"):
        return f"Load noise σ_max = {_sigma(name):g} kW"
    if name == PROFILE:
        return PROFILE_LABEL
    return T.FORECASTER_LABEL.get(name, name)


def _lost(noise: pd.DataFrame, real: pd.DataFrame | None) -> dict[str, float]:
    """Value lost against the reference that isolates each run's input."""
    out = {}
    n = noise.set_index("forecaster")[PROFIT]
    for f, v in n.items():
        out[f] = float(n["perfect"] - v)
    if real is not None:
        r = real.set_index("forecaster")[PROFIT]
        if PROFILE in r.index:
            for f in REAL_PRICE:
                if f in r.index:
                    out[f] = float(r[PROFILE] - r[f])
            if "perfect" in r.index:
                out[PROFILE] = float(r["perfect"] - r[PROFILE])
    return out


def _legend_outside(p):
    p.add_layout(p.legend[0], "right")


# ---------------------------------------------------------------------------
# Forecasts compared
# ---------------------------------------------------------------------------

GUIDE = {
    "noise_price": ("Synthetic price noise",
                    "The actual price path plus a random error path, added to asinh(price/100): roughly additive near $0 and "
                    "proportional on high prices, so a spike keeps its timing and only its size is wrong. The error is smooth along "
                    "the horizon (AR(1), ρ = 0.95 per 5 min), near zero for the next interval and growing as √lead to σ_max at 24 h "
                    "ahead. A fresh path is drawn at every re-plan, so the error has no bias and does not persist from one plan to "
                    "the next. Levels: σ_max = {levels}."),
    "noise_net": ("Synthetic load noise",
                  "The actual net-local power plus an error path of the same shape, added in kW. Levels: σ_max = {levels} kW."),
    "naive": ("Seasonal naive", "The price in the same half-hour slot on the previous day."),
    "aemo": ("AEMO pre-dispatch", "The latest AEMO pre-dispatch price run published before the forecast is issued (half-hourly), "
                                  "filled with the seasonal naive beyond the end of the run."),
    "lstm": ("LSTM", "A 2-layer LSTM over the previous 7 days of price, system demand and calendar features, predicting the next 48 "
                     "half-hour prices in asinh(price/100) with Huber loss. Trained on 2015–2023, validated on 2024."),
    PROFILE: (PROFILE_LABEL, "Net-local power forecast as the mean of the same 5-minute slot over the previous 7 days. Every real "
                             "price forecaster uses it for the household, so their value lost is measured against perfect price "
                             "with this profile and is the cost of the price forecast alone."),
}


def _forecast_guide(price_n, load_n, real_names, have_profile: bool):
    """Swatch, name and one-paragraph description of each kind of forecast on the page."""
    levels = lambda names: ", ".join(f"{_sigma(n):g}" for n in names)
    rows = []
    if price_n:
        rows.append((GREYS[1], "Price", GUIDE["noise_price"][0], GUIDE["noise_price"][1].format(levels=levels(price_n))))
    rows += [(T.FORECASTER_COLOUR[n], "Price", *GUIDE[n]) for n in real_names]
    if load_n:
        rows.append((GREYS[1], "Household", GUIDE["noise_net"][0], GUIDE["noise_net"][1].format(levels=levels(load_n))))
    if have_profile:
        rows.append((PROFILE_COLOUR, "Household", *GUIDE[PROFILE]))
    td = f'padding:6px 14px 6px 0;vertical-align:top;border-bottom:1px solid {T.GRID}'
    body = "".join(
        f'<tr><td style="{td};white-space:nowrap;color:{T.MUTED}">{inp}</td>'
        f'<td style="{td};white-space:nowrap;color:{T.INK};font-weight:600">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{c};margin-right:7px"></span>{name}</td>'
        f'<td style="{td};color:{T.INK2}">{text}</td></tr>' for c, inp, name, text in rows)
    return Div(text=f'<table style="font:12px/1.45 {T.FONT};border-collapse:collapse;max-width:1100px;margin-top:4px">{body}</table>',
               sizing_mode="stretch_width")


# ---------------------------------------------------------------------------
# Value lost by run
# ---------------------------------------------------------------------------

def _reference(name: str) -> str:
    return "perfect price" if name in REAL_PRICE else "perfect foresight"


def _lost_bars(noise: pd.DataFrame, real: pd.DataFrame | None, lost: dict, names: list[str], noise_names: list[str],
               kind: str, title: str, x_end: float):
    """kind 'price' or 'net'. One bar per run: synthetic in greys, real forecasters in colour."""
    col, unit, fmt = ("price_mae", "$/MWh", ".0f") if kind == "price" else ("net_local_mae_kw", "kW", ".2f")
    mae = pd.concat([noise.set_index("forecaster")[col]] + ([real.set_index("forecaster")[col]] if real is not None else []))
    mae = mae[~mae.index.duplicated()]
    labels = [_label(n) for n in names]
    src = ColumnDataSource(dict(
        label=labels, value=[lost[n] for n in names],
        colour=[_grey(noise_names, n) if n in noise_names else (PROFILE_COLOUR if n == PROFILE else T.FORECASTER_COLOUR[n]) for n in names],
        text=[f"${lost[n]:.2f}   (MAE {mae[n]:{fmt}} {unit}, against {_reference(n)})" for n in names],
    ))
    p = T.make_figure(height=60 + 34 * len(names), title=title, y_range=FactorRange(factors=labels[::-1]), tools="save")
    r = p.hbar(y="label", right="value", height=0.55, source=src, color="colour")
    p.add_layout(LabelSet(x="value", y="label", text="text", source=src, x_offset=8, text_baseline="middle",
                          text_font=T.FONT, text_font_size="11px", text_color=T.INK2))
    p.add_tools(HoverTool(renderers=[r], tooltips=[("", "@label"), ("value lost", "$@value{0.00}")]))
    p.x_range.start, p.x_range.end = 0, x_end
    p.xaxis.axis_label = "Value lost ($ over JAN25)"
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = T.GRID
    p.yaxis.major_label_text_color = T.INK
    T.style(p, legend=False)
    return p


# ---------------------------------------------------------------------------
# 0. What the forecasts look like
# ---------------------------------------------------------------------------

def _example_panel(ex: dict, kind: str, names: list[str], noise_names: list[str], title: str, x_range=None):
    """kind 'price' (asinh axis) or 'net' (kW). Actual in ink, synthetic in greys, real in colour."""
    actual = ex[f"actual_{kind}"]
    to_y = T.price_to_axis if kind == "price" else (lambda v: np.asarray(v, dtype=float))
    data = {"time": ex["time"], "actual": actual, "actual_y": to_y(actual)}
    p = T.make_figure(height=340, title=title, x_axis_type="datetime", **({"x_range": x_range} if x_range is not None else {}))
    for n in names:
        if n not in ex[kind]:
            continue
        data[n], data[f"{n}_y"] = ex[kind][n], to_y(ex[kind][n])
    src = ColumnDataSource(data)
    for n in names:
        if n not in ex[kind]:
            continue
        if n in noise_names:
            style = {"color": _grey(noise_names, n), "line_width": 1.5}
        elif n == PROFILE:
            style = {"color": PROFILE_COLOUR, "line_width": 2.5}
        else:
            style = {"line_width": 2, **T.line_style(n)}
        p.line("time", f"{n}_y", source=src, legend_label=_label(n), **style)
    rend = p.line("time", "actual_y", source=src, color=T.INK, line_width=2.5, legend_label="Actual")
    unit = "$/MWh" if kind == "price" else "kW"
    fmt = "{0}" if kind == "price" else "{0.00}"
    T.line_hover(p, [rend], [("time", "@time{%d %b %H:%M}"), ("actual", f"@actual{fmt} {unit}")]
                 + [(_label(n), f"@{{{n}}}{fmt} {unit}") for n in names if n in ex[kind]], formatters={"@time": "datetime"})
    p.add_layout(Span(location=ex["issue"].timestamp() * 1000, dimension="height", line_color=T.MUTED, line_dash="dashed", line_width=1))
    if kind == "price":
        T.asinh_price_axis(p)
        lo = min(-50.0, min(float(np.min(v)) for v in [actual] + [ex[kind][n] for n in names if n in ex[kind]]) - 10)
        p.y_range.start, p.y_range.end = float(T.price_to_axis(lo)), float(T.price_to_axis(30000))
    else:
        p.yaxis.axis_label = "Net-local power G − A (kW, + = surplus)"
        T.zero_line(p)
    T.style(p)
    _legend_outside(p)
    return p


# ---------------------------------------------------------------------------
# 1. Error by lead time
# ---------------------------------------------------------------------------

def _lead_panel(lead: pd.DataFrame, names: list[str], col: str, title: str, y_label: str, noise_names: list[str],
                log_y: bool, first_step: int, bin_steps: int = 1):
    kw = {"y_axis_type": "log"} if log_y else {}
    p = T.make_figure(height=320, title=title, **kw)
    rends = []
    for n in names:
        d = lead[(lead["forecaster"] == n) & (lead["step"] >= first_step)]
        if d.empty:
            continue
        if bin_steps > 1:                                  # mean over each half hour, plotted at the bin centre
            d = d.assign(step=(d["step"] // bin_steps) * bin_steps + bin_steps / 2).groupby("step", as_index=False)[col].mean()
        colour = _grey(noise_names, n) if n in noise_names or n == "perfect" else (PROFILE_COLOUR if n == PROFILE else T.FORECASTER_COLOUR[n])
        dash = "solid" if n in noise_names or n == PROFILE else T.FORECASTER_DASH.get(n, "solid")
        src = ColumnDataSource(dict(h=d["step"].to_numpy() * INTERVAL_HOURS, y=d[col].to_numpy(), name=[_label(n)] * len(d)))
        rends.append(p.line("h", "y", source=src, color=colour, line_dash=dash, line_width=2.5 if n == PROFILE else 2,
                            legend_label=_label(n)))
    p.add_tools(HoverTool(renderers=rends, tooltips=[("", "@name"), ("lead", "@h{0.00} h"), ("MAE", "@y{0.00}")], line_policy="nearest"))
    p.xaxis.axis_label = "Lead time (hours ahead of the interval being dispatched)"
    p.yaxis.axis_label = y_label
    p.x_range.start, p.x_range.end = 0, 24
    p.xaxis.ticker = [0, 1, 4, 8, 12, 16, 20, 24]
    if not log_y:
        p.y_range.start = 0
    T.style(p)
    _legend_outside(p)
    return p


# ---------------------------------------------------------------------------
# 2. Value lost against error at matched lead times
# ---------------------------------------------------------------------------

LABEL_SIDE = {"lstm": ("right", -12, 0), "naive": ("left", 12, 0), "aemo": ("right", 6, 16), PROFILE: ("right", -12, 0)}   # align, dx, dy


def _measure(lead: pd.DataFrame, name: str, col: str, how) -> float:
    d = lead[lead["forecaster"] == name]
    if d.empty:
        return float("nan")
    return float(d[col].mean()) if how == "mean" else float(d.loc[d["step"] == how, col].iloc[0])


def _value_panel(lead, lost, noise_names, real_names, col, how, title, x_label, y_end, show_legend):
    p = T.make_figure(height=320, title=title, tools="save")
    curve = ["perfect"] + noise_names
    xs = [_measure(lead, n, col, how) for n in curve]
    ys = [lost.get(n, np.nan) for n in curve]
    src = ColumnDataSource(dict(x=xs, y=ys, name=[_label(n) for n in curve], colour=[_grey(noise_names, n) for n in curve]))
    legend = {"legend_label": "Synthetic noise on perfect foresight"} if show_legend else {}
    p.line("x", "y", source=src, color=T.INK2, line_width=2, **legend)
    r = p.scatter("x", "y", source=src, size=9, color="colour", line_color=T.SURFACE, line_width=2)
    hover = [r]
    for n in real_names:
        x, y = _measure(lead, n, col, how), lost.get(n, np.nan)
        if np.isnan(x) or np.isnan(y):
            continue
        colour = PROFILE_COLOUR if n == PROFILE else T.FORECASTER_COLOUR[n]
        s = ColumnDataSource(dict(x=[x], y=[y], name=[_label(n)]))
        hover.append(p.scatter("x", "y", source=s, size=14, marker="diamond", color=colour,
                               line_color=T.INK2 if n == PROFILE else T.SURFACE, line_width=1 if n == PROFILE else 2))
        align, dx, dy = LABEL_SIDE.get(n, ("left", 12, 0))
        T.value_label(p, x, y, _label(n), x_offset=dx, y_offset=dy, align=align, baseline="middle")
        xs.append(x)
    p.add_tools(HoverTool(renderers=hover, tooltips=[("", "@name"), ("error", "@x{0.00}"), ("value lost", "$@y{0.00}")]))
    x_end = np.nanmax(xs) * 1.4 or 1.0                      # room for the direct labels
    p.xaxis.axis_label = x_label
    p.yaxis.axis_label = "Value lost ($ over JAN25)"
    p.x_range.start, p.x_range.end = -0.03 * x_end, x_end
    p.y_range.start, p.y_range.end = -0.04 * y_end, y_end
    p.xgrid.grid_line_color = T.GRID
    T.zero_line(p)
    T.style(p)
    if show_legend:
        p.legend.location = "bottom_right"
    return p


# ---------------------------------------------------------------------------
# Per-interval value (price-band table)
# ---------------------------------------------------------------------------

def _interval_net(df: pd.DataFrame, deg_scale: float = 1.0) -> np.ndarray:
    """
    Per-interval grid profit minus aging cost ($). Rainflow life loss has no per-interval
    value, so the optimiser's piecewise-linear cost is used as the time profile, scaled by
    deg_scale = rainflow total / model total so that the month sums to the rainflow figure.
    """
    n = BatteryParams().network_tariff
    rrp = df["rrp"].to_numpy() / 1000
    return (df["grid_export_kwh"] * rrp - df["grid_import_kwh"] * (rrp + n) - deg_scale * df["degradation_cost"]).to_numpy()


def _deg_scales(*summaries) -> dict[str, float]:
    out = {}
    for s in summaries:
        if s is None:
            continue
        for _, r in s.iterrows():
            if r["degradation_cost_model"] > 0:
                out[r["forecaster"]] = float(r["degradation_cost_rainflow"] / r["degradation_cost_model"])
    return out


# ---------------------------------------------------------------------------
# 3. Mechanism
# ---------------------------------------------------------------------------

def _mechanism_panel(noise: pd.DataFrame):
    names = ["perfect"] + _noise_names(noise, "noise_price_")
    s = noise.set_index("forecaster").loc[names]
    src = ColumnDataSource(dict(x=s["price_mae"].to_numpy(), gross=s["net_profit_ex_degradation"].to_numpy(),
                                deg=s["degradation_cost_rainflow"].to_numpy(), net=s[PROFIT].to_numpy(),
                                efc=s["equivalent_full_cycles"].to_numpy()))
    p = T.make_figure(height=320, title="Trading profit and degradation cost against price error", tools="save")
    series = [("gross", T.BLUE, "Trading profit before degradation"), ("deg", T.ORANGE, "Rainflow degradation cost"),
              ("net", T.INK, "Net profit")]
    rs = []
    for key, colour, label in series:
        p.line("x", key, source=src, color=colour, line_width=2, legend_label=label)
        rs.append(p.scatter("x", key, source=src, size=8, color=colour, line_color=T.SURFACE, line_width=2))
    p.add_tools(HoverTool(renderers=[rs[0]], mode="vline", tooltips=[
        ("price MAE", "@x{0.0} $/MWh"), ("trading profit", "$@gross{0.00}"), ("degradation", "$@deg{0.00}"),
        ("net profit", "$@net{0.00}"), ("EFC", "@efc{0.0}")]))
    p.xaxis.axis_label = "Price forecast MAE, 24 h average ($/MWh)"
    p.yaxis.axis_label = "$ over JAN25"
    p.xgrid.grid_line_color = T.GRID
    T.zero_line(p)
    T.style(p)
    p.legend.location = "bottom_left"
    return p


DEPTH_EDGES = np.linspace(0, 1, 11)


def _cycle_depth_panels(results, names, r_cell: float):
    params = BatteryParams()
    centres = (DEPTH_EDGES[:-1] + DEPTH_EDGES[1:]) / 2
    data = {"d": centres, "lab": [f"{a:.1f} – {b:.1f}" for a, b in zip(DEPTH_EDGES[:-1], DEPTH_EDGES[1:])]}
    for n in names:
        if n not in results:
            continue
        _, cycles = rainflow_life_loss(results[n]["battery_state"].to_numpy(), params.e_max)
        depth = np.array([c[0] for c in cycles]); count = np.array([c[2] for c in cycles])
        idx = np.clip(np.digitize(depth, DEPTH_EDGES) - 1, 0, len(centres) - 1)
        data[f"n_{n}"] = np.bincount(idx, weights=count, minlength=len(centres))
        data[f"c_{n}"] = np.bincount(idx, weights=r_cell * count * stress_function(depth), minlength=len(centres))
    src = ColumnDataSource(data)
    panels = []
    noise_names = [n for n in names if n != "perfect"]
    for key, title, y_label, fmt in (("n", "Rainflow cycles by depth", "Rainflow cycle count (half cycles = 0.5)", "{0.0}"),
                                     ("c", "Degradation cost by cycle depth", "$ over JAN25", "${0.00}")):
        p = T.make_figure(height=300, title=title, tools="save")
        rends = []
        for n in names:
            if f"{key}_{n}" not in data:
                continue
            c = _grey(noise_names, n)
            p.line("d", f"{key}_{n}", source=src, color=c, line_width=2, legend_label=_label(n))
            rends.append(p.scatter("d", f"{key}_{n}", source=src, size=7, color=c, line_color=T.SURFACE, line_width=1.5))
        p.add_tools(HoverTool(renderers=rends[:1], mode="vline",
                              tooltips=[("depth", "@lab")] + [(_label(n), f"@{key}_{n}{fmt}") for n in names if f"{key}_{n}" in data]))
        p.xaxis.axis_label = "Cycle depth (fraction of usable capacity)"
        p.yaxis.axis_label = y_label
        p.y_range.start = 0
        T.style(p)
        _legend_outside(p)
        panels.append(p)
    return panels


# ---------------------------------------------------------------------------
# 4. Price bands
# ---------------------------------------------------------------------------

def _short(name: str) -> str:
    if name.startswith("noise_price_"):
        return f"Price σ {_sigma(name):g}"
    if name.startswith("noise_net_"):
        return f"Load σ {_sigma(name):g} kW"
    return {"naive": "Naive*", "aemo": "AEMO*", "lstm": "LSTM*", PROFILE: "Profile*"}.get(name, name)


def _band_table(results, pairs, scales):
    rrp = results["perfect"]["rrp"].to_numpy()
    rows = []
    for lo, hi, lab in BANDS:
        m = (rrp >= lo) & (rrp < hi) if np.isfinite(lo) else rrp < hi
        rec = {"band": lab, "intervals": int(m.sum())}
        for run, ref in pairs:
            if run in results and ref in results:
                rec[run] = float((_interval_net(results[ref], scales.get(ref, 1.0)) - _interval_net(results[run], scales.get(run, 1.0)))[m].sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    tot = {"band": "Total", "intervals": int(df["intervals"].sum()), **{c: float(df[c].sum()) for c in df.columns[2:]}}
    df = pd.concat([df, pd.DataFrame([tot])], ignore_index=True)
    cols = [("band", "RRP band ($/MWh)", None), ("intervals", "Intervals", "0")] + \
           [(run, _short(run), "0.00") for run, _ in pairs if run in df]
    return T.summary_table(df, cols, height=48 + 28 * len(df), text_width=120, number_width=86)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    ("forecaster", "Run", None), ("price_mae", "Price MAE $/MWh", "0.0"), ("price_mae_1h", "Price MAE 1 h", "0.0"),
    ("net_local_mae_kw", "Net-local MAE kW", "0.00"), ("net_profit_ex_degradation", "Trading profit $", "0.00"),
    ("degradation_cost_rainflow", "Degradation $", "0.00"), (PROFIT, "Net profit $", "0.00"),
    ("value_lost", "Value lost $", "0.00"), ("reference", "Against", None), ("equivalent_full_cycles", "EFC", "0.0"),
]


def _run_table(noise: pd.DataFrame, real: pd.DataFrame | None, lost: dict) -> pd.DataFrame:
    """Every run, synthetic then real, with value lost against the reference that isolates its input."""
    rows = noise if real is None else pd.concat([noise, real[real["forecaster"].isin(lost) & (real["forecaster"] != "perfect")]],
                                                ignore_index=True)
    rows = rows.assign(value_lost=rows["forecaster"].map(lost), reference=rows["forecaster"].map(_reference))
    rows.loc[rows["forecaster"] == "perfect", "reference"] = "–"
    rows["forecaster"] = [_label(n) + ("*" if n in REAL_PRICE or n == PROFILE else "") for n in rows["forecaster"]]
    return rows


def plot_noise_study(noise: pd.DataFrame, real: pd.DataFrame | None, results: dict, lead: pd.DataFrame,
                     title: str, output_path: str, r_cell: float = 12_000.0, example: dict | None = None):
    """
    noise   : results/forecast_noise_summary.csv
    real    : results/forecast_summary.csv, or None
    results : run name -> per-interval results frame (results/mpc_household_J4_<name>.csv)
    lead    : forecast MAE per 5-min step for each run (forecaster, step, price_mae, net_mae)
    example : forecasts at one issue time (forecast_trading.example_forecasts), or None
    """
    price_n = _noise_names(noise, "noise_price_")
    load_n = _noise_names(noise, "noise_net_")
    lost = _lost(noise, real)
    have_real = [n for n in REAL_PRICE if n in lost]
    scales = _deg_scales(noise, real)
    table = _run_table(noise, real, lost)

    y_price = max(lost[n] for n in price_n + have_real) * 1.12
    y_load = max([lost[n] for n in load_n] + [lost.get(PROFILE, 0.0)]) * 1.12
    price_measures = [(1, "Next interval (5 min ahead)"), (12, "1 h ahead"), ("mean", "24 h average")]
    load_measures = [(0, "This interval (committed)"), (12, "1 h ahead"), ("mean", "24 h average")]

    children = [
        T.heading(title, "Household MILP (J = 4, R_cell = 12,000) re-planned every 5 minutes on a 24 h horizon over January 2025. "
                         "How much trading value is lost as the price or household forecast gets worse, for synthetic error of a "
                         "controlled size and for the real forecasters."),
        T.section("Forecasts compared",
                  "Every run is the same controller given a different forecast of one input. The price of the interval being dispatched "
                  "is always the actual price (AEMO publishes it before the interval starts); everything further ahead is forecast. "
                  "The household's net-local power is forecast for every interval, including the current one."),
        _forecast_guide(price_n, load_n, have_real, PROFILE in lost),
        T.section("Value lost by run",
                  "Net profit given up against the reference that isolates each forecast: perfect foresight for the synthetic runs and "
                  "for the 7-day load profile, perfect price with the 7-day load profile for the real price forecasters (so their bars "
                  "are the cost of the price forecast alone). Grey bars are synthetic noise, coloured bars are the real forecasters*."),
        row(_lost_bars(noise, real, lost, price_n + have_real, price_n, "price", "Price forecast", y_price * 2.1),
            _lost_bars(noise, real, lost, load_n + ([PROFILE] if PROFILE in lost else []), load_n, "net", "Household (net-local) forecast",
                       y_load * 2.1),
            sizing_mode="stretch_width"),
    ]
    if example is not None:
        issue = example["issue"]
        p_syn = _example_panel(example, "price", price_n, price_n, f"Synthetic price forecasts issued {issue:%-d %b %H:%M}")
        p_real = _example_panel(example, "price", have_real, price_n, f"Real price forecasts issued {issue:%-d %b %H:%M}", x_range=p_syn.x_range)
        p_net = _example_panel(example, "net", load_n + [PROFILE], load_n, f"Household forecasts issued {issue:%-d %b %H:%M}",
                               x_range=p_syn.x_range)
        children += [
            T.section("0. What the forecasts look like",
                      f"The 24 h forecast each run planned on at {issue:%-d %B %H:%M}, ten hours before the 17,500 $/MWh spike, against what "
                      "happened (ink). The price axis is asinh-scaled: linear near zero, logarithmic in the spikes. The synthetic forecasts follow "
                      "the actual path, including the spike's timing, and drift further from it the further ahead they look; the real forecasts "
                      "carry no knowledge of the actual path. The first point is the actual price in every case. "
                      "Each re-plan five minutes later draws a fresh error path."),
            row(p_syn, p_real, sizing_mode="stretch_width"),
            p_net,
        ]
    children += [
        T.section("1. How the error grows with lead time",
                  "The synthetic error is near zero for the interval being dispatched and grows to σ_max a day ahead. The real forecasters "
                  "are about as wrong in the next hour as tomorrow, and the 7-day load profile is already wrong by ~1.15 kW in the interval the "
                  "controller commits. A single 24 h-average MAE therefore does not place the two kinds of forecast on a common scale."),
        row(_lead_panel(lead, ["perfect"] + price_n + have_real, "price_mae", "Price forecast MAE by lead time", "$/MWh (log scale)",
                        price_n, log_y=True, first_step=1, bin_steps=6),
            _lead_panel(lead, load_n + ([PROFILE] if PROFILE in set(lead["forecaster"]) else []), "net_mae",
                        "Net-local forecast MAE by lead time", "kW", load_n, log_y=False, first_step=0),
            sizing_mode="stretch_width"),
        T.section("2. Value lost against forecast error, measured at three lead times",
                  "Does a real forecaster lose more or less than random error of the same size? The grey curve is the synthetic runs: "
                  "value lost (y) against forecast MAE (x). Each diamond is a real forecaster. A diamond above the curve loses more than "
                  "its error size explains; below, less. The y values are identical in all three panels of a row. Only x changes: the MAE "
                  "of the forecast for the next interval, for 1 h ahead, or averaged over the 24 h horizon. That matters because synthetic "
                  "error is small at short leads and large at long ones while real error is nearly flat (section 1), so the verdict depends "
                  "on the lead time used to match them. Matched on the 24 h average, the LSTM, naive and load profile sit above the curve "
                  "and AEMO below it. Matched at 1 h or at the next interval, every real price forecaster sits far below the curve: it is "
                  "as wrong at short leads as the heaviest synthetic noise but loses a fraction as much. The load profile's error in the "
                  "committed interval (1.18 kW) is beyond anything the synthetic runs reach (0.14 kW), so it cannot be placed on that curve."),
        row(*[_value_panel(lead, lost, price_n, have_real, "price_mae", how, f"Price: {t}", "Price MAE ($/MWh)", y_price, i == 0)
              for i, (how, t) in enumerate(price_measures)], sizing_mode="stretch_width"),
        row(*[_value_panel(lead, lost, load_n, [PROFILE] if PROFILE in lost else [], "net_mae", how, f"Household: {t}",
                           "Net-local MAE (kW)", y_load, i == 0)
              for i, (how, t) in enumerate(load_measures)], sizing_mode="stretch_width"),
        T.section("3. Mechanism: price error buys cycling",
                  "Up to σ_max = 1 the MILP trades more, not worse: trading profit rises slightly while cycling and degradation rise faster. "
                  "The cycle-depth panels show where the extra cycles and their cost fall."),
        _mechanism_panel(noise),
        row(*_cycle_depth_panels(results, ["perfect"] + price_n, r_cell), sizing_mode="stretch_width"),
        T.section("4. Value lost by price band ($)",
                  "Which prices the value is lost at: per-interval value lost against each run's reference, summed by the spot price of "
                  "the interval. It separates missed spike revenue (> 1000) from ordinary arbitrage (the middle bands) and missed "
                  "negative-price charging (< 0). Aging cost per interval is the optimiser's piecewise-linear cost rescaled so each "
                  "run's month total equals its rainflow cost. A negative entry means the run did better than its reference in that band."),
        _band_table(results, [(n, "perfect") for n in price_n + load_n] + [(n, PROFILE) for n in have_real]
                    + ([(PROFILE, "perfect")] if PROFILE in lost else []), scales),
        T.note("* old-join runs until forecast_trading.py is re-run; the price forecasters are measured against perfect price. "
               "The Profile column compares the old-join perfect-price run with the corrected perfect-foresight run, so it totals "
               "less than the old-join figure in the bar chart."),
        T.section("Table view"),
        T.summary_table(table, TABLE_COLUMNS, height=48 + 28 * len(table), text_width=190, number_width=112),
        T.note("* old-join runs. Net profit = grid revenue − grid cost (incl. network tariff on imports) − rainflow life loss × R_cell. "
               "MAE is over every 24 h forecast issued at half-hour boundaries in the test month; the price at step 0 is the actual. "
               "One noise seed per level. " + STALE_NOTE),
    ]
    T.save_page(children, title=title, output_path=output_path)
