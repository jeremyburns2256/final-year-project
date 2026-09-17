"""
milp_sweep_plot.py

One page for the MILP cycle-aging study (milp/MODEL_NOTES.md, thesis Sec. 6.2),
laid out as two columns, BESS-only and household, so each scenario reads
top-down and the two can be compared side by side:

  1. Headline: net profit after the ex-post rainflow degradation cost for every
     configuration, including the two baselines (state machine; MILP with the
     aging cost switched off).
  2. Gross grid profit split into what the battery kept and what cycling cost.
  3. The optimiser's piecewise-linear aging cost against the rainflow cost it
     was approximating, with the Xu et al. (2018) Eq. (26) relative error.
  4. Cycling: life consumed and equivalent full cycles against J.
  5. The model itself: the stress function Phi(delta) and the marginal segment
     costs c_j for each J.
  6. Table view.

J is an ordered choice, so it takes one blue ramp (light J=1 -> dark J=16)
wherever the segment counts appear together.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, FactorRange, LabelSet, Legend, LegendItem

from milp.degradation import stress_function
from milp.model import BatteryParams
from plotting import theme as T

SCENARIOS = ("bess_only", "household")
J_VALUES = (1, 2, 4, 8, 16)


def _bar_labels(p, src, x, y, text, **kw):
    base = dict(text_font=T.FONT, text_font_size="11px", text_color=T.INK2)
    base.update(kw)
    p.add_layout(LabelSet(x=x, y=y, text=text, source=src, **base))


def _money(v: float) -> str:
    return f"−${abs(v):,.2f}" if v < 0 else f"${v:,.2f}"


def _rows(summary: pd.DataFrame, scenario: str) -> pd.DataFrame:
    d = summary[(summary["scenario"] == scenario) & (summary["r_cell"] > 0)].sort_values("n_segments")
    return d


# ---------------------------------------------------------------------------
# 1. Headline
# ---------------------------------------------------------------------------

def _headline(summary: pd.DataFrame, scenario: str):
    labels, values, colours = [], [], []
    sm = summary[summary["scenario"] == "state_machine"]
    if scenario == "household" and len(sm):
        labels.append("State machine, Table 5.1 thresholds (20 kWh)")
        values.append(float(sm["net_profit_incl_degradation"].iloc[0]))
        colours.append(T.MUTED)
    r0 = summary[(summary["scenario"] == scenario) & (summary["r_cell"] == 0)]
    if len(r0):
        labels.append("MILP, aging cost ignored (R_cell = 0)")
        values.append(float(r0["net_profit_incl_degradation"].iloc[0]))
        colours.append(T.MUTED)
    for _, r in _rows(summary, scenario).iterrows():
        j = int(r["n_segments"])
        labels.append(f"MILP, J = {j}")
        values.append(float(r["net_profit_incl_degradation"]))
        colours.append(T.colour_for_j(j))
    src = ColumnDataSource(dict(label=labels, value=values, colour=colours, text=[_money(v) for v in values],
                                x_text=[max(v, 0) for v in values]))
    p = T.make_figure(height=70 + 34 * len(labels), title=f"{T.SCENARIO_LABEL[scenario]}: net profit after rainflow degradation cost",
                      y_range=FactorRange(factors=labels[::-1]), tools="save")
    p.hbar(y="label", right="value", height=0.55, source=src, color="colour")
    T.zero_line(p, "height")
    _bar_labels(p, src, "x_text", "label", "text", x_offset=6, text_baseline="middle")
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    span = hi - lo
    p.x_range.start, p.x_range.end = lo - 0.04 * span, hi + 0.22 * span
    p.xaxis.axis_label = "$ over JAN25 (grid revenue − grid cost − rainflow life loss × R_cell)"
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = T.GRID
    p.yaxis.major_label_text_color = T.INK
    T.style(p, legend=False)
    return p


# ---------------------------------------------------------------------------
# 2. Where the gross profit goes
# ---------------------------------------------------------------------------

def _profit_split(summary: pd.DataFrame, scenario: str):
    d = _rows(summary, scenario)
    js = [f"J = {int(j)}" for j in d["n_segments"]]
    net = d["net_profit_incl_degradation"].to_numpy(dtype=float)
    deg = d["degradation_cost_rainflow"].to_numpy(dtype=float)
    gross = d["net_profit_ex_degradation"].to_numpy(dtype=float)
    src = ColumnDataSource(dict(j=js, net=net, deg=deg, gross=gross, top=net + deg,
                                t_gross=[f"${g:,.1f}" for g in gross], t_net=[f"${n:,.1f}" for n in net], t_deg=[f"−${x:,.1f}" for x in deg]))
    p = T.make_figure(height=300, title="Gross grid profit = kept (dark) + rainflow degradation cost (light)", x_range=FactorRange(factors=js), tools="save,hover")
    r_net = p.vbar(x="j", bottom=0, top="net", width=0.55, source=src, color=T.BLUE_RAMP[4], line_color=T.SURFACE, line_width=2)
    r_deg = p.vbar(x="j", bottom="net", top="top", width=0.55, source=src, color=T.BLUE_RAMP[1], line_color=T.SURFACE, line_width=2)
    p.hover.tooltips = [("J", "@j"), ("gross grid profit", "$@gross{0.00}"), ("rainflow degradation", "$@deg{0.00}"), ("net", "$@net{0.00}")]
    p.hover.renderers = [r_net, r_deg]
    _bar_labels(p, src, "j", "top", "t_gross", y_offset=4, text_align="center", text_baseline="bottom")
    _bar_labels(p, src, "j", "net", "t_net", y_offset=-4, text_align="center", text_baseline="top", text_color=T.SURFACE)
    p.add_layout(Legend(items=[LegendItem(label="net profit after degradation", renderers=[r_net]),
                               LegendItem(label="rainflow degradation cost", renderers=[r_deg])], location="top_left"))
    p.y_range.start, p.y_range.end = 0, float((net + deg).max()) * 1.5
    p.yaxis.axis_label = "$ over JAN25"
    p.xaxis.axis_label = "cycle-depth segments"
    p.ygrid.grid_line_color = T.GRID
    T.style(p)
    return p


# ---------------------------------------------------------------------------
# 3. Model cost vs rainflow
# ---------------------------------------------------------------------------

def _model_vs_rainflow(summary: pd.DataFrame, scenario: str):
    from bokeh.transform import dodge

    d = _rows(summary, scenario)
    js = [f"J = {int(j)}" for j in d["n_segments"]]
    model = d["degradation_cost_model"].to_numpy(dtype=float)
    rf = d["degradation_cost_rainflow"].to_numpy(dtype=float)
    err = d["rainflow_relative_error"].to_numpy(dtype=float)
    src = ColumnDataSource(dict(j=js, model=model, rf=rf, err=err, top=np.maximum(model, rf),
                                t_err=[f"{100 * e:.0f}% error" for e in err]))
    p = T.make_figure(height=300, title="Optimiser's piecewise-linear aging cost against the ex-post rainflow cost", x_range=FactorRange(factors=js), tools="save,hover")
    r1 = p.vbar(x=dodge("j", -0.16, range=p.x_range), top="model", width=0.28, source=src, color=T.BLUE, line_color=T.SURFACE, line_width=2)
    r2 = p.vbar(x=dodge("j", 0.16, range=p.x_range), top="rf", width=0.28, source=src, color=T.ORANGE, line_color=T.SURFACE, line_width=2)
    p.hover.tooltips = [("J", "@j"), ("model (PWL)", "$@model{0.00}"), ("rainflow", "$@rf{0.00}"), ("relative error, Xu Eq. 26", "@err{0.000}")]
    _bar_labels(p, src, "j", "top", "t_err", y_offset=4, text_align="center", text_baseline="bottom")
    p.add_layout(Legend(items=[LegendItem(label="model cost charged by the optimiser", renderers=[r1]),
                               LegendItem(label="rainflow cost of the SoC trajectory", renderers=[r2])], location="top_right"))
    p.y_range.start, p.y_range.end = 0, float(np.maximum(model, rf).max()) * 1.3
    p.yaxis.axis_label = "$ over JAN25"
    p.xaxis.axis_label = "cycle-depth segments"
    p.ygrid.grid_line_color = T.GRID
    T.style(p)
    return p


# ---------------------------------------------------------------------------
# 4. Cycling
# ---------------------------------------------------------------------------

def _cycling(summary: pd.DataFrame, scenario: str, field: str, title: str, unit: str, fmt: str):
    d = _rows(summary, scenario)
    js = [f"J = {int(j)}" for j in d["n_segments"]]
    vals = d[field].to_numpy(dtype=float)
    src = ColumnDataSource(dict(j=js, v=vals, colour=[T.colour_for_j(int(j)) for j in d["n_segments"]], t=[fmt.format(v) for v in vals],
                                efc=d["equivalent_full_cycles"].to_numpy(), depth=d["mean_cycle_depth"].to_numpy(), cyc=d["rainflow_cycles"].to_numpy()))
    r0 = summary[(summary["scenario"] == scenario) & (summary["r_cell"] == 0)]
    sub = f"   (aging cost ignored: {fmt.format(float(r0[field].iloc[0]))})" if len(r0) else ""
    p = T.make_figure(height=240, title=title + sub, x_range=FactorRange(factors=js), tools="save,hover")
    p.vbar(x="j", top="v", width=0.55, source=src, color="colour", line_color=T.SURFACE, line_width=2)
    p.hover.tooltips = [("J", "@j"), (title, "@t"), ("EFC", "@efc{0.0}"), ("rainflow cycles", "@cyc{0.0}"), ("mean depth", "@depth{0.00}")]
    _bar_labels(p, src, "j", "v", "t", y_offset=4, text_align="center", text_baseline="bottom")
    p.y_range.start, p.y_range.end = 0, float(vals.max()) * 1.3
    p.yaxis.axis_label = unit
    p.ygrid.grid_line_color = T.GRID
    T.style(p, legend=False)
    return p


# ---------------------------------------------------------------------------
# 5. The model
# ---------------------------------------------------------------------------

def _stress_function(params: BatteryParams):
    delta = np.linspace(0, 1, 201)
    phi = stress_function(delta, params.phi_a, params.phi_k)
    src = ColumnDataSource(dict(delta=delta, phi=phi, phi_pct=100 * phi, cycles=np.where(phi > 0, 1 / np.maximum(phi, 1e-12), np.nan)))
    p = T.make_figure(height=300, title=f"Cycle-depth stress function Φ(δ) = {params.phi_a:.2e}·δ^{params.phi_k:g}  (Eq. 3.8, NMC)")
    r = p.line("delta", "phi_pct", source=src, color=T.INK, line_width=2)
    T.line_hover(p, [r], [("depth δ", "@delta{0.00}"), ("life per cycle Φ", "@phi_pct{0.0000}%"), ("cycles to end of life", "@cycles{0,0}")])
    p.xaxis.axis_label = "cycle depth δ (fraction of usable capacity)"
    p.yaxis.axis_label = "% of cell life consumed per cycle"
    p.x_range.start, p.x_range.end = 0, 1
    p.y_range.start = 0
    p.ygrid.grid_line_color = T.GRID
    T.style(p, legend=False)
    return p


def _segment_costs(params: BatteryParams, j_values=J_VALUES):
    p = T.make_figure(height=300, title=f"Marginal aging cost c_j by segment (Eq. 2.3 with the J factor), R_cell = {params.r_cell:,.0f}")
    items = []
    for j in j_values:
        pr = BatteryParams(**{**params.__dict__, "n_segments": j})
        c = pr.segment_costs
        edges = np.linspace(0, 1, j + 1)
        xs = np.repeat(edges, 2)[1:-1]
        ys = np.repeat(c, 2)
        r = p.line(xs, ys, color=T.colour_for_j(j, j_values), line_width=2)
        items.append(LegendItem(label=f"J = {j}", renderers=[r]))
    p.add_layout(Legend(items=items, location="top_left"))
    p.xaxis.axis_label = "depth of discharge covered by the segment"
    p.yaxis.axis_label = "$ per kWh discharged"
    p.x_range.start, p.x_range.end = 0, 1
    p.y_range.start = 0
    p.ygrid.grid_line_color = T.GRID
    T.style(p)
    return p


# ---------------------------------------------------------------------------
# 6. Table
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    ("config", "Configuration", None),
    ("net_profit_incl_degradation", "Net profit ($)", "0.00"),
    ("net_profit_ex_degradation", "Gross profit ($)", "0.00"),
    ("degradation_cost_rainflow", "Rainflow ($)", "0.00"),
    ("degradation_cost_model", "Model ($)", "0.00"),
    ("rainflow_relative_error", "Rel. error", "0.000"),
    ("life_loss_pct", "Life loss (%)", "0.000"),
    ("equivalent_full_cycles", "EFC", "0.0"),
    ("rainflow_cycles", "Cycles", "0.0"),
    ("mean_cycle_depth", "Mean depth", "0.00"),
    ("grid_revenue", "Revenue ($)", "0.00"),
    ("grid_cost", "Cost ($)", "0.00"),
    ("solve_seconds", "Solve (s)", "0.0"),
]


def _config_name(r) -> str:
    if r["scenario"] == "state_machine":
        return "State machine (household)"
    sc = T.SCENARIO_LABEL[r["scenario"]]
    return f"{sc}, aging ignored" if r["r_cell"] == 0 else f"{sc}, J = {int(r['n_segments'])}"


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def plot_j_sweep(summary: pd.DataFrame, *, title: str, output_path: str, params: BatteryParams | None = None) -> None:
    """summary: one row per run with scenario, r_cell, n_segments and the metrics from rolling.summarise (state_machine rows optional)."""
    params = params or BatteryParams(r_cell=float(summary.loc[summary["r_cell"] > 0, "r_cell"].max()))
    if "life_loss_pct" not in summary:
        summary = summary.assign(life_loss_pct=100 * summary["life_loss_fraction"])
    scen = [s for s in SCENARIOS if (summary["scenario"] == s).any()]

    tiles = []
    for s in scen:
        d = _rows(summary, s)
        if len(d):
            best = d.loc[d["net_profit_incl_degradation"].idxmax()]
            tiles.append({"label": f"{T.SCENARIO_LABEL[s]}: best net profit", "value": f"${best['net_profit_incl_degradation']:,.2f}",
                          "sub": f"J = {int(best['n_segments'])}, life loss {best['life_loss_pct']:.3f}%"})
        r0 = summary[(summary["scenario"] == s) & (summary["r_cell"] == 0)]
        if len(r0):
            tiles.append({"label": f"{T.SCENARIO_LABEL[s]}: aging cost ignored", "value": _money(float(r0["net_profit_incl_degradation"].iloc[0])),
                          "sub": f"life loss {float(r0['life_loss_pct'].iloc[0]):.2f}%, {float(r0['equivalent_full_cycles'].iloc[0]):.0f} EFC",
                          "tone": "bad" if float(r0["net_profit_incl_degradation"].iloc[0]) < 0 else None})
    sm = summary[summary["scenario"] == "state_machine"]
    if len(sm):
        tiles.append({"label": "State machine baseline", "value": _money(float(sm["net_profit_incl_degradation"].iloc[0])),
                      "sub": f"life loss {float(sm['life_loss_pct'].iloc[0]):.2f}%", "tone": "bad" if float(sm["net_profit_incl_degradation"].iloc[0]) < 0 else None})

    children = [
        T.heading(title, "Perfect-foresight MILP, 48 h windows committed 24 h at a time, with the Xu et al. (2018) piecewise-linear cycle-aging "
                         "cost in the objective. Every configuration is charged ex post for the life its SoC trajectory actually consumed "
                         "(rainflow count × Φ × R_cell), so runs that ignore aging in the optimiser still pay for it here."),
        T.stat_tiles(tiles),
        T.section("Headline: net profit after degradation", "Grey bars are the baselines: the threshold state machine on its own 20 kWh battery, and the MILP with the aging cost switched off."),
        row(*[_headline(summary, s) for s in scen], sizing_mode="stretch_width"),
        T.section("Where the gross grid profit goes"),
        row(*[_profit_split(summary, s) for s in scen], sizing_mode="stretch_width"),
        T.section("How well the optimiser priced its own cycling",
                  "The piecewise-linear cost is charged per segment during the solve; rainflow counting on the resulting SoC path gives the cost it should have paid. "
                  "Relative error is |model − rainflow| / rainflow at the R_cell the optimiser used (Xu et al. Eq. 26)."),
        row(*[_model_vs_rainflow(summary, s) for s in scen], sizing_mode="stretch_width"),
        T.section("Cycling"),
        row(*[_cycling(summary, s, "life_loss_pct", "Cycle life consumed", "% of cell life", "{:.3f}%") for s in scen], sizing_mode="stretch_width"),
        row(*[_cycling(summary, s, "equivalent_full_cycles", "Equivalent full cycles", "EFC over JAN25", "{:.1f}") for s in scen], sizing_mode="stretch_width"),
        T.section("The degradation model", "Φ(δ) is convex, so shallow cycles are cheap per kWh and deep ones expensive; J segments approximate its slope in steps."),
        row(_stress_function(params), _segment_costs(params), sizing_mode="stretch_width"),
    ]
    tbl = summary.copy()
    tbl["config"] = [_config_name(r) for _, r in tbl.iterrows()]
    children += [T.section("Table view"), T.summary_table(tbl, TABLE_COLUMNS, height=48 + 26 * len(tbl), text_width=230, number_width=88),
                 T.note("BESS-only: arbitrage on the NSW spot price with no household load or solar. Household: the same battery behind the metered "
                        "net-local power. State machine: thesis Table 5.1 thresholds on a 20 kWh lossless battery, degradation costed ex post at the same R_cell. "
                        "Use the toolbar's save button on any panel to export it as PNG.")]
    T.save_page(children, title=title, output_path=output_path)


if __name__ == "__main__":
    plot_j_sweep(pd.read_csv("results/milp_summary.csv"), title="MILP degradation study JAN25", output_path="plots/milp_j_sweep.html")
