"""
MILP battery trading using the thesis Chapter 2 model with Xu et al. (2018)
piecewise-linear cycle aging cost. Perfect foresight, rolling horizon.

Run from the trading/ directory:
    python milp_trading.py                 # full experiment matrix on JAN25
    python milp_trading.py --quick         # 3 days, J in {1, 4}, for a smoke test

Companion to state_machine_trading.py; run_milp_simulation mirrors run_trading_simulation.
"""

from __future__ import annotations

import argparse
import os
import time

import pandas as pd

from milp.model import BatteryParams
from milp.rolling import simulate_milp, summarise
from plotting.battery_plot import plot_battery_trading
from plotting.milp_sweep_plot import plot_j_sweep
from utils.data import merge_optional_csv

TEST_CSV = "data/price_JAN25.csv"
TEST_EXPORT_CSV = "data/export_JAN25.csv"
TEST_IMPORT_CSV = "data/import_JAN25.csv"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
J_SWEEP = (1, 2, 4, 8, 16)
R_CELL = 12_000.0  # AUD, placeholder replacement cost for a Powerwall 3
EXPORT_LIMITS_KW = (None, 10.0, 5.0)  # grid export limit study; None = unlimited (the headline runs)
SM_BUY_THRESHOLD, SM_SELL_THRESHOLD = 69.70, 127.20  # thesis Table 5.1 (DEC24 grid search)
SPIKE_RRP = 1000.0  # $/MWh, the price-spike band used in RESULTS.md
KEEP_RUN_PLOTS = {"household_J4_R12000"}   # per-run pages drawn by default; the rest need --per-run


def load_test_data(test_csv=TEST_CSV, export_csv=TEST_EXPORT_CSV, import_csv=TEST_IMPORT_CSV, household=True, n_days=None):
    df = pd.read_csv(test_csv)
    if household:
        df = merge_optional_csv(df, export_csv, "EXPORT_KW")
        df = merge_optional_csv(df, import_csv, "IMPORT_KW")
    if n_days is not None:
        df = df.iloc[: int(n_days * 288)].reset_index(drop=True)
    return df


def run_milp_simulation(
    params: BatteryParams,
    household: bool = True,
    test_csv=TEST_CSV,
    test_export_csv=TEST_EXPORT_CSV,
    test_import_csv=TEST_IMPORT_CSV,
    n_days=None,
    window_hours=48.0,
    step_hours=24.0,
    solver_name="HiGHS",
    verbose=True,
    plot=True,
    plot_title=None,
    plot_output_path=None,
):
    """Run the rolling MILP on one dataset with one parameter set. Returns dict(results_df, metrics, params)."""
    df = load_test_data(test_csv, test_export_csv, test_import_csv, household, n_days)
    label = f"{'household' if household else 'bess_only'}_J{params.n_segments}_R{int(params.r_cell)}"
    if verbose:
        print(f"\n=== MILP run: {label} ({len(df)} intervals, window {window_hours}h / step {step_hours}h, {solver_name}) ===")
        print(f"    segment costs c_j ($/kWh): {[round(float(c), 4) for c in params.segment_costs]}")

    results_df, metrics = simulate_milp(
        df,
        params,
        export_col="EXPORT_KW" if household else None,
        import_col="IMPORT_KW" if household else None,
        window_hours=window_hours,
        step_hours=step_hours,
        solver_name=solver_name,
        verbose=verbose,
        r_cell_valuation=R_CELL,
    )
    if verbose:
        print_metrics(label, metrics)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(f"{RESULTS_DIR}/milp_{label}.csv", index=False)
    if plot:
        plot_milp_run(results_df, label, params.e_max, title=plot_title, output_path=plot_output_path)
    return {"results_df": results_df, "metrics": metrics, "params": params, "label": label}


def run_title(label: str) -> str:
    scenario, j, r = label.split("_J")[0], label.split("_J")[1].split("_R")[0], label.split("_R")[1]
    sc = "Household (solar + load)" if scenario == "household" else "BESS only (arbitrage)"
    aging = "aging cost ignored (R_cell = 0)" if float(r) == 0 else f"J = {j} cycle-depth segments, R_cell = {float(r):,.0f}"
    return f"Perfect-foresight MILP, {sc}, {aging}"


def plot_milp_run(results_df: pd.DataFrame, label: str, e_max: float, title=None, output_path=None) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_battery_trading(results_df, title=title or run_title(label), output_path=output_path or f"{PLOTS_DIR}/milp_{label}.html",
                         bess_size=e_max, show_plot=False)


def print_metrics(label: str, m: dict) -> None:
    print(f"{'-' * 44}\n  {label}\n{'-' * 44}")
    print(f"Grid cost:                    ${m['grid_cost']:9.2f}")
    print(f"Grid revenue:                 ${m['grid_revenue']:9.2f}")
    print(f"Net profit excl. degradation: ${m['net_profit_ex_degradation']:9.2f}")
    print(f"Degradation cost (model):     ${m['degradation_cost_model']:9.2f}   (optimiser's internal PWL cost)")
    print(f"Degradation cost (rainflow):  ${m['degradation_cost_rainflow']:9.2f}   rel. error {m['rainflow_relative_error']:.3f}")
    print(f"Net profit incl. degradation: ${m['net_profit_incl_degradation']:9.2f}   (uses rainflow cost at R_cell={R_CELL:.0f})")
    print(f"Life loss: {100 * m['life_loss_fraction']:.3f}%   rainflow cycles: {m['rainflow_cycles']:.1f}   "
          f"mean depth {m['mean_cycle_depth']:.2f}   EFC {m['equivalent_full_cycles']:.1f}")
    print(f"Final SoC: {m['final_soc_kwh']:.2f} kWh   solve time {m.get('solve_seconds', float('nan')):.1f}s")


def state_machine_baseline(n_days=None, verbose=True):
    """
    State machine on the same data with the thesis Table 5.1 thresholds, plus an
    ex-post rainflow degradation cost on its 20 kWh battery so it sits on the same footing.
    """
    from state_machine_trading import run_trading_simulation
    from utils import bess_simulator

    out = run_trading_simulation(
        buy_threshold=SM_BUY_THRESHOLD, sell_threshold=SM_SELL_THRESHOLD, optimise_thresholds=False, verbose=False, plot=False
    )
    results_df = out["results_df"]
    if n_days is not None:
        results_df = results_df.iloc[: int(n_days * 288)].reset_index(drop=True)
    sm_params = BatteryParams.state_machine_equivalent(r_cell=R_CELL, e_initial=0.0)
    results_df = results_df.copy()
    results_df["degradation_cost"] = 0.0
    results_df["discharge_kw"] = (-results_df["battery_state"].diff().fillna(results_df["battery_state"].iloc[0])).clip(lower=0) / (5 / 60)
    m = summarise(results_df, sm_params, r_cell_valuation=R_CELL)
    m["degradation_cost_model"] = float("nan")
    m["rainflow_relative_error"] = float("nan")
    if verbose:
        print_metrics(f"state_machine (buy {SM_BUY_THRESHOLD} / sell {SM_SELL_THRESHOLD}, {bess_simulator.BESS_SIZE} kWh)", m)
    return m


def run_experiment_matrix(j_values=J_SWEEP, n_days=None, solver_name="HiGHS", plot=True, verbose=True, per_run_plots=False):
    """
    The agreed matrix: BESS-only and household, each at R_cell=0 (J=1) and R_cell=R_CELL over the J sweep.
    plot draws the study page (and the KEEP_RUN_PLOTS pages); per_run_plots draws a page for every run.
    """
    rows = []
    t0 = time.time()
    for household in (False, True):
        scenario = "household" if household else "bess_only"
        configs = [(0.0, 1)] + [(R_CELL, j) for j in j_values]
        for r_cell, j in configs:
            params = BatteryParams(r_cell=r_cell, n_segments=j)
            label = f"{scenario}_J{j}_R{int(r_cell)}"
            out = run_milp_simulation(params, household=household, n_days=n_days, solver_name=solver_name,
                                      plot=plot and (per_run_plots or label in KEEP_RUN_PLOTS), verbose=verbose)
            rows.append({"scenario": scenario, "r_cell": r_cell, "n_segments": j, **out["metrics"]})

    sm = state_machine_baseline(n_days=n_days, verbose=verbose)
    rows.append({"scenario": "state_machine", "r_cell": R_CELL, "n_segments": 0, **sm})

    summary = pd.DataFrame(rows)
    summary["life_loss_pct"] = 100 * summary["life_loss_fraction"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{n_days}d" if n_days else ""
    csv_path = f"{RESULTS_DIR}/milp_summary{suffix}.csv"
    summary.to_csv(csv_path, index=False)

    cols = ["scenario", "r_cell", "n_segments", "grid_cost", "grid_revenue", "net_profit_ex_degradation",
            "degradation_cost_model", "degradation_cost_rainflow", "rainflow_relative_error",
            "net_profit_incl_degradation", "life_loss_pct", "equivalent_full_cycles", "final_soc_kwh", "solve_seconds"]
    pd.set_option("display.width", 250)
    print(f"\n\n==== Summary (JAN25{suffix}) ====")
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nWritten {csv_path}   total wall time {time.time() - t0:.0f}s")

    if plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_j_sweep(summary, title=f"MILP degradation study JAN25{suffix}", output_path=f"{PLOTS_DIR}/milp_j_sweep{suffix}.html")
    return summary


def run_export_limit_study(limits=EXPORT_LIMITS_KW, n_segments=4, n_days=None, solver_name="HiGHS", verbose=True):
    """
    Household, perfect foresight, J = n_segments, under a grid export limit.

    The headline runs enforce no limit and export up to ~16 kW during price spikes
    (battery at 11.04 kW on top of the solar surplus). This measures how much of the
    value added survives a 10 kW or 5 kW connection limit. Solar curtailment is not
    modelled, so under a limit the battery must keep headroom for any surplus above it.
    Written to results/milp_export_limit.csv, separate from the J sweep.
    """
    from milp.model import INTERVAL_HOURS

    df = load_test_data(household=True, n_days=n_days)
    rrp = df["RRP"].to_numpy(dtype=float)
    net = (df["EXPORT_KW"] - df["IMPORT_KW"]).to_numpy(dtype=float)
    tariff = BatteryParams().network_tariff
    no_batt = (net.clip(min=0) * rrp / 1000 - (-net).clip(min=0) * (rrp / 1000 + tariff)) * INTERVAL_HOURS
    spike = rrp > SPIKE_RRP
    rows = []
    for lim in limits:
        params = BatteryParams(r_cell=R_CELL, n_segments=n_segments, export_limit_kw=lim)
        results_df, m = simulate_milp(df, params, export_col="EXPORT_KW", import_col="IMPORT_KW",
                                      solver_name=solver_name, verbose=False, r_cell_valuation=R_CELL)
        at_meter = (results_df["grid_export_kwh"] * rrp / 1000 - results_df["grid_import_kwh"] * (rrp / 1000 + tariff)).to_numpy()
        added = at_meter - no_batt
        rows.append({"export_limit_kw": lim, "n_segments": n_segments, **m,
                     "value_added": float(added.sum()), "value_added_spike": float(added[spike].sum()),
                     "value_added_non_spike": float(added[~spike].sum()),
                     "max_grid_export_kw": float(results_df["grid_export_kwh"].max() / INTERVAL_HOURS)})
    summary = pd.DataFrame(rows)
    suffix = f"_{n_days}d" if n_days else ""
    path = f"{RESULTS_DIR}/milp_export_limit{suffix}.csv"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary.to_csv(path, index=False)
    if verbose:
        cols = ["export_limit_kw", "net_profit_ex_degradation", "degradation_cost_rainflow", "net_profit_incl_degradation",
                "value_added", "value_added_spike", "value_added_non_spike", "max_grid_export_kw"]
        print(f"\n==== Export limit study (household, J={n_segments}) ====")
        print(summary[cols].round(2).to_string(index=False))
        print(f"Written {path}")
    return summary


def replot(n_days=None, per_run_plots=False) -> None:
    """Redraw the study page (and KEEP_RUN_PLOTS, or every run with per_run_plots) from results/ without solving."""
    import glob

    suffix = f"_{n_days}d" if n_days else ""
    summary = pd.read_csv(f"{RESULTS_DIR}/milp_summary{suffix}.csv")
    plot_j_sweep(summary, title=f"MILP degradation study JAN25{suffix}", output_path=f"{PLOTS_DIR}/milp_j_sweep{suffix}.html")
    e_max = BatteryParams().e_max
    for path in sorted(glob.glob(f"{RESULTS_DIR}/milp_*_J*_R*.csv")):
        label = os.path.basename(path)[len("milp_"):-len(".csv")]
        if per_run_plots or label in KEEP_RUN_PLOTS:
            plot_milp_run(pd.read_csv(path), label, e_max)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 days, J in {1,4}")
    ap.add_argument("--days", type=float, default=None, help="limit to the first N days of the test data")
    ap.add_argument("--solver", default="HiGHS", help="HiGHS (default), GUROBI or CBC")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--plot-only", action="store_true", help="redraw the plots from results/ without solving")
    ap.add_argument("--per-run", action="store_true", help="also draw a page for every individual run (default: study page + household J=4)")
    ap.add_argument("--export-limits", action="store_true", help="run only the grid export limit study (household, J=4)")
    args = ap.parse_args()
    n_days = 3 if args.quick else args.days
    if args.plot_only:
        replot(n_days, per_run_plots=args.per_run)
        return
    if args.export_limits:
        run_export_limit_study(n_days=n_days, solver_name=args.solver, verbose=not args.quiet)
        return
    j_values = (1, 4) if args.quick else J_SWEEP
    run_experiment_matrix(j_values=j_values, n_days=n_days, solver_name=args.solver, plot=not args.no_plot, verbose=not args.quiet,
                          per_run_plots=args.per_run)
    run_export_limit_study(n_days=n_days, solver_name=args.solver, verbose=not args.quiet)


if __name__ == "__main__":
    main()
