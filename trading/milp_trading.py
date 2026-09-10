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
    if plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_battery_trading(
            results_df,
            title=plot_title or f"MILP {label}",
            output_path=plot_output_path or f"{PLOTS_DIR}/milp_{label}.html",
            bess_size=params.e_max,
            show_plot=False,
        )
    return {"results_df": results_df, "metrics": metrics, "params": params, "label": label}


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
        buy_threshold=68.22, sell_threshold=127.20, optimise_thresholds=False, verbose=False, plot=False
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
        print_metrics(f"state_machine (buy 68.22 / sell 127.20, {bess_simulator.BESS_SIZE} kWh)", m)
    return m


def run_experiment_matrix(j_values=J_SWEEP, n_days=None, solver_name="HiGHS", plot=True, verbose=True):
    """The agreed matrix: BESS-only and household, each at R_cell=0 (J=1) and R_cell=R_CELL over the J sweep."""
    rows = []
    t0 = time.time()
    for household in (False, True):
        scenario = "household" if household else "bess_only"
        configs = [(0.0, 1)] + [(R_CELL, j) for j in j_values]
        for r_cell, j in configs:
            params = BatteryParams(r_cell=r_cell, n_segments=j)
            out = run_milp_simulation(params, household=household, n_days=n_days, solver_name=solver_name, plot=plot, verbose=verbose)
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
        plot_j_sweep(summary[summary["scenario"] != "state_machine"], title=f"MILP J sweep JAN25{suffix}", output_path=f"{PLOTS_DIR}/milp_j_sweep{suffix}.html")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="3 days, J in {1,4}")
    ap.add_argument("--days", type=float, default=None, help="limit to the first N days of the test data")
    ap.add_argument("--solver", default="HiGHS", help="HiGHS (default), GUROBI or CBC")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    n_days = 3 if args.quick else args.days
    j_values = (1, 4) if args.quick else J_SWEEP
    run_experiment_matrix(j_values=j_values, n_days=n_days, solver_name=args.solver, plot=not args.no_plot, verbose=not args.quiet)


if __name__ == "__main__":
    main()
