"""
Forecast-driven MILP battery trading (thesis Section 6.4, FORECAST_NOTES.md).

Runs the Chapter 2 MILP as a 24 h / 5-min MPC loop under several forecasters and
reports the dispatch value of each relative to perfect foresight.

Run from the trading/ directory:
    python forecast_trading.py                       # all forecasters, full JAN25
    python forecast_trading.py --quick               # perfect + naive, 2 days
    python forecast_trading.py --forecasters perfect perfect_price naive aemo lstm --days 7
    python forecast_trading.py --reuse             # only forecasters without a results/*.json are solved

Companion to milp_trading.py (perfect-foresight rolling horizon).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from forecasting import load_frame
from milp.model import BatteryParams
from milp.rolling import simulate_milp_mpc
from milp_trading import R_CELL, print_metrics
from plotting.battery_plot import plot_battery_trading
from plotting.forecast_study_plot import plot_forecast_study

RESULTS_DIR = "results"
PLOTS_DIR = "plots"
N_SEGMENTS = 4
HORIZON_HOURS = 24.0
ALL_FORECASTERS = ("perfect", "perfect_price", "naive", "aemo", "lstm")


def make_forecaster(name: str, frame, household: bool = True):
    if name == "perfect":
        from forecasting.naive import PerfectForecaster
        return PerfectForecaster(frame)
    if name == "perfect_price":
        from forecasting.naive import PerfectPriceForecaster
        return PerfectPriceForecaster(frame)
    if name == "naive":
        from forecasting.naive import SeasonalNaiveForecaster
        return SeasonalNaiveForecaster(frame)
    if name == "aemo":
        from forecasting.aemo import AemoPredispatchForecaster
        return AemoPredispatchForecaster(frame)
    if name == "lstm":
        from forecasting.lstm import LstmPriceForecaster
        return LstmPriceForecaster(frame)
    raise ValueError(f"unknown forecaster {name!r}; choose from {ALL_FORECASTERS}")


def forecast_errors(forecaster, frame, horizon: int = 288, step0_actual: bool = True) -> dict:
    """
    Ex-post forecast error over the test period: MAE/RMSE in $/MWh on the price
    path the MILP actually saw, plus relative MAE against the seasonal naive at
    1 h and 24 h ahead, and net-local MAE in kW.
    """
    from forecasting.naive import SeasonalNaiveForecaster

    naive = forecaster if forecaster.name == "naive" else SeasonalNaiveForecaster(frame)
    t0, T = frame.test_start, frame.n
    errs, errs_naive, net_errs = [], [], []
    err_1h, err_24h, nerr_1h, nerr_24h = [], [], [], []
    for t in range(t0, T - horizon + 1, 6):           # every half hour is enough for the statistics
        actual = frame.rrp[t : t + horizon]
        fc = np.asarray(forecaster.price(t, horizon), dtype=float)
        nfc = np.asarray(naive.price(t, horizon), dtype=float)
        if step0_actual:
            fc[0] = nfc[0] = actual[0]
        e, ne = fc - actual, nfc - actual
        errs.append(e); errs_naive.append(ne)
        err_1h.append(e[12]); err_24h.append(e[-1]); nerr_1h.append(ne[12]); nerr_24h.append(ne[-1])
        net_errs.append(np.asarray(forecaster.net_local(t, horizon)) - frame.net_local[t : t + horizon])
    e = np.concatenate(errs); ne = np.concatenate(errs_naive); n = np.concatenate(net_errs)
    fc_all = np.concatenate([np.asarray(forecaster.price(t, horizon), dtype=float) for t in range(t0, T - horizon + 1, 6)])
    act_all = np.concatenate([frame.rrp[t : t + horizon] for t in range(t0, T - horizon + 1, 6)])
    return {
        "price_mae": float(np.abs(e).mean()),
        "price_median_ae": float(np.median(np.abs(e))),
        "price_mae_clip1000": float(np.abs(np.minimum(fc_all, 1000.0) - act_all).mean()),
        "price_fc_spike_frac": float((fc_all > 1000.0).mean()),
        "price_actual_spike_frac": float((act_all > 1000.0).mean()),
        "price_rmse": float(np.sqrt((e ** 2).mean())),
        "price_rmae_vs_naive": float(np.abs(e).mean() / np.abs(ne).mean()),
        "price_mae_1h": float(np.abs(err_1h).mean()),
        "price_mae_24h": float(np.abs(err_24h).mean()),
        "price_rmae_1h": float(np.abs(err_1h).mean() / np.abs(nerr_1h).mean()),
        "price_rmae_24h": float(np.abs(err_24h).mean() / np.abs(nerr_24h).mean()),
        "net_local_mae_kw": float(np.abs(n).mean()),
    }


def run_one(name: str, household: bool, n_days, solver_name: str, plot: bool, verbose: bool) -> dict:
    """One forecaster end to end. Safe to call in a worker process."""
    frame = load_frame(household=household, n_test_days=n_days)
    params = BatteryParams(r_cell=R_CELL, n_segments=N_SEGMENTS)
    forecaster = make_forecaster(name, frame, household)
    label = f"mpc_{'household' if household else 'bess_only'}_J{N_SEGMENTS}_{name}"
    if verbose:
        print(f"\n=== {label}: {frame.n - frame.test_start} intervals, horizon {HORIZON_HOURS}h, {solver_name} ===")
    results_df, metrics = simulate_milp_mpc(
        frame, params, forecaster, horizon_hours=HORIZON_HOURS, solver_name=solver_name,
        verbose=verbose, r_cell_valuation=R_CELL,
    )
    metrics.update(forecast_errors(forecaster, frame))
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df.to_csv(f"{RESULTS_DIR}/{label}.csv", index=False)
    row = {"forecaster": name, "scenario": "household" if household else "bess_only",
           "n_segments": N_SEGMENTS, "r_cell": R_CELL, "n_days": n_days, **metrics}
    with open(f"{RESULTS_DIR}/{label}{'_' + str(n_days) + 'd' if n_days else ''}.json", "w") as fh:
        json.dump(row, fh, indent=2)
    if plot:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plot_battery_trading(results_df, title=label, output_path=f"{PLOTS_DIR}/{label}.html",
                             bess_size=params.e_max, show_plot=False)
    if verbose:
        print_metrics(label, metrics)
        print(f"Price MAE {metrics['price_mae']:.1f} $/MWh (rMAE vs naive {metrics['price_rmae_vs_naive']:.3f}; "
              f"1h {metrics['price_mae_1h']:.1f}, 24h {metrics['price_mae_24h']:.1f})   "
              f"net-local MAE {metrics['net_local_mae_kw']:.2f} kW")
    return row


def load_saved_row(name: str, household: bool, n_days) -> dict | None:
    label = f"mpc_{'household' if household else 'bess_only'}_J{N_SEGMENTS}_{name}"
    path = f"{RESULTS_DIR}/{label}{'_' + str(n_days) + 'd' if n_days else ''}.json"
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def run_study(forecasters=ALL_FORECASTERS, household=True, n_days=None, solver_name="HiGHS",
              plot=True, verbose=True, parallel=True, reuse=False, per_run_plots=False) -> pd.DataFrame:
    """
    reuse=True picks up results/<label>.json from earlier runs instead of re-solving those forecasters.
    plot draws the study page; per_run_plots also draws a page per forecaster.
    """
    t0 = time.time()
    saved = {f: load_saved_row(f, household, n_days) for f in forecasters} if reuse else {}
    todo = [f for f in forecasters if saved.get(f) is None]
    if reuse and verbose:
        print(f"reusing saved results for {[f for f in forecasters if saved.get(f)]}, running {todo}")
    args = [(f, household, n_days, solver_name, plot and per_run_plots, verbose) for f in todo]
    if not args:
        rows = []
    elif parallel and len(args) > 1:
        with ProcessPoolExecutor(max_workers=len(args)) as ex:
            rows = list(ex.map(run_one, *zip(*args)))
    else:
        rows = [run_one(*a) for a in args]
    rows = [saved[f] if saved.get(f) else next(r for r in rows if r["forecaster"] == f) for f in forecasters]
    summary = pd.DataFrame(rows)
    perfect = summary.loc[summary["forecaster"] == "perfect", "net_profit_incl_degradation"]
    if len(perfect):
        summary["profit_gap_to_perfect"] = float(perfect.iloc[0]) - summary["net_profit_incl_degradation"]
        summary["profit_pct_of_perfect"] = 100 * summary["net_profit_incl_degradation"] / float(perfect.iloc[0])
    summary["life_loss_pct"] = 100 * summary["life_loss_fraction"]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{n_days}d" if n_days else ""
    path = f"{RESULTS_DIR}/forecast_summary{suffix}.csv"
    summary.to_csv(path, index=False)
    cols = ["forecaster", "net_profit_ex_degradation", "degradation_cost_rainflow", "net_profit_incl_degradation",
            "profit_gap_to_perfect", "profit_pct_of_perfect", "life_loss_pct", "equivalent_full_cycles",
            "price_mae", "price_median_ae", "price_mae_clip1000", "price_rmae_vs_naive", "price_mae_1h", "price_mae_24h",
            "price_fc_spike_frac", "net_local_mae_kw", "solve_seconds"]
    cols = [c for c in cols if c in summary]
    pd.set_option("display.width", 250)
    print(f"\n\n==== Forecast study (JAN25{suffix}, household J={N_SEGMENTS}, R_cell={R_CELL:.0f}) ====")
    print(summary[cols].round(3).to_string(index=False))
    print(f"\nWritten {path}   wall time {time.time() - t0:.0f}s")
    if plot:
        plot_study(summary, forecasters, household=household, n_days=n_days)   # per-run pages were drawn in run_one
    return summary


def plot_study(summary: pd.DataFrame, forecasters=ALL_FORECASTERS, household: bool = True, n_days=None,
               per_run: bool = False) -> None:
    """Study page from a summary frame and the saved per-interval CSVs. per_run=True also redraws each forecaster's own plot."""
    frame = load_frame(household=household, n_test_days=n_days)
    scenario = "household" if household else "bess_only"
    names = [f for f in forecasters if f in set(summary["forecaster"])]
    fcs = {f: make_forecaster(f, frame, household) for f in names}
    res = {f: pd.read_csv(f"{RESULTS_DIR}/mpc_{scenario}_J{N_SEGMENTS}_{f}.csv") for f in names}
    day = str(frame.start_times[frame.test_start].normalize().date()) if n_days else "2025-01-15"
    suffix = f"_{n_days}d" if n_days else ""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if per_run:
        e_max = BatteryParams(r_cell=R_CELL, n_segments=N_SEGMENTS).e_max
        for f in names:
            label = f"mpc_{scenario}_J{N_SEGMENTS}_{f}"
            plot_battery_trading(res[f], title=f"MPC, {scenario}, J={N_SEGMENTS}, {f} forecaster",
                                 output_path=f"{PLOTS_DIR}/{label}.html", bess_size=e_max, show_plot=False)
    plot_forecast_study(summary, res, frame, fcs, title=f"Forecast study JAN25{suffix}",
                        output_path=f"{PLOTS_DIR}/forecast_study{suffix}.html", day=day)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--forecasters", nargs="+", default=list(ALL_FORECASTERS))
    ap.add_argument("--quick", action="store_true", help="perfect + naive on 2 days")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--bess-only", action="store_true")
    ap.add_argument("--solver", default="HiGHS")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--serial", action="store_true", help="run forecasters one after another")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="reuse results/*.json from earlier runs where present")
    ap.add_argument("--plot-only", action="store_true", help="redraw the plots from results/ without solving")
    ap.add_argument("--per-run", action="store_true", help="also draw a page for every forecaster's run (default: study page only)")
    a = ap.parse_args()
    forecasters = ["perfect", "naive"] if a.quick else a.forecasters
    n_days = 2 if a.quick else a.days
    if a.plot_only:
        suffix = f"_{n_days}d" if n_days else ""
        summary = pd.read_csv(f"{RESULTS_DIR}/forecast_summary{suffix}.csv")
        plot_study(summary, forecasters, household=not a.bess_only, n_days=n_days, per_run=a.per_run)
        return
    run_study(forecasters, household=not a.bess_only, n_days=n_days, solver_name=a.solver,
              plot=not a.no_plot, verbose=not a.quiet, parallel=not a.serial, reuse=a.reuse, per_run_plots=a.per_run)


if __name__ == "__main__":
    main()
