"""
rolling.py

Rolling-horizon driver: solve a window of `window_intervals`, commit the first
`step_intervals`, carry the per-segment energy E_{t,j} forward, repeat.

The terminal constraint (Eq. 2.6k) is applied at the end of each solved window,
so the committed intervals are free to carry charge into the next day; only the
final window's terminal constraint binds on the actual end of the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from milp.degradation import rainflow_aging_cost, relative_error
from milp.model import INTERVAL_HOURS, BatteryParams, build_and_solve_window, make_solver


def simulate_milp(
    price_df: pd.DataFrame,
    params: BatteryParams,
    export_col: str | None = None,
    import_col: str | None = None,
    window_hours: float = 48.0,
    step_hours: float = 24.0,
    terminal_soc_kwh: float | None = None,
    solver_name: str = "HiGHS",
    solver_time_limit: float | None = 120.0,
    verbose: bool = True,
    r_cell_valuation: float | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run the rolling MILP over a full price DataFrame.

    Returns (results_df, metrics). results_df has the same columns as the state
    machine simulator (so plotting/battery_plot.py works unchanged) plus
    charge_kw, discharge_kw, degradation_cost, cumulative_degradation.

    r_cell_valuation: replacement cost used to price the ex-post rainflow life loss
    in the metrics (defaults to params.r_cell). Pass the real R_cell so that runs
    optimised with a different (e.g. zero) aging cost are still charged for the
    life they actually consumed.
    """
    rrp = price_df["RRP"].to_numpy(dtype=float)
    times = price_df["SETTLEMENTDATE"].to_numpy()
    export_kw = price_df[export_col].to_numpy(dtype=float) if export_col else np.zeros(len(rrp))
    import_kw = price_df[import_col].to_numpy(dtype=float) if import_col else np.zeros(len(rrp))
    net_local = export_kw - import_kw  # G_t - A_t

    T = len(rrp)
    win = int(round(window_hours / INTERVAL_HOURS))
    step = int(round(step_hours / INTERVAL_HOURS))
    e_terminal = params.e_initial if terminal_soc_kwh is None else terminal_soc_kwh
    solver = make_solver(solver_name, time_limit=solver_time_limit)

    charge = np.zeros(T)
    discharge = np.zeros(T)
    grid_import = np.zeros(T)
    grid_export = np.zeros(T)
    soc = np.zeros(T)
    deg = np.zeros(T)
    seg_state = params.initial_segment_energy()
    n_windows = 0
    total_solve = 0.0

    for start in range(0, T, step):
        end = min(start + win, T)
        commit_end = min(start + step, T)
        res = build_and_solve_window(
            rrp[start:end], net_local[start:end], params, seg_state, e_terminal, solver=solver
        )
        n = commit_end - start
        charge[start:commit_end] = res.charge_kw[:n]
        discharge[start:commit_end] = res.discharge_kw[:n]
        grid_import[start:commit_end] = res.grid_import_kw[:n]
        grid_export[start:commit_end] = res.grid_export_kw[:n]
        soc[start:commit_end] = res.soc_kwh[:n]
        deg[start:commit_end] = res.degradation_cost[:n]
        seg_state = res.segment_soc_kwh[n - 1].copy()
        n_windows += 1
        total_solve += res.solve_seconds
        if verbose:
            print(
                f"  window {n_windows:3d}: t={start:5d}..{end:5d} commit {n:3d}  "
                f"obj={res.objective:9.3f}  solve={res.solve_seconds:5.2f}s  SoC_end={seg_state.sum():5.2f} kWh"
            )

    dt = INTERVAL_HOURS
    cost = grid_import * dt * (rrp / 1000 + params.network_tariff)
    revenue = grid_export * dt * rrp / 1000

    results = pd.DataFrame(
        {
            "time": times,
            "battery_state": soc,
            "rrp": rrp,
            "export_kw": export_kw,
            "import_kw": import_kw,
            "charge_kw": charge,
            "discharge_kw": discharge,
            "grid_import_kwh": grid_import * dt,
            "grid_export_kwh": grid_export * dt,
            "degradation_cost": deg,
            "cumulative_cost": np.cumsum(cost),
            "cumulative_revenue": np.cumsum(revenue),
            "cumulative_degradation": np.cumsum(deg),
        }
    )
    results["cumulative_profit"] = results["cumulative_revenue"] - results["cumulative_cost"]

    metrics = summarise(results, params, r_cell_valuation=r_cell_valuation)
    metrics.update({"n_windows": n_windows, "solve_seconds": total_solve})
    return results, metrics


def summarise(results: pd.DataFrame, params: BatteryParams, r_cell_valuation: float | None = None) -> dict:
    """
    Metrics on a results DataFrame, including the ex-post rainflow check.

    Two degradation figures are reported:
      degradation_cost_model    the piecewise-linear cost the optimiser charged itself
                                (zero when params.r_cell == 0). Used only for the
                                Xu Eq. (26) validation against rainflow at the same R.
      degradation_cost_rainflow the real cost: rainflow life loss x r_cell_valuation.
    net_profit_incl_degradation always subtracts the real (rainflow) cost so that
    runs with different optimiser aging costs are comparable.
    """
    r_val = params.r_cell if r_cell_valuation is None else r_cell_valuation
    soc_path = np.concatenate([[params.e_initial], results["battery_state"].to_numpy()])
    rf = rainflow_aging_cost(soc_path, params.e_max, r_val, params.phi_a, params.phi_k)
    model_deg = float(results["degradation_cost"].sum()) if "degradation_cost" in results else float("nan")
    # Validate the piecewise-linear approximation at the R the optimiser actually used.
    rainflow_at_model_r = params.r_cell * rf["life_loss_fraction"]
    grid_cost = float(results["cumulative_cost"].iloc[-1])
    grid_rev = float(results["cumulative_revenue"].iloc[-1])
    discharged = float(results["discharge_kw"].sum() * INTERVAL_HOURS) if "discharge_kw" in results else float("nan")
    return {
        "grid_cost": grid_cost,
        "grid_revenue": grid_rev,
        "net_profit_ex_degradation": grid_rev - grid_cost,
        "degradation_cost_model": model_deg,
        "degradation_cost_rainflow": rf["rainflow_cost"],
        "net_profit_incl_degradation": grid_rev - grid_cost - rf["rainflow_cost"],
        "rainflow_relative_error": relative_error(model_deg, rainflow_at_model_r),
        "life_loss_fraction": rf["life_loss_fraction"],
        "rainflow_cycles": rf["n_cycles"],
        "mean_cycle_depth": rf["mean_cycle_depth"],
        "max_cycle_depth": rf["max_cycle_depth"],
        "discharged_kwh": discharged,
        "equivalent_full_cycles": discharged / params.e_max if discharged == discharged else float("nan"),
        "final_soc_kwh": float(results["battery_state"].iloc[-1]),
    }


# ---------------------------------------------------------------------------
# Forecast-driven MPC loop (FORECAST_NOTES.md)
# ---------------------------------------------------------------------------

def simulate_milp_mpc(
    frame,
    params: BatteryParams,
    forecaster,
    horizon_hours: float = 24.0,
    step0_actual_price: bool = True,
    solver_name: str = "HiGHS",
    solver_time_limit: float | None = 30.0,
    verbose: bool = True,
    r_cell_valuation: float | None = None,
    progress_every: int = 288,
) -> tuple[pd.DataFrame, dict]:
    """
    Plan on forecasts, settle on actuals, re-plan every 5-min interval.

    At each test interval t:
      1. price_fc  = forecaster.price(t, H);  price_fc[0] = actual RRP (if step0_actual_price)
         net_fc    = forecaster.net_local(t, H)
      2. solve the Chapter 2 MILP over the H-interval horizon
      3. commit P_c, P_d for interval t only; recompute D_i, D_e from the balance
         with the *actual* net-local; cost with the *actual* RRP
      4. carry per-segment SoC forward

    The horizon shrinks at the end of the data so the terminal constraint
    (Eq. 2.6k) binds on the true last interval, as in simulate_milp.

    Returns (results_df, metrics) in the same layout as simulate_milp, plus
    price_fc_next (forecast for t+1 issued at t) and net_local_fc (forecast for t)
    columns for ex-post forecast error analysis.
    """
    rrp = frame.rrp
    net_actual = frame.net_local
    t0, T = frame.test_start, frame.n
    n = T - t0
    H = int(round(horizon_hours / INTERVAL_HOURS))
    e_terminal = params.e_initial
    solver = make_solver(solver_name, time_limit=solver_time_limit)
    dt = INTERVAL_HOURS

    charge = np.zeros(n)
    discharge = np.zeros(n)
    grid_import = np.zeros(n)
    grid_export = np.zeros(n)
    soc = np.zeros(n)
    deg = np.zeros(n)
    price_fc_next = np.full(n, np.nan)
    net_fc_now = np.zeros(n)
    seg_state = params.initial_segment_energy()
    total_solve = 0.0
    c = params.segment_costs

    for i, t in enumerate(range(t0, T)):
        h = min(H, T - t)
        price_fc = np.asarray(forecaster.price(t, h), dtype=float)
        net_fc = np.asarray(forecaster.net_local(t, h), dtype=float)
        if step0_actual_price:
            price_fc[0] = rrp[t]
        res = build_and_solve_window(price_fc, net_fc, params, seg_state, e_terminal, solver=solver)

        pc, pd_ = res.charge_kw[0], res.discharge_kw[0]
        grid = pc - pd_ - net_actual[t]              # Eq. 2.8 with actual household power
        charge[i], discharge[i] = pc, pd_
        grid_import[i], grid_export[i] = max(grid, 0.0), max(-grid, 0.0)
        seg_state = res.segment_soc_kwh[0].copy()
        soc[i] = seg_state.sum()
        deg[i] = res.degradation_cost[0]
        price_fc_next[i] = price_fc[1] if h > 1 else np.nan
        net_fc_now[i] = net_fc[0]
        total_solve += res.solve_seconds
        if verbose and (i % progress_every == 0 or i == n - 1):
            print(f"  [{forecaster.name}] t={i:5d}/{n}  SoC={soc[i]:5.2f} kWh  cum solve={total_solve:7.1f}s")

    r = slice(t0, T)
    cost = grid_import * dt * (rrp[r] / 1000 + params.network_tariff)
    revenue = grid_export * dt * rrp[r] / 1000
    results = pd.DataFrame(
        {
            "time": (frame.start_times[r] + pd.Timedelta(minutes=5)).strftime("%-d/%m/%Y %-H:%M"),
            "battery_state": soc,
            "rrp": rrp[r],
            "export_kw": frame.export_kw[r],
            "import_kw": frame.import_kw[r],
            "charge_kw": charge,
            "discharge_kw": discharge,
            "grid_import_kwh": grid_import * dt,
            "grid_export_kwh": grid_export * dt,
            "degradation_cost": deg,
            "cumulative_cost": np.cumsum(cost),
            "cumulative_revenue": np.cumsum(revenue),
            "cumulative_degradation": np.cumsum(deg),
            "price_fc_next": price_fc_next,
            "net_local_fc": net_fc_now,
        }
    )
    results["cumulative_profit"] = results["cumulative_revenue"] - results["cumulative_cost"]
    metrics = summarise(results, params, r_cell_valuation=r_cell_valuation)
    metrics.update({"n_windows": n, "solve_seconds": total_solve, "forecaster": forecaster.name})
    return results, metrics
