"""
bess_simulator.py

Physical constants and simulation engine for BESS price-arbitrage trading.
Any strategy plugs in via a callback:
    strategy_fn(rrp, battery_state, export_kw, load_kw) -> 'buy'|'sell'|'hold'

The simulator enforces SoC bounds and energy-per-interval limits; the strategy
only expresses intent.

Export and load are optional — omitting both reproduces the original
arbitrage-only behaviour.


Energy flow model (each 5-min interval)
----------------------------------------
    net_local  = export_kw * INTERVAL_HOURS   [kWh]
                 (export_kw = solar - load from meter B1 data)

    Strategy intent maps to a battery delta:
        'buy'  → bess_delta = +min(room_in_battery, inverter_limit)   [charge]
        'sell' → bess_delta = −min(stored_energy,   inverter_limit)   [discharge]
        'hold' → bess_delta determined by net_local (self-consumption):
                   net_local > 0 : surplus (export) charges battery (up to limits)
                   net_local < 0 : deficit (import) draws from battery (up to limits)
                   net_local = 0 : no battery movement

    Grid position:
        grid_net = −net_local + bess_delta
        grid_net > 0 → net import from grid  (cost)
        grid_net < 0 → net export to grid    (revenue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Physical / market constants ───────────────────────────────────────────────
BESS_SIZE = 20  # kWh  — usable capacity
BESS_INVERTER_CAPACITY = 11.04  # kW   — max charge/discharge rate for BESS
BESS_INVERTER_CAPACITY_SOLAR = 20  # kW   — max charge rate from solar (
INTERVAL_HOURS = 5 / 60  # h    — 5-minute settlement periods

_ENERGY_PER_INTERVAL_BESS = BESS_INVERTER_CAPACITY * INTERVAL_HOURS  # kWh
_ENERGY_PER_INTERVAL_SOLAR = BESS_INVERTER_CAPACITY_SOLAR * INTERVAL_HOURS  # kWh


# ── Full-detail simulation ────────────────────────────────────────────────────
def simulate(
    price_df: pd.DataFrame,
    strategy_fn,
    export_col: str | None = None,
    import_col: str | None = None,
    network_tariff: float = 0.0,
) -> pd.DataFrame:
    """
    General simulation with grid import and export data.

    Calls strategy_fn(rrp, battery_state, export_kw, import_kw) -> (grid_net, bess_delta)
    at every interval. Enforces SoC bounds and energy-per-interval limits for solar inverter and battery inverter.

    Parameters
    ----------
    price_df       : DataFrame with columns SETTLEMENTDATE, RRP, TOTALDEMAND.
                     May also contain export_col and/or load_col columns.
    strategy_fn    : callable(rrp, battery_state, export_kw, import_kw) -> str
    export_col     : Column name for net export (solar - load) in kW. None = no export data.
    import_col       : Column name for grid import in kW.  None = no import data.
    network_tariff : Fixed network charge in cents/kWh on grid imports. Default = 0.0.

    Returns
    -------
    DataFrame with per-interval results including grid import/export columns.
    """
    battery_state = 0.0
    cumulative_cost = 0.0
    cumulative_revenue = 0.0

    results = []
    for _, row in price_df.iterrows():
        time = row["SETTLEMENTDATE"]
        rrp = row["RRP"]
        export_kw = float(row[export_col]) if export_col is not None else 0.0
        import_kw = float(row[import_col]) if import_col is not None else 0.0

        grid_net, bess_delta = strategy_fn(rrp, battery_state, export_kw, import_kw)

        battery_state += bess_delta

        # Grid position: positive = import (cost), negative = export (revenue)
        grid_import = max(0.0, grid_net)
        grid_export = max(0.0, -grid_net)

        # Cost = wholesale price + network tariff (both converted to $/kWh)
        cumulative_cost += grid_import * (rrp / 1000 + network_tariff / 100)
        cumulative_revenue += grid_export * rrp / 1000

        results.append(
            {
                "time": time,
                "battery_state": battery_state,
                "rrp": rrp,
                "export_kw": export_kw,
                "import_kw": import_kw,
                "grid_import_kwh": grid_import,
                "grid_export_kwh": grid_export,
                "cumulative_cost": cumulative_cost,
                "cumulative_revenue": cumulative_revenue,
                "cumulative_profit": cumulative_revenue - cumulative_cost,
            }
        )

    return pd.DataFrame(results)


# ── Fast profit-only simulation for optimisation loops ────────────────────────
def simulate_profit(
    rrp_arr: np.ndarray,
    strategy_fn,
    export_arr: np.ndarray | None = None,
    import_arr: np.ndarray | None = None,
    network_tariff: float = 0.0,
) -> float:
    """
    Fast profit-only simulation for optimisation loops.

    NOTE: export_arr represents net export = solar - load (B1 meter data), not solar generation.

    Parameters
    ----------
    rrp_arr        : 1-D numpy array of RRP values ($/MWh), one per interval.
    strategy_fn    : callable(rrp, battery_state, export_kw, load_kw) -> str
    export_arr     : 1-D numpy array of net export (solar - load) in kW. None = zeros.
    load_arr       : 1-D numpy array of household load (kW).  None = zeros.
    network_tariff : Fixed network charge in cents/kWh on grid imports. Default = 0.0.

    Returns
    -------
    Net profit in dollars (export revenue minus import cost).
    """
    n = len(rrp_arr)
    if export_arr is None:
        export_arr = np.zeros(n)
    if import_arr is None:
        import_arr = np.zeros(n)

    battery_state = 0.0
    cost = 0.0
    revenue = 0.0

    for i in range(n):
        rrp = rrp_arr[i]
        export_kw = export_arr[i]
        import_kw = import_arr[i]

        grid_net, bess_delta = strategy_fn(rrp, battery_state, export_kw, import_kw)

        battery_state += bess_delta

        if grid_net > 0:
            # Cost = wholesale price + network tariff (both converted to $/kWh)
            cost += grid_net * (rrp / 1000 + network_tariff / 100)
        else:
            revenue += (-grid_net) * rrp / 1000

    return revenue - cost
