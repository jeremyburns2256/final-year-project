"""
bess_simulator.py

Physical constants and simulation engine for BESS price-arbitrage trading.
Any strategy plugs in via a callback:
    strategy_fn(rrp, battery_state, solar_kw, load_kw) -> 'buy'|'sell'|'hold'

The simulator enforces SoC bounds and energy-per-interval limits; the strategy
only expresses intent.

Solar and load are optional — omitting both reproduces the original
arbitrage-only behaviour.

Energy flow model (each 5-min interval)
----------------------------------------
    net_local  = (solar_kw - load_kw) * INTERVAL_HOURS   [kWh]

    Strategy intent maps to a battery delta:
        'buy'  → bess_delta = +min(room_in_battery, inverter_limit)   [charge]
        'sell' → bess_delta = −min(stored_energy,   inverter_limit)   [discharge]
        'hold' → bess_delta determined by net_local (self-consumption):
                   net_local > 0 : surplus solar charges battery (up to limits)
                   net_local < 0 : load deficit draws from battery (up to limits)
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
BESS_SIZE              = 20       # kWh  — usable capacity
BESS_INVERTER_CAPACITY = 5        # kW   — max charge/discharge rate
INTERVAL_HOURS         = 5 / 60  # h    — 5-minute settlement periods

_ENERGY_PER_INTERVAL = BESS_INVERTER_CAPACITY * INTERVAL_HOURS  # kWh


# ── Full-detail simulation ────────────────────────────────────────────────────
def simulate(
    price_df: pd.DataFrame,
    strategy_fn,
    solar_col: str | None = None,
    load_col: str | None = None,
) -> pd.DataFrame:
    """
    General simulation with optional solar generation and household load.

    Calls strategy_fn(rrp, battery_state, solar_kw, load_kw) -> 'buy'|'sell'|'hold'
    at every interval. Enforces SoC bounds and energy-per-interval limits.

    Parameters
    ----------
    price_df   : DataFrame with columns SETTLEMENTDATE, RRP, TOTALDEMAND.
                 May also contain solar_col and/or load_col columns.
    strategy_fn: callable(rrp, battery_state, solar_kw, load_kw) -> str
    solar_col  : Column name for solar generation in kW. None = no solar.
    load_col   : Column name for household load in kW.  None = no load.

    Returns
    -------
    DataFrame with per-interval results including grid import/export columns.
    """
    battery_state      = 0.0
    cumulative_cost    = 0.0
    cumulative_revenue = 0.0

    results = []
    for _, row in price_df.iterrows():
        time     = row["SETTLEMENTDATE"]
        rrp      = row["RRP"]
        demand   = row["TOTALDEMAND"]
        solar_kw = float(row[solar_col]) if solar_col is not None else 0.0
        load_kw  = float(row[load_col])  if load_col  is not None else 0.0

        # Net local generation surplus this interval (kWh)
        net_local = (solar_kw - load_kw) * INTERVAL_HOURS

        intent = strategy_fn(rrp, battery_state, solar_kw, load_kw)

        if intent == "buy" and battery_state < BESS_SIZE:
            bess_delta = min(BESS_SIZE - battery_state, _ENERGY_PER_INTERVAL)
            action = "buy"

        elif intent == "sell" and battery_state > 0:
            bess_delta = -min(battery_state, _ENERGY_PER_INTERVAL)
            action = "sell"

        else:
            # Hold: passive self-consumption flows
            if net_local > 0 and battery_state < BESS_SIZE:
                bess_delta = min(net_local, _ENERGY_PER_INTERVAL, BESS_SIZE - battery_state)
            elif net_local < 0 and battery_state > 0:
                bess_delta = max(net_local, -_ENERGY_PER_INTERVAL, -battery_state)
            else:
                bess_delta = 0.0
            action = "hold"

        battery_state += bess_delta

        # Grid position: positive = import (cost), negative = export (revenue)
        grid_net    = -net_local + bess_delta
        grid_import = max(0.0, grid_net)
        grid_export = max(0.0, -grid_net)

        cumulative_cost    += grid_import * rrp / 1000
        cumulative_revenue += grid_export * rrp / 1000

        results.append({
            "time":               time,
            "battery_state":      battery_state,
            "rrp":                rrp,
            "demand":             demand,
            "solar_kw":           solar_kw,
            "load_kw":            load_kw,
            "action":             action,
            "grid_import_kwh":    grid_import,
            "grid_export_kwh":    grid_export,
            "cumulative_cost":    cumulative_cost,
            "cumulative_revenue": cumulative_revenue,
            "cumulative_profit":  cumulative_revenue - cumulative_cost,
        })

    return pd.DataFrame(results)


# ── Fast profit-only simulation for optimisation loops ────────────────────────
def simulate_profit(
    rrp_arr: np.ndarray,
    strategy_fn,
    solar_arr: np.ndarray | None = None,
    load_arr: np.ndarray | None = None,
) -> float:
    """
    Fast profit-only simulation for optimisation loops.

    Parameters
    ----------
    rrp_arr    : 1-D numpy array of RRP values ($/MWh), one per interval.
    strategy_fn: callable(rrp, battery_state, solar_kw, load_kw) -> str
    solar_arr  : 1-D numpy array of solar generation (kW). None = zeros.
    load_arr   : 1-D numpy array of household load (kW).  None = zeros.

    Returns
    -------
    Net profit in dollars (export revenue minus import cost).
    """
    n = len(rrp_arr)
    if solar_arr is None:
        solar_arr = np.zeros(n)
    if load_arr is None:
        load_arr = np.zeros(n)

    battery_state = 0.0
    cost          = 0.0
    revenue       = 0.0

    for i in range(n):
        rrp      = rrp_arr[i]
        solar_kw = solar_arr[i]
        load_kw  = load_arr[i]

        net_local = (solar_kw - load_kw) * INTERVAL_HOURS
        intent    = strategy_fn(rrp, battery_state, solar_kw, load_kw)

        if intent == "buy" and battery_state < BESS_SIZE:
            bess_delta = min(BESS_SIZE - battery_state, _ENERGY_PER_INTERVAL)

        elif intent == "sell" and battery_state > 0:
            bess_delta = -min(battery_state, _ENERGY_PER_INTERVAL)

        else:
            if net_local > 0 and battery_state < BESS_SIZE:
                bess_delta = min(net_local, _ENERGY_PER_INTERVAL, BESS_SIZE - battery_state)
            elif net_local < 0 and battery_state > 0:
                bess_delta = max(net_local, -_ENERGY_PER_INTERVAL, -battery_state)
            else:
                bess_delta = 0.0

        battery_state += bess_delta

        grid_net = -net_local + bess_delta
        if grid_net > 0:
            cost    += grid_net * rrp / 1000
        else:
            revenue += (-grid_net) * rrp / 1000

    return revenue - cost
