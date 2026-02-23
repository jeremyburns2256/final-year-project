"""
bess_simulator.py

Physical constants and simulation engine for BESS price-arbitrage trading.
Any strategy plugs in via a callback: strategy_fn(rrp, battery_state) -> 'buy'|'sell'|'hold'.
The simulator enforces SoC bounds and energy-per-interval limits; the strategy only
expresses intent.
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
def simulate(price_df: pd.DataFrame, strategy_fn) -> pd.DataFrame:
    """
    General simulation. Calls strategy_fn(rrp, battery_state) -> 'buy'|'sell'|'hold'
    at every interval. Enforces SoC bounds and energy-per-interval limits.
    Returns a per-interval results DataFrame.

    Required columns in price_df: SETTLEMENTDATE, RRP, TOTALDEMAND.

    Returns a DataFrame with columns:
        time, battery_state, rrp, demand, action,
        cumulative_cost, cumulative_revenue, cumulative_profit
    """
    battery_state = 0.0
    buy_cost      = 0.0
    sell_revenue  = 0.0

    results = []
    for _, row in price_df.iterrows():
        time   = row["SETTLEMENTDATE"]
        rrp    = row["RRP"]
        demand = row["TOTALDEMAND"]

        intent = strategy_fn(rrp, battery_state)

        if intent == "buy" and battery_state < BESS_SIZE:
            delta          = min(BESS_SIZE - battery_state, _ENERGY_PER_INTERVAL)
            battery_state += delta
            buy_cost      += delta * rrp / 1000   # $/MWh -> $
            action         = "buy"

        elif intent == "sell" and battery_state > 0:
            delta          = min(battery_state, _ENERGY_PER_INTERVAL)
            battery_state -= delta
            sell_revenue  += delta * rrp / 1000
            action         = "sell"

        else:
            action = "hold"

        results.append({
            "time":               time,
            "battery_state":      battery_state,
            "rrp":                rrp,
            "demand":             demand,
            "action":             action,
            "cumulative_cost":    buy_cost,
            "cumulative_revenue": sell_revenue,
            "cumulative_profit":  sell_revenue - buy_cost,
        })

    return pd.DataFrame(results)


# ── Fast profit-only simulation for optimisation loops ────────────────────────
def simulate_profit(rrp_arr: np.ndarray, strategy_fn) -> float:
    """
    Fast profit-only simulation for optimisation loops.
    Loops over a numpy array, calls strategy_fn per interval, returns net profit.

    Uses a plain Python loop over a numpy array — fast enough for grid search
    (3 600+ evaluations over ~8 900 intervals each completes in a few seconds).

    Parameters
    ----------
    rrp_arr     : 1-D numpy array of RRP values ($/MWh), one per interval.
    strategy_fn : callable(rrp, battery_state) -> 'buy'|'sell'|'hold'

    Returns
    -------
    Net profit in dollars (sell revenue minus buy cost).
    """
    battery_state = 0.0
    buy_cost      = 0.0
    sell_revenue  = 0.0

    for rrp in rrp_arr:
        intent = strategy_fn(rrp, battery_state)

        if intent == "buy" and battery_state < BESS_SIZE:
            delta          = min(BESS_SIZE - battery_state, _ENERGY_PER_INTERVAL)
            battery_state += delta
            buy_cost      += delta * rrp / 1000

        elif intent == "sell" and battery_state > 0:
            delta          = min(battery_state, _ENERGY_PER_INTERVAL)
            battery_state -= delta
            sell_revenue  += delta * rrp / 1000

    return sell_revenue - buy_cost
