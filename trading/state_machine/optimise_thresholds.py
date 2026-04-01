"""
optimise_thresholds.py

Grid-searches buy/sell price thresholds to maximise net profit for state-machine
BESS trading. Supports optional export and import columns for prosumer scenarios.

"""

import numpy as np
import pandas as pd

from utils.bess_simulator import simulate_profit
from state_machine.strategy_state_machine import make_strategy


def optimise_thresholds_brute(
    price_df: pd.DataFrame,
    export_col: str | None = None,
    import_col: str | None = None,
    n_steps: int = 60,
    verbose: bool = True,
    network_tariff: float = 0.0,
) -> tuple[float, float, float]:
    """
    Grid-search buy/sell thresholds to maximise net profit.


    Parameters
    ----------
    price_df       : DataFrame with a 'RRP' column ($/MWh) and optionally
                     export_col and/or import_col columns.
    export_col     : Column name for net export (solar - load) in kW. None = no export data.
    import_col       : Column name for household import (kW).  None = no import data.
    n_steps        : Number of candidate values along each axis of the grid.
    verbose        : Print progress and results to stdout.
    network_tariff : Fixed network charge in cents/kWh on grid imports. Default = 0.0.

    Returns
    -------
    (best_buy_threshold, best_sell_threshold, best_profit)
    """
    rrp_arr = price_df["RRP"].to_numpy(dtype=np.float64)
    rrp_series = price_df["RRP"]

    export_arr = price_df[export_col].to_numpy(dtype=np.float64) if export_col else None
    import_arr = price_df[import_col].to_numpy(dtype=np.float64) if import_col else None

    buy_values = np.linspace(rrp_arr.min(), rrp_series.quantile(0.50), n_steps)
    sell_values = np.linspace(
        rrp_series.quantile(0.50), rrp_series.quantile(0.99), n_steps
    )

    if verbose:
        print(
            f"Grid search: {len(buy_values)} buy x {len(sell_values)} sell thresholds"
        )
        print(f"Buy  range:  ${buy_values[0]:.2f} to ${buy_values[-1]:.2f}")
        print(f"Sell range:  ${sell_values[0]:.2f} to ${sell_values[-1]:.2f}")
        print()

    best_profit = -np.inf
    best_buy_threshold = None
    best_sell_threshold = None

    for buy_t in buy_values:
        for sell_t in sell_values:
            if buy_t >= sell_t:
                continue
            profit = simulate_profit(
                rrp_arr,
                make_strategy(buy_t, sell_t),
                export_arr=export_arr,
                import_arr=import_arr,
                network_tariff=network_tariff,
            )
            if profit > best_profit:
                best_profit = profit
                best_buy_threshold = buy_t
                best_sell_threshold = sell_t

    if verbose:
        print("-- Optimised thresholds ----------------------------")
        print(f"  Buy threshold:   ${best_buy_threshold:.2f}")
        print(f"  Sell threshold:  ${best_sell_threshold:.2f}")
        print(f"  Net profit:      ${best_profit:.2f}")
        print()

    return best_buy_threshold, best_sell_threshold, best_profit
