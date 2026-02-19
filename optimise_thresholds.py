"""
optimise_thresholds.py

Grid-searches buy/sell price thresholds to maximise net profit for state-machine
BESS trading on the January 2026 NEM data, then plots the optimal result.
"""

import numpy as np
import pandas as pd

from battery_plot import plot_battery_trading
from bess_simulator import BESS_SIZE, simulate, simulate_profit
from strategy_state_machine import make_strategy

# Original hardcoded thresholds (for comparison)
ORIGINAL_BUY_THRESHOLD  = 20   # $/MWh
ORIGINAL_SELL_THRESHOLD = 70   # $/MWh

# ── Load data ─────────────────────────────────────────────────────────────────
price_df   = pd.read_csv("data/price_JAN26.csv")
rrp_arr    = price_df["RRP"].to_numpy(dtype=np.float64)
rrp_series = price_df["RRP"]

# ── Grid search bounds ────────────────────────────────────────────────────────
buy_values  = np.linspace(rrp_arr.min(),             rrp_series.quantile(0.50), 60)
sell_values = np.linspace(rrp_series.quantile(0.50), rrp_series.quantile(0.99), 60)

print(f"Grid search: {len(buy_values)} buy x {len(sell_values)} sell thresholds")
print(f"Buy  range:  ${buy_values[0]:.2f} to ${buy_values[-1]:.2f}")
print(f"Sell range:  ${sell_values[0]:.2f} to ${sell_values[-1]:.2f}")
print()

# ── Run grid search ───────────────────────────────────────────────────────────
best_profit         = -np.inf
best_buy_threshold  = None
best_sell_threshold = None

for buy_t in buy_values:
    for sell_t in sell_values:
        if buy_t >= sell_t:
            continue
        profit = simulate_profit(rrp_arr, make_strategy(buy_t, sell_t))
        if profit > best_profit:
            best_profit         = profit
            best_buy_threshold  = buy_t
            best_sell_threshold = sell_t

# ── Original thresholds result (for comparison) ───────────────────────────────
original_profit = simulate_profit(rrp_arr, make_strategy(ORIGINAL_BUY_THRESHOLD, ORIGINAL_SELL_THRESHOLD))

# ── Print results ─────────────────────────────────────────────────────────────
print("-- Original thresholds -----------------------------")
print(f"  Buy threshold:   ${ORIGINAL_BUY_THRESHOLD}")
print(f"  Sell threshold:  ${ORIGINAL_SELL_THRESHOLD}")
print(f"  Net profit:      ${original_profit:.2f}")
print()
print("-- Optimised thresholds ----------------------------")
print(f"  Buy threshold:   ${best_buy_threshold:.2f}")
print(f"  Sell threshold:  ${best_sell_threshold:.2f}")
print(f"  Net profit:      ${best_profit:.2f}")
print(f"  Improvement:     ${best_profit - original_profit:.2f}")
print()

# ── Full-detail simulation for best params -> plot ────────────────────────────
results_df = simulate(price_df, make_strategy(best_buy_threshold, best_sell_threshold))

print(f"Final battery state: {results_df['battery_state'].iloc[-1]:.2f} kWh")
print(f"Total buy cost:      ${results_df['cumulative_cost'].iloc[-1]:.2f}")
print(f"Total sell revenue:  ${results_df['cumulative_revenue'].iloc[-1]:.2f}")
buy_count  = (results_df["action"] == "buy").sum()
sell_count = (results_df["action"] == "sell").sum()
hold_count = (results_df["action"] == "hold").sum()
print(f"Buy / Sell / Hold:   {buy_count} / {sell_count} / {hold_count}")
print()

plot_battery_trading(
    results_df,
    title=(
        f"Optimised State Machine Trading -- January 2026  "
        f"(buy<${best_buy_threshold:.1f}, sell>=${best_sell_threshold:.1f})"
    ),
    output_path="optimised_jan26.html",
    bess_size=BESS_SIZE,
    buy_threshold=best_buy_threshold,
    sell_threshold=best_sell_threshold,
)
