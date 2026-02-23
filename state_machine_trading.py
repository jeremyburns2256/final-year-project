"""
Basic BESS trading using a state machine.
No household consumption or generation.
No efficiency losses.
"""

import pandas as pd

from plotting.battery_plot import plot_battery_trading
from utils.bess_simulator import BESS_SIZE, simulate
from state_machine.optimise_thresholds import optimise_thresholds_brute
from state_machine.strategy_state_machine import make_strategy
from utils.data import remove_outliers

BUY_THRESHOLD  = 20   # $/MWh
SELL_THRESHOLD = 70   # $/MWh
OPTIMISE_THRESHOLDS = True
REMOVE_OUTLIERS_OPTIMISATION = True
REMOVE_OUTLIERS_SIMULATION = False
CSV_PATH = "data/price_JAN26.csv"


if __name__ == "__main__":
    price_df   = pd.read_csv(CSV_PATH)
    price_outliers_removed = remove_outliers(price_df, column="RRP", lower_quantile=0.15, upper_quantile=0.85)
    

    if OPTIMISE_THRESHOLDS:
        BUY_THRESHOLD, SELL_THRESHOLD, _ = optimise_thresholds_brute(
            price_outliers_removed if REMOVE_OUTLIERS_OPTIMISATION else price_df,
            n_steps=60,
            verbose=True,
        )

    strategy   = make_strategy(BUY_THRESHOLD, SELL_THRESHOLD)
    results_df = simulate(price_df if not REMOVE_OUTLIERS_SIMULATION else price_outliers_removed, strategy)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"Final Battery State: {results_df['battery_state'].iloc[-1]:.2f} kWh")
    print(f"Total Buy Cost:      ${results_df['cumulative_cost'].iloc[-1]:.2f}")
    print(f"Total Sell Revenue:  ${results_df['cumulative_revenue'].iloc[-1]:.2f}")
    print(f"Net Profit:          ${results_df['cumulative_profit'].iloc[-1]:.2f}")
    print()
    buy_count  = (results_df["action"] == "buy").sum()
    sell_count = (results_df["action"] == "sell").sum()
    hold_count = (results_df["action"] == "hold").sum()
    print(f"Buy Actions:  {buy_count}")
    print(f"Sell Actions: {sell_count}")
    print(f"Hold Actions: {hold_count}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_battery_trading(
        results_df,
        title="State Machine Trading",
        output_path=f"plots/state_machine_{CSV_PATH.split('/')[-1].replace('.csv', '.html')}",
        bess_size=BESS_SIZE,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
    )
