"""
Basic BESS trading using a state machine.
No household consumption or generation.
No efficiency losses.
"""

import pandas as pd

from battery_plot import plot_battery_trading
from bess_simulator import BESS_SIZE, simulate
from strategy_state_machine import make_strategy

BUY_THRESHOLD  = 20   # $/MWh
SELL_THRESHOLD = 70   # $/MWh

if __name__ == "__main__":
    price_df   = pd.read_csv("data/price_JAN26.csv")
    strategy   = make_strategy(BUY_THRESHOLD, SELL_THRESHOLD)
    results_df = simulate(price_df, strategy)

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
        title="State Machine Trading -- January 2026",
        output_path="state_machine_jan26.html",
        bess_size=BESS_SIZE,
        buy_threshold=BUY_THRESHOLD,
        sell_threshold=SELL_THRESHOLD,
    )
