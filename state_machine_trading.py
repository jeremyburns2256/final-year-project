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

BUY_THRESHOLD = 20  # $/MWh
SELL_THRESHOLD = 70  # $/MWh
OPTIMISE_THRESHOLDS = True
REMOVE_OUTLIERS_OPTIMISATION = True
REMOVE_OUTLIERS_SIMULATION = False
TRAIN_CSV = "data/price_JAN26.csv"
TEST_CSV = "data/price_FEB26.csv"


def run_trading_simulation(
    train_csv=TRAIN_CSV,
    test_csv=TEST_CSV,
    buy_threshold=BUY_THRESHOLD,
    sell_threshold=SELL_THRESHOLD,
    optimise_thresholds=OPTIMISE_THRESHOLDS,
    remove_outliers_optimisation=REMOVE_OUTLIERS_OPTIMISATION,
    remove_outliers_simulation=REMOVE_OUTLIERS_SIMULATION,
    lower_quantile=0.15,
    upper_quantile=0.85,
    n_steps=60,
    verbose=True,
    plot=True,
    plot_title="State Machine Trading",
    plot_output_path=None,
):
    """
    Run the state machine trading simulation with configurable parameters.

    Args:
        train_csv: Path to training CSV (for threshold optimization)
        test_csv: Path to test CSV (for evaluation)
        buy_threshold: Initial buy threshold in $/MWh
        sell_threshold: Initial sell threshold in $/MWh
        optimise_thresholds: Whether to optimise thresholds on training data
        remove_outliers_optimisation: Remove outliers for optimisation
        remove_outliers_simulation: Remove outliers for simulation
        lower_quantile: Lower quantile for outlier removal
        upper_quantile: Upper quantile for outlier removal
        n_steps: Number of steps for brute force optimisation
        verbose: Print verbose output
        plot: Whether to generate plot
        plot_title: Title for the plot
        plot_output_path: Path to save the plot (auto-generated if None)

    Returns:
        dict: Dictionary containing:
            - results_df: Simulation results DataFrame
            - buy_threshold: Final buy threshold used
            - sell_threshold: Final sell threshold used
            - metrics: Dict with profit, costs, revenue, action counts
    """
    # Load training and test data
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    final_buy_threshold = buy_threshold
    final_sell_threshold = sell_threshold

    # Optimize thresholds on training data
    if optimise_thresholds:
        if verbose:
            print(f"Optimizing thresholds on {train_csv.split('/')[-1]}...")

        train_data = train_df
        if remove_outliers_optimisation:
            train_data = remove_outliers(
                train_df,
                column="RRP",
                lower_quantile=lower_quantile,
                upper_quantile=upper_quantile,
            )

        final_buy_threshold, final_sell_threshold, _ = optimise_thresholds_brute(
            train_data,
            n_steps=n_steps,
            verbose=verbose,
        )

        if verbose:
            print(f"Optimized thresholds: Buy=${final_buy_threshold:.2f}, Sell=${final_sell_threshold:.2f}\n")

    # Evaluate on test data
    if verbose:
        print(f"Evaluating on {test_csv.split('/')[-1]}...")

    test_data = test_df
    if remove_outliers_simulation:
        test_data = remove_outliers(
            test_df,
            column="RRP",
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    strategy = make_strategy(final_buy_threshold, final_sell_threshold)
    results_df = simulate(test_data, strategy)

    # Calculate metrics
    metrics = {
        "final_battery_state": results_df["battery_state"].iloc[-1],
        "total_cost": results_df["cumulative_cost"].iloc[-1],
        "total_revenue": results_df["cumulative_revenue"].iloc[-1],
        "net_profit": results_df["cumulative_profit"].iloc[-1],
        "buy_count": (results_df["action"] == "buy").sum(),
        "sell_count": (results_df["action"] == "sell").sum(),
        "hold_count": (results_df["action"] == "hold").sum(),
    }

    # Print summary if verbose
    if verbose:
        print(f"\n{'─'*40}")
        print(f"  State Machine Results ({test_csv.split('/')[-1]})")
        print(f"{'─'*40}")
        print(f"Final Battery State: {metrics['final_battery_state']:.2f} kWh")
        print(f"Total Buy Cost:      ${metrics['total_cost']:.2f}")
        print(f"Total Sell Revenue:  ${metrics['total_revenue']:.2f}")
        print(f"Net Profit:          ${metrics['net_profit']:.2f}")
        print()
        print(f"Buy Actions:  {metrics['buy_count']}")
        print(f"Sell Actions: {metrics['sell_count']}")
        print(f"Hold Actions: {metrics['hold_count']}")

    # Generate plot
    if plot:
        if plot_output_path is None:
            plot_output_path = f"plots/state_machine_{test_csv.split('/')[-1].replace('.csv', '.html')}"

        plot_battery_trading(
            results_df,
            title=plot_title,
            output_path=plot_output_path,
            bess_size=BESS_SIZE,
            buy_threshold=final_buy_threshold,
            sell_threshold=final_sell_threshold,
        )

    return {
        "results_df": results_df,
        "buy_threshold": final_buy_threshold,
        "sell_threshold": final_sell_threshold,
        "metrics": metrics,
    }


def main():
    """Run the state machine trading simulation with default settings."""
    run_trading_simulation()


if __name__ == "__main__":
    main()
