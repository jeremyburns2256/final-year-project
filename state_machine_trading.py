"""
Basic BESS trading using a state machine.
Supports optional export and import data via separate CSV files.
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
NETWORK_TARIFF = 10.8007  # cents/kWh on grid imports. AusGrid EA010 Tarrif Code
OPTIMISE_THRESHOLDS = True
REMOVE_OUTLIERS_OPTIMISATION = True
REMOVE_OUTLIERS_SIMULATION = False
TRAIN_CSV = "data/price_DEC24.csv"
TEST_CSV = "data/price_JAN25.csv"
TRAIN_EXPORT_CSV = "data/export_DEC24.csv"
TRAIN_IMPORT_CSV = "data/import_DEC24.csv"
TEST_EXPORT_CSV = "data/export_JAN25.csv"
TEST_IMPORT_CSV = "data/import_JAN25.csv"


def _merge_optional_csv(
    price_df: pd.DataFrame, csv_path: str, col: str
) -> pd.DataFrame:
    """
    Load an optional single-column CSV and left-join it onto price_df by timestamp.

    The join is done on parsed datetimes, so timestamp string formatting differences
    between files (e.g. zero-padded vs not) are handled automatically. The original
    SETTLEMENTDATE strings in price_df are preserved.
    """
    extra = pd.read_csv(csv_path)[["SETTLEMENTDATE", col]]

    price_indexed = price_df.copy()
    price_indexed["_dt"] = pd.to_datetime(price_df["SETTLEMENTDATE"], dayfirst=True)

    extra_indexed = extra[[col]].copy()
    extra_indexed["_dt"] = pd.to_datetime(extra["SETTLEMENTDATE"], dayfirst=True)

    merged = price_indexed.merge(extra_indexed, on="_dt", how="left").fillna({col: 0.0})
    return merged.drop(columns="_dt")


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
    train_export_csv=TRAIN_EXPORT_CSV,
    train_import_csv=TRAIN_IMPORT_CSV,
    test_export_csv=TEST_EXPORT_CSV,
    test_import_csv=TEST_IMPORT_CSV,
    network_tariff=NETWORK_TARIFF,
    verbose=True,
    plot=True,
    plot_title="State Machine Trading",
    plot_output_path=None,
):
    """
    Run the state machine trading simulation with configurable parameters.

    Export and import CSVs are specified separately for training and test so that
    the optimiser trains on the correct period's generation/consumption profile.

    NOTE: Export data (B1) represents net export = solar - load, not solar generation.

    Args:
        train_csv: Path to training price CSV (for threshold optimisation)
        test_csv: Path to test price CSV (for evaluation)
        buy_threshold: Initial buy threshold in $/MWh
        sell_threshold: Initial sell threshold in $/MWh
        optimise_thresholds: Whether to optimise thresholds on training data
        remove_outliers_optimisation: Remove price outliers for optimisation
        remove_outliers_simulation: Remove price outliers for simulation
        lower_quantile: Lower quantile for outlier removal
        upper_quantile: Upper quantile for outlier removal
        n_steps: Number of steps for brute-force optimisation
        train_export_csv: Export CSV to merge with training data. None = no export data.
        train_import_csv:  import CSV to merge with training data. None = no import.
        test_export_csv:  Export CSV to merge with test data. None = no export data.
        test_import_csv:   import CSV to merge with test data. None = no import.
        network_tariff: Fixed network charge in cents/kWh on grid imports. Default = 0.0.
        verbose: Print verbose output
        plot: Whether to generate plot
        plot_title: Title for the plot
        plot_output_path: Path to save the plot (auto-generated if None)

    Returns:
        dict with keys:
            results_df    — per-interval simulation DataFrame
            buy_threshold — final buy threshold used
            sell_threshold — final sell threshold used

    Example — full prosumer run:
        run_trading_simulation(
            train_export_csv="data/export_JAN26.csv",
            train_import_csv="data/import_JAN26.csv",
            test_export_csv="data/export_FEB26.csv",
            test_import_csv="data/import_FEB26.csv",
        )
    """
    train_export_col = "EXPORT_KW" if train_export_csv else None
    train_import_col = "IMPORT_KW" if train_import_csv else None
    test_export_col = "EXPORT_KW" if test_export_csv else None
    test_import_col = "IMPORT_KW" if test_import_csv else None

    # ── import and merge training data ──────────────────────────────────────────
    train_df = pd.read_csv(train_csv)
    if train_export_csv:
        train_df = _merge_optional_csv(train_df, train_export_csv, "EXPORT_KW")
    if train_import_csv:
        train_df = _merge_optional_csv(train_df, train_import_csv, "IMPORT_KW")

    # ── Load and merge test data ───────────────────────────────────────────────
    test_df = pd.read_csv(test_csv)
    if test_export_csv:
        test_df = _merge_optional_csv(test_df, test_export_csv, "EXPORT_KW")
    if test_import_csv:
        test_df = _merge_optional_csv(test_df, test_import_csv, "IMPORT_KW")

    final_buy_threshold = buy_threshold
    final_sell_threshold = sell_threshold

    # ── Optimise thresholds on training data ──────────────────────────────────
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
            export_col=train_export_col,
            import_col=train_import_col,
            n_steps=n_steps,
            verbose=verbose,
            network_tariff=network_tariff,
        )

        if verbose:
            print(
                f"Optimized thresholds: Buy=${final_buy_threshold:.2f}, Sell=${final_sell_threshold:.2f}\n"
            )

    # ── Evaluate on test data ─────────────────────────────────────────────────
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
    results_df = simulate(
        test_data,
        strategy,
        export_col=test_export_col,
        import_col=test_import_col,
        network_tariff=network_tariff,
    )

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics = {
        "final_battery_state": results_df["battery_state"].iloc[-1],
        "total_cost": results_df["cumulative_cost"].iloc[-1],
        "total_revenue": results_df["cumulative_revenue"].iloc[-1],
        "net_profit": results_df["cumulative_profit"].iloc[-1],
    }

    # ── Verbose summary ───────────────────────────────────────────────────────
    if verbose:
        mode_parts = []
        if test_export_csv:
            mode_parts.append("export")
        if test_import_csv:
            mode_parts.append("import")
        mode_str = f" [{', '.join(mode_parts)}]" if mode_parts else " [arbitrage only]"

        print(f"\n{'-'*40}")
        print(f"  State Machine Results ({test_csv.split('/')[-1]}){mode_str}")
        print(f"{'-'*40}")
        print(f"Final Battery State: {metrics['final_battery_state']:.2f} kWh")
        print(f"Total Grid Cost:     ${metrics['total_cost']:.2f}")
        print(f"Total Grid Revenue:  ${metrics['total_revenue']:.2f}")
        print(f"Net Profit:          ${metrics['net_profit']:.2f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
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
