"""
DQN-based BESS Trading

Train and evaluate a Deep Q-Network agent for battery energy storage trading.
Provides similar interface to state_machine_trading.py for easy comparison.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

from dqn.agent import DQNAgent
from dqn.environment import BESSEnvironment
from plotting.battery_plot import plot_battery_trading
from utils.bess_simulator import BESS_SIZE
from utils.data import remove_outliers


# Default parameters
TRAIN_CSV = "data/price_DEC24.csv"
TEST_CSV = "data/price_JAN25.csv"
TRAIN_EXPORT_CSV = "data/export_DEC24.csv"
TRAIN_IMPORT_CSV = "data/import_DEC24.csv"
TEST_EXPORT_CSV = "data/export_JAN25.csv"
TEST_IMPORT_CSV = "data/import_JAN25.csv"
NETWORK_TARIFF = 10.8007  # cents/kWh


def _merge_optional_csv(
    price_df: pd.DataFrame, csv_path: str, col: str
) -> pd.DataFrame:
    """
    Load an optional single-column CSV and left-join it onto price_df by timestamp.

    Same function as in state_machine_trading.py for consistency.
    """
    extra = pd.read_csv(csv_path)[["SETTLEMENTDATE", col]]

    price_indexed = price_df.copy()
    price_indexed["_dt"] = pd.to_datetime(price_df["SETTLEMENTDATE"], dayfirst=True)

    extra_indexed = extra[[col]].copy()
    extra_indexed["_dt"] = pd.to_datetime(extra["SETTLEMENTDATE"], dayfirst=True)

    merged = price_indexed.merge(extra_indexed, on="_dt", how="left").fillna({col: 0.0})
    return merged.drop(columns="_dt")


def train_dqn_agent(
    train_csv=TRAIN_CSV,
    train_export_csv=TRAIN_EXPORT_CSV,
    train_import_csv=TRAIN_IMPORT_CSV,
    network_tariff=NETWORK_TARIFF,
    remove_outliers_training=True,
    lower_quantile=0.15,
    upper_quantile=0.85,
    episodes=50,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    memory_size=10000,
    update_target_every=5,
    hidden_layers=[128, 128, 64],
    dropout_rate=0.2,
    save_model=True,
    model_path="models/dqn_bess.weights.h5",
    verbose=True,
):
    """
    Train a DQN agent on historical trading data.

    Args:
        train_csv: Path to training price CSV
        train_export_csv: Export CSV to merge with training data
        train_import_csv: Import CSV to merge with training data
        network_tariff: Fixed network charge in cents/kWh on grid imports
        remove_outliers_training: Remove price outliers for training
        lower_quantile: Lower quantile for outlier removal
        upper_quantile: Upper quantile for outlier removal
        episodes: Number of training episodes
        learning_rate: Adam optimizer learning rate
        gamma: Discount factor for future rewards
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Multiplicative decay for epsilon
        batch_size: Batch size for experience replay
        memory_size: Maximum size of replay memory
        update_target_every: Update target network every N episodes
        hidden_layers: List of neurons per hidden layer
        dropout_rate: Dropout rate for regularization
        save_model: Whether to save trained model
        model_path: Path to save model weights
        verbose: Print training progress

    Returns:
        Trained DQNAgent
    """
    train_export_col = "EXPORT_KW" if train_export_csv else None
    train_import_col = "IMPORT_KW" if train_import_csv else None

    # Load and merge training data
    train_df = pd.read_csv(train_csv)
    if train_export_csv:
        train_df = _merge_optional_csv(train_df, train_export_csv, "EXPORT_KW")
    if train_import_csv:
        train_df = _merge_optional_csv(train_df, train_import_csv, "IMPORT_KW")

    # Remove outliers if requested
    if remove_outliers_training:
        train_df = remove_outliers(
            train_df,
            column="RRP",
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    if verbose:
        print(f"Training DQN agent on {train_csv.split('/')[-1]}...")
        print(f"Training samples: {len(train_df)}")
        print(f"Episodes: {episodes}")
        print()

    # Initialize environment
    env = BESSEnvironment(
        train_df,
        export_col=train_export_col,
        import_col=train_import_col,
        network_tariff=network_tariff,
    )

    # Initialize agent
    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        memory_size=memory_size,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    )

    # Training history
    episode_profits = []
    episode_revenues = []
    episode_costs = []
    best_profit = -np.inf

    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_profit = 0
        step_count = 0

        # Episode loop
        while True:
            # Agent chooses action
            action = agent.act(state, training=True)

            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            total_profit += reward
            step_count += 1

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train on batch
            loss = agent.replay()

            state = next_state

            if done:
                break

        # Update target network periodically
        if (episode + 1) % update_target_every == 0:
            agent.update_target_model()

        # Record episode statistics
        episode_profits.append(total_profit)
        episode_revenues.append(env.cumulative_revenue)
        episode_costs.append(env.cumulative_cost)

        # Save best model
        if save_model and total_profit > best_profit:
            best_profit = total_profit
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            agent.save(model_path)

        # Print progress
        if verbose:
            avg_loss = np.mean(agent.losses[-step_count:]) if agent.losses else 0
            print(
                f"Episode {episode+1:3d}/{episodes} | "
                f"Profit: ${total_profit:8.2f} | "
                f"Revenue: ${env.cumulative_revenue:8.2f} | "
                f"Cost: ${env.cumulative_cost:8.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Memory: {agent.get_memory_size()}"
            )

    if verbose:
        print()
        print("="*70)
        print("Training Complete")
        print("="*70)
        print(f"Best Profit: ${best_profit:.2f}")
        print(f"Final Epsilon: {agent.epsilon:.3f}")
        print(f"Memory Size: {agent.get_memory_size()}/{memory_size}")
        if save_model:
            print(f"Model saved to: {model_path}")
        print()

    return agent, {
        'episode_profits': episode_profits,
        'episode_revenues': episode_revenues,
        'episode_costs': episode_costs,
    }


def evaluate_dqn_agent(
    agent,
    test_csv=TEST_CSV,
    test_export_csv=TEST_EXPORT_CSV,
    test_import_csv=TEST_IMPORT_CSV,
    network_tariff=NETWORK_TARIFF,
    remove_outliers_eval=False,
    lower_quantile=0.15,
    upper_quantile=0.85,
    verbose=True,
    plot=True,
    plot_title="DQN Trading",
    plot_output_path=None,
):
    """
    Evaluate a trained DQN agent on test data.

    Args:
        agent: Trained DQNAgent
        test_csv: Path to test price CSV
        test_export_csv: Export CSV to merge with test data
        test_import_csv: Import CSV to merge with test data
        network_tariff: Fixed network charge in cents/kWh on grid imports
        remove_outliers_eval: Remove price outliers for evaluation
        lower_quantile: Lower quantile for outlier removal
        upper_quantile: Upper quantile for outlier removal
        verbose: Print evaluation results
        plot: Whether to generate plot
        plot_title: Title for the plot
        plot_output_path: Path to save the plot

    Returns:
        Dictionary with results_df and metrics
    """
    test_export_col = "EXPORT_KW" if test_export_csv else None
    test_import_col = "IMPORT_KW" if test_import_csv else None

    # Load and merge test data
    test_df = pd.read_csv(test_csv)
    if test_export_csv:
        test_df = _merge_optional_csv(test_df, test_export_csv, "EXPORT_KW")
    if test_import_csv:
        test_df = _merge_optional_csv(test_df, test_import_csv, "IMPORT_KW")

    # Remove outliers if requested
    if remove_outliers_eval:
        test_df = remove_outliers(
            test_df,
            column="RRP",
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
        )

    if verbose:
        print(f"Evaluating DQN agent on {test_csv.split('/')[-1]}...")

    # Initialize environment
    env = BESSEnvironment(
        test_df,
        export_col=test_export_col,
        import_col=test_import_col,
        network_tariff=network_tariff,
    )

    # Evaluation loop (greedy policy, no exploration)
    state = env.reset()
    while True:
        action = agent.act(state, training=False)
        next_state, reward, done, info = env.step(action)
        state = next_state

        if done:
            break

    # Get results
    results_df = env.get_episode_dataframe()

    # Metrics
    metrics = {
        "final_battery_state": env.battery_state,
        "total_cost": env.cumulative_cost,
        "total_revenue": env.cumulative_revenue,
        "net_profit": env.cumulative_profit,
    }

    # Verbose summary
    if verbose:
        mode_parts = []
        if test_export_csv:
            mode_parts.append("export")
        if test_import_csv:
            mode_parts.append("import")
        mode_str = f" [{', '.join(mode_parts)}]" if mode_parts else " [arbitrage only]"

        print(f"\n{'-'*40}")
        print(f"  DQN Results ({test_csv.split('/')[-1]}){mode_str}")
        print(f"{'-'*40}")
        print(f"Final Battery State: {metrics['final_battery_state']:.2f} kWh")
        print(f"Total Grid Cost:     ${metrics['total_cost']:.2f}")
        print(f"Total Grid Revenue:  ${metrics['total_revenue']:.2f}")
        print(f"Net Profit:          ${metrics['net_profit']:.2f}")

    # Plot
    if plot:
        if plot_output_path is None:
            plot_output_path = f"plots/dqn_{test_csv.split('/')[-1].replace('.csv', '.html')}"

        # Convert to format expected by plot_battery_trading
        plot_df = pd.DataFrame({
            'time': results_df['time'],
            'battery_state': results_df['battery_state'],
            'rrp': results_df['rrp'],
            'export_kw': results_df['export_kw'],
            'import_kw': results_df['import_kw'],
            'grid_import_kwh': results_df.apply(lambda x: max(0, x['grid_net']), axis=1),
            'grid_export_kwh': results_df.apply(lambda x: max(0, -x['grid_net']), axis=1),
            'cumulative_cost': env.cumulative_cost - results_df['reward'].iloc[::-1].cumsum().iloc[::-1],
            'cumulative_revenue': results_df['reward'].where(results_df['reward'] > 0, 0).cumsum(),
            'cumulative_profit': results_df['reward'].cumsum(),
        })

        plot_battery_trading(
            plot_df,
            title=plot_title,
            output_path=plot_output_path,
            bess_size=BESS_SIZE,
        )

    return {
        "results_df": results_df,
        "metrics": metrics,
    }


def run_dqn_trading(
    train_csv=TRAIN_CSV,
    test_csv=TEST_CSV,
    train_export_csv=TRAIN_EXPORT_CSV,
    train_import_csv=TRAIN_IMPORT_CSV,
    test_export_csv=TEST_EXPORT_CSV,
    test_import_csv=TEST_IMPORT_CSV,
    network_tariff=NETWORK_TARIFF,
    remove_outliers_training=True,
    remove_outliers_eval=False,
    lower_quantile=0.15,
    upper_quantile=0.85,
    episodes=50,
    learning_rate=0.0005,
    gamma=0.99,
    epsilon=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    batch_size=64,
    memory_size=10000,
    update_target_every=5,
    hidden_layers=[128, 128, 64],
    dropout_rate=0.2,
    save_model=True,
    model_path="models/dqn_bess.weights.h5",
    load_model=False,
    verbose=True,
    plot=True,
    plot_title="DQN Trading",
    plot_output_path=None,
):
    """
    Complete DQN training and evaluation pipeline.

    Mirrors the interface of run_trading_simulation() from state_machine_trading.py
    for easy comparison.

    Args:
        train_csv: Path to training price CSV
        test_csv: Path to test price CSV
        train_export_csv: Training export CSV
        train_import_csv: Training import CSV
        test_export_csv: Test export CSV
        test_import_csv: Test import CSV
        network_tariff: Network charge in cents/kWh
        remove_outliers_training: Remove outliers from training data
        remove_outliers_eval: Remove outliers from evaluation data
        lower_quantile: Lower quantile for outlier removal
        upper_quantile: Upper quantile for outlier removal
        episodes: Number of training episodes
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay rate
        batch_size: Batch size
        memory_size: Replay memory size
        update_target_every: Update target network every N episodes
        hidden_layers: Hidden layer sizes
        dropout_rate: Dropout rate
        save_model: Save trained model
        model_path: Path to save/load model
        load_model: Load existing model instead of training
        verbose: Print verbose output
        plot: Generate plot
        plot_title: Plot title
        plot_output_path: Path to save plot

    Returns:
        Dictionary with results_df and metrics
    """
    if load_model:
        # Load existing model
        if verbose:
            print(f"Loading model from {model_path}...")

        # Initialize a dummy environment to get state/action sizes
        dummy_df = pd.read_csv(train_csv).head(100)
        dummy_env = BESSEnvironment(dummy_df, network_tariff=network_tariff)

        agent = DQNAgent(
            state_size=dummy_env.get_state_size(),
            action_size=dummy_env.get_action_size(),
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon_min,  # No exploration for loaded model
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
        )
        agent.load(model_path)

        training_history = None
    else:
        # Train new model
        agent, training_history = train_dqn_agent(
            train_csv=train_csv,
            train_export_csv=train_export_csv,
            train_import_csv=train_import_csv,
            network_tariff=network_tariff,
            remove_outliers_training=remove_outliers_training,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile,
            episodes=episodes,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            batch_size=batch_size,
            memory_size=memory_size,
            update_target_every=update_target_every,
            hidden_layers=hidden_layers,
            dropout_rate=dropout_rate,
            save_model=save_model,
            model_path=model_path,
            verbose=verbose,
        )

    # Evaluate on test data
    results = evaluate_dqn_agent(
        agent=agent,
        test_csv=test_csv,
        test_export_csv=test_export_csv,
        test_import_csv=test_import_csv,
        network_tariff=network_tariff,
        remove_outliers_eval=remove_outliers_eval,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        verbose=verbose,
        plot=plot,
        plot_title=plot_title,
        plot_output_path=plot_output_path,
    )

    results['agent'] = agent
    results['training_history'] = training_history

    return results


def main():
    """Run DQN trading with default settings."""
    run_dqn_trading()


if __name__ == "__main__":
    main()
