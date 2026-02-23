"""
dqn_arbitrage.py

Deep Q-Network (DQN) approach to battery arbitrage.
- Directly optimizes for profit (no proxy labels)
- Learns Q(state, action) = expected cumulative profit
- State includes battery level + price features
- Uses experience replay and target network for stability
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import random

from utils.bess_simulator import BESS_SIZE, simulate

# ── Config ─────────────────────────────────────────────────────────────────────
TRAIN_CSV = "data/price_JAN26.csv"
TEST_CSV = "data/price_FEB26.csv"
N_LAGS = 12  # how many past prices to use as features
EPISODES = 30  # number of training episodes (reduced for faster training)
BATCH_SIZE = 64
GAMMA = 0.95  # discount factor for future rewards
EPSILON_START = 1.0  # initial exploration rate
EPSILON_END = 0.01  # final exploration rate
EPSILON_DECAY = 0.95  # decay rate per episode (faster decay)
LEARNING_RATE = 0.001
MEMORY_SIZE = 5000  # experience replay buffer size (reduced)
TARGET_UPDATE_FREQ = 5  # update target network every N episodes (more frequent)
SEED = 42


# ── Battery Environment ────────────────────────────────────────────────────────
class BatteryEnv:
    """Battery trading environment with profit-based rewards."""

    def __init__(self, price_df, n_lags=12):
        self.prices = price_df["RRP"].values.astype(float)
        self.n_lags = n_lags
        self.reset()

    def reset(self):
        """Reset environment to initial state."""
        self.timestep = self.n_lags  # start after lag window
        self.battery_state = BESS_SIZE / 2  # start at 50%
        self.total_profit = 0.0
        self.total_cost = 0.0
        self.total_revenue = 0.0
        return self._get_state()

    def _get_state(self):
        """
        Get current state vector:
        - Battery level (normalized 0-1)
        - Current price (normalized)
        - Last n_lags prices (normalized)
        - Rolling mean and std
        """
        # Battery level (0-1)
        battery_norm = self.battery_state / BESS_SIZE

        # Price window
        price_window = self.prices[self.timestep - self.n_lags : self.timestep]

        # Normalize prices (simple min-max on window)
        price_min = price_window.min()
        price_max = price_window.max()
        price_range = price_max - price_min if price_max > price_min else 1.0

        prices_norm = (price_window - price_min) / price_range
        current_price_norm = (self.prices[self.timestep] - price_min) / price_range

        # Stats
        mean_norm = prices_norm.mean()
        std_norm = prices_norm.std()

        # State vector
        state = np.array([
            battery_norm,
            current_price_norm,
            *prices_norm,
            mean_norm,
            std_norm
        ], dtype=np.float32)

        return state

    def step(self, action):
        """
        Take action and return (next_state, reward, done).

        Actions: 0=buy, 1=hold, 2=sell
        """
        current_price = self.prices[self.timestep]
        reward = 0.0

        # Execute action
        if action == 0:  # buy
            if self.battery_state < BESS_SIZE:
                # Buy 1 kWh
                cost = current_price / 1000  # convert $/MWh to $/kWh
                self.battery_state = min(self.battery_state + 1, BESS_SIZE)
                self.total_cost += cost
                reward = -cost  # negative reward for spending

        elif action == 2:  # sell
            if self.battery_state > 0:
                # Sell 1 kWh
                revenue = current_price / 1000  # convert $/MWh to $/kWh
                self.battery_state = max(self.battery_state - 1, 0)
                self.total_revenue += revenue
                reward = revenue  # positive reward for earning

        # Update profit
        self.total_profit = self.total_revenue - self.total_cost

        # Move to next timestep
        self.timestep += 1
        done = self.timestep >= len(self.prices) - 1

        next_state = self._get_state() if not done else None

        return next_state, reward, done

    def get_state_size(self):
        """Return size of state vector."""
        return 2 + self.n_lags + 2  # battery + current_price + lags + mean + std

    def get_action_size(self):
        """Return number of actions."""
        return 3  # buy, hold, sell


# ── DQN Agent ──────────────────────────────────────────────────────────────────
class DQNAgent:
    """Deep Q-Network agent with experience replay."""

    def __init__(self, state_size, action_size, learning_rate=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Experience replay memory
        self.memory = deque(maxlen=MEMORY_SIZE)

        # Exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY

        # Q-Networks
        self.model = self._build_model(learning_rate)
        self.target_model = self._build_model(learning_rate)
        self.update_target_model()

    def _build_model(self, learning_rate):
        """Build neural network for Q-function approximation."""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')  # Q-values
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse'
        )

        return model

    def update_target_model(self):
        """Copy weights from model to target_model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state
            training: If True, use epsilon-greedy; if False, use greedy
        """
        if training and np.random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Predict Q-values
        state_batch = np.array([state])
        q_values = self.model.predict(state_batch, verbose=0)[0]

        return np.argmax(q_values)

    def replay(self, batch_size):
        """Train on a batch of experiences from memory."""
        if len(self.memory) < batch_size:
            return

        # Sample random batch
        batch = random.sample(self.memory, batch_size)

        states = np.array([exp[0] for exp in batch])
        actions = np.array([exp[1] for exp in batch])
        rewards = np.array([exp[2] for exp in batch])
        next_states = np.array([exp[3] if exp[3] is not None else np.zeros(self.state_size) for exp in batch])
        dones = np.array([exp[4] for exp in batch])

        # Predict Q-values for current states
        q_values = self.model.predict(states, verbose=0)

        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values using Bellman equation
        for i in range(batch_size):
            if dones[i]:
                q_values[i][actions[i]] = rewards[i]
            else:
                q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train model
        self.model.fit(states, q_values, epochs=1, verbose=0)

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ── Training Function ──────────────────────────────────────────────────────────
def train_dqn(
    train_csv=TRAIN_CSV,
    n_lags=N_LAGS,
    episodes=EPISODES,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    target_update_freq=TARGET_UPDATE_FREQ,
    seed=SEED,
    verbose=True
):
    """
    Train DQN agent on price data.

    Returns:
        trained agent and training history
    """
    # Set seeds
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)

    # Load data and create environment
    train_df = pd.read_csv(train_csv)
    env = BatteryEnv(train_df, n_lags=n_lags)

    # Create agent
    state_size = env.get_state_size()
    action_size = env.get_action_size()
    agent = DQNAgent(state_size, action_size, learning_rate=learning_rate, gamma=gamma)

    # Training history
    history = {
        'episode': [],
        'profit': [],
        'epsilon': [],
        'avg_profit': []
    }

    if verbose:
        print(f"Training DQN on {train_csv.split('/')[-1]}")
        print(f"State size: {state_size}, Action size: {action_size}")
        print(f"Episodes: {episodes}, Batch size: {batch_size}")
        print(f"Gamma: {gamma}, Learning rate: {learning_rate}\n")

    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        # Run episode
        while True:
            action = agent.act(state, training=True)
            next_state, reward, done = env.step(action)

            total_reward += reward

            # Store experience
            agent.remember(state, action, reward, next_state, done)

            # Train on batch
            if len(agent.memory) >= batch_size:
                agent.replay(batch_size)

            if done:
                break

            state = next_state

        # Update target network periodically
        if (episode + 1) % target_update_freq == 0:
            agent.update_target_model()

        # Decay exploration
        agent.decay_epsilon()

        # Record history
        history['episode'].append(episode)
        history['profit'].append(env.total_profit)
        history['epsilon'].append(agent.epsilon)

        # Calculate moving average
        recent_profits = history['profit'][-10:]
        avg_profit = np.mean(recent_profits)
        history['avg_profit'].append(avg_profit)

        if verbose and (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{episodes} | "
                  f"Profit: ${env.total_profit:.2f} | "
                  f"Avg(10): ${avg_profit:.2f} | "
                  f"ε: {agent.epsilon:.3f}")

    return agent, history, env


# ── Evaluation Function ────────────────────────────────────────────────────────
def evaluate_dqn(agent, test_csv, n_lags=N_LAGS, verbose=True):
    """
    Evaluate trained DQN agent on test data.

    Returns:
        results dictionary
    """
    test_df = pd.read_csv(test_csv)
    env = BatteryEnv(test_df, n_lags=n_lags)

    state = env.reset()
    actions_taken = []

    # Run episode without exploration
    while True:
        action = agent.act(state, training=False)  # greedy policy
        actions_taken.append(action)
        next_state, reward, done = env.step(action)

        if done:
            break

        state = next_state

    # Map actions to strings
    ACTION_MAP = {0: "buy", 1: "hold", 2: "sell"}

    # Count actions
    buy_count = actions_taken.count(0)
    hold_count = actions_taken.count(1)
    sell_count = actions_taken.count(2)

    if verbose:
        print(f"\n{'─'*40}")
        print(f"  DQN Results ({test_csv.split('/')[-1]})")
        print(f"{'─'*40}")
        print(f"  Buy cost:    ${env.total_cost:.2f}")
        print(f"  Sell revenue:${env.total_revenue:.2f}")
        print(f"  Net profit:  ${env.total_profit:.2f}")
        print(f"  Actions — buy:{buy_count}  sell:{sell_count}  hold:{hold_count}")

    return {
        "profit": env.total_profit,
        "cost": env.total_cost,
        "revenue": env.total_revenue,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "actions": actions_taken,
    }


# ── Main Callable Function ─────────────────────────────────────────────────────
def run_dqn_arbitrage(
    train_csv=TRAIN_CSV,
    test_csv=TEST_CSV,
    n_lags=N_LAGS,
    episodes=EPISODES,
    batch_size=BATCH_SIZE,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    target_update_freq=TARGET_UPDATE_FREQ,
    seed=SEED,
    verbose=True
):
    """
    Run DQN battery arbitrage: train on training data, evaluate on test data.

    Args:
        train_csv: Path to training data CSV
        test_csv: Path to test data CSV
        n_lags: Number of past prices to use as features
        episodes: Number of training episodes
        batch_size: Training batch size
        gamma: Discount factor for future rewards
        learning_rate: Learning rate for optimizer
        target_update_freq: Update target network every N episodes
        seed: Random seed for reproducibility
        verbose: Print detailed output

    Returns:
        dict: Dictionary containing:
            - agent: Trained DQN agent
            - train_history: Training history
            - test_results: Test evaluation results
            - metrics: Summary metrics
    """
    # Train
    agent, train_history, _ = train_dqn(
        train_csv=train_csv,
        n_lags=n_lags,
        episodes=episodes,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        target_update_freq=target_update_freq,
        seed=seed,
        verbose=verbose
    )

    # Evaluate on test set
    test_results = evaluate_dqn(agent, test_csv, n_lags=n_lags, verbose=verbose)

    return {
        "agent": agent,
        "train_history": train_history,
        "test_results": test_results,
        "metrics": {
            "test_profit": test_results["profit"],
            "test_cost": test_results["cost"],
            "test_revenue": test_results["revenue"],
            "final_train_profit": train_history["profit"][-1],
        }
    }


def main():
    """Run DQN arbitrage with default settings."""
    run_dqn_arbitrage()


if __name__ == "__main__":
    main()
