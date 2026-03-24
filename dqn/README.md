# DQN-based BESS Trading

Deep Q-Network (DQN) reinforcement learning implementation for Battery Energy Storage System (BESS) arbitrage trading.

## Overview

This implementation uses a DQN agent to learn optimal battery trading strategies from historical electricity price data. The agent learns when to buy (charge), sell (discharge), or hold based on:
- Current battery state
- Electricity prices (RRP)
- Solar export and household load
- Time of day
- Price history

## Files

- **`dqn_agent.py`**: DQN agent with experience replay and target network
- **`bess_environment.py`**: RL environment wrapper for BESS trading
- **`dqn_trading.py`**: Training and evaluation functions
- **`compare_strategies.py`**: Compare DQN vs state machine strategies
- **`example_dqn.py`**: Example usage scripts

## Quick Start

### 1. Train a DQN Agent

```python
from dqn.training import run_dqn_trading

results = run_dqn_trading(
    train_csv="data/price_DEC24.csv",
    test_csv="data/price_JAN25.csv",
    train_export_csv="data/export_DEC24.csv",
    train_import_csv="data/import_DEC24.csv",
    test_export_csv="data/export_JAN25.csv",
    test_import_csv="data/import_JAN25.csv",
    episodes=50,
    verbose=True,
    plot=True,
)

print(f"Net Profit: ${results['metrics']['net_profit']:.2f}")
```

### 2. Compare DQN vs State Machine

```python
from examples.compare_strategies import compare_strategies

results = compare_strategies(
    dqn_episodes=50,
    sm_optimise_thresholds=True,
    verbose=True,
    plot=True,
)

print(f"DQN Profit: ${results['dqn']['metrics']['net_profit']:.2f}")
print(f"SM Profit: ${results['state_machine']['metrics']['net_profit']:.2f}")
```

### 3. Run from Command Line

```bash
# Activate virtual environment
source .venv/bin/activate

# Run DQN training
python -m dqn.training

# Run comparison
python -m examples.compare_strategies

# Run examples
python -m examples.dqn_example
```

## How It Works

### State Space (9 features)

The DQN agent observes:
1. **Battery SoC**: State of charge (0-1)
2. **Current RRP**: Electricity price (normalized)
3. **Export kW**: Solar export power (normalized)
4. **Import kW**: Grid import power (normalized)
5. **Time of day**: Normalized time (0-1)
6. **RRP lag 1**: Previous interval price
7. **RRP lag 2**: Two intervals ago price
8. **RRP MA 12**: 1-hour moving average
9. **RRP MA 48**: 4-hour moving average

### Action Space

Three discrete actions:
- **0 (Buy)**: Charge battery from grid
- **1 (Sell)**: Discharge battery to grid
- **2 (Hold)**: No active trading, self-consumption only

### Reward Function

Reward = Revenue - Cost for each interval:
- **Cost**: Grid imports × (wholesale price + network tariff)
- **Revenue**: Grid exports × wholesale price

### Neural Network Architecture

Default architecture:
```
Input (9 features)
  ↓
Dense(128, relu) → Dropout(0.2)
  ↓
Dense(128, relu) → Dropout(0.2)
  ↓
Dense(64, relu)
  ↓
Dense(3, linear)  # Q-values for [buy, sell, hold]
```

## Hyperparameters

Key hyperparameters and their defaults:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodes` | 50 | Number of training episodes |
| `learning_rate` | 0.0005 | Adam optimizer learning rate |
| `gamma` | 0.99 | Discount factor for future rewards |
| `epsilon` | 1.0 | Initial exploration rate |
| `epsilon_min` | 0.01 | Minimum exploration rate |
| `epsilon_decay` | 0.995 | Exploration decay per episode |
| `batch_size` | 64 | Experience replay batch size |
| `memory_size` | 10000 | Replay memory capacity |
| `update_target_every` | 5 | Update target network every N episodes |
| `hidden_layers` | [128,128,64] | Hidden layer sizes |
| `dropout_rate` | 0.2 | Dropout for regularization |

## Tuning Guide

### For Better Performance

1. **Increase episodes**: 100-200 episodes for more training
2. **Larger network**: `hidden_layers=[256, 256, 128]`
3. **More memory**: `memory_size=20000`
4. **Slower exploration decay**: `epsilon_decay=0.99`

### For Faster Training

1. **Fewer episodes**: 20-30 episodes for quick tests
2. **Smaller network**: `hidden_layers=[64, 64]`
3. **Larger batches**: `batch_size=128`

### For Stability

1. **Lower learning rate**: `learning_rate=0.0001`
2. **More regularization**: `dropout_rate=0.3`
3. **Frequent target updates**: `update_target_every=3`

## Data Requirements

The implementation expects CSV files with:
- **Price data**: `SETTLEMENTDATE`, `RRP`, `TOTALDEMAND`
- **Export data** (optional): `SETTLEMENTDATE`, `EXPORT_KW`
- **Import data** (optional): `SETTLEMENTDATE`, `IMPORT_KW`

Export represents net export (solar - load) from B1 meter data.

## Physical Constraints

The environment enforces realistic constraints:
- Battery capacity: 20 kWh (configurable in `utils/bess_simulator.py`)
- Battery inverter: 11.04 kW max charge/discharge rate
- Solar inverter: 20 kW max charge rate from solar
- 5-minute trading intervals

## Comparison with State Machine

| Aspect | State Machine | DQN |
|--------|---------------|-----|
| **Strategy** | Fixed thresholds | Learned policy |
| **Optimization** | Brute-force grid search | Gradient descent |
| **Adaptability** | Static thresholds | Adapts to patterns |
| **Training time** | ~minutes | ~minutes to hours |
| **Interpretability** | High (clear thresholds) | Low (black box) |
| **Price patterns** | Simple threshold logic | Can learn complex patterns |

## Output

### Training Output

```
Episode   1/50 | Profit: $  123.45 | Revenue: $  456.78 | Cost: $  333.33 | Epsilon: 0.995 | Loss: 0.0234 | Memory: 8640
Episode   2/50 | Profit: $  145.67 | Revenue: $  478.90 | Cost: $  333.23 | Epsilon: 0.990 | Loss: 0.0198 | Memory: 10000
...
```

### Evaluation Output

```
----------------------------------------
  DQN Results (price_JAN25.csv) [export, import]
----------------------------------------
Final Battery State: 10.23 kWh
Total Grid Cost:     $1234.56
Total Grid Revenue:  $2345.67
Net Profit:          $1111.11
```

### Comparison Output

```
Metric                   State Machine             DQN      Difference
----------------------------------------------------------------------
Net Profit                    $1000.00        $1111.11         $111.11
Total Revenue                 $2200.00        $2345.67         $145.67
Total Cost                    $1200.00        $1234.56          $34.56
Final Battery (kWh)              12.50           10.23           -2.27
----------------------------------------------------------------------

DQN vs State Machine: +11.11% profit change
```

## Visualizations

Generated plots (saved to `plots/` directory):
- Battery state over time
- Price levels and actions
- Cumulative profit
- Grid import/export

## Advanced Usage

### Custom Environment

```python
from dqn.environment import BESSEnvironment
import pandas as pd

# Create custom environment
df = pd.read_csv("data/price_DEC24.csv")
env = BESSEnvironment(
    df,
    export_col="EXPORT_KW",
    import_col="IMPORT_KW",
    network_tariff=10.8007,
    max_rrp=500.0,  # Custom normalization
)

# Manual control
state = env.reset()
for _ in range(100):
    action = 0  # Buy
    next_state, reward, done, info = env.step(action)
    if done:
        break
```

### Custom Agent

```python
from dqn.agent import DQNAgent

agent = DQNAgent(
    state_size=9,
    action_size=3,
    learning_rate=0.001,
    hidden_layers=[256, 256, 128, 64],
    dropout_rate=0.3,
)

# Manual training loop
# ... your custom training code ...
```

### Save/Load Models

```python
# Save
agent.save("models/my_model.weights.h5")

# Load
agent.load("models/my_model.weights.h5")
```

## Troubleshooting

### Low Profit
- Increase training episodes
- Reduce learning rate
- Try different hyperparameters
- Check data quality

### Unstable Training
- Reduce learning rate
- Increase target update frequency
- Add more dropout
- Reduce network size

### Memory Issues
- Reduce `memory_size`
- Reduce `batch_size`
- Reduce network size

## References

- DQN Paper: [Mnih et al., 2015](https://arxiv.org/abs/1312.5602)
- TensorFlow: https://www.tensorflow.org/
- Reinforcement Learning: Sutton & Barto (2018)

## License

Part of final-year-project research on battery energy storage optimization.
