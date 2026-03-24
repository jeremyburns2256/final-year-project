"""
BESS Trading Environment for Reinforcement Learning

Provides a standard RL environment interface for the battery trading problem.
Compatible with the existing BESS simulator and physical constraints.
"""

import numpy as np
import pandas as pd
from utils.bess_simulator import (
    BESS_SIZE,
    BESS_INVERTER_CAPACITY,
    BESS_INVERTER_CAPACITY_SOLAR,
    INTERVAL_HOURS,
)


class BESSEnvironment:
    """
    Battery Energy Storage System trading environment.

    State space: [battery_soc, rrp, export_kw, import_kw, time_of_day,
                  rrp_lag_1, rrp_lag_2, rrp_ma_12, rrp_ma_48]
    Action space: 0=buy, 1=sell, 2=hold
    Reward: Profit (revenue - cost) for the interval
    """

    def __init__(
        self,
        price_df,
        export_col=None,
        import_col=None,
        network_tariff=10.8007,
        max_rrp=300.0,
        max_power=20.0,
    ):
        """
        Initialize BESS trading environment.

        Args:
            price_df: DataFrame with columns SETTLEMENTDATE, RRP, and optional export/import
            export_col: Column name for net export (solar - load) in kW
            import_col: Column name for grid import in kW
            network_tariff: Fixed network charge in cents/kWh on grid imports
            max_rrp: Maximum RRP for normalization ($/MWh)
            max_power: Maximum power for normalization (kW)
        """
        self.price_df = price_df.copy()
        self.export_col = export_col
        self.import_col = import_col
        self.network_tariff = network_tariff
        self.max_rrp = max_rrp
        self.max_power = max_power

        # Physical constants
        self.bess_size = BESS_SIZE
        self.bess_inverter_capacity = BESS_INVERTER_CAPACITY
        self.solar_inverter_capacity = BESS_INVERTER_CAPACITY_SOLAR
        self.interval_hours = INTERVAL_HOURS
        self.energy_per_interval_bess = BESS_INVERTER_CAPACITY * INTERVAL_HOURS
        self.energy_per_interval_solar = BESS_INVERTER_CAPACITY_SOLAR * INTERVAL_HOURS

        # Precompute features
        self._precompute_features()

        # Episode state
        self.current_step = 0
        self.battery_state = 0.0
        self.cumulative_profit = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_revenue = 0.0

        # History for tracking
        self.episode_history = []

        # State size: [battery_soc, rrp, export, import, time, rrp_lag_1, rrp_lag_2, rrp_ma_12, rrp_ma_48]
        self.state_size = 9
        self.action_size = 3

    def _precompute_features(self):
        """Precompute RRP lag and moving average features."""
        rrp = self.price_df['RRP'].values

        # Lag features
        self.price_df['rrp_lag_1'] = np.roll(rrp, 1)
        self.price_df['rrp_lag_2'] = np.roll(rrp, 2)

        # Moving averages (12 intervals = 1 hour, 48 intervals = 4 hours)
        self.price_df['rrp_ma_12'] = (
            self.price_df['RRP'].rolling(window=12, min_periods=1).mean()
        )
        self.price_df['rrp_ma_48'] = (
            self.price_df['RRP'].rolling(window=48, min_periods=1).mean()
        )

        # Set initial lags to current value (no lookahead)
        self.price_df.loc[0, 'rrp_lag_1'] = rrp[0]
        self.price_df.loc[0, 'rrp_lag_2'] = rrp[0]
        self.price_df.loc[1, 'rrp_lag_2'] = rrp[1]

        # Parse timestamps for time-of-day feature
        try:
            timestamps = pd.to_datetime(self.price_df['SETTLEMENTDATE'], dayfirst=True)
            # Hour of day (0-23)
            self.price_df['hour'] = timestamps.dt.hour
            # Minute of day (0-1440)
            self.price_df['minute_of_day'] = timestamps.dt.hour * 60 + timestamps.dt.minute
        except Exception:
            # Fallback: use step index
            self.price_df['hour'] = 0
            self.price_df['minute_of_day'] = (
                np.arange(len(self.price_df)) % 288
            ) * 5  # 288 intervals per day

    def reset(self):
        """
        Reset environment to initial state.

        Returns:
            Initial state observation
        """
        self.current_step = 0
        self.battery_state = 0.0
        self.cumulative_profit = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_revenue = 0.0
        self.episode_history = []

        return self._get_state()

    def _get_state(self):
        """
        Extract and normalize current state features.

        Returns:
            Numpy array of normalized state features
        """
        if self.current_step >= len(self.price_df):
            return np.zeros(self.state_size)

        row = self.price_df.iloc[self.current_step]

        # Battery state of charge (0-1)
        battery_soc = self.battery_state / self.bess_size

        # Current price (normalized)
        rrp_normalized = row['RRP'] / self.max_rrp

        # Export and import (normalized)
        export_kw = (
            row[self.export_col] / self.max_power if self.export_col else 0.0
        )
        import_kw = (
            row[self.import_col] / self.max_power if self.import_col else 0.0
        )

        # Time of day (0-1, cyclical via minute of day)
        time_normalized = row['minute_of_day'] / 1440.0

        # Price history features (normalized)
        rrp_lag_1 = row['rrp_lag_1'] / self.max_rrp
        rrp_lag_2 = row['rrp_lag_2'] / self.max_rrp
        rrp_ma_12 = row['rrp_ma_12'] / self.max_rrp
        rrp_ma_48 = row['rrp_ma_48'] / self.max_rrp

        state = np.array([
            battery_soc,
            rrp_normalized,
            export_kw,
            import_kw,
            time_normalized,
            rrp_lag_1,
            rrp_lag_2,
            rrp_ma_12,
            rrp_ma_48,
        ])

        return state

    def step(self, action):
        """
        Execute action and return (next_state, reward, done, info).

        Actions:
            0: buy (charge from grid)
            1: sell (discharge to grid)
            2: hold (no grid trading, self-consumption only)

        Args:
            action: Integer action (0, 1, or 2)

        Returns:
            next_state: Next state observation
            reward: Reward (profit for this interval)
            done: Whether episode has ended
            info: Dictionary with additional information
        """
        row = self.price_df.iloc[self.current_step]
        rrp = row['RRP']
        export_kw = row[self.export_col] if self.export_col else 0.0
        import_kw = row[self.import_col] if self.import_col else 0.0

        net_export = export_kw * self.interval_hours
        net_import = import_kw * self.interval_hours

        # Execute action with physical constraints
        bess_delta, grid_net = self._execute_action(
            action, export_kw, import_kw, net_export, net_import
        )

        # Update battery state
        old_battery_state = self.battery_state
        self.battery_state += bess_delta

        # Ensure battery stays within bounds (safety check)
        self.battery_state = np.clip(self.battery_state, 0.0, self.bess_size)

        # Calculate reward (profit for this interval)
        reward, cost, revenue = self._calculate_reward(grid_net, rrp)
        self.cumulative_profit += reward
        self.cumulative_cost += cost
        self.cumulative_revenue += revenue

        # Store history
        self.episode_history.append({
            'step': self.current_step,
            'time': row['SETTLEMENTDATE'],
            'action': action,
            'battery_state': self.battery_state,
            'bess_delta': bess_delta,
            'rrp': rrp,
            'export_kw': export_kw,
            'import_kw': import_kw,
            'grid_net': grid_net,
            'reward': reward,
            'cumulative_profit': self.cumulative_profit,
        })

        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.price_df)

        next_state = self._get_state() if not done else np.zeros(self.state_size)

        info = {
            'battery_state': self.battery_state,
            'cumulative_profit': self.cumulative_profit,
            'cumulative_cost': self.cumulative_cost,
            'cumulative_revenue': self.cumulative_revenue,
            'rrp': rrp,
            'action_name': ['buy', 'sell', 'hold'][action],
            'bess_delta': bess_delta,
            'grid_net': grid_net,
        }

        return next_state, reward, done, info

    def _execute_action(self, action, export_kw, import_kw, net_export, net_import):
        """
        Map DQN action to battery delta and grid position.

        Enforces physical constraints:
        - Battery capacity (0 to BESS_SIZE)
        - Inverter power limits
        - Energy per interval limits

        Args:
            action: 0=buy, 1=sell, 2=hold
            export_kw: Net export power (kW)
            import_kw: Grid import power (kW)
            net_export: Net export energy this interval (kWh)
            net_import: Grid import energy this interval (kWh)

        Returns:
            (bess_delta, grid_net): Battery energy change and grid position (kWh)
        """
        if action == 0:  # Buy (charge from grid)
            # Charge battery from grid at maximum rate
            room_in_battery = self.bess_size - self.battery_state
            bess_delta = min(self.energy_per_interval_bess, room_in_battery)

            # Grid imports for load + battery charge
            grid_net = bess_delta + net_import

        elif action == 1:  # Sell (discharge to grid)
            # Discharge battery to grid at maximum rate
            available_energy = self.battery_state
            bess_delta = -min(available_energy, self.energy_per_interval_bess)

            # Grid position: battery discharge (negative) + any import - any export
            grid_net = bess_delta + net_import - net_export

        else:  # Hold (no grid trading)
            # No active grid trading, handle local generation/consumption only
            if net_export > 0:  # Solar surplus
                # Charge battery with excess solar if there's room
                room_in_battery = self.bess_size - self.battery_state
                bess_delta = min(
                    net_export,
                    self.energy_per_interval_solar,
                    room_in_battery
                )
                # Grid exports remaining surplus after battery charge
                grid_net = -(net_export - bess_delta)
            elif net_import > 0:  # Load deficit
                # Discharge battery to meet load if possible
                available_energy = self.battery_state
                bess_delta = -min(available_energy, self.energy_per_interval_bess, net_import)
                # Grid imports remaining deficit after battery discharge
                grid_net = net_import + bess_delta
            else:
                # No net export or import
                bess_delta = 0.0
                grid_net = 0.0

        return bess_delta, grid_net

    def _calculate_reward(self, grid_net, rrp):
        """
        Calculate profit/loss for this interval.

        Grid position interpretation:
            grid_net > 0: Import from grid (cost)
            grid_net < 0: Export to grid (revenue)

        Args:
            grid_net: Net grid position (kWh)
            rrp: Regional reference price ($/MWh)

        Returns:
            (reward, cost, revenue): Profit, cost, and revenue for this interval
        """
        cost = 0.0
        revenue = 0.0

        if grid_net > 0:  # Import (cost)
            # Cost = wholesale price + network tariff (converted to $/kWh)
            cost = grid_net * (rrp / 1000 + self.network_tariff / 100)
        else:  # Export (revenue)
            # Revenue = wholesale price only (converted to $/kWh)
            revenue = (-grid_net) * (rrp / 1000)

        reward = revenue - cost
        return reward, cost, revenue

    def get_episode_dataframe(self):
        """
        Get episode history as a pandas DataFrame.

        Returns:
            DataFrame with episode history
        """
        return pd.DataFrame(self.episode_history)

    def get_state_size(self):
        """Return dimension of state space."""
        return self.state_size

    def get_action_size(self):
        """Return number of possible actions."""
        return self.action_size

    def get_current_rrp(self):
        """Return current RRP if episode not done."""
        if self.current_step < len(self.price_df):
            return self.price_df.iloc[self.current_step]['RRP']
        return 0.0

    def render(self, mode='human'):
        """
        Render current state (optional, for debugging).

        Args:
            mode: Rendering mode ('human' or 'ansi')
        """
        if self.current_step >= len(self.price_df):
            print("Episode complete")
            return

        row = self.price_df.iloc[self.current_step]
        print(f"\n{'='*50}")
        print(f"Step: {self.current_step}/{len(self.price_df)}")
        print(f"Time: {row['SETTLEMENTDATE']}")
        print(f"Battery SoC: {self.battery_state:.2f} kWh ({100*self.battery_state/self.bess_size:.1f}%)")
        print(f"RRP: ${row['RRP']:.2f}/MWh")
        if self.export_col:
            print(f"Export: {row[self.export_col]:.2f} kW")
        if self.import_col:
            print(f"Import: {row[self.import_col]:.2f} kW")
        print(f"Cumulative Profit: ${self.cumulative_profit:.2f}")
        print(f"{'='*50}")
