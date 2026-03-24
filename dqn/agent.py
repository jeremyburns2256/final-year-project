"""
Deep Q-Network (DQN) Agent for BESS Trading

Implements a DQN agent with experience replay and target network
for learning optimal battery trading strategies.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from collections import deque
import random


class DQNAgent:
    """
    Deep Q-Network agent for discrete action spaces.

    Uses experience replay and a target network for stable training.
    Epsilon-greedy exploration strategy with decay.
    """

    def __init__(
        self,
        state_size,
        action_size=3,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        hidden_layers=[128, 128, 64],
        dropout_rate=0.2,
    ):
        """
        Initialize DQN agent.

        Args:
            state_size: Dimension of state space
            action_size: Number of possible actions (default 3: buy/sell/hold)
            learning_rate: Adam optimizer learning rate
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Multiplicative decay factor for epsilon
            memory_size: Maximum size of replay memory
            batch_size: Number of experiences to sample per training step
            hidden_layers: List of neurons per hidden layer
            dropout_rate: Dropout rate for regularization
        """
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        # Experience replay memory
        self.memory = deque(maxlen=memory_size)

        # Training history
        self.losses = []

        # Main Q-network
        self.model = self._build_model()

        # Target Q-network (for stable training)
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """
        Build the neural network for Q-value approximation.

        Architecture:
            - Input layer (state_size)
            - Hidden layers with ReLU activation and dropout
            - Output layer (action_size) with linear activation

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential()

        # Input layer
        model.add(keras.layers.Input(shape=(self.state_size,)))

        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            model.add(keras.layers.Dense(units, activation='relu'))
            if self.dropout_rate > 0:
                model.add(keras.layers.Dropout(self.dropout_rate))

        # Output layer: Q-values for each action
        model.add(keras.layers.Dense(self.action_size, activation='linear'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return model

    def update_target_model(self):
        """
        Copy weights from main model to target model.

        This provides stability during training by keeping the target
        Q-values consistent for a period of time.
        """
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay memory.

        Args:
            state: Current state observation
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.

        Args:
            state: Current state observation (1D numpy array)
            training: If False, use greedy policy (no exploration)

        Returns:
            Action index (0, 1, or 2 for buy/sell/hold)
        """
        # Exploration: choose random action
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation: choose action with highest Q-value
        state_input = state.reshape(1, -1)
        q_values = self.model.predict(state_input, verbose=0)
        return np.argmax(q_values[0])

    def get_q_values(self, state):
        """
        Get Q-values for all actions given a state.

        Args:
            state: State observation

        Returns:
            Array of Q-values for each action
        """
        state_input = state.reshape(1, -1)
        return self.model.predict(state_input, verbose=0)[0]

    def replay(self):
        """
        Train on a batch of experiences from memory.

        Uses the Bellman equation to update Q-values:
            Q(s,a) = r + gamma * max(Q(s',a'))

        Returns:
            Average loss for the batch (None if not enough samples)
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        # Extract components
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])

        # Predict Q-values for current states
        current_q_values = self.model.predict(states, verbose=0)

        # Predict Q-values for next states using target network
        next_q_values = self.target_model.predict(next_states, verbose=0)

        # Update Q-values using Bellman equation
        for i in range(self.batch_size):
            if dones[i]:
                # Terminal state: no future rewards
                target = rewards[i]
            else:
                # Non-terminal: include discounted future reward
                target = rewards[i] + self.gamma * np.max(next_q_values[i])

            current_q_values[i][actions[i]] = target

        # Train the model
        history = self.model.fit(states, current_q_values, epochs=1, verbose=0)
        loss = history.history['loss'][0]
        self.losses.append(loss)

        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, filepath):
        """
        Save model weights to file.

        Args:
            filepath: Path to save weights (e.g., 'models/dqn_weights.h5')
        """
        self.model.save_weights(filepath)
        print(f"Model weights saved to {filepath}")

    def load(self, filepath):
        """
        Load model weights from file.

        Args:
            filepath: Path to load weights from
        """
        self.model.load_weights(filepath)
        self.update_target_model()
        print(f"Model weights loaded from {filepath}")

    def save_full_model(self, filepath):
        """
        Save complete model architecture and weights.

        Args:
            filepath: Path to save model (e.g., 'models/dqn_model.keras')
        """
        self.model.save(filepath)
        print(f"Full model saved to {filepath}")

    def get_memory_size(self):
        """Return current size of replay memory."""
        return len(self.memory)

    def clear_memory(self):
        """Clear replay memory."""
        self.memory.clear()

    def get_config(self):
        """
        Get agent configuration.

        Returns:
            Dictionary of agent parameters
        """
        return {
            'state_size': self.state_size,
            'action_size': self.action_size,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'memory_size': self.memory.maxlen,
        }
