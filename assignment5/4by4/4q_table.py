"""
Q-Table Module

This module contains the QTable class which implements Q-learning
for tic-tac-toe using TD(0) updates.
"""

import pickle
import random
from typing import Dict, List, Optional, Tuple


class QTable:
    """Q-learning table using TD(0) updates.
    Stores Q-values for state-action pairs and provides learning methods."""

    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        """Initialize Q-table with hyperparameters (alpha, gamma, epsilon).
        Note: epsilon defaults to 0.2 (higher than 3x3) due to larger state space."""
        self.table: Dict[Tuple[tuple, str], float] = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def state_to_tuple(self, state: dict) -> tuple:
        """Convert state dict to hashable tuple for dictionary key.
        Returns tuple of state values in consistent order.
        Updated for 4x4 board with qty_middle_zone (0-4) instead of center boolean."""
        return (
            state['can_win_this_turn'],
            state['must_block_this_turn'],
            state['qty_middle_zone_available'],
            state['qty_middle_zone_owned'],
            state['qty_corners_available'],
            state['qty_edge_mids_available'],
            state['total_pieces_placed']
        )

    def get_q_value(self, state: dict, action: str) -> float:
        """Get Q-value for state-action pair (0.0 if never seen)."""
        state_tuple = self.state_to_tuple(state)
        key = (state_tuple, action)
        return self.table.get(key, 0.0)

    def set_q_value(self, state: dict, action: str, value: float):
        """Set Q-value for state-action pair."""
        state_tuple = self.state_to_tuple(state)
        key = (state_tuple, action)
        self.table[key] = value

    def get_best_action(self, state: dict, valid_actions: List[str]) -> str:
        """Return action with highest Q-value from valid actions.
        Ties broken randomly."""
        if not valid_actions:
            raise ValueError("No valid actions provided")

        # Get Q-values for all valid actions
        q_values = [(action, self.get_q_value(state, action)) for action in valid_actions]

        # Return action with max Q-value (ties broken randomly)
        max_q = max(q_val for _, q_val in q_values)
        best_actions = [action for action, q_val in q_values if q_val == max_q]

        return random.choice(best_actions)

    def get_max_q_value(self, state: dict, valid_actions: List[str]) -> float:
        """Get maximum Q-value for a state over valid actions."""
        if not valid_actions:
            return 0.0

        return max(self.get_q_value(state, action) for action in valid_actions)

    def epsilon_greedy_select(self, state: dict, valid_actions: List[str]) -> str:
        """ε-greedy action selection: explore with probability ε, else exploit."""
        if not valid_actions:
            raise ValueError("No valid actions provided")

        # Explore: choose random action
        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploit: choose best action
        return self.get_best_action(state, valid_actions)

    def update_q_value(self, state: dict, action: str, reward: float,
                      next_state: Optional[dict], next_valid_actions: Optional[List[str]]):
        """Apply TD(0) Q-learning update rule: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]"""
        # Get current Q-value
        current_q = self.get_q_value(state, action)

        # Calculate TD target
        if next_state is None or next_valid_actions is None:
            # Terminal state: no future value
            td_target = reward
        else:
            # Bootstrap from next state's max Q-value
            max_next_q = self.get_max_q_value(next_state, next_valid_actions)
            td_target = reward + self.gamma * max_next_q

        # TD error
        td_error = td_target - current_q

        # Update Q-value
        new_q = current_q + self.alpha * td_error
        self.set_q_value(state, action, new_q)

    def save(self, filepath: str):
        """Save Q-table to disk using pickle."""
        data = {
            'table': self.table,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> 'QTable':
        """Load Q-table from disk and return QTable instance."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        q_table = cls(alpha=data['alpha'], gamma=data['gamma'], epsilon=data['epsilon'])
        q_table.table = data['table']
        return q_table

    def get_table_size(self) -> int:
        """Get number of state-action pairs in table."""
        return len(self.table)
