"""
Metrics Module

This module tracks and displays learning metrics during Q-learning training.
"""

from typing import Dict, List
from collections import defaultdict


class LearningMetrics:
    """Tracks learning metrics during Q-learning training."""

    def __init__(self):
        """Initialize metrics tracking."""
        # Action selection counts
        self.action_counts: Dict[str, int] = defaultdict(int)

        # Strategic best actions over time (stores tuples of (best_action, q_value))
        self.can_win_best: List[tuple] = []
        self.center_available_best: List[tuple] = []
        self.must_block_best: List[tuple] = []

        # Q-table size over time
        self.qtable_sizes: List[int] = []

    def record_action(self, action: str):
        """Record an action selection."""
        self.action_counts[action] += 1

    def record_strategic_qvalues(self, q_table, game_number: int):
        """Record best action and its Q-value for strategic state features."""
        # Best action when can_win=True
        can_win_best = self._get_best_action_for_feature(
            q_table, True, feature_index=0
        )
        self.can_win_best.append(can_win_best)

        # Best action when center_available=True
        center_best = self._get_best_action_for_feature(
            q_table, True, feature_index=2
        )
        self.center_available_best.append(center_best)

        # Best action when must_block=True
        block_best = self._get_best_action_for_feature(
            q_table, True, feature_index=1
        )
        self.must_block_best.append(block_best)

    def _get_best_action_for_feature(self, q_table, feature_value: bool,
                                       feature_index: int) -> tuple:
        """Get best action (highest avg Q-value) for states matching a feature.

        Returns: (action_name, avg_q_value) or ('none', 0.0) if no matches

        State tuple order: (can_win, must_block, center_available, center_owned,
                           qty_corners_available, qty_edge_mids_available, total_pieces_placed)
        """
        # Collect Q-values by action
        action_qvalues = defaultdict(list)

        for (state_tuple, act), qval in q_table.table.items():
            # Check if this state matches our feature
            if state_tuple[feature_index] == feature_value:
                action_qvalues[act].append(qval)

        # Calculate average Q-value for each action
        if not action_qvalues:
            return ('none', 0.0)

        action_averages = {
            action: sum(qvals) / len(qvals)
            for action, qvals in action_qvalues.items()
        }

        # Return action with highest average Q-value
        best_action = max(action_averages.items(), key=lambda x: x[1])
        return best_action

    def record_qtable_size(self, size: int):
        """Record Q-table size."""
        self.qtable_sizes.append(size)

    def display_metrics(self, game_number: int, total_games: int):
        """Display current learning metrics."""
        print(f"\n{'='*60}")
        print(f"LEARNING METRICS - Game {game_number}/{total_games}")
        print(f"{'='*60}")

        # Q-Table Size
        if self.qtable_sizes:
            print(f"\nQ-Table Size: {self.qtable_sizes[-1]:,} state-action pairs")

        # Strategic Best Actions
        print(f"\nStrategic Learning (Best Action by State Feature):")
        if self.can_win_best:
            action, qval = self.can_win_best[-1]
            print(f"  'Can Win' --> {action:20s} (Q={qval:>7.4f})")
        if self.center_available_best:
            action, qval = self.center_available_best[-1]
            print(f"  'Center Available' --> {action:11s} (Q={qval:>7.4f})")
        if self.must_block_best:
            action, qval = self.must_block_best[-1]
            print(f"  'Must Block' --> {action:17s} (Q={qval:>7.4f})")

        # Action Distribution
        print(f"\nAction Selection Distribution:")
        total_actions = sum(self.action_counts.values())
        if total_actions > 0:
            # Sort actions by count (descending)
            sorted_actions = sorted(self.action_counts.items(), key=lambda x: x[1], reverse=True)
            for action, count in sorted_actions:
                percentage = (count / total_actions) * 100
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = '█' * bar_length
                print(f"  {action:20s} {count:6d} ({percentage:5.1f}%) {bar}")

        print(f"{'='*60}\n")

    def reset_action_counts(self):
        """Reset action counts (useful for interval-based tracking)."""
        self.action_counts = defaultdict(int)
