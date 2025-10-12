"""
Agent Module

This module contains the TicTacAgent class which implements
a reinforcement learning agent for tic-tac-toe.
"""

import os
import random
import importlib.util
from typing import Optional, Tuple, List

# Import modules with "4" prefix
def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

board_module = load_module("board", "4board.py")
players_module = load_module("players", "4players.py")

Board = board_module.Board
Player = players_module.Player


class TicTacAgent(Player):
    """TicTacAgent that uses reinforcement learning.
    Assesses game state features to inform decision making."""

    def __init__(self, symbol: str, name: str, q_table=None, learning_enabled: bool = True):
        """Initialize agent with symbol, name, and optional Q-learning."""
        super().__init__(symbol, name)
        # State features that will be assessed each turn for RL
        self.current_state = {
            'can_win_this_turn': False,
            'must_block_this_turn': False,
            'qty_middle_zone_available': 0,
            'qty_middle_zone_owned': 0,
            'qty_corners_available': 0,
            'qty_edge_mids_available': 0,
            'total_pieces_placed': 0
        }

        # Q-learning attributes
        self.q_table = q_table
        self.learning_enabled = learning_enabled
        self.last_state: Optional[dict] = None
        self.last_action: Optional[str] = None
        self.metrics = None  # Optional metrics tracker

    def _can_win_this_turn(self, board: Board) -> bool:
        """Check if agent can win on this turn (4-in-a-row).
        Returns True if there's a winning move available."""
        for combo in board.winning_combinations:
            symbols = [board.board[pos] for pos in combo]
            # Count own symbols (including dual cells 'B' which count for both)
            own_count = sum(1 for s in symbols if s == self.symbol or s == 'B')
            empty_count = symbols.count(None)
            # Can win if we have 3 of 4 (including duals) and 1 empty
            if own_count == 3 and empty_count == 1:
                return True
        return False

    def _must_block_this_turn(self, board: Board) -> bool:
        """Check if opponent can win next turn (must block) (4-in-a-row).
        Returns True if opponent has a winning move available."""
        # Get opponent's symbol
        opponent_symbol = 'O' if self.symbol == 'X' else 'X'

        for combo in board.winning_combinations:
            symbols = [board.board[pos] for pos in combo]
            # Count opponent symbols (including dual cells 'B' which count for both)
            opp_count = sum(1 for s in symbols if s == opponent_symbol or s == 'B')
            empty_count = symbols.count(None)
            # Must block if opponent has 3 of 4 (including duals) and 1 empty
            if opp_count == 3 and empty_count == 1:
                return True
        return False

    def _qty_middle_zone_available(self, board: Board) -> int:
        """Count how many middle zone positions (5, 6, 9, 10) are available.
        Returns 0-4 representing number of empty middle zone cells."""
        middle_zone = [5, 6, 9, 10]
        return sum(1 for pos in middle_zone if board.board[pos] is None)

    def _qty_middle_zone_owned(self, board: Board) -> int:
        """Count how many middle zone positions (5, 6, 9, 10) agent owns.
        Returns 0-4 representing number of middle zone cells with agent's symbol."""
        middle_zone = [5, 6, 9, 10]
        return sum(1 for pos in middle_zone if board.board[pos] == self.symbol)

    def _qty_corners_available(self, board: Board) -> int:
        """Count how many corner positions are available (4x4 grid).
        Returns 0-4 representing number of empty corners (0, 3, 12, 15)."""
        corners = [0, 3, 12, 15]
        return sum(1 for corner in corners if board.board[corner] is None)

    def _qty_edge_mids_available(self, board: Board) -> int:
        """Count how many edge middle positions are available (4x4 grid).
        Returns 0-8 representing number of empty edge mids (1, 2, 4, 7, 8, 11, 13, 14)."""
        edge_mids = [1, 2, 4, 7, 8, 11, 13, 14]
        return sum(1 for edge in edge_mids if board.board[edge] is None)

    def _total_pieces_placed(self, board: Board) -> int:
        """Count total number of pieces on the board (excluding dual cells).
        Returns 0-14 representing total X's and O's placed (not counting 'B')."""
        return sum(1 for cell in board.board if cell in ['X', 'O'])

    def assess_game_state(self, board: Board) -> dict:
        """Assess the current game state and return feature dictionary.
        This captures the state for reinforcement learning."""
        self.current_state = {
            'can_win_this_turn': self._can_win_this_turn(board),
            'must_block_this_turn': self._must_block_this_turn(board),
            'qty_middle_zone_available': self._qty_middle_zone_available(board),
            'qty_middle_zone_owned': self._qty_middle_zone_owned(board),
            'qty_corners_available': self._qty_corners_available(board),
            'qty_edge_mids_available': self._qty_edge_mids_available(board),
            'total_pieces_placed': self._total_pieces_placed(board)
        }
        return self.current_state

    # Action methods - these define what the agent can do
    def win_now(self, board: Board) -> Optional[int]:
        """Complete a winning line (3 own symbols/duals + 1 empty) for 4-in-a-row.
        Returns the winning position, or None if no winning move exists."""
        for combo in board.winning_combinations:
            symbols = [board.board[pos] for pos in combo]
            # Count own symbols (including dual cells)
            own_count = sum(1 for s in symbols if s == self.symbol or s == 'B')
            empty_count = symbols.count(None)
            # Can win if 3 of 4 are ours/dual and 1 is empty
            if own_count == 3 and empty_count == 1:
                # Find and return the empty position
                for pos in combo:
                    if board.board[pos] is None:
                        return pos
        return None

    def block_opponent(self, board: Board) -> Optional[int]:
        """Block opponent's winning move (3 opponent symbols/duals + 1 empty) for 4-in-a-row.
        Returns the blocking position, or None if no block needed."""
        opponent_symbol = 'O' if self.symbol == 'X' else 'X'

        for combo in board.winning_combinations:
            symbols = [board.board[pos] for pos in combo]
            # Count opponent symbols (including dual cells)
            opp_count = sum(1 for s in symbols if s == opponent_symbol or s == 'B')
            empty_count = symbols.count(None)
            # Must block if opponent has 3 of 4 (including duals) and 1 is empty
            if opp_count == 3 and empty_count == 1:
                # Find and return the empty position to block
                for pos in combo:
                    if board.board[pos] is None:
                        return pos
        return None

    def take_center(self, board: Board) -> Optional[int]:
        """Play middle zone position (5, 6, 9, or 10) if available.
        Returns random middle zone position if available, None otherwise."""
        middle_zone = [5, 6, 9, 10]
        available_middle = [pos for pos in middle_zone if board.board[pos] is None]
        if available_middle:
            return random.choice(available_middle)
        return None

    def take_edge_mid(self, board: Board) -> Optional[int]:
        """Randomly select from available edge mids (1, 2, 4, 7, 8, 11, 13, 14).
        Returns random edge mid position, or None if none available."""
        edge_mids = [1, 2, 4, 7, 8, 11, 13, 14]
        available_edges = [pos for pos in edge_mids if board.board[pos] is None]
        if available_edges:
            return random.choice(available_edges)
        return None

    def take_corner(self, board: Board) -> Optional[int]:
        """Randomly select from available corners (0, 3, 12, 15).
        Returns random corner position, or None if none available."""
        corners = [0, 3, 12, 15]
        available_corners = [pos for pos in corners if board.board[pos] is None]
        if available_corners:
            return random.choice(available_corners)
        return None

    def take_random(self, board: Board) -> Optional[int]:
        """Randomly select from all available positions.
        Returns random available position, or None if board is full."""
        available_positions = board.get_available_positions()
        if available_positions:
            return random.choice(available_positions)
        return None

    def get_valid_actions(self, board: Board) -> List[str]:
        """Determine which actions are feasible in current state.
        Returns list of valid action names."""
        valid = []
        if self.win_now(board) is not None:
            valid.append('win_now')
        if self.block_opponent(board) is not None:
            valid.append('block_opponent')
        if self.take_center(board) is not None:
            valid.append('take_center')
        if self.take_edge_mid(board) is not None:
            valid.append('take_edge_mid')
        if self.take_corner(board) is not None:
            valid.append('take_corner')
        if self.take_random(board) is not None:
            valid.append('take_random')
        return valid

    def select_action_via_q_learning(self, board: Board, state: dict) -> Tuple[str, int]:
        """Use Q-table for ε-greedy action selection. Returns (action_name, position)."""
        valid_actions = self.get_valid_actions(board)

        # Use ε-greedy to select action name
        action_name = self.q_table.epsilon_greedy_select(state, valid_actions)

        # Execute the selected action to get position
        action_map = {
            'win_now': self.win_now,
            'block_opponent': self.block_opponent,
            'take_center': self.take_center,
            'take_edge_mid': self.take_edge_mid,
            'take_corner': self.take_corner,
            'take_random': self.take_random
        }
        position = action_map[action_name](board)

        return action_name, position

    def update_from_transition(self, reward: float, next_state: Optional[dict],
                               next_board: Optional[Board]):
        """Perform TD(0) Q-learning update after observing transition."""
        if not self.q_table or not self.learning_enabled:
            return

        if self.last_state is None or self.last_action is None:
            return  # First move, nothing to update yet

        # Get valid actions for next state (None if terminal)
        next_valid_actions = None
        if next_state is not None and next_board is not None:
            next_valid_actions = self.get_valid_actions(next_board)

        # Apply TD(0) update
        self.q_table.update_q_value(
            state=self.last_state,
            action=self.last_action,
            reward=reward,
            next_state=next_state,
            next_valid_actions=next_valid_actions
        )

    def reset_episode(self):
        """Reset episode tracking. Called at start of new game."""
        self.last_state = None
        self.last_action = None

    def get_move(self, board: Board) -> int:
        """Select a move using Q-learning if enabled, else random."""
        # Assess current game state
        state = self.assess_game_state(board)

        if self.q_table and self.learning_enabled:
            # Q-learning mode
            action_name, position = self.select_action_via_q_learning(board, state)

            # Store state and action for next TD update
            self.last_state = state.copy()  # Important: copy the dict
            self.last_action = action_name

            # Record action in metrics if available
            if self.metrics:
                self.metrics.record_action(action_name)

            print(f"{self.name} ({self.symbol}) chooses position {position} via {action_name}")
            return position
        else:
            # Fallback: random selection
            available_positions = board.get_available_positions()
            move = random.choice(available_positions)
            print(f"{self.name} ({self.symbol}) chooses position {move}")
            return move
