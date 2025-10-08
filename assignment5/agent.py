"""
Agent Module

This module contains the TicTacAgent class which implements
a reinforcement learning agent for tic-tac-toe.
"""

import random
from board import Board
from players import Player


class TicTacAgent(Player):
    """TicTacAgent that uses reinforcement learning.
    Assesses game state features to inform decision making."""

    def __init__(self, symbol: str, name: str):
        """Initialize agent with symbol and name."""
        super().__init__(symbol, name)
        # State features that will be assessed each turn for RL
        self.current_state = {
            'can_win_this_turn': False,
            'must_block_this_turn': False,
            'center_available': False,
            'corner_available': False
        }

    def _can_win_this_turn(self, board: Board) -> bool:
        """Check if agent can win on this turn.
        Returns True if there's a winning move available."""
        for combo in board.winning_combinations:
            symbols = [board.board[pos] for pos in combo]
            # Check if two positions have our symbol and one is empty
            if symbols.count(self.symbol) == 2 and symbols.count(None) == 1:
                return True
        return False

    def _must_block_this_turn(self, board: Board) -> bool:
        """Check if opponent can win next turn (must block).
        Returns True if opponent has a winning move available."""
        # Get opponent's symbol
        opponent_symbol = 'O' if self.symbol == 'X' else 'X'

        for combo in board.winning_combinations:
            symbols = [board.board[pos] for pos in combo]
            # Check if two positions have opponent's symbol and one is empty
            if symbols.count(opponent_symbol) == 2 and symbols.count(None) == 1:
                return True
        return False

    def _center_available(self, board: Board) -> bool:
        """Check if center position (4) is available.
        Returns True if center is empty."""
        return board.board[4] is None

    def _corner_available(self, board: Board) -> bool:
        """Check if any corner position is available.
        Returns True if at least one corner (0, 2, 6, 8) is empty."""
        corners = [0, 2, 6, 8]
        return any(board.board[corner] is None for corner in corners)

    def assess_game_state(self, board: Board) -> dict:
        """Assess the current game state and return feature dictionary.
        This captures the state for reinforcement learning."""
        self.current_state = {
            'can_win_this_turn': self._can_win_this_turn(board),
            'must_block_this_turn': self._must_block_this_turn(board),
            'center_available': self._center_available(board),
            'corner_available': self._corner_available(board)
        }
        return self.current_state

    def get_move(self, board: Board) -> int:
        """Select a move based on assessed game state.
        Currently uses random selection, will be enhanced with RL."""
        # Assess game state for this turn
        state = self.assess_game_state(board)

        # For now, still make random moves (RL will be added later)
        available_positions = board.get_available_positions()
        move = random.choice(available_positions)

        # Provide feedback about the agent's choice
        print(f"{self.name} ({self.symbol}) chooses position {move}")

        return move
