"""
Players Module

This module contains all player types for the tic-tac-toe game.
Includes the abstract Player base class and concrete implementations
for Human and NPC players.
"""

import random
from abc import ABC, abstractmethod
from board import Board


class Player(ABC):
    """Abstract base class for all player types.
    Defines the interface that all players must implement."""

    def __init__(self, symbol: str, name: str):
        """Initialize a player with symbol ('X' or 'O') and display name."""
        self.symbol = symbol
        self.name = name

    @abstractmethod
    def get_move(self, board: Board) -> int:
        """Get the player's next move (position 0-8).
        Must be implemented by all subclasses."""
        pass


class HumanPlayer(Player):
    """Human player that gets moves from user input.
    Prompts user, validates input, and handles errors gracefully."""

    def get_move(self, board: Board) -> int:
        """Prompt the human player for their move and validate it.
        Returns the validated position chosen by the player."""
        while True:
            try:
                # Prompt for input
                move_input = input(f"{self.name} ({self.symbol}), enter your move (0-8): ")
                position = int(move_input)

                # Validate the move
                if board.is_valid_move(position):
                    return position
                else:
                    print("Invalid move! That position is either taken or out of bounds.")
            except ValueError:
                print("Invalid input! Please enter a number between 0 and 8.")
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise


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
