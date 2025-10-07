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


class NPCPlayer(Player):
    """NPC (Non-Player Character) that makes random moves.
    Basic implementation - can be enhanced with minimax or other AI strategies."""

    def get_move(self, board: Board) -> int:
        """Select a random valid move from available positions.
        Returns a random available position."""
        available_positions = board.get_available_positions()

        # Choose randomly from available positions
        move = random.choice(available_positions)

        # Provide feedback about the NPC's choice
        print(f"{self.name} ({self.symbol}) chooses position {move}")

        return move
