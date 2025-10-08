"""
Players Module

This module contains the base Player class and HumanPlayer implementation.
The TicTacAgent is in a separate module (agent.py).
"""

import os
import importlib.util
from abc import ABC, abstractmethod

# Import modules with "4" prefix
def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

board_module = load_module("board", "4board.py")
Board = board_module.Board


class Player(ABC):
    """Abstract base class for all player types.
    Defines the interface that all players must implement."""

    def __init__(self, symbol: str, name: str):
        """Initialize a player with symbol ('X' or 'O') and display name."""
        self.symbol = symbol
        self.name = name

    @abstractmethod
    def get_move(self, board: Board) -> int:
        """Get the player's next move (position 0-15).
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
                move_input = input(f"{self.name} ({self.symbol}), enter your move (0-15): ")
                position = int(move_input)

                # Validate the move
                if board.is_valid_move(position):
                    return position
                else:
                    if position in board.dual_cell_positions:
                        print("Invalid move! That position is a dual cell and cannot be played on.")
                    else:
                        print("Invalid move! That position is either taken or out of bounds.")
            except ValueError:
                print("Invalid input! Please enter a number between 0 and 15.")
            except KeyboardInterrupt:
                print("\nGame interrupted by user.")
                raise
