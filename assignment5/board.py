"""
Board Module

This module contains the Board class which manages the tic-tac-toe
board state and game rules.
"""

from typing import Optional, List


class Board:
    """Manages the tic-tac-toe board state and game rules.
    Handles board tracking, move validation, and win/draw conditions."""

    def __init__(self):
        """Initialize an empty 3x3 board."""
        # Board is represented as a list of 9 positions (0-8)
        # None = empty, 'X' = player X, 'O' = player O
        self.board = [None] * 9

        # Possible winning combinations (indices)
        self.winning_combinations = [
            [0, 1, 2],  # Top row
            [3, 4, 5],  # Middle row
            [6, 7, 8],  # Bottom row
            [0, 3, 6],  # Left column
            [1, 4, 7],  # Middle column
            [2, 5, 8],  # Right column
            [0, 4, 8],  # Diagonal top-left to bottom-right
            [2, 4, 6],  # Diagonal top-right to bottom-left
        ]

    def is_valid_move(self, position: int) -> bool:
        """Check if a move is valid (position 0-8, empty).
        Returns True if valid, False otherwise."""
        return 0 <= position < 9 and self.board[position] is None

    def make_move(self, position: int, symbol: str) -> bool:
        """Place a symbol ('X' or 'O') at the given position (0-8).
        Returns True if successful, False otherwise."""
        if self.is_valid_move(position):
            self.board[position] = symbol
            return True
        return False

    def get_available_positions(self) -> List[int]:
        """Get all empty positions on the board as a list of indices."""
        return [i for i, cell in enumerate(self.board) if cell is None]

    def check_winner(self) -> Optional[str]:
        """Check if there's a winner. Returns 'X', 'O', or None."""
        for combo in self.winning_combinations:
            # Check if all three positions in this combination have the same non-None symbol
            if (self.board[combo[0]] is not None and
                self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]]):
                return self.board[combo[0]]
        return None

    def is_full(self) -> bool:
        """Check if the board is full (no empty spaces)."""
        return all(cell is not None for cell in self.board)

    def is_game_over(self) -> bool:
        """Check if the game is over (winner or draw)."""
        return self.check_winner() is not None or self.is_full()

    def display(self) -> str:
        """Generate a formatted string showing current board and position reference side by side."""
        # Create game board representation (with X's and O's)
        game_board = []
        for cell in self.board:
            if cell is None:
                game_board.append(' ')
            else:
                game_board.append(cell)

        # Position reference board (always 0-8)
        pos_board = ['0', '1', '2', '3', '4', '5', '6', '7', '8']

        # Format both boards side by side
        board_str = "\n"
        board_str += "Current Board       Position Reference\n"
        board_str += f" {game_board[0]} | {game_board[1]} | {game_board[2]}              {pos_board[0]} | {pos_board[1]} | {pos_board[2]} \n"
        board_str += "-----------            -----------\n"
        board_str += f" {game_board[3]} | {game_board[4]} | {game_board[5]}              {pos_board[3]} | {pos_board[4]} | {pos_board[5]} \n"
        board_str += "-----------            -----------\n"
        board_str += f" {game_board[6]} | {game_board[7]} | {game_board[8]}              {pos_board[6]} | {pos_board[7]} | {pos_board[8]} \n"

        return board_str

    def reset(self):
        """Reset the board to its initial empty state."""
        self.board = [None] * 9
