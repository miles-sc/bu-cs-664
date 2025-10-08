"""
Board Module

This module contains the Board class which manages the tic-tac-toe
board state and game rules.
"""

import random
from typing import Optional, List


class Board:
    """Manages the tic-tac-toe board state and game rules.
    Handles board tracking, move validation, and win/draw conditions."""

    def __init__(self, num_dual_cells: int = 2):
        """Initialize an empty 4x4 board with dual cells.

        Args:
            num_dual_cells: Number of dual cells (cells that count as both X and O). Default is 2.
        """
        # Board is represented as a list of 16 positions (0-15)
        # None = empty, 'X' = player X, 'O' = player O, 'B' = dual cell (both)
        self.board = [None] * 16
        self.num_dual_cells = num_dual_cells
        self.dual_cell_positions: List[int] = []

        # Auto-generate 4x4 winning combinations (indices)
        self.winning_combinations = []

        # Rows (4 rows)
        for row in range(4):
            self.winning_combinations.append([row * 4 + i for i in range(4)])

        # Columns (4 columns)
        for col in range(4):
            self.winning_combinations.append([col + row * 4 for row in range(4)])

        # Diagonals (2 diagonals)
        self.winning_combinations.append([0, 5, 10, 15])  # Top-left to bottom-right
        self.winning_combinations.append([3, 6, 9, 12])   # Top-right to bottom-left

    def assign_dual_cells(self):
        """Randomly assign dual cell positions for this game.
        Dual cells are fixed for the entire game and cannot be played on."""
        if self.num_dual_cells > 0:
            self.dual_cell_positions = random.sample(range(16), self.num_dual_cells)
            for pos in self.dual_cell_positions:
                self.board[pos] = 'B'

    def is_valid_move(self, position: int) -> bool:
        """Check if a move is valid (position 0-15, empty, not a dual cell).
        Returns True if valid, False otherwise."""
        return 0 <= position < 16 and self.board[position] is None

    def make_move(self, position: int, symbol: str) -> bool:
        """Place a symbol ('X' or 'O') at the given position (0-15).
        Returns True if successful, False otherwise.
        Cannot place on dual cells."""
        if self.is_valid_move(position):
            self.board[position] = symbol
            return True
        return False

    def get_available_positions(self) -> List[int]:
        """Get all empty positions on the board as a list of indices."""
        return [i for i, cell in enumerate(self.board) if cell is None]

    def check_winner(self) -> Optional[str]:
        """Check if there's a winner. Returns 'X', 'O', or None.
        Dual cells ('B') count as wildcards - they satisfy both X and O."""
        for combo in self.winning_combinations:
            # Get symbols in this combination
            symbols = [self.board[pos] for pos in combo]

            # Skip if any position is empty (None)
            if None in symbols:
                continue

            # Check for X win (all X or B)
            if all(s in ['X', 'B'] for s in symbols) and 'X' in symbols:
                return 'X'

            # Check for O win (all O or B)
            if all(s in ['O', 'B'] for s in symbols) and 'O' in symbols:
                return 'O'

        return None

    def is_full(self) -> bool:
        """Check if the board is full (no empty spaces)."""
        return all(cell is not None for cell in self.board)

    def is_game_over(self) -> bool:
        """Check if the game is over (winner or draw)."""
        return self.check_winner() is not None or self.is_full()

    def display(self) -> str:
        """Generate a formatted string showing current board and position reference side by side (4x4)."""
        # Create game board representation (with X's, O's, and * for dual cells)
        game_board = []
        for cell in self.board:
            if cell is None:
                game_board.append(' ')
            elif cell == 'B':
                game_board.append('*')  # Display dual cells as *
            else:
                game_board.append(cell)

        # Position reference board (always 0-15)
        pos_board = [f'{i:2d}' for i in range(16)]

        # Format both boards side by side (4x4 layout)
        board_str = "\n"
        board_str += "Current Board                  Position Reference\n"
        for row in range(4):
            # Game board row
            board_str += " "
            for col in range(4):
                idx = row * 4 + col
                board_str += f" {game_board[idx]} "
                if col < 3:
                    board_str += "|"
            board_str += "              "
            # Position reference row
            for col in range(4):
                idx = row * 4 + col
                board_str += f" {pos_board[idx]} "
                if col < 3:
                    board_str += "|"
            board_str += "\n"
            # Separator line (except after last row)
            if row < 3:
                board_str += "---------------              ------------------\n"

        return board_str

    def reset(self):
        """Reset the board to its initial empty state and reassign new dual cells."""
        self.board = [None] * 16
        self.assign_dual_cells()
