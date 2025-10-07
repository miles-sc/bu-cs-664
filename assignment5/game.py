"""
Game Module

This module contains the Game class which orchestrates the
tic-tac-toe game flow and coordinates between players and the board.
"""

from board import Board
from players import Player


class Game:
    """Orchestrates the tic-tac-toe game flow.
    Manages game loop, coordinates players and board, displays state, and declares results."""

    def __init__(self, player1: Player, player2: Player):
        """Initialize a new game with player1 ('X') and player2 ('O')."""
        self.board = Board()
        self.players = [player1, player2]
        self.current_player_index = 0

    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_player_index]

    def switch_player(self):
        """Switch to the other player."""
        self.current_player_index = 1 - self.current_player_index

    def play_turn(self):
        """Execute a single turn: display board, get move, and update board."""
        # Show the current board state
        print(self.board.display())

        # Get the current player's move
        current_player = self.get_current_player()
        position = current_player.get_move(self.board)

        # Make the move
        self.board.make_move(position, current_player.symbol)

    def play(self):
        """Main game loop. Runs until game over, then displays final result."""
        print("\n=== TIC-TAC-TOE ===\n")
        print(f"{self.players[0].name} is {self.players[0].symbol}")
        print(f"{self.players[1].name} is {self.players[1].symbol}")
        print("\nPositions are numbered 0-8:")
        print(" 0 | 1 | 2 ")
        print("-----------")
        print(" 3 | 4 | 5 ")
        print("-----------")
        print(" 6 | 7 | 8 \n")

        # Main game loop
        while not self.board.is_game_over():
            self.play_turn()
            self.switch_player()

        # Game is over - display final board and result
        print("\n=== GAME OVER ===")
        print(self.board.display())

        winner = self.board.check_winner()
        if winner:
            # Find which player won
            winning_player = next(p for p in self.players if p.symbol == winner)
            print(f"\n{winning_player.name} ({winner}) wins!")
        else:
            print("\nIt's a draw!")