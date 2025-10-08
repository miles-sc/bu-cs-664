"""
Game Module

This module contains the Game class which orchestrates the
tic-tac-toe game flow and coordinates between players and the board.
"""

from typing import TYPE_CHECKING
from board import Board
from players import Player

if TYPE_CHECKING:
    from agent import TicTacAgent


class Game:
    """Orchestrates the tic-tac-toe game flow.
    Manages game loop, coordinates players and board, displays state, and declares results."""

    def __init__(self, player1: Player, player2: Player, q_table=None):
        """Initialize a new game with player1 ('X'), player2 ('O'), and optional Q-table."""
        self.board = Board()
        self.players = [player1, player2]
        self.current_player_index = 0
        self.q_table = q_table

        # Import here to avoid circular dependency
        from agent import TicTacAgent

        # If agents have Q-learning, ensure they reference the same Q-table
        for player in self.players:
            if isinstance(player, TicTacAgent) and player.q_table is None:
                player.q_table = q_table

    def get_current_player(self) -> Player:
        """Get the player whose turn it is."""
        return self.players[self.current_player_index]

    def switch_player(self):
        """Switch to the other player."""
        self.current_player_index = 1 - self.current_player_index

    def play_turn(self):
        """Execute a single turn with TD(0) updates."""
        from agent import TicTacAgent

        # Show the current board state
        print(self.board.display())

        current_player = self.get_current_player()

        # Player makes move (stores state and action internally)
        position = current_player.get_move(self.board)
        self.board.make_move(position, current_player.symbol)

        # Check if game ended
        game_over = self.board.is_game_over()
        winner = self.board.check_winner()

        # Determine reward for current player
        if game_over:
            if winner == current_player.symbol:
                reward = 1.0  # Win
            elif winner is None:
                reward = 0.0  # Draw
            else:
                reward = -1.0  # Loss (shouldn't happen here)
            next_state = None
            next_board = None
        else:
            reward = 0.0  # Intermediate move
            if isinstance(current_player, TicTacAgent):
                next_state = current_player.assess_game_state(self.board)
            else:
                next_state = None
            next_board = self.board

        # TD(0) update for current player's move
        if isinstance(current_player, TicTacAgent):
            current_player.update_from_transition(reward, next_state, next_board)

        # If game over, also update opponent with their loss/draw reward
        if game_over:
            opponent = self.players[1 - self.current_player_index]
            if isinstance(opponent, TicTacAgent):
                if winner == current_player.symbol:
                    opponent_reward = -1.0  # Opponent lost
                elif winner is None:
                    opponent_reward = 0.0   # Draw
                else:
                    opponent_reward = 1.0   # Opponent won (shouldn't happen)
                opponent.update_from_transition(opponent_reward, None, None)

    def play(self):
        """Main game loop with Q-learning support."""
        from agent import TicTacAgent

        print("\n=== TIC-TAC-TOE ===\n")
        print(f"{self.players[0].name} is {self.players[0].symbol}")
        print(f"{self.players[1].name} is {self.players[1].symbol}")
        print("\nPositions are numbered 0-8:")
        print(" 0 | 1 | 2 ")
        print("-----------")
        print(" 3 | 4 | 5 ")
        print("-----------")
        print(" 6 | 7 | 8 \n")

        # Reset episode for all agents
        for player in self.players:
            if isinstance(player, TicTacAgent):
                player.reset_episode()

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