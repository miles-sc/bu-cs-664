"""
Game Module

This module contains the Game class which orchestrates the
tic-tac-toe game flow and coordinates between players and the board.
"""

import os
import importlib.util
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from typing import TYPE_CHECKING
    agent_module = load_module("agent", "4agent.py")
    TicTacAgent = agent_module.TicTacAgent


class Game:
    """Orchestrates the tic-tac-toe game flow.
    Manages game loop, coordinates players and board, displays state, and declares results."""

    def __init__(self, player1: Player, player2: Player, q_table=None, num_dual_cells: int = 2):
        """Initialize a new game with player1 ('X'), player2 ('O'), optional Q-table, and dual cells."""
        self.board = Board(num_dual_cells=num_dual_cells)
        self.players = [player1, player2]
        self.current_player_index = 0
        self.q_table = q_table

        # Import here to avoid circular dependency
        agent_module = load_module("agent", "4agent.py")
        TicTacAgent = agent_module.TicTacAgent

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
        agent_module = load_module("agent", "4agent.py")
        TicTacAgent = agent_module.TicTacAgent

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
        agent_module = load_module("agent", "4agent.py")
        TicTacAgent = agent_module.TicTacAgent

        print("\n=== 4x4 TIC-TAC-TOE ===\n")
        print(f"{self.players[0].name} is {self.players[0].symbol}")
        print(f"{self.players[1].name} is {self.players[1].symbol}")
        print("\nPositions are numbered 0-15:")
        print("  0 |  1 |  2 |  3 ")
        print("------------------")
        print("  4 |  5 |  6 |  7 ")
        print("------------------")
        print("  8 |  9 | 10 | 11 ")
        print("------------------")
        print(" 12 | 13 | 14 | 15 \n")

        # Assign dual cells for this game
        self.board.assign_dual_cells()
        if self.board.dual_cell_positions:
            print(f"Dual cells (count as both X and O) at positions: {sorted(self.board.dual_cell_positions)}")
            print("These positions cannot be played on.\n")

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