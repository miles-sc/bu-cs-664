"""
Main Entry Point for Tic-Tac-Toe Game

This script provides a simple command-line interface to start
a game with different player configurations, including Q-learning training.
"""

import os
from game import Game
from players import HumanPlayer
from agent import TicTacAgent
from q_table import QTable


def main():
    """Entry point for the game. Sets up players and starts a new game."""
    print("Welcome to Tic-Tac-Toe!")
    print("\nChoose mode:")
    print("1. Play (no learning)")
    print("2. Train agents with Q-learning")

    try:
        mode = input("\nEnter mode (1-2): ")

        if mode == "2":
            # Training mode with Q-learning
            train_agents()
        else:
            # Regular play mode
            play_game()

    except KeyboardInterrupt:
        print("\n\nGame terminated by user. Goodbye!")


def play_game():
    """Regular game play without Q-learning."""
    print("\nChoose game mode:")
    print("1. Human vs Human")
    print("2. Human vs Agent (random)")
    print("3. Agent vs Agent (random)")
    print("4. Human vs Agent (trained)")

    choice = input("\nEnter your choice (1-4): ")

    q_table = None
    learning_enabled = False

    # Option 4: Load trained agent
    if choice == "4":
        if os.path.exists("q_table.pkl"):
            q_table = QTable.load("q_table.pkl")
            learning_enabled = False  # Don't update during play
            print("Loaded trained Q-table")
        else:
            print("No trained Q-table found! Using random agent instead.")

    if choice == "1":
        player1 = HumanPlayer("X", "Player 1")
        player2 = HumanPlayer("O", "Player 2")
    elif choice == "2" or (choice == "4" and q_table is None):
        player1 = HumanPlayer("X", "Player 1")
        player2 = TicTacAgent("O", "Agent", q_table=None, learning_enabled=False)
    elif choice == "3":
        player1 = TicTacAgent("X", "Agent 1", q_table=None, learning_enabled=False)
        player2 = TicTacAgent("O", "Agent 2", q_table=None, learning_enabled=False)
        input("\nPress Enter to watch the agents play...")
    elif choice == "4" and q_table is not None:
        player1 = HumanPlayer("X", "Player 1")
        player2 = TicTacAgent("O", "Trained Agent", q_table=q_table, learning_enabled=False)
    else:
        print("Invalid choice! Defaulting to Human vs Agent")
        player1 = HumanPlayer("X", "Player 1")
        player2 = TicTacAgent("O", "Agent", q_table=None, learning_enabled=False)

    # Create and start the game
    game = Game(player1, player2, q_table=q_table)
    game.play()


def train_agents():
    """Train agents using Q-learning."""
    print("\n=== Q-Learning Training Mode ===\n")

    # Try to load existing Q-table
    q_table = None
    if os.path.exists("q_table.pkl"):
        load_existing = input("Found existing Q-table. Load it? (y/n): ")
        if load_existing.lower() == 'y':
            q_table = QTable.load("q_table.pkl")
            print(f"Loaded Q-table with {q_table.get_table_size()} entries")
            print(f"Current hyperparameters: alpha={q_table.alpha}, gamma={q_table.gamma}, epsilon={q_table.epsilon}")

    # Create new Q-table if not loaded
    if q_table is None:
        print("\nConfigure hyperparameters:")
        alpha = float(input("Learning rate (alpha) [0.1]: ") or "0.1")
        gamma = float(input("Discount factor (gamma) [0.9]: ") or "0.9")
        epsilon = float(input("Exploration rate (epsilon) [0.3]: ") or "0.3")
        q_table = QTable(alpha=alpha, gamma=gamma, epsilon=epsilon)
        print("Created new Q-table")

    # Training configuration
    num_games = int(input("\nNumber of games to train: "))
    silent_mode = input("Silent mode? (only show progress, not game boards) (y/n): ")
    silent = silent_mode.lower() == 'y'

    # Track statistics
    wins_x = 0
    wins_o = 0
    draws = 0

    print(f"\nTraining for {num_games} games...")
    print(f"Hyperparameters: alpha={q_table.alpha}, gamma={q_table.gamma}, epsilon={q_table.epsilon}\n")

    for i in range(num_games):
        agent1 = TicTacAgent("X", "Agent 1", q_table=q_table, learning_enabled=True)
        agent2 = TicTacAgent("O", "Agent 2", q_table=q_table, learning_enabled=True)

        # Create game with silent mode option
        game = Game(agent1, agent2, q_table=q_table)

        # Temporarily redirect output if silent
        if silent:
            import sys
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        game.play()

        # Restore output and track stats
        if silent:
            sys.stdout = old_stdout

        # Track statistics
        winner = game.board.check_winner()
        if winner == 'X':
            wins_x += 1
        elif winner == 'O':
            wins_o += 1
        else:
            draws += 1

        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{num_games} games")
            print(f"  Stats: X wins: {wins_x}, O wins: {wins_o}, Draws: {draws}")
            print(f"  Q-table size: {q_table.get_table_size()} entries")

    # Final statistics
    print(f"\n=== Training Complete ===")
    print(f"Total games: {num_games}")
    print(f"X wins: {wins_x} ({100*wins_x/num_games:.1f}%)")
    print(f"O wins: {wins_o} ({100*wins_o/num_games:.1f}%)")
    print(f"Draws: {draws} ({100*draws/num_games:.1f}%)")
    print(f"Final Q-table size: {q_table.get_table_size()} entries")

    # Save Q-table
    save_choice = input("\nSave Q-table? (y/n): ")
    if save_choice.lower() == 'y':
        q_table.save("q_table.pkl")
        print("Q-table saved to q_table.pkl!")


if __name__ == "__main__":
    main()