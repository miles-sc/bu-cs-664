"""
Main Entry Point for Tic-Tac-Toe Game

This script provides a simple command-line interface to start
a game with different player configurations.
"""

from game import Game
from players import HumanPlayer, TicTacAgent


def main():
    """Entry point for the game. Sets up players and starts a new game."""
    print("Welcome to Tic-Tac-Toe!")
    print("\nChoose game mode:")
    print("1. Human vs Human")
    print("2. Human vs Agent")
    print("3. Agent vs Agent")

    try:
        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            player1 = HumanPlayer("X", "Player 1")
            player2 = HumanPlayer("O", "Player 2")
        elif choice == "2":
            player1 = HumanPlayer("X", "Player 1")
            player2 = TicTacAgent("O", "Agent")
        elif choice == "3":
            player1 = TicTacAgent("X", "Agent 1")
            player2 = TicTacAgent("O", "Agent 2")
            # Add a small delay for Agent vs Agent to make it watchable
            input("\nPress Enter to watch the agents play...")
        else:
            print("Invalid choice! Defaulting to Human vs Agent")
            player1 = HumanPlayer("X", "Player 1")
            player2 = TicTacAgent("O", "Agent")

        # Create and start the game
        game = Game(player1, player2)
        game.play()

    except KeyboardInterrupt:
        print("\n\nGame terminated by user. Goodbye!")


if __name__ == "__main__":
    main()