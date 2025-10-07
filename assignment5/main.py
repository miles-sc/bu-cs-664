"""
Main Entry Point for Tic-Tac-Toe Game

This script provides a simple command-line interface to start
a game with different player configurations.
"""

from game import Game
from players import HumanPlayer, NPCPlayer


def main():
    """Entry point for the game. Sets up players and starts a new game."""
    print("Welcome to Tic-Tac-Toe!")
    print("\nChoose game mode:")
    print("1. Human vs Human")
    print("2. Human vs NPC")
    print("3. NPC vs NPC")

    try:
        choice = input("\nEnter your choice (1-3): ")

        if choice == "1":
            player1 = HumanPlayer("X", "Player 1")
            player2 = HumanPlayer("O", "Player 2")
        elif choice == "2":
            player1 = HumanPlayer("X", "Player 1")
            player2 = NPCPlayer("O", "Computer")
        elif choice == "3":
            player1 = NPCPlayer("X", "Computer 1")
            player2 = NPCPlayer("O", "Computer 2")
            # Add a small delay for NPC vs NPC to make it watchable
            input("\nPress Enter to watch the computers play...")
        else:
            print("Invalid choice! Defaulting to Human vs NPC")
            player1 = HumanPlayer("X", "Player 1")
            player2 = NPCPlayer("O", "Computer")

        # Create and start the game
        game = Game(player1, player2)
        game.play()

    except KeyboardInterrupt:
        print("\n\nGame terminated by user. Goodbye!")


if __name__ == "__main__":
    main()