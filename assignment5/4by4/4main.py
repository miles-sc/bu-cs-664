"""
Main Entry Point for Tic-Tac-Toe Game

This script provides a simple command-line interface to start
a game with different player configurations, including Q-learning training.
"""

import os
import importlib.util

# Import modules with "4" prefix by loading them dynamically
def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, file_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load all modules
game_module = load_module("game", os.path.join(script_dir, "4game.py"))
players_module = load_module("players", os.path.join(script_dir, "4players.py"))
agent_module = load_module("agent", os.path.join(script_dir, "4agent.py"))
qtable_module = load_module("q_table", os.path.join(script_dir, "4q_table.py"))
metrics_module = load_module("metrics", os.path.join(script_dir, "4metrics.py"))

# Extract classes
Game = game_module.Game
HumanPlayer = players_module.HumanPlayer
TicTacAgent = agent_module.TicTacAgent
QTable = qtable_module.QTable
LearningMetrics = metrics_module.LearningMetrics


def main():
    """Entry point for the game. Sets up players and starts a new game."""
    print("Welcome to 4x4 Tic-Tac-Toe!")
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

    # Configure number of dual cells
    num_dual_cells = int(input("Number of dual cells [2]: ") or "2")

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
    game = Game(player1, player2, q_table=q_table, num_dual_cells=num_dual_cells)
    game.play()


def train_agents():
    """Train agents using Q-learning with configurable hyperparameters and progress tracking."""
    print("\n=== Q-Learning Training Mode (4x4) ===\n")

    # Configure number of dual cells
    num_dual_cells = int(input("Number of dual cells per game [2]: ") or "2")
    print(f"Using {num_dual_cells} dual cells per game\n")

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
        epsilon = float(input("Exploration rate (epsilon) [0.2]: ") or "0.2")
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

    # Initialize metrics tracking
    metrics = LearningMetrics()
    interval = max(1, num_games // 5)  # Display metrics 5 times during training

    print(f"\nTraining for {num_games} games...")
    print(f"Hyperparameters: alpha={q_table.alpha}, gamma={q_table.gamma}, epsilon={q_table.epsilon}")
    print(f"Metrics will be displayed every {interval} games\n")

    for i in range(num_games):
        agent1 = TicTacAgent("X", "Agent 1", q_table=q_table, learning_enabled=True)
        agent2 = TicTacAgent("O", "Agent 2", q_table=q_table, learning_enabled=True)

        # Pass metrics to agents for action tracking
        agent1.metrics = metrics
        agent2.metrics = metrics

        # Create game with silent mode option and dual cells
        game = Game(agent1, agent2, q_table=q_table, num_dual_cells=num_dual_cells)

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

        # Display metrics at intervals
        if (i + 1) % interval == 0:
            metrics.record_strategic_qvalues(q_table, i + 1)
            metrics.record_qtable_size(q_table.get_table_size())
            metrics.display_metrics(i + 1, num_games)
            metrics.reset_action_counts()  # Reset for next interval

    # Final statistics and metrics
    print(f"\n{'='*60}")
    print(f"=== Training Complete ===")
    print(f"{'='*60}")
    print(f"Total games: {num_games}")
    print(f"X wins: {wins_x} ({100*wins_x/num_games:.1f}%)")
    print(f"O wins: {wins_o} ({100*wins_o/num_games:.1f}%)")
    print(f"Draws: {draws} ({100*draws/num_games:.1f}%)")
    print(f"\nFinal Q-table size: {q_table.get_table_size()} entries")

    # Show final metrics
    metrics.record_strategic_qvalues(q_table, num_games)
    metrics.record_qtable_size(q_table.get_table_size())
    # print(f"\nFinal Strategic Learning:")
    # if metrics.can_win_best:
    #     action, qval = metrics.can_win_best[-1]
    #     print(f"  'Can Win' --> {action:20s} (Q={qval:>7.4f})")
    # if metrics.center_available_best:
    #     action, qval = metrics.center_available_best[-1]
    #     print(f"  'Middle Zone Available' --> {action:20s} (Q={qval:>7.4f})")
    # if metrics.must_block_best:
    #     action, qval = metrics.must_block_best[-1]
    #     print(f"  'Must Block' --> {action:20s} (Q={qval:>7.4f})")
    # print(f"{'='*60}\n")

    # Save Q-table
    save_choice = input("\nSave Q-table? (y/n): ")
    if save_choice.lower() == 'y':
        q_table.save("q_table.pkl")
        print("Q-table saved to q_table.pkl!")


if __name__ == "__main__":
    main()
