"""
Interactive CLI chatbot interface for the Natlang customer service agent.
Run this script to have a conversation with the chatbot.
"""

# Suppress gRPC and Google Cloud warnings
import os
import sys
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

# Suppress gRPC/ALTS warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GRPC_ENABLE_FORK_SUPPORT'] = '0'

# Redirect stderr temporarily to suppress initialization warnings
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

from natlang import Natlang

# Restore stderr after import
sys.stderr.close()
sys.stderr = stderr


def print_welcome():
    """Print welcome message and instructions."""
    print("\n" + "="*80)
    print(" " * 20 + "NATLANG: CUSTOMER SERVICE CHATBOT")
    print("="*80)
    print("\nWelcome! I'm here to help you with any questions or issues.")
    print("\nType 'quit' or 'exit' to end the conversation.")
    print("="*80 + "\n")


def print_bot_response(response: str):
    """Print the chatbot's response in a formatted way."""
    print("\n[NATLANG]")
    print(response)
    print()


def main():
    """Run the interactive chatbot CLI."""
    print_welcome()

    try:
        # Initialize the chatbot
        chatbot = Natlang()

        # Main conversation loop
        while True:
            # Get user input
            try:
                user_input = input("[YOU] ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                print("\n[CHATBOT]")
                print("Thank you for contacting customer service. Have a great day!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            try:
                # Process the input and get response
                response = chatbot.process_input(user_input)
                print_bot_response(response)

                # Check if the chatbot routed to a human agent
                if chatbot.routed_to_human:
                    print("="*80)
                    print("Session ended - transferring to live agent.")
                    print("="*80 + "\n")
                    break

            except Exception as e:
                print("\n[ERROR]")
                print(f"I apologize, but I encountered an error processing your request: {e}")
                print("Please try again or type 'quit' to exit.\n")

    except Exception as e:
        print(f"\n[FATAL ERROR]")
        print(f"Failed to initialize chatbot: {e}")
        print("Please check your API key configuration and try again.\n")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
