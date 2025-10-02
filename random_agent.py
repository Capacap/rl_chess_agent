
"""
random_agent.py
A baseline agent that selects a legal move uniformly at random.
"""

import random
import chess
from agent_interface import Agent

class RandomAgent(Agent):
    """
    An agent that picks a random move from the list of legal moves.
    This agent ignores the board state and the time limit.
    """

    def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Chooses a random legal move.

        Args:
        board (chess.Board): The current game board.
        time_limit (float): Ignored by this agent.

        Returns:
        chess.Move: A randomly selected legal move.
        """
        # Get a list of all legal moves for the current position
        legal_moves = list(board.legal_moves)

        # Randomly choose one of the legal moves
        chosen_move = random.choice(legal_moves)

        return chosen_move

# Example of how to test the agent
if __name__ == "__main__":
    # Create a new board
    test_board = chess.Board()
    # Create an instance of the RandomAgent playing as White
    agent = RandomAgent(test_board, chess.WHITE)
    # Get a move with a 2-second time limit
    move = agent.make_move(test_board, 2.0)
    print(f"RandomAgent chose move: {move}")
