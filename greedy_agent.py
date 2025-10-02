"""
greedy_agent.py
A baseline agent that uses a simple material-counting heuristic.
It always captures the highest-value piece available. If no capture is available, it moves randomly.
"""

import random
from agent_interface import Agent
import chess

class GreedyAgent(Agent):
    """
    An agent that uses a greedy material heuristic.
    Piece Values: Pawn=1, Knight=3, Bishop=3, Rook=5, Queen=9, King=0 (infinite, don't capture)
    """

    # Class-level dictionary for piece values
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0 # Ignore king "value" for captures to avoid bad trades
    }

    def __init__(self, board: chess.Board, color: chess.Color):
        super().__init__(board, color)

    def _get_piece_value(self, piece_type: chess.PieceType) -> int:
        """Returns the material value of a piece type."""
        return self.PIECE_VALUES.get(piece_type, 0)

    def _score_move(self, board: chess.Board, move: chess.Move) -> int:
        """
        Scores a move based on material gain.
        Returns the value of the captured piece, if any.
        Returns 0 for non-captures or if we move into a capture (we only count the immediate capture).
        """
        # Check if the move is a capture
        if board.is_capture(move):
            # Use the `capture()` method for a more reliable way to get the captured piece
            captured_piece_type = board.piece_type_at(move.to_square)
            if captured_piece_type:
                return self._get_piece_value(captured_piece_type)
        return 0

    def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Chooses the move that captures the highest-value piece.
        If multiple moves capture the same value, it chooses randomly among them.
        If no capturing moves are available, it chooses a random move.

        Args:
        board (chess.Board): The current game board.
        time_limit (float): Ignored by this agent.

        Returns:
        chess.Move: The chosen move.
        """
        legal_moves = list(board.legal_moves)
        best_move_score = -1
        best_moves = [] # List to hold all moves with the best score

        # Score every move
        for move in legal_moves:
            score = self._score_move(board, move)
            # If this move has a higher score than current best, start a new list
            if score > best_move_score:
                best_move_score = score
                best_moves = [move]
            # If it ties the best score, add it to the list
            elif score == best_move_score:
                best_moves.append(move)

        # If we have capturing moves (best_move_score > 0), choose the best one randomly
        if best_move_score > 0:
            chosen_move = random.choice(best_moves)
        else:
            # Otherwise, just move randomly
            chosen_move = random.choice(legal_moves)

        return chosen_move

# Example of how to test the agent
if __name__ == "__main__":
    test_board = chess.Board()
    agent = GreedyAgent(test_board, chess.WHITE)
    move = agent.make_move(test_board, 2.0)
    print(f"GreedyAgent chose move: {move}")