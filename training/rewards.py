"""
rewards.py
Shaped reward functions for bootstrapping chess learning.

Combines outcome-based rewards with intermediate progress signals:
- Material balance (don't lose pieces)
- Pawn advancement (push for promotion)
- Piece activity (avoid stagnation)
- Game outcome (final result)

This hybrid approach helps the model bootstrap from a random initialization
by providing learning signal even in drawn positions.
"""

import chess
import numpy as np
from typing import Dict

# Piece values (standard chess values)
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # King not counted in material
}

# Reward component weights (tunable)
# Increased intermediate rewards to provide stronger learning signal during bootstrapping
REWARD_WEIGHTS = {
    'material': 2.0,        # Core signal - don't lose pieces (increased from 1.0)
    'pawn_advancement': 0.6,  # Encourage pushing pawns (increased from 0.3)
    'piece_activity': 0.4,   # Encourage active play (increased from 0.2)
    'outcome': 0.3           # Final game result (reduced from 0.5 to emphasize position value)
}


def compute_position_value(board: chess.Board) -> float:
    """
    Compute shaped reward for a board position.

    Args:
        board: Chess board position

    Returns:
        Value from white's perspective in [-1, 1]
    """
    material = compute_material_balance(board)
    pawns = compute_pawn_advancement(board)
    activity = compute_piece_activity(board)

    value = (
        REWARD_WEIGHTS['material'] * material +
        REWARD_WEIGHTS['pawn_advancement'] * pawns +
        REWARD_WEIGHTS['piece_activity'] * activity
    )

    # Normalize to [-1, 1] range
    # Max possible: 2.0 + 0.6 + 0.4 = 3.0
    value = value / 3.0

    return float(np.clip(value, -1.0, 1.0))


def compute_material_balance(board: chess.Board) -> float:
    """
    Compute material balance from white's perspective.

    Args:
        board: Chess board position

    Returns:
        Material balance in [-1, 1], positive = white ahead
    """
    white_material = sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == chess.WHITE
    )
    black_material = sum(
        PIECE_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.color == chess.BLACK
    )

    # Starting material: 39 points each side
    # (8 pawns + 2 rooks + 2 knights + 2 bishops + 1 queen)
    balance = (white_material - black_material) / 39.0
    return float(np.clip(balance, -1.0, 1.0))


def compute_pawn_advancement(board: chess.Board) -> float:
    """
    Compute pawn advancement score from white's perspective.
    Rewards pawns that are closer to promotion.

    Args:
        board: Chess board position

    Returns:
        Pawn advancement score in [-1, 1]
    """
    score = 0.0

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            rank = chess.square_rank(square)
            if piece.color == chess.WHITE:
                # White pawns: rank 1 (start) to rank 7 (near promotion)
                # Rank 1 = 0, Rank 7 = 1.0
                score += (rank - 1) / 6.0
            else:
                # Black pawns: rank 6 (start) to rank 0 (near promotion)
                # Rank 6 = 0, Rank 0 = 1.0
                score -= (6 - rank) / 6.0

    # Normalize by max possible (8 pawns * 1.0 = 8)
    return float(np.clip(score / 8.0, -1.0, 1.0))


def compute_piece_activity(board: chess.Board) -> float:
    """
    Compute piece mobility/activity from white's perspective.
    More legal moves = better position, more active pieces.

    Args:
        board: Chess board position

    Returns:
        Activity score in [-1, 1]
    """
    try:
        # Current player's mobility
        current_mobility = board.legal_moves.count()

        # Opponent's mobility (simulate their turn)
        # Use null move if legal
        board_copy = board.copy()
        try:
            board_copy.push(chess.Move.null())
            opponent_mobility = board_copy.legal_moves.count()
        except:
            # Null move not legal (in check), just count without null move
            board_copy.turn = not board_copy.turn
            opponent_mobility = board_copy.legal_moves.count()

        # Mobility difference (typical range: 20-40 moves)
        mobility_diff = (current_mobility - opponent_mobility) / 50.0

        # Convert to white's perspective
        if board.turn == chess.WHITE:
            return float(np.clip(mobility_diff, -1.0, 1.0))
        else:
            return float(np.clip(-mobility_diff, -1.0, 1.0))
    except:
        # Fallback if anything fails
        return 0.0


def compute_game_value(
    board: chess.Board,
    game_outcome: float,
    use_shaped_rewards: bool = True
) -> float:
    """
    Compute final value for a game position.

    Args:
        board: Current board position
        game_outcome: Final game result (1.0, -1.0, or 0.0) from white's perspective
        use_shaped_rewards: If True, blend outcome with position value

    Returns:
        Value from white's perspective in [-1, 1]
    """
    if not use_shaped_rewards:
        return game_outcome

    # Blend position value with outcome
    position_val = compute_position_value(board)

    # Weighted combination
    value = (
        REWARD_WEIGHTS['outcome'] * game_outcome +
        (1 - REWARD_WEIGHTS['outcome']) * position_val
    )

    return float(np.clip(value, -1.0, 1.0))


def test_rewards():
    """Test reward functions on sample positions."""
    print("Testing reward functions:")
    print("=" * 60)

    # Test 1: Starting position (should be neutral)
    board = chess.Board()
    print("\n1. Starting position:")
    print(f"   Material: {compute_material_balance(board):.3f} (expect ~0.0)")
    print(f"   Pawn advancement: {compute_pawn_advancement(board):.3f} (expect ~0.0)")
    print(f"   Piece activity: {compute_piece_activity(board):.3f} (expect ~0.0)")
    print(f"   Overall value: {compute_position_value(board):.3f} (expect ~0.0)")

    # Test 2: White up a pawn
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    board.push_uci("e4d5")  # White captures pawn
    print("\n2. White captures pawn:")
    print(f"   Material: {compute_material_balance(board):.3f} (expect >0)")
    print(f"   Overall value: {compute_position_value(board):.3f} (expect >0)")

    # Test 3: Advanced pawn
    board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")  # White pawn on 7th rank
    print("\n3. White pawn near promotion:")
    print(f"   Pawn advancement: {compute_pawn_advancement(board):.3f} (expect >0)")
    print(f"   Overall value: {compute_position_value(board):.3f} (expect >0)")

    print("\n" + "=" * 60)
    print("Tests complete. Values should be sensible.")


if __name__ == "__main__":
    test_rewards()
