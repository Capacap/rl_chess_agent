"""
move.py
Encode chess moves as action indices for neural network output.

Encoding scheme:
- Flat 4096 action space: from_square * 64 + to_square
- from_square, to_square ∈ [0, 63]
- Underpromotions handled separately if needed (reserved space up to 4288)
"""

import chess
from typing import Optional


def encode_move(move: chess.Move) -> int:
    """
    Map chess.Move to action index [0, 4095].

    Args:
        move: Chess move to encode

    Returns:
        Action index in range [0, 4095]
        - Standard moves: from_square * 64 + to_square
        - Queen promotions: Same encoding
        - Underpromotions: Extended encoding (4096+)

    Examples:
        e2e4 (white pawn): encode_move(Move.from_uci("e2e4")) -> 12*64 + 28 = 796
    """
    from_square = move.from_square
    to_square = move.to_square

    # Base encoding: from * 64 + to
    action = from_square * 64 + to_square

    # TODO: Handle underpromotions (knight, bishop, rook)
    # if move.promotion and move.promotion != chess.QUEEN:
    #     action = 4096 + offset

    return action


def decode_move(action: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Map action index to chess.Move. Returns None if illegal.

    Args:
        action: Index in [0, 4095] (or extended range for underpromotions)
        board: Board state for legality check

    Returns:
        chess.Move if legal, None otherwise

    Validation:
        - Checks if decoded move is in board.legal_moves
        - Returns None for illegal moves (parse, don't validate)
    """
    # Handle underpromotions in extended range [4096, 4288)
    if action >= 4096:
        # Extended encoding for underpromotions
        # Simplified: treat as out of range for now
        return None

    # Standard decoding
    from_square = action // 64
    to_square = action % 64

    # Validate square indices
    if not (0 <= from_square < 64 and 0 <= to_square < 64):
        return None

    try:
        # Try basic move first
        move = chess.Move(from_square, to_square)
        if move in board.legal_moves:
            return move

        # Check for promotion moves
        # Promotion rank: rank 7 for white (squares 56-63), rank 0 for black (squares 0-7)
        to_rank = chess.square_rank(to_square)
        from_rank = chess.square_rank(from_square)

        # Check if this could be a pawn promotion
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            # White pawn promoting (moving to rank 7)
            if piece.color == chess.WHITE and to_rank == 7:
                # Try queen promotion (default)
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                if move in board.legal_moves:
                    return move

            # Black pawn promoting (moving to rank 0)
            elif piece.color == chess.BLACK and to_rank == 0:
                # Try queen promotion (default)
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                if move in board.legal_moves:
                    return move

        return None
    except:
        return None


def create_legal_move_mask(board: chess.Board) -> list[bool]:
    """
    Create boolean mask for legal moves in current position.

    Args:
        board: Current board state

    Returns:
        List of 4096 booleans, True if move is legal

    Usage:
        Used to mask neural network policy output via masked_fill
    """
    mask = [False] * 4096

    # Mark all legal moves as True
    for move in board.legal_moves:
        action = encode_move(move)
        if 0 <= action < 4096:
            mask[action] = True

    return mask
