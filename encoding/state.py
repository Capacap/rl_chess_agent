"""
state.py
Encode chess board state as tensor representation for neural network input.

Encoding scheme:
- 13 planes of 8x8 each
- Planes 0-5: White pieces (P, N, B, R, Q, K)
- Planes 6-11: Black pieces (P, N, B, R, Q, K)
- Plane 12: Metadata (castling rights, side to move)
"""

import chess
import torch
import numpy as np


def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode board state as 13×8×8 tensor.

    Args:
        board: Chess board state to encode

    Returns:
        Tensor of shape [1, 13, 8, 8], dtype float32
        - Batch dimension added for compatibility with neural network
        - Values are binary (0.0 or 1.0) except metadata plane

    Invariant:
        Output is deterministic for same board state
    """
    # Initialize tensor: [13, 8, 8]
    planes = torch.zeros((13, 8, 8), dtype=torch.float32)

    # Encode pieces into planes 0-11
    _encode_pieces(board, planes)

    # Encode metadata into plane 12
    _encode_metadata(board, planes)

    # Add batch dimension: [1, 13, 8, 8]
    return planes.unsqueeze(0)


def _encode_pieces(board: chess.Board, planes: torch.Tensor) -> None:
    """
    Helper: Encode piece positions into planes 0-11.

    Args:
        board: Chess board state
        planes: Tensor to fill (modified in-place)
    """
    # Map piece types to plane indices
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Iterate through all squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            # Determine plane index
            plane_idx = piece_to_plane[piece.piece_type]
            if piece.color == chess.BLACK:
                plane_idx += 6  # Black pieces in planes 6-11

            # Convert square to (rank, file) coordinates
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            # Set plane value to 1.0
            planes[plane_idx, rank, file] = 1.0


def _encode_metadata(board: chess.Board, planes: torch.Tensor) -> None:
    """
    Helper: Encode metadata into plane 12.

    Args:
        board: Chess board state
        planes: Tensor to fill (modified in-place)
    """
    # Encode metadata as binary features tiled across 8x8
    # Bit 0: White kingside castling
    # Bit 1: White queenside castling
    # Bit 2: Black kingside castling
    # Bit 3: Black queenside castling
    # Bit 4: Side to move (1=white, 0=black)

    value = 0.0

    # Castling rights (4 bits, each worth 0.0625)
    if board.has_kingside_castling_rights(chess.WHITE):
        value += 0.0625
    if board.has_queenside_castling_rights(chess.WHITE):
        value += 0.0625
    if board.has_kingside_castling_rights(chess.BLACK):
        value += 0.0625
    if board.has_queenside_castling_rights(chess.BLACK):
        value += 0.0625

    # Side to move (1 bit worth 0.25)
    if board.turn == chess.WHITE:
        value += 0.25

    # Tile value across entire 8x8 plane
    planes[12, :, :] = value
