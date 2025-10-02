"""
test_encoding.py
Unit tests for state and move encoding.
"""

import chess
import torch
import pytest
from encoding.state import encode_board
from encoding.move import encode_move, decode_move, create_legal_move_mask


class TestStateEncoding:
    """Test board state encoding."""

    def test_encode_board_shape(self):
        """Verify output shape is [1, 13, 8, 8]."""
        board = chess.Board()
        state = encode_board(board)
        assert state.shape == (1, 13, 8, 8)
        assert state.dtype == torch.float32

    def test_encode_board_determinism(self):
        """Same board state should produce identical tensors."""
        board = chess.Board()
        state1 = encode_board(board)
        state2 = encode_board(board)
        assert torch.equal(state1, state2)

    def test_encode_board_starting_position(self):
        """Test encoding of starting chess position."""
        # TODO: Verify piece positions are encoded correctly
        # TODO: Check white pieces in planes 0-5
        # TODO: Check black pieces in planes 6-11
        # TODO: Verify metadata plane
        pass

    def test_encode_board_midgame(self):
        """Test encoding after some moves."""
        # TODO: Play a few moves and verify encoding
        pass

    def test_encode_board_endgame(self):
        """Test encoding in endgame position."""
        # TODO: Set up endgame position and verify
        pass


class TestMoveEncoding:
    """Test move encoding and decoding."""

    def test_encode_move_basic(self):
        """Test basic move encoding."""
        move = chess.Move.from_uci("e2e4")
        action = encode_move(move)
        assert 0 <= action < 4096

    def test_decode_move_basic(self):
        """Test basic move decoding."""
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        action = encode_move(move)
        decoded = decode_move(action, board)
        # TODO: Uncomment when implemented
        # assert decoded == move

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode is identity for legal moves."""
        board = chess.Board()
        for move in board.legal_moves:
            action = encode_move(move)
            decoded = decode_move(action, board)
            # TODO: Uncomment when implemented
            # assert decoded == move

    def test_decode_illegal_move(self):
        """Decoding illegal move should return None."""
        board = chess.Board()
        # Action for move that's illegal in starting position
        # TODO: Find an illegal action and test
        pass

    def test_create_legal_move_mask(self):
        """Test legal move mask creation."""
        board = chess.Board()
        mask = create_legal_move_mask(board)
        assert len(mask) == 4096
        assert isinstance(mask[0], bool)
        # TODO: Verify correct moves are masked
        # TODO: Count should match len(board.legal_moves)


class TestPromotions:
    """Test move encoding for promotions."""

    def test_encode_queen_promotion(self):
        """Test queen promotion encoding."""
        # TODO: Set up position with pawn on 7th rank
        # TODO: Test promotion move
        pass

    def test_encode_underpromotion(self):
        """Test knight/bishop/rook promotion."""
        # TODO: Test underpromotion moves
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
