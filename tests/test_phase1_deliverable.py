"""
test_phase1_deliverable.py
Integration test for Phase 1 deliverable.

Tests full pipeline: Board → Encode → Network → Decode → Move
"""

import chess
import torch
import pytest
from encoding.state import encode_board
from encoding.move import encode_move, decode_move, create_legal_move_mask
from model.network import ChessNet


class TestPhase1Integration:
    """End-to-end integration tests."""

    def test_full_pipeline_starting_position(self):
        """Test complete pipeline from board to move selection."""
        board = chess.Board()

        # Step 1: Encode board state
        state = encode_board(board)
        assert state.shape == (1, 13, 8, 8)

        # Step 2: Create legal move mask
        legal_moves = list(board.legal_moves)
        legal_mask = torch.zeros(1, 4096, dtype=torch.bool)
        for move in legal_moves:
            action = encode_move(move)
            legal_mask[0, action] = True

        # Step 3: Get network prediction (untrained)
        net = ChessNet(channels=64, num_blocks=4)
        net.eval()

        # TODO: Uncomment when implemented
        # with torch.no_grad():
        #     policy, value = net(state, legal_mask)

        # Step 4: Sample move from policy
        # TODO: Uncomment when implemented
        # action = torch.multinomial(policy, 1).item()
        # selected_move = decode_move(action, board)

        # Step 5: Verify move is legal
        # TODO: Uncomment when implemented
        # assert selected_move is not None
        # assert selected_move in board.legal_moves

    def test_pipeline_midgame_position(self):
        """Test pipeline after several moves."""
        board = chess.Board()
        # Play some moves
        board.push_san("e4")
        board.push_san("e5")
        board.push_san("Nf3")
        board.push_san("Nc6")

        # TODO: Run same pipeline as above
        pass

    def test_pipeline_does_not_select_illegal_move(self):
        """Verify network never selects illegal moves."""
        board = chess.Board()

        # Run pipeline 100 times
        # TODO: Verify all selected moves are legal
        pass

    def test_pipeline_handles_checkmate(self):
        """Test pipeline handles checkmate positions."""
        # Fool's mate position
        board = chess.Board()
        board.push_san("f3")
        board.push_san("e5")
        board.push_san("g4")
        board.push_san("Qh4#")

        assert board.is_checkmate()

        # Encoding should still work
        state = encode_board(board)
        assert state.shape == (1, 13, 8, 8)

    def test_pipeline_handles_stalemate(self):
        """Test pipeline handles stalemate positions."""
        # Set up stalemate position
        # TODO: Create stalemate position
        # TODO: Verify encoding works
        pass

    def test_value_head_produces_scores(self):
        """Value head should produce meaningful scores."""
        board = chess.Board()
        state = encode_board(board)
        legal_mask = torch.ones(1, 4096, dtype=torch.bool)

        net = ChessNet()
        net.eval()

        # TODO: Uncomment when implemented
        # with torch.no_grad():
        #     _, value = net(state, legal_mask)
        #
        # assert -1.0 <= value.item() <= 1.0
        # # Starting position should be close to 0 (equal)
        # # (untrained network might not satisfy this)


class TestPhase1Performance:
    """Performance tests for Phase 1."""

    def test_inference_speed_requirement(self):
        """Network inference must be <50ms on CPU."""
        import time

        net = ChessNet(channels=64, num_blocks=4)
        net.eval()

        board = chess.Board()
        state = encode_board(board)
        legal_mask = torch.ones(1, 4096, dtype=torch.bool)

        # TODO: Uncomment when implemented
        # # Warmup
        # with torch.no_grad():
        #     _ = net(state, legal_mask)
        #
        # # Measure 100 iterations
        # times = []
        # for _ in range(100):
        #     start = time.perf_counter()
        #     with torch.no_grad():
        #         _ = net(state, legal_mask)
        #     times.append(time.perf_counter() - start)
        #
        # avg_time = sum(times) / len(times)
        # print(f"\nAverage inference time: {avg_time*1000:.1f}ms")
        # assert avg_time < 0.050, f"Inference too slow: {avg_time*1000:.1f}ms"

    def test_memory_usage(self):
        """Network should fit in <2GB RAM."""
        # TODO: Measure memory usage
        # TODO: Assert < 2GB
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
