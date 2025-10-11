"""
test_agent_integration.py
Integration tests for tournament agent.
"""

import pytest
import chess
import time
import os
from pathlib import Path


class TestAgentIntegration:
    """Integration tests for MyAwesomeAgent."""

    def test_agent_can_load_checkpoint(self):
        """Test: Agent can load a pickle checkpoint."""
        # Ensure we have a checkpoint
        checkpoint_files = list(Path("checkpoints").glob("**/*.pkl"))
        if not checkpoint_files:
            pytest.skip("No .pkl checkpoint available for testing")

        from my_agent import MyAwesomeAgent

        board = chess.Board()
        agent = MyAwesomeAgent(board, chess.WHITE)

        assert agent.network is not None, "Network not loaded"
        assert agent.network.channels > 0, "Invalid network architecture"

    def test_agent_makes_legal_moves(self):
        """Test: Agent produces only legal moves."""
        checkpoint_files = list(Path("checkpoints").glob("**/*.pkl"))
        if not checkpoint_files:
            pytest.skip("No .pkl checkpoint available for testing")

        from my_agent import MyAwesomeAgent

        board = chess.Board()
        agent = MyAwesomeAgent(board, chess.WHITE)

        # Test multiple positions
        for _ in range(5):
            move = agent.make_move(board, time_limit=2.0)
            assert move in board.legal_moves, f"Illegal move: {move}"
            board.push(move)

    def test_agent_respects_time_limit(self):
        """Test: Agent completes moves within time limit."""
        checkpoint_files = list(Path("checkpoints").glob("**/*.pkl"))
        if not checkpoint_files:
            pytest.skip("No .pkl checkpoint available for testing")

        from my_agent import MyAwesomeAgent

        board = chess.Board()
        agent = MyAwesomeAgent(board, chess.WHITE)

        # Test with strict time limit
        time_limit = 2.0
        start = time.time()
        move = agent.make_move(board, time_limit=time_limit)
        elapsed = time.time() - start

        assert elapsed < time_limit, f"Agent exceeded time limit: {elapsed:.2f}s > {time_limit}s"
        assert move in board.legal_moves, "Move is illegal"

    def test_agent_handles_different_positions(self):
        """Test: Agent handles various game positions."""
        checkpoint_files = list(Path("checkpoints").glob("**/*.pkl"))
        if not checkpoint_files:
            pytest.skip("No .pkl checkpoint available for testing")

        from my_agent import MyAwesomeAgent

        test_positions = [
            # Starting position
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            # Midgame
            "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            # Endgame
            "8/5k2/8/5K2/8/8/8/8 w - - 0 1",
        ]

        for fen in test_positions:
            board = chess.Board(fen)
            agent = MyAwesomeAgent(board, board.turn)
            move = agent.make_move(board, time_limit=2.0)
            assert move in board.legal_moves, f"Illegal move in position: {fen}"

    def test_checkpoint_size_within_limits(self):
        """Test: Checkpoint file size is within tournament limits."""
        checkpoint_files = list(Path("checkpoints").glob("**/*.pkl"))
        if not checkpoint_files:
            pytest.skip("No .pkl checkpoint available for testing")

        size_limit_bytes = 2 * 1024 * 1024 * 1024  # 2GB

        for checkpoint_file in checkpoint_files:
            file_size = os.path.getsize(checkpoint_file)
            assert file_size < size_limit_bytes, \
                f"Checkpoint {checkpoint_file} exceeds 2GB limit: {file_size / (1024**3):.2f} GB"
