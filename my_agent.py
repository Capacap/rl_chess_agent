# my_agent.py
"""
Tournament submission agent - loads trained model from pickle checkpoint.
"""

from agent_interface import Agent
import chess
import torch
import pickle
import os
from pathlib import Path


class MyAwesomeAgent(Agent):
    """Chess agent using trained neural network with MCTS."""

    def __init__(self, board, color):
        super().__init__(board, color)

        # Load trained model from pickle checkpoint
        checkpoint_path = self._find_checkpoint()
        self.network = self._load_model(checkpoint_path)

        # CPU-only inference (tournament requirement)
        self.network.to("cpu")
        self.network.eval()
        torch.set_num_threads(1)  # CPU optimization

        # MCTS parameters (tune based on time budget)
        self.num_simulations = 20  # Adjust based on 2s time limit

    def _find_checkpoint(self) -> str:
        """
        Find the latest trained model checkpoint.

        Returns:
            Path to .pkl checkpoint file

        Raises:
            FileNotFoundError: If no checkpoint found
        """
        # Look for checkpoints in common locations
        search_paths = [
            "checkpoints/production_run",  # Production training
            "checkpoints",  # Any checkpoint directory
            ".",  # Current directory
        ]

        for search_dir in search_paths:
            if not os.path.exists(search_dir):
                continue

            # Find all .pkl files
            pkl_files = list(Path(search_dir).glob("**/*.pkl"))
            if pkl_files:
                # Return the latest one (highest iteration number)
                pkl_files.sort()
                return str(pkl_files[-1])

        raise FileNotFoundError(
            "No trained model checkpoint (.pkl) found. "
            "Run training first: python train.py"
        )

    def _load_model(self, checkpoint_path: str):
        """
        Load model from pickle checkpoint.

        Args:
            checkpoint_path: Path to .pkl file

        Returns:
            Loaded ChessNet model
        """
        # Import here to avoid circular dependencies
        from model.network import ChessNet

        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)

        # Reconstruct network
        network = ChessNet(
            channels=checkpoint_data['channels'],
            num_blocks=checkpoint_data['num_blocks']
        )

        # Load weights
        network.load_state_dict(checkpoint_data['state_dict'])

        return network

    def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        Select best move using MCTS + neural network.

        Args:
            board: Current board position
            time_limit: Maximum time allowed (seconds)

        Returns:
            Selected move (guaranteed legal)
        """
        # Import here to avoid issues if not needed
        from training.mcts_nn import mcts_search

        try:
            # Run MCTS search with time budget
            # Reserve 10% for overhead (network loading, move selection, etc.)
            search_time = time_limit * 0.9

            visit_counts = mcts_search(
                board,
                network=self.network,
                num_simulations=self.num_simulations,
                c_puct=1.0,
                time_budget=search_time
            )

            # Select most visited move
            if visit_counts:
                best_move = max(visit_counts, key=visit_counts.get)
                return best_move

        except Exception as e:
            # Fallback: if anything goes wrong, return a legal move
            print(f"Warning: MCTS failed ({e}), using fallback")

        # Fallback: return any legal move (should rarely happen)
        legal_moves = list(board.legal_moves)
        if legal_moves:
            return legal_moves[0]

        # This should never happen (board.legal_moves always returns something if not game over)
        raise ValueError("No legal moves available")