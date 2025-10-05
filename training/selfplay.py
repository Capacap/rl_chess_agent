"""
selfplay.py
Self-play game generation for training data.

Policy-only mode: Direct network policy sampling (no MCTS).
Generates experience tuples: (fen, selected_move, game_outcome)
Uses temperature-based sampling for exploration vs exploitation balance.
"""

import chess
import random
import numpy as np
import torch
import time
from typing import Dict, List, Optional
from model.network import ChessNet
from encoding.state import encode_board
from encoding.move import encode_move, create_legal_move_mask, decode_move


# Temperature schedule: {move_number: temperature}
# High temp (1.0) = exploration, low temp (0.1) = exploitation
DEFAULT_TEMP_SCHEDULE = {
    0: 1.0,   # Moves 0-9: High exploration (diverse openings)
    10: 0.5,  # Moves 10-19: Moderate
    20: 0.1   # Moves 20+: Near-deterministic (strong endgame)
}


class Experience:
    """Single training experience from self-play."""

    def __init__(self, fen: str, policy: np.ndarray, value: float, move: Optional[chess.Move] = None):
        """
        Args:
            fen: Board position as FEN string
            policy: Policy target (one-hot encoded move or distribution) [4096]
            value: Game outcome from this position's perspective
            move: Selected move (for one-hot encoding, optional)
        """
        self.fen = fen
        self.policy = policy
        self.value = value
        self.move = move


class SelfPlayWorker:
    """Generates training data via self-play games (policy-only, no MCTS)."""

    def __init__(
        self,
        network: ChessNet,
        temp_schedule: Dict[int, float] = None,
        num_simulations: int = 0,  # Unused in policy-only mode
        c_puct: float = 1.0  # Unused in policy-only mode
    ):
        """
        Args:
            network: Neural network for direct policy inference
            temp_schedule: Temperature schedule {move_num: temp}
            num_simulations: Ignored (kept for API compatibility)
            c_puct: Ignored (kept for API compatibility)
        """
        self.network = network
        self.network.eval()  # Set to eval mode for inference
        self.temp_schedule = temp_schedule or DEFAULT_TEMP_SCHEDULE
        self.device = next(network.parameters()).device

    def play_game(self, max_moves: int = 200) -> List[Experience]:
        """
        Play one self-play game using direct policy sampling (no MCTS).

        Args:
            max_moves: Maximum moves before declaring draw

        Returns:
            List of experiences with filled game outcomes
        """
        board = chess.Board()
        experiences: List[Dict] = []
        move_count = 0

        with torch.no_grad():  # Inference mode
            while not board.is_game_over() and move_count < max_moves:
                # Get network policy for current position
                policy_probs = self._get_policy(board)

                # Sample move with temperature
                temp = self._get_temperature(move_count)
                move = self._sample_move_from_policy(board, policy_probs, temperature=temp)

                # Record experience with one-hot policy target
                experiences.append({
                    'fen': board.fen(),
                    'move': move,  # Store selected move for one-hot encoding
                    'value': None  # Placeholder
                })

                # Execute move
                board.push(move)
                move_count += 1

        # Backfill game outcome
        if board.is_game_over():
            outcome = self._get_outcome(board)
        else:
            outcome = 0.0  # Draw by move limit

        result = []
        for i, exp in enumerate(experiences):
            # Flip outcome for alternating players
            player_outcome = outcome if i % 2 == 0 else -outcome

            # Create one-hot policy target
            policy_target = np.zeros(4096, dtype=np.float32)
            action_idx = encode_move(exp['move'])
            policy_target[action_idx] = 1.0

            result.append(Experience(
                fen=exp['fen'],
                policy=policy_target,
                value=player_outcome,
                move=exp['move']
            ))

        return result

    def generate_batch(
        self,
        num_games: int,
        max_moves: int = 200,
        num_workers: int = 1
    ) -> List[Experience]:
        """
        Generate multiple self-play games sequentially.

        Args:
            num_games: Number of games to generate
            max_moves: Maximum moves per game
            num_workers: Ignored (kept for compatibility)

        Returns:
            Flattened list of all experiences from all games
        """
        return self._generate_sequential(num_games, max_moves)

    def _generate_sequential(self, num_games: int, max_moves: int) -> List[Experience]:
        """Sequential game generation."""
        import time
        all_experiences = []
        start_time = time.time()

        for game_idx in range(num_games):
            game_start = time.time()
            game_experiences = self.play_game(max_moves=max_moves)
            game_time = time.time() - game_start
            all_experiences.extend(game_experiences)

            # Progress every game (more granular)
            elapsed = time.time() - start_time
            avg_time = elapsed / (game_idx + 1)
            eta = avg_time * (num_games - game_idx - 1)

            print(f"  Game {game_idx + 1}/{num_games}: {len(game_experiences)} exp, "
                  f"{game_time:.1f}s (avg {avg_time:.1f}s/game, ETA {eta/60:.1f}min)")

        return all_experiences


    def _get_policy(self, board: chess.Board) -> np.ndarray:
        """
        Get network policy for board position.

        Args:
            board: Current board state

        Returns:
            Policy probabilities [4096], masked to legal moves
        """
        # Encode board state (already has batch dim [1, 13, 8, 8])
        state_tensor = encode_board(board).to(self.device)

        # Create legal move mask
        legal_mask = create_legal_move_mask(board)
        legal_mask_tensor = torch.tensor(legal_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        # Forward pass
        policy, _ = self.network(state_tensor, legal_mask_tensor)

        # Convert to numpy
        return policy.cpu().numpy()[0]

    def _sample_move_from_policy(
        self,
        board: chess.Board,
        policy_probs: np.ndarray,
        temperature: float
    ) -> chess.Move:
        """
        Sample move from policy distribution with temperature.

        Args:
            board: Current board position
            policy_probs: Policy probabilities [4096]
            temperature: Sampling temperature
                - 0: Deterministic (argmax)
                - 1: Sample proportionally
                - >1: More exploratory

        Returns:
            Sampled legal move
        """
        legal_moves = list(board.legal_moves)
        legal_indices = [encode_move(m) for m in legal_moves]
        legal_probs = policy_probs[legal_indices]

        if temperature < 1e-3:
            # Deterministic: pick highest probability
            max_prob = np.max(legal_probs)
            best_indices = [i for i, p in enumerate(legal_probs) if p == max_prob]
            return legal_moves[random.choice(best_indices)]

        # Apply temperature
        probs_temp = legal_probs ** (1.0 / temperature)
        probs_temp /= probs_temp.sum()

        # Sample
        chosen_idx = np.random.choice(len(legal_moves), p=probs_temp)
        return legal_moves[chosen_idx]

    def _get_temperature(self, move_count: int) -> float:
        """
        Get temperature for current move number.

        Args:
            move_count: Current move number (0-indexed)

        Returns:
            Temperature value for move sampling
        """
        # Find highest threshold <= move_count
        applicable_temps = [(t, temp) for t, temp in self.temp_schedule.items() if move_count >= t]
        if applicable_temps:
            return max(applicable_temps, key=lambda x: x[0])[1]
        # Default to lowest temperature
        return min(self.temp_schedule.values())

    def _get_outcome(self, board: chess.Board) -> float:
        """
        Get game outcome from white's perspective.

        Args:
            board: Final board position

        Returns:
            1.0 (white win), -1.0 (black win), 0.0 (draw)
        """
        outcome = board.outcome()

        if outcome.winner is None:
            return 0.0  # Draw

        return 1.0 if outcome.winner == chess.WHITE else -1.0


