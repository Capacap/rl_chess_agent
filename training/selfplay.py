"""
selfplay.py
Self-play game generation for training data.

Generates experience tuples: (fen, mcts_policy, value)
Uses temperature-based sampling for exploration vs exploitation balance.

Supports both pure outcome-based rewards and shaped rewards that include
intermediate signals (material balance, pawn advancement, piece activity).
"""

import chess
import random
import numpy as np
import time
from typing import Dict, List
from model.network import ChessNet
from training.mcts_nn import mcts_search
from encoding.move import encode_move


# Temperature schedule: {move_number: temperature}
# High temp (1.5) = exploration, low temp (0.3) = exploitation
# Increased for better exploration during bootstrapping
DEFAULT_TEMP_SCHEDULE = {
    0: 1.5,   # Moves 0-19: High exploration (diverse openings)
    20: 1.0,  # Moves 20-39: Moderate exploration
    40: 0.3   # Moves 40+: Low temperature (endgame)
}


class Experience:
    """Single training experience from self-play."""

    def __init__(self, fen: str, policy: np.ndarray, value: float):
        """
        Args:
            fen: Board position as FEN string
            policy: MCTS policy distribution [4096]
            value: Game outcome from this position's perspective
        """
        self.fen = fen
        self.policy = policy
        self.value = value


class SelfPlayWorker:
    """Generates training data via self-play games."""

    def __init__(
        self,
        network: ChessNet,
        temp_schedule: Dict[int, float] = None,
        num_simulations: int = 40,
        c_puct: float = 1.0
    ):
        """
        Args:
            network: Neural network for MCTS guidance
            temp_schedule: Temperature schedule {move_num: temp}
            num_simulations: MCTS simulations per move
            c_puct: MCTS exploration constant
        """
        self.network = network
        self.temp_schedule = temp_schedule or DEFAULT_TEMP_SCHEDULE
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def play_game(self, max_moves: int = 200, use_shaped_rewards: bool = True) -> List[Experience]:
        """
        Play one self-play game to completion.

        Args:
            max_moves: Maximum moves before declaring draw
            use_shaped_rewards: If True, blend game outcome with position values

        Returns:
            List of experiences with filled values
        """
        from training.rewards import compute_position_value, REWARD_WEIGHTS

        board = chess.Board()
        experiences: List[Dict] = []  # Store temporarily without values
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            # Run MCTS from current position
            visit_counts = mcts_search(
                board,
                network=self.network,
                num_simulations=self.num_simulations,
                c_puct=self.c_puct
            )

            # Convert visit counts to policy distribution
            mcts_policy = self._visit_counts_to_policy(visit_counts)

            # Sample move with temperature
            temp = self._get_temperature(move_count)
            move = self._sample_move(visit_counts, temperature=temp)

            # Record experience (value filled later)
            experiences.append({
                'fen': board.fen(),
                'policy': mcts_policy,
                'board': board.copy(),  # Store board for reward computation
                'player': board.turn    # Track which player's move
            })

            # Execute move
            board.push(move)
            move_count += 1

        # Compute final outcome
        if board.is_game_over():
            outcome = self._get_outcome(board)
        else:
            # Hit move limit - treat as draw
            outcome = 0.0

        # Backfill values with shaped rewards or pure outcome
        result = []
        for exp in experiences:
            if use_shaped_rewards:
                # Compute position value
                position_val = compute_position_value(exp['board'])

                # Convert to player's perspective
                if exp['player'] == chess.WHITE:
                    position_val_player = position_val
                    outcome_player = outcome
                else:
                    position_val_player = -position_val
                    outcome_player = -outcome

                # Weighted combination of outcome and position
                value = (
                    REWARD_WEIGHTS['outcome'] * outcome_player +
                    (1 - REWARD_WEIGHTS['outcome']) * position_val_player
                )
            else:
                # Pure outcome-based (original approach)
                # Determine player index (0 = white, 1 = black)
                player_idx = 0 if exp['player'] == chess.WHITE else 1
                # Flip outcome for black
                value = outcome if player_idx == 0 else -outcome

            result.append(Experience(
                fen=exp['fen'],
                policy=exp['policy'],
                value=float(value)
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


    def _get_temperature(self, move_count: int) -> float:
        """
        Get temperature for current move number.

        Args:
            move_count: Current move number (0-indexed)

        Returns:
            Temperature value for move sampling
        """
        # Find the largest threshold <= move_count
        applicable_temp = None
        for threshold, temp in sorted(self.temp_schedule.items(), reverse=True):
            if move_count >= threshold:
                applicable_temp = temp
                break

        # Fallback to highest temperature (shouldn't happen with threshold 0)
        if applicable_temp is None:
            applicable_temp = max(self.temp_schedule.values())

        return applicable_temp

    def _sample_move(
        self,
        visit_counts: Dict[chess.Move, int],
        temperature: float
    ) -> chess.Move:
        """
        Sample move from visit distribution with temperature.

        Args:
            visit_counts: {move: visit_count} from MCTS
            temperature: Sampling temperature
                - 0: Deterministic (argmax)
                - 1: Proportional to visits
                - >1: More exploratory

        Returns:
            Sampled move
        """
        moves = list(visit_counts.keys())
        visits = [visit_counts[m] for m in moves]

        if temperature < 1e-3:
            # Deterministic: pick most visited
            max_visits = max(visits)
            best_moves = [m for m, v in zip(moves, visits) if v == max_visits]
            return random.choice(best_moves)

        # Apply temperature
        visits_temp = [v ** (1.0 / temperature) for v in visits]
        total = sum(visits_temp)
        probs = [v / total for v in visits_temp]

        return random.choices(moves, weights=probs, k=1)[0]

    def _visit_counts_to_policy(
        self,
        visit_counts: Dict[chess.Move, int]
    ) -> np.ndarray:
        """
        Convert MCTS visit counts to policy distribution.

        Args:
            visit_counts: {move: visits} dictionary

        Returns:
            Policy array [4096] with probabilities summing to 1.0
        """
        policy = np.zeros(4096, dtype=np.float32)

        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return policy  # All zeros (shouldn't happen)

        for move, visits in visit_counts.items():
            action_idx = encode_move(move)
            policy[action_idx] = visits / total_visits

        return policy

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


