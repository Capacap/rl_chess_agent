"""
arena.py
Model evaluation via head-to-head competition.

Pits challenger vs champion networks to determine if new model should replace current best.
Uses Wilson score interval for statistical significance.
"""

import chess
import random
import time
import math
from typing import Dict
from model.network import ChessNet
from training.mcts_nn import mcts_search


class Arena:
    """Pit two neural networks against each other."""

    def __init__(self, num_simulations: int = 40, time_per_move: float = 2.0):
        """
        Args:
            num_simulations: MCTS simulations per move
            time_per_move: Time budget per move (seconds)
        """
        self.num_simulations = num_simulations
        self.time_per_move = time_per_move

    def compete(
        self,
        challenger: ChessNet,
        champion: ChessNet,
        num_games: int = 50
    ) -> Dict[str, float]:
        """
        Play games between two networks, alternating colors.

        Args:
            challenger: New network to evaluate
            champion: Current best network
            num_games: Number of games to play

        Returns:
            results: {wins, losses, draws, win_rate}
                - win_rate includes draws as 0.5
        """
        results = {'wins': 0, 'losses': 0, 'draws': 0}

        for game_idx in range(num_games):
            # Alternate who plays white
            if game_idx % 2 == 0:
                white_net, black_net = challenger, champion
                challenger_color = chess.WHITE
            else:
                white_net, black_net = champion, challenger
                challenger_color = chess.BLACK

            # Play game
            outcome = self._play_game(white_net, black_net)

            # Record result from challenger's perspective
            if outcome == 'white' and challenger_color == chess.WHITE:
                results['wins'] += 1
            elif outcome == 'black' and challenger_color == chess.BLACK:
                results['wins'] += 1
            elif outcome == 'draw':
                results['draws'] += 1
            else:
                results['losses'] += 1

            # Progress logging
            if (game_idx + 1) % 10 == 0:
                current_rate = (results['wins'] + 0.5 * results['draws']) / (game_idx + 1)
                print(f"Games {game_idx + 1}/{num_games}: "
                      f"W{results['wins']}-L{results['losses']}-D{results['draws']} "
                      f"({current_rate:.1%})")

        # Calculate win rate (draws = 0.5)
        results['win_rate'] = (results['wins'] + 0.5 * results['draws']) / num_games

        return results

    def _play_game(
        self,
        white_net: ChessNet,
        black_net: ChessNet
    ) -> str:
        """
        Play one game between two networks.

        Args:
            white_net: Network playing white
            black_net: Network playing black

        Returns:
            'white', 'black', or 'draw'
        """
        board = chess.Board()
        move_count = 0
        max_moves = 200  # Prevent infinite games

        while not board.is_game_over() and move_count < max_moves:
            current_net = white_net if board.turn == chess.WHITE else black_net

            # Run MCTS with time limit
            start = time.time()
            visit_counts = mcts_search(
                board,
                current_net,
                num_simulations=self.num_simulations
            )
            elapsed = time.time() - start

            # Safety check: if MCTS took too long, reduce sims next time
            if elapsed > self.time_per_move:
                print(f"Warning: MCTS took {elapsed:.2f}s (limit: {self.time_per_move}s)")

            # Select most visited move
            if not visit_counts:
                # Fallback: random move if MCTS failed
                move = random.choice(list(board.legal_moves))
            else:
                move = max(visit_counts.keys(), key=lambda m: visit_counts[m])

            board.push(move)
            move_count += 1

        # Determine outcome
        if board.is_game_over():
            outcome = board.outcome()
            if outcome.winner is None:
                return 'draw'
            return 'white' if outcome.winner == chess.WHITE else 'black'
        else:
            # Hit move limit - count as draw
            return 'draw'


def should_replace(
    challenger_win_rate: float,
    num_games: int,
    threshold: float = 0.55
) -> bool:
    """
    Decide if challenger should replace champion using statistical test.

    Uses Wilson score interval to ensure significance.

    Args:
        challenger_win_rate: Win rate from arena (0.0 to 1.0)
        num_games: Number of games played
        threshold: Minimum win rate to consider replacement

    Returns:
        True if challenger should replace champion
    """
    if challenger_win_rate < threshold:
        return False

    # Wilson score interval (95% confidence)
    # Check if lower bound > 0.5 (better than random)
    z = 1.96  # 95% confidence
    p = challenger_win_rate
    n = num_games

    # Wilson score lower bound
    numerator = p + z*z/(2*n) - z * math.sqrt((p*(1-p) + z*z/(4*n))/n)
    denominator = 1 + z*z/n
    lower_bound = numerator / denominator

    # Replace if statistically better than 50%
    return lower_bound > 0.5


def evaluate_vs_baseline(
    network: ChessNet,
    baseline_type: str = 'random',
    num_games: int = 20
) -> float:
    """
    Evaluate network against baseline agent.

    Args:
        network: Neural network to evaluate
        baseline_type: 'random' or 'greedy'
        num_games: Number of games to play

    Returns:
        Win rate against baseline
    """
    # TODO: Implement baseline agents for evaluation
    # For now, return placeholder
    raise NotImplementedError("Baseline evaluation not yet implemented")
