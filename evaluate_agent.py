#!/usr/bin/env python3
"""
evaluate_agent.py
Evaluate trained agent against baseline opponents.

Usage:
    python evaluate_agent.py --checkpoint checkpoints/run1/iteration_10.pt --opponent greedy --games 50
"""

import argparse
import torch
import chess
from model.network import ChessNet
from training.mcts_nn import mcts_search
from greedy_agent import GreedyAgent
from random_agent import RandomAgent
from mcts_agent import MCTSAgent


def load_trained_agent(checkpoint_path: str, num_simulations: int = 20) -> tuple:
    """Load trained network and create inference function."""
    # Load checkpoint
    network = ChessNet(channels=64, num_blocks=4)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Handle both full checkpoint and state_dict formats
    if isinstance(checkpoint, dict) and 'network' in checkpoint:
        network.load_state_dict(checkpoint['network'])
    else:
        network.load_state_dict(checkpoint)

    network.eval()
    network.cpu()

    def make_move(board: chess.Board, time_limit: float = 2.0) -> chess.Move:
        """Make move using MCTS + NN."""
        visit_counts = mcts_search(
            board,
            network=network,
            num_simulations=num_simulations,
            c_puct=1.0
        )
        # Pick most visited move
        best_move = max(visit_counts.items(), key=lambda x: x[1])[0]
        return best_move

    return network, make_move


def play_game(white_agent, black_agent, max_moves: int = 200) -> float:
    """
    Play one game between two agents.

    Args:
        white_agent: Function (board, time_limit) -> move
        black_agent: Function (board, time_limit) -> move
        max_moves: Maximum moves before declaring draw

    Returns:
        1.0 if white wins, -1.0 if black wins, 0.0 for draw
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        agent = white_agent if board.turn == chess.WHITE else black_agent
        move = agent(board, time_limit=2.0)
        board.push(move)
        move_count += 1

    if board.is_game_over():
        outcome = board.outcome()
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == chess.WHITE else -1.0

    return 0.0  # Draw by move limit


def evaluate(
    trained_agent,
    opponent_agent,
    num_games: int = 50,
    as_white: bool = True
) -> dict:
    """
    Evaluate trained agent against opponent.

    Args:
        trained_agent: Trained agent function
        opponent_agent: Baseline agent function
        num_games: Number of games to play
        as_white: If True, trained agent plays white

    Returns:
        dict with wins, losses, draws, win_rate
    """
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(num_games):
        if as_white:
            result = play_game(trained_agent, opponent_agent)
        else:
            result = -play_game(opponent_agent, trained_agent)

        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            draws += 1

        if (game_num + 1) % 10 == 0:
            win_rate = wins / (game_num + 1)
            print(f"  Games {game_num + 1}/{num_games}: "
                  f"W{wins}-L{losses}-D{draws} ({win_rate:.1%})")

    win_rate = wins / num_games

    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained agent against baselines"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        choices=['greedy', 'random', 'mcts'],
        default='greedy',
        help="Baseline opponent type"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games to play"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=20,
        help="MCTS simulations for trained agent"
    )
    parser.add_argument(
        "--color",
        type=str,
        choices=['white', 'black', 'both'],
        default='both',
        help="Color for trained agent"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Agent Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Opponent: {args.opponent}")
    print(f"Games: {args.games}")
    print(f"MCTS simulations: {args.simulations}")
    print("=" * 70)
    print()

    # Load trained agent
    print("Loading trained agent...")
    network, trained_agent = load_trained_agent(args.checkpoint, args.simulations)
    print("✓ Loaded\n")

    # Create opponent
    if args.opponent == 'greedy':
        opponent = GreedyAgent()
        opponent_fn = lambda board, time_limit: opponent.make_move(board, time_limit)
    elif args.opponent == 'random':
        opponent = RandomAgent()
        opponent_fn = lambda board, time_limit: opponent.make_move(board, time_limit)
    elif args.opponent == 'mcts':
        opponent = MCTSAgent(simulations=20, time_limit=2.0)
        opponent_fn = lambda board, time_limit: opponent.make_move(board, time_limit)

    # Evaluate
    if args.color in ['white', 'both']:
        print(f"Playing as WHITE against {args.opponent.upper()}...")
        results_white = evaluate(trained_agent, opponent_fn, args.games, as_white=True)
        print(f"\nResults (as WHITE):")
        print(f"  Wins: {results_white['wins']}")
        print(f"  Losses: {results_white['losses']}")
        print(f"  Draws: {results_white['draws']}")
        print(f"  Win rate: {results_white['win_rate']:.1%}")
        print()

    if args.color in ['black', 'both']:
        print(f"Playing as BLACK against {args.opponent.upper()}...")
        results_black = evaluate(trained_agent, opponent_fn, args.games, as_white=False)
        print(f"\nResults (as BLACK):")
        print(f"  Wins: {results_black['wins']}")
        print(f"  Losses: {results_black['losses']}")
        print(f"  Draws: {results_black['draws']}")
        print(f"  Win rate: {results_black['win_rate']:.1%}")
        print()

    if args.color == 'both':
        total_wins = results_white['wins'] + results_black['wins']
        total_losses = results_white['losses'] + results_black['losses']
        total_draws = results_white['draws'] + results_black['draws']
        total_games = args.games * 2
        overall_win_rate = total_wins / total_games

        print("=" * 70)
        print("OVERALL RESULTS")
        print("=" * 70)
        print(f"Total games: {total_games}")
        print(f"Wins: {total_wins}")
        print(f"Losses: {total_losses}")
        print(f"Draws: {total_draws}")
        print(f"Win rate: {overall_win_rate:.1%}")
        print("=" * 70)


if __name__ == "__main__":
    main()
