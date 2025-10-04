#!/usr/bin/env python3
"""
train.py
CLI entrypoint for training the chess RL agent.

Usage:
    python train.py                    # Default settings
    python train.py --iterations 20    # Custom iterations
    python train.py --resume checkpoints/iteration_10.pt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from training.pipeline import training_pipeline, resume_training
from model.network import ChessNet


def main():
    parser = argparse.ArgumentParser(
        description="Train chess RL agent using AlphaZero-style self-play",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--games-per-iter",
        type=int,
        default=100,
        help="Self-play games per iteration"
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=40,
        help="MCTS simulations per move"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Training epochs per iteration"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--arena-games",
        type=int,
        default=50,
        help="Arena games for evaluation"
    )

    # Network architecture
    parser.add_argument(
        "--channels",
        type=int,
        default=64,
        help="Number of channels in residual blocks"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=4,
        help="Number of residual blocks"
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (default: checkpoints/YYYYMMDD_HHMMSS)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint path"
    )

    args = parser.parse_args()

    # Setup checkpoint directory
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.checkpoint_dir = f"checkpoints/{timestamp}"

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=" * 70)
    print("Chess RL Agent Training")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games_per_iter}")
    print(f"MCTS simulations: {args.simulations}")
    print(f"Network: {args.channels} channels, {args.blocks} blocks")
    print(f"Training: batch_size={args.batch_size}, epochs={args.epochs}, lr={args.lr}")
    print(f"Arena: {args.arena_games} games")
    print("=" * 70)
    print()

    try:
        if args.resume:
            # Resume from checkpoint
            print(f"Resuming from: {args.resume}\n")
            champion = resume_training(
                checkpoint_path=args.resume,
                num_iterations=args.iterations,
                games_per_iter=args.games_per_iter,
                num_simulations=args.simulations,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                arena_games=args.arena_games,
                checkpoint_dir=args.checkpoint_dir
            )
        else:
            # New training run
            initial_network = ChessNet(channels=args.channels, num_blocks=args.blocks)
            champion = training_pipeline(
                initial_network=initial_network,
                num_iterations=args.iterations,
                games_per_iter=args.games_per_iter,
                num_simulations=args.simulations,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                arena_games=args.arena_games,
                checkpoint_dir=args.checkpoint_dir
            )

        print("\n" + "=" * 70)
        print("Training completed successfully!")
        print(f"Final champion saved in: {args.checkpoint_dir}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Checkpoints saved in: {args.checkpoint_dir}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
