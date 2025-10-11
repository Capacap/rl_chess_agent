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

    # Network architecture (optimized defaults)
    parser.add_argument(
        "--channels",
        type=int,
        default=48,
        help="Number of channels in residual blocks (default: 48 for faster training)"
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=3,
        help="Number of residual blocks (default: 3 for faster training)"
    )

    # Optimization flags
    parser.add_argument(
        "--use-adaptive-schedule",
        action="store_true",
        default=True,
        help="Use progressive training schedule from config.py (recommended)"
    )
    parser.add_argument(
        "--no-adaptive-schedule",
        action="store_false",
        dest="use_adaptive_schedule",
        help="Disable adaptive schedule, use fixed parameters"
    )
    parser.add_argument(
        "--simulations-arena",
        type=int,
        default=None,
        help="MCTS simulations for arena (default: 40 for asymmetric optimization)"
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="enable_early_stopping",
        default=True,
        help="Disable early stopping in training"
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
    parser.add_argument(
        "--gdrive-backup-dir",
        type=str,
        default=None,
        help="Google Drive backup directory (Colab only)"
    )

    args = parser.parse_args()

    # Setup checkpoint directory
    if args.checkpoint_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.checkpoint_dir = f"checkpoints/{timestamp}"

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Set arena simulations default if not provided
    if args.simulations_arena is None:
        args.simulations_arena = 40  # Use higher sims for arena by default

    # Print configuration
    print("=" * 70)
    print("Chess RL Agent Training (Optimized)")
    print("=" * 70)
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Iterations: {args.iterations}")
    print(f"Network: {args.channels} channels, {args.blocks} blocks (~500K params)")
    print(f"MCTS: {args.simulations} sims (self-play), {args.simulations_arena} sims (arena)")
    print(f"Adaptive schedule: {'Enabled' if args.use_adaptive_schedule else 'Disabled'}")
    print(f"Early stopping: {'Enabled' if args.enable_early_stopping else 'Disabled'}")
    if not args.use_adaptive_schedule:
        print(f"Fixed params: games={args.games_per_iter}, epochs={args.epochs}, lr={args.lr}")
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
                num_simulations_arena=args.simulations_arena,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                arena_games=args.arena_games,
                checkpoint_dir=args.checkpoint_dir,
                gdrive_backup_dir=args.gdrive_backup_dir,
                use_adaptive_schedule=args.use_adaptive_schedule,
                enable_early_stopping=args.enable_early_stopping
            )
        else:
            # New training run
            initial_network = ChessNet(channels=args.channels, num_blocks=args.blocks)
            champion = training_pipeline(
                initial_network=initial_network,
                num_iterations=args.iterations,
                games_per_iter=args.games_per_iter,
                num_simulations=args.simulations,
                num_simulations_arena=args.simulations_arena,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                arena_games=args.arena_games,
                checkpoint_dir=args.checkpoint_dir,
                gdrive_backup_dir=args.gdrive_backup_dir,
                use_adaptive_schedule=args.use_adaptive_schedule,
                enable_early_stopping=args.enable_early_stopping
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
