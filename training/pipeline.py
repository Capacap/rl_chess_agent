"""
pipeline.py
End-to-end training pipeline: self-play → train → arena → iterate.

Orchestrates the full AlphaZero-style training loop.
"""

import torch
import os
import time
from pathlib import Path
from typing import Optional
from model.network import ChessNet
from training.selfplay import SelfPlayWorker, DEFAULT_TEMP_SCHEDULE
from training.train import train_iteration, save_checkpoint, load_checkpoint
from training.arena import Arena, should_replace
from training.logger import setup_logger


def training_pipeline(
    initial_network: Optional[ChessNet] = None,
    num_iterations: int = 10,
    games_per_iter: int = 100,
    num_simulations: int = 40,
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-3,
    arena_games: int = 50,
    checkpoint_dir: str = "checkpoints"
) -> ChessNet:
    """
    Full self-play training pipeline.

    Process:
        1. Generate self-play games with current champion
        2. Train challenger network on experiences
        3. Arena: challenger vs champion
        4. Replace champion if challenger wins >55% (statistically significant)
        5. Repeat

    Args:
        initial_network: Starting network (None = create new)
        num_iterations: Number of training iterations
        games_per_iter: Self-play games per iteration
        num_simulations: MCTS simulations per move
        batch_size: Training batch size
        epochs: Training epochs per iteration
        lr: Learning rate
        arena_games: Number of arena games for evaluation
        checkpoint_dir: Directory for saving checkpoints

    Returns:
        Best network found
    """
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)

    # Setup logging
    log_file = f"{checkpoint_dir}/training.log"
    logger = setup_logger("training", log_file=log_file)
    logger.info("="*60)
    logger.info("Training Pipeline Started")
    logger.info("="*60)
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Iterations: {num_iterations}")
    logger.info(f"Games per iteration: {games_per_iter}")
    logger.info(f"MCTS simulations: {num_simulations}")
    logger.info(f"Training: batch_size={batch_size}, epochs={epochs}, lr={lr}")
    logger.info(f"Arena games: {arena_games}")

    # Initialize champion network
    if initial_network is None:
        champion = ChessNet(channels=64, num_blocks=4)
        print("Created new ChessNet (64 channels, 4 blocks)")
    else:
        champion = initial_network
        print("Using provided initial network")

    # Save initial network
    save_checkpoint(champion, f"{checkpoint_dir}/iteration_0.pt")

    # Training loop
    for iteration in range(num_iterations):
        iter_start = time.time()

        logger.info("")
        logger.info("="*60)
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info("="*60)

        # Step 1: Self-play
        logger.info("")
        logger.info(f"[1/4] Generating {games_per_iter} self-play games...")
        selfplay_start = time.time()

        worker = SelfPlayWorker(
            network=champion,
            temp_schedule=DEFAULT_TEMP_SCHEDULE,
            num_simulations=num_simulations
        )

        experiences = worker.generate_batch(
            games_per_iter,
            max_moves=100,
            num_workers=1
        )
        selfplay_time = time.time() - selfplay_start
        logger.info(f"Collected {len(experiences)} experiences ({selfplay_time/60:.1f} min)")

        # Step 2: Train challenger
        logger.info("")
        logger.info(f"[2/4] Training challenger network...")
        train_start = time.time()
        challenger = _clone_network(champion)
        history = train_iteration(
            network=challenger,
            experiences=experiences,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr
        )
        train_time = time.time() - train_start
        logger.info(f"Training complete ({train_time:.1f}s)")

        # Step 3: Arena evaluation
        logger.info("")
        logger.info(f"[3/4] Arena: Challenger vs Champion ({arena_games} games)...")
        arena_start = time.time()
        # Move networks to CPU for arena (MCTS runs on CPU)
        challenger.cpu()
        champion.cpu()
        arena = Arena(num_simulations=num_simulations)
        results = arena.compete(challenger, champion, num_games=arena_games)
        arena_time = time.time() - arena_start

        logger.info("")
        logger.info(f"Arena results:")
        logger.info(f"  Wins: {results['wins']}")
        logger.info(f"  Losses: {results['losses']}")
        logger.info(f"  Draws: {results['draws']}")
        logger.info(f"  Win rate: {results['win_rate']:.1%}")
        logger.info(f"  Time: {arena_time/60:.1f} min")

        # Step 4: Replacement decision
        logger.info("")
        logger.info(f"[4/4] Evaluating replacement...")
        replaced = should_replace(results['win_rate'], arena_games)
        if replaced:
            logger.info("✓ Challenger promoted to champion!")
            champion = challenger
            save_checkpoint(champion, f"{checkpoint_dir}/iteration_{iteration + 1}.pt")
        else:
            logger.info("✗ Champion retained")
            # Save challenger anyway for analysis
            save_checkpoint(challenger, f"{checkpoint_dir}/iteration_{iteration + 1}_challenger.pt")

        # Iteration summary
        iter_time = time.time() - iter_start
        logger.info("")
        logger.info(f"Iteration {iteration + 1} complete")
        logger.info(f"  Champion: iteration_{iteration + 1 if replaced else iteration}")
        logger.info(f"  Time breakdown:")
        logger.info(f"    - Self-play: {selfplay_time/60:.1f} min ({selfplay_time/iter_time*100:.1f}%)")
        logger.info(f"    - Training: {train_time:.1f}s ({train_time/iter_time*100:.1f}%)")
        logger.info(f"    - Arena: {arena_time/60:.1f} min ({arena_time/iter_time*100:.1f}%)")
        logger.info(f"  Total: {iter_time/60:.1f} min")

    # Determine which iteration is the final champion
    champion_files = sorted(Path(checkpoint_dir).glob("iteration_*.pt"))
    champion_files = [f for f in champion_files if "_challenger" not in f.name]
    final_champion_iter = max([int(f.stem.split('_')[1]) for f in champion_files])

    logger.info("")
    logger.info("="*60)
    logger.info(f"Training complete: {num_iterations} iterations")
    logger.info(f"Final champion: {checkpoint_dir}/iteration_{final_champion_iter}.pt")
    logger.info(f"All checkpoints: {len(list(Path(checkpoint_dir).glob('*.pt')))} files saved")
    logger.info("="*60)

    return champion


def _clone_network(network: ChessNet) -> ChessNet:
    """
    Create a copy of network with same architecture and weights.

    Args:
        network: Network to clone

    Returns:
        New network with copied weights
    """
    # Create new network with same architecture
    clone = ChessNet(
        channels=network.channels,
        num_blocks=network.num_blocks
    )

    # Copy weights
    clone.load_state_dict(network.state_dict())

    return clone


def resume_training(
    checkpoint_path: str,
    num_iterations: int = 10,
    **kwargs
) -> ChessNet:
    """
    Resume training from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        num_iterations: Additional iterations to run
        **kwargs: Additional args passed to training_pipeline

    Returns:
        Updated champion network
    """
    # Load checkpoint
    network = ChessNet()
    load_checkpoint(network, checkpoint_path)

    # Continue training
    return training_pipeline(
        initial_network=network,
        num_iterations=num_iterations,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage: minimal training run
    print("Running minimal training pipeline...")

    best_network = training_pipeline(
        num_iterations=2,
        games_per_iter=10,
        num_simulations=20,  # Reduced for speed
        batch_size=64,
        epochs=3,
        arena_games=10
    )

    print("Pipeline test complete")
