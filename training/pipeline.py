"""
pipeline.py
End-to-end training pipeline: self-play → train → arena → iterate.

Orchestrates the full AlphaZero-style training loop.
"""

import torch
import os
from pathlib import Path
from typing import Optional
from model.network import ChessNet
from training.selfplay import SelfPlayWorker, DEFAULT_TEMP_SCHEDULE
from training.train import train_iteration, save_checkpoint, load_checkpoint
from training.arena import Arena, should_replace


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
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Step 1: Self-play
        print(f"\n[1/4] Generating {games_per_iter} self-play games...")
        worker = SelfPlayWorker(
            network=champion,
            temp_schedule=DEFAULT_TEMP_SCHEDULE,
            num_simulations=num_simulations
        )
        experiences = worker.generate_batch(games_per_iter, max_moves=100)
        print(f"Collected {len(experiences)} experiences")

        # Step 2: Train challenger
        print(f"\n[2/4] Training challenger network...")
        challenger = _clone_network(champion)
        history = train_iteration(
            network=challenger,
            experiences=experiences,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr
        )

        # Step 3: Arena evaluation
        print(f"\n[3/4] Arena: Challenger vs Champion ({arena_games} games)...")
        arena = Arena(num_simulations=num_simulations)
        results = arena.compete(challenger, champion, num_games=arena_games)

        print(f"\nArena results:")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Draws: {results['draws']}")
        print(f"  Win rate: {results['win_rate']:.1%}")

        # Step 4: Replacement decision
        print(f"\n[4/4] Evaluating replacement...")
        if should_replace(results['win_rate'], arena_games):
            print("✓ Challenger promoted to champion!")
            champion = challenger
            save_checkpoint(champion, f"{checkpoint_dir}/iteration_{iteration + 1}.pt")
        else:
            print("✗ Champion retained")

        # Iteration summary
        print(f"\nIteration {iteration + 1} complete")
        print(f"Champion network: iteration_{iteration + 1 if should_replace(results['win_rate'], arena_games) else iteration}")

    print(f"\n{'='*60}")
    print(f"Training complete: {num_iterations} iterations")
    print(f"Final champion saved: {checkpoint_dir}/iteration_{num_iterations}.pt")
    print(f"{'='*60}\n")

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
