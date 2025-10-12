"""
config.py
Optimized training configuration for fast iteration with shaped rewards.

Key optimizations:
1. Smaller network (48 channels, 3 blocks) for faster inference
2. Asymmetric MCTS (fewer sims in self-play, more in arena)
3. Progressive schedule (more games/epochs as training progresses)
4. Early stopping to avoid wasted epochs
5. Smaller arena with adjusted thresholds
"""

# Network architecture (optimized for speed)
CHANNELS = 48   # Down from 64 (~60% fewer parameters)
BLOCKS = 3      # Down from 4

# MCTS configuration
SELFPLAY_SIMULATIONS = 12   # Reduced from 20 for speed
ARENA_SIMULATIONS = 40       # Keep high for accurate evaluation
C_PUCT = 1.0

# Progressive training schedule
# Format: {iteration: {'games': int, 'epochs': int, 'lr': float, 'batch_size': int}}
SCHEDULE = {
    # Early iterations: Fast bootstrap, higher LR
    1:  {'games': 25, 'epochs': 4, 'lr': 2e-3, 'batch_size': 256},
    2:  {'games': 30, 'epochs': 4, 'lr': 2e-3, 'batch_size': 256},

    # Mid iterations: More data, moderate LR
    3:  {'games': 35, 'epochs': 5, 'lr': 1e-3, 'batch_size': 256},
    4:  {'games': 40, 'epochs': 5, 'lr': 1e-3, 'batch_size': 256},
    5:  {'games': 45, 'epochs': 6, 'lr': 8e-4, 'batch_size': 256},

    # Late iterations: Refinement, lower LR
    6:  {'games': 50, 'epochs': 6, 'lr': 8e-4, 'batch_size': 256},
    7:  {'games': 50, 'epochs': 7, 'lr': 5e-4, 'batch_size': 256},
    8:  {'games': 50, 'epochs': 7, 'lr': 5e-4, 'batch_size': 256},
    9:  {'games': 50, 'epochs': 8, 'lr': 3e-4, 'batch_size': 256},
    10: {'games': 50, 'epochs': 8, 'lr': 3e-4, 'batch_size': 256},
}

# Arena configuration (smaller for speed)
ARENA_GAMES = 20  # Increased from 16 for better statistical power

# Replacement thresholds (adjusted for shaped rewards bootstrapping)
# With 20 games, need 10-11 wins (52%) which is reasonable with shaped rewards
EARLY_THRESHOLD = 0.52   # Iterations 1-5 (lowered to allow more updates)
LATE_THRESHOLD = 0.56    # Iterations 6-10 (slightly stricter when converging)

# Early stopping configuration
ENABLE_EARLY_STOPPING = True
MIN_EPOCHS = 3                # Always train at least this many epochs (was 2)
MAX_EPOCHS = 10               # Never train more than this (fallback)
PATIENCE = 3                  # Stop after this many epochs without improvement (was 2)
MIN_IMPROVEMENT = 0.005       # Minimum relative improvement (0.5%, was 1%)

# Game limits
MAX_MOVES_SELFPLAY = 200
MAX_MOVES_ARENA = 200

# Temperature schedule (exploration)
TEMP_SCHEDULE = {
    0: 1.5,   # High exploration early game
    20: 1.0,  # Moderate mid-game
    40: 0.3   # Low temperature endgame
}


def get_iteration_config(iteration: int) -> dict:
    """
    Get configuration for a specific iteration.

    Args:
        iteration: Iteration number (1-indexed)

    Returns:
        Dictionary with 'games', 'epochs', 'lr', 'batch_size'
    """
    # Return config if exists, otherwise use last iteration's config
    if iteration in SCHEDULE:
        return SCHEDULE[iteration]

    # Fallback: use config from highest iteration
    max_iter = max(SCHEDULE.keys())
    return SCHEDULE[max_iter]


def get_replacement_threshold(iteration: int) -> float:
    """
    Get arena replacement threshold for a specific iteration.

    Args:
        iteration: Iteration number (1-indexed)

    Returns:
        Threshold value (0.0 to 1.0)
    """
    return EARLY_THRESHOLD if iteration <= 5 else LATE_THRESHOLD


def estimate_iteration_time(iteration: int) -> float:
    """
    Estimate time (in minutes) for a single iteration.

    Args:
        iteration: Iteration number (1-indexed)

    Returns:
        Estimated time in minutes
    """
    config = get_iteration_config(iteration)

    # Time estimates (rough)
    # Self-play: ~1.5 min/game with 12 sims and smaller network
    # Training: ~0.2 min/epoch with smaller network
    # Arena: ~2 min/game with 40 sims (20 games now)

    selfplay_time = config['games'] * 1.5
    training_time = config['epochs'] * 0.2
    arena_time = ARENA_GAMES * 2

    return selfplay_time + training_time + arena_time


def print_training_summary(num_iterations: int = 10):
    """Print summary of training configuration."""
    print("=" * 70)
    print("OPTIMIZED TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"\nNetwork: {CHANNELS} channels, {BLOCKS} blocks (~500K parameters)")
    print(f"MCTS: {SELFPLAY_SIMULATIONS} sims (self-play), {ARENA_SIMULATIONS} sims (arena)")
    print(f"Arena: {ARENA_GAMES} games, threshold {EARLY_THRESHOLD:.0%}/{LATE_THRESHOLD:.0%} (early/late)")
    print(f"Early stopping: {'Enabled' if ENABLE_EARLY_STOPPING else 'Disabled'}")

    print(f"\nProgressive schedule:")
    print(f"{'Iter':<6} {'Games':<8} {'Epochs':<8} {'LR':<10} {'Est. Time':<12}")
    print("-" * 70)

    total_time = 0
    for i in range(1, num_iterations + 1):
        config = get_iteration_config(i)
        est_time = estimate_iteration_time(i)
        total_time += est_time

        print(f"{i:<6} {config['games']:<8} {config['epochs']:<8} "
              f"{config['lr']:<10.0e} {est_time:<12.1f} min")

    print("-" * 70)
    print(f"Total estimated time: {total_time:.1f} min ({total_time/60:.1f} hours)")
    print("=" * 70)


if __name__ == "__main__":
    # Print configuration summary
    print_training_summary(10)
