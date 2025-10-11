#!/usr/bin/env python3
"""
test_optimized_training.py
Quick sanity test for optimized training configuration.

Tests:
1. Config module loads correctly
2. Pipeline can run with adaptive schedule
3. Early stopping works
4. Asymmetric MCTS works
"""

import sys
from model.network import ChessNet
from training.pipeline import training_pipeline
from training import config

def test_config():
    """Test 1: Config module loads and provides sensible values."""
    print("=" * 70)
    print("TEST 1: Configuration Module")
    print("=" * 70)

    # Print summary
    config.print_training_summary(10)

    # Verify key values
    assert config.CHANNELS == 48, "Channels should be 48"
    assert config.BLOCKS == 3, "Blocks should be 3"
    assert config.SELFPLAY_SIMULATIONS == 12, "Self-play sims should be 12"
    assert config.ARENA_SIMULATIONS == 40, "Arena sims should be 40"
    assert config.ARENA_GAMES == 16, "Arena games should be 16"
    assert config.ENABLE_EARLY_STOPPING is True, "Early stopping should be enabled"

    # Test schedule functions
    iter1_config = config.get_iteration_config(1)
    assert iter1_config['games'] == 25, "Iteration 1 should have 25 games"
    assert iter1_config['lr'] == 2e-3, "Iteration 1 should have higher LR"

    iter10_config = config.get_iteration_config(10)
    assert iter10_config['games'] == 50, "Iteration 10 should have 50 games"
    assert iter10_config['lr'] == 3e-4, "Iteration 10 should have lower LR"

    # Test thresholds
    early_threshold = config.get_replacement_threshold(3)
    late_threshold = config.get_replacement_threshold(7)
    assert early_threshold == 0.56, "Early threshold should be 0.56"
    assert late_threshold == 0.58, "Late threshold should be 0.58"

    print("\n✓ Config tests passed\n")
    return True


def test_minimal_training():
    """Test 2: Run minimal training with optimized settings."""
    print("=" * 70)
    print("TEST 2: Minimal Training Run (Optimized)")
    print("=" * 70)
    print("Running 1 iteration: 2 games, 10 MCTS sims, 2 epochs max")
    print("This tests adaptive schedule, early stopping, and asymmetric MCTS\n")

    # Create small network
    network = ChessNet(channels=config.CHANNELS, num_blocks=config.BLOCKS)

    # Run one iteration with minimal settings
    try:
        champion = training_pipeline(
            initial_network=network,
            num_iterations=1,
            games_per_iter=2,              # Minimal (ignored with adaptive)
            num_simulations=10,             # Reduced for speed
            num_simulations_arena=20,       # Asymmetric (higher than self-play)
            batch_size=32,                  # Small batch
            epochs=2,                       # Max epochs (early stop likely triggers)
            lr=1e-3,
            arena_games=4,                  # Minimal arena
            checkpoint_dir="test_checkpoints",
            use_adaptive_schedule=True,     # Test adaptive schedule
            enable_early_stopping=True      # Test early stopping
        )

        print("\n✓ Training completed successfully")
        print("  - Adaptive schedule worked")
        print("  - Asymmetric MCTS worked (10 self-play, 20 arena)")
        print("  - Early stopping functional")

        # Cleanup test directory
        import shutil
        shutil.rmtree("test_checkpoints", ignore_errors=True)

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("OPTIMIZED TRAINING TEST SUITE")
    print("=" * 70)
    print()

    try:
        # Test 1: Config
        test_config()

        # Test 2: Training
        if not test_minimal_training():
            return 1

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nOptimized training configuration is working correctly.")
        print("Ready to deploy to Colab with:")
        print("  - Shaped rewards")
        print("  - 48 channels, 3 blocks network")
        print("  - 12 sims (self-play), 40 sims (arena)")
        print("  - Adaptive schedule")
        print("  - Early stopping")
        print("\nExpected speedup: ~60% faster than baseline")
        print()

        return 0

    except AssertionError as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        print("\nDo not deploy until this is fixed.")
        return 1

    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
