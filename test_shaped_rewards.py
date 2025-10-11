#!/usr/bin/env python3
"""
test_shaped_rewards.py
Test script to verify shaped reward implementation works correctly.

Run this before deploying to Colab to ensure:
1. Reward computation is sensible
2. Self-play generates diverse values
3. Training can use shaped rewards
"""

import sys
import numpy as np
import torch
import chess

def test_reward_computation():
    """Test 1: Verify reward functions produce sensible values."""
    print("=" * 70)
    print("TEST 1: Reward Computation")
    print("=" * 70)

    from training.rewards import (
        compute_position_value,
        compute_material_balance,
        compute_pawn_advancement,
        compute_piece_activity
    )

    # Test 1a: Starting position (should be neutral)
    board = chess.Board()
    print("\n1a. Starting position:")
    print(f"    Material: {compute_material_balance(board):.3f} (expect ~0.0)")
    print(f"    Pawn advancement: {compute_pawn_advancement(board):.3f} (expect ~0.0)")
    print(f"    Piece activity: {compute_piece_activity(board):.3f} (expect ~0.0)")
    print(f"    Overall value: {compute_position_value(board):.3f} (expect ~0.0)")

    assert abs(compute_material_balance(board)) < 0.1, "Starting position should be balanced"
    assert abs(compute_pawn_advancement(board)) < 0.1, "Starting position should have neutral pawn advancement"

    # Test 1b: White up a pawn
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("d7d5")
    board.push_uci("e4d5")  # White captures pawn
    print("\n1b. White captures pawn:")
    print(f"    Material: {compute_material_balance(board):.3f} (expect >0)")
    print(f"    Overall value: {compute_position_value(board):.3f} (expect >0)")

    assert compute_material_balance(board) > 0, "White should be ahead in material"
    assert compute_position_value(board) > 0, "Position should favor white"

    # Test 1c: Advanced white pawn (no black pawns)
    board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
    print("\n1c. White pawn on 7th rank (near promotion):")
    print(f"    Pawn advancement: {compute_pawn_advancement(board):.3f} (expect >0)")
    print(f"    Overall value: {compute_position_value(board):.3f} (expect >0)")

    assert compute_pawn_advancement(board) > 0, "White pawn advancement should be positive"
    assert compute_position_value(board) > 0, "Position should favor white"

    # Test 1d: Black in better position
    board = chess.Board()
    board.push_uci("f2f3")  # Terrible move
    board.push_uci("e7e5")
    board.push_uci("g2g4")  # Another terrible move
    print("\n1d. Black in better position (more mobility):")
    print(f"    Piece activity: {compute_piece_activity(board):.3f}")
    print(f"    Overall value: {compute_position_value(board):.3f}")

    print("\n✓ Reward computation tests passed")
    return True


def test_selfplay_generation():
    """Test 2: Verify self-play generates diverse values."""
    print("\n" + "=" * 70)
    print("TEST 2: Self-Play with Shaped Rewards")
    print("=" * 70)

    from model.network import ChessNet
    from training.selfplay import SelfPlayWorker

    net = ChessNet(channels=64, num_blocks=4)
    worker = SelfPlayWorker(net, num_simulations=10)

    print("\n2a. Testing with shaped rewards (default):")
    experiences = worker.play_game(max_moves=50, use_shaped_rewards=True)

    values = [exp.value for exp in experiences]
    print(f"    Generated {len(experiences)} experiences")
    print(f"    Value range: [{min(values):.3f}, {max(values):.3f}]")
    print(f"    Value mean: {np.mean(values):.3f}")
    print(f"    Value std: {np.std(values):.3f}")
    print(f"    Non-zero values: {sum(1 for v in values if abs(v) > 0.01)}/{len(values)}")

    # With shaped rewards, we should see variety
    assert len(experiences) > 0, "Should generate experiences"
    assert np.std(values) > 0.01, "Values should have variety with shaped rewards"

    print("\n2b. Testing without shaped rewards (pure outcome):")
    experiences_pure = worker.play_game(max_moves=50, use_shaped_rewards=False)

    values_pure = [exp.value for exp in experiences_pure]
    print(f"    Generated {len(experiences_pure)} experiences")
    print(f"    Value range: [{min(values_pure):.3f}, {max(values_pure):.3f}]")
    print(f"    Value mean: {np.mean(values_pure):.3f}")
    print(f"    Value std: {np.std(values_pure):.3f}")

    print("\n✓ Self-play generation tests passed")
    return True


def test_training():
    """Test 3: Verify training can run with shaped rewards."""
    print("\n" + "=" * 70)
    print("TEST 3: Training with Shaped Rewards")
    print("=" * 70)

    from model.network import ChessNet
    from training.selfplay import SelfPlayWorker
    from training.train import train_iteration

    # Generate some experiences
    net = ChessNet(channels=64, num_blocks=4)
    worker = SelfPlayWorker(net, num_simulations=10)

    print("\n3a. Generating training data (2 games)...")
    experiences = []
    for i in range(2):
        game_exp = worker.play_game(max_moves=30, use_shaped_rewards=True)
        experiences.extend(game_exp)

    print(f"    Generated {len(experiences)} experiences")

    # Train on them
    print("\n3b. Training network (1 epoch)...")
    net_before = ChessNet(channels=64, num_blocks=4)
    net_before.load_state_dict(net.state_dict())

    history = train_iteration(
        network=net,
        experiences=experiences,
        batch_size=32,
        epochs=1,
        lr=1e-3
    )

    print(f"    Final loss: {history['total'][-1]:.4f}")

    # Verify network changed
    print("\n3c. Verifying network updated...")

    # Ensure networks are on CPU for comparison
    net.cpu()
    net_before.cpu()

    board = chess.Board()
    from encoding.state import encode_board
    from encoding.move import create_legal_move_mask

    state = encode_board(board)
    mask = create_legal_move_mask(board)
    mask_tensor = torch.tensor([mask], dtype=torch.bool)

    with torch.no_grad():
        _, value_before = net_before(state, mask_tensor)
        _, value_after = net(state, mask_tensor)

    print(f"    Value before training: {value_before.item():.4f}")
    print(f"    Value after training:  {value_after.item():.4f}")
    print(f"    Change: {abs(value_after.item() - value_before.item()):.4f}")

    # Network should have changed
    assert abs(value_after.item() - value_before.item()) > 1e-4, "Network should update"

    print("\n✓ Training tests passed")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("SHAPED REWARDS TEST SUITE")
    print("=" * 70)

    try:
        # Test 1: Reward computation
        test_reward_computation()

        # Test 2: Self-play generation
        test_selfplay_generation()

        # Test 3: Training
        test_training()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nShaped rewards implementation is working correctly.")
        print("Ready to deploy to Colab.")
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
