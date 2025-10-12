#!/usr/bin/env python3
"""
Quick verification that shaped rewards are producing diverse values.
Run this before starting long training to ensure rewards are working.

Expected:
- Value std > 0.10 (ideally 0.15-0.25)
- Decisive positions: 20-40%
- Value range spanning most of [-1, 1]
"""

from training.selfplay import SelfPlayWorker, DEFAULT_TEMP_SCHEDULE
from model.network import ChessNet
import numpy as np

def verify_shaped_rewards():
    print("=" * 70)
    print("Shaped Rewards Verification")
    print("=" * 70)
    print("\nCreating test network (48 channels, 3 blocks)...")
    net = ChessNet(channels=48, num_blocks=3)

    print("Creating self-play worker (12 sims)...")
    worker = SelfPlayWorker(net, DEFAULT_TEMP_SCHEDULE, num_simulations=12)

    print("\nPlaying 3 test games with shaped rewards...")
    print("(This will take ~3-5 minutes)\n")

    all_values = []
    for i in range(3):
        experiences = worker.play_game(max_moves=200, use_shaped_rewards=True)
        values = [exp.value for exp in experiences]
        all_values.extend(values)

        print(f"Game {i+1}:")
        print(f"  Positions: {len(experiences)}")
        print(f"  Value mean: {np.mean(values):+.3f}")
        print(f"  Value std:  {np.std(values):.3f}")
        print(f"  Value range: [{min(values):+.3f}, {max(values):+.3f}]")

    print("\n" + "=" * 70)
    print("Overall Statistics")
    print("=" * 70)

    value_std = np.std(all_values)
    value_mean = np.mean(all_values)
    decisive_count = sum(1 for v in all_values if abs(v) > 0.3)
    decisive_pct = decisive_count / len(all_values)
    near_draw_count = sum(1 for v in all_values if abs(v) < 0.1)
    near_draw_pct = near_draw_count / len(all_values)

    print(f"Total positions: {len(all_values)}")
    print(f"Value mean: {value_mean:+.3f}")
    print(f"Value std:  {value_std:.3f}")
    print(f"Value range: [{min(all_values):+.3f}, {max(all_values):+.3f}]")
    print(f"Decisive positions (|v| > 0.3): {decisive_count}/{len(all_values)} ({decisive_pct:.1%})")
    print(f"Near-draw positions (|v| < 0.1): {near_draw_count}/{len(all_values)} ({near_draw_pct:.1%})")

    print("\n" + "=" * 70)
    print("Assessment")
    print("=" * 70)

    passed = True

    # Check 1: Value std
    if value_std < 0.05:
        print("❌ FAIL: Value std too low (<0.05)")
        print("   Shaped rewards may not be working correctly")
        passed = False
    elif value_std < 0.10:
        print("⚠️  WARNING: Value std marginal (0.05-0.10)")
        print("   Consider checking reward computation")
    else:
        print(f"✓ PASS: Value std is {value_std:.3f} (>0.10)")

    # Check 2: Decisive positions
    if decisive_pct < 0.10:
        print("❌ FAIL: Too few decisive positions (<10%)")
        print("   Shaped rewards not providing enough gradient")
        passed = False
    elif decisive_pct < 0.20:
        print("⚠️  WARNING: Low decisive positions (10-20%)")
        print("   Training may be slow")
    else:
        print(f"✓ PASS: {decisive_pct:.1%} decisive positions")

    # Check 3: Value range
    value_range = max(all_values) - min(all_values)
    if value_range < 0.5:
        print(f"❌ FAIL: Value range too narrow ({value_range:.2f})")
        print("   Not exploring enough of the value space")
        passed = False
    else:
        print(f"✓ PASS: Value range is {value_range:.2f}")

    print("\n" + "=" * 70)
    if passed:
        print("✓ VERIFICATION PASSED")
        print("Shaped rewards are producing diverse values.")
        print("Safe to proceed with training.")
    else:
        print("❌ VERIFICATION FAILED")
        print("Do not proceed with long training.")
        print("Debug shaped rewards first.")
    print("=" * 70)

    return passed

if __name__ == "__main__":
    try:
        passed = verify_shaped_rewards()
        exit(0 if passed else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
