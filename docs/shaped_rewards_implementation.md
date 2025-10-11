# Shaped Rewards Implementation

**Date:** October 11, 2025
**Status:** Active
**Version:** 2.0 (replaces pure AlphaZero approach)

## Executive Summary

The training pipeline now uses **shaped rewards** to help the model bootstrap from random initialization. This hybrid approach combines game outcomes with intermediate position evaluations to provide richer learning signal.

**Key change:** Instead of only learning from final game results (win/loss/draw), the model now also learns from position quality during the game (material balance, pawn advancement, piece activity).

## Why This Change?

### Problem with Pure AlphaZero Approach

The original implementation used only sparse game outcomes:
- Win = +1.0
- Loss = -1.0
- Draw = 0.0

**Issue:** Early training produces mostly draws → all values near 0.0 → weak learning signal → champion never updates → training stagnates.

### Shaped Rewards Solution

Blend game outcomes with intermediate rewards:
```
value = 0.5 * game_outcome + 0.5 * position_value

where position_value combines:
  - Material balance (1.0 weight)
  - Pawn advancement (0.3 weight)
  - Piece activity (0.2 weight)
```

**Benefit:** Even in drawn games, positions have different values, providing gradient for learning.

## Reward Components

### 1. Material Balance (Weight: 1.0)

Standard chess piece values:
- Pawn: 1
- Knight/Bishop: 3
- Rook: 5
- Queen: 9

**Normalized:** Difference divided by 39 (starting material per side), clamped to [-1, 1].

**Purpose:** Teaches "don't lose pieces for nothing."

### 2. Pawn Advancement (Weight: 0.3)

Measures how close pawns are to promotion:
- White pawns: score increases from rank 2 → rank 7
- Black pawns: score increases from rank 7 → rank 2

**Normalized:** Sum of all pawn scores divided by 8 (max pawns).

**Purpose:** Incentivizes pushing pawns toward promotion, addresses "no promotions" concern.

### 3. Piece Activity (Weight: 0.2)

Mobility measure:
- Count legal moves for current player
- Count legal moves for opponent
- Difference normalized by 50 (typical move count)

**Purpose:** Encourages active play, discourages stagnant positions.

### 4. Game Outcome (Weight: 0.5)

Final game result from white's perspective:
- Win: +1.0
- Loss: -1.0
- Draw: 0.0

**Reduced from 1.0 to 0.5** to balance with intermediate rewards.

**Purpose:** Still teaches that winning matters, but not exclusively.

## Implementation Details

### Code Structure

- `training/rewards.py`: Reward computation functions
- `training/selfplay.py`: Modified to use shaped rewards
- `training/pipeline.py`: Adds diagnostics for reward distribution

### Self-Play

```python
# In selfplay.py
worker.play_game(max_moves=200, use_shaped_rewards=True)
```

For each position in the game:
1. Store board state and player
2. After game ends, compute:
   - Position value from board features
   - Game outcome
3. Blend: `value = 0.5 * outcome + 0.5 * position_value`
4. Convert to player's perspective

### Toggle Option

Set `use_shaped_rewards=False` to revert to pure outcome-based:
```python
worker.play_game(max_moves=200, use_shaped_rewards=False)
```

## Other Fixes

### 1. Increased Exploration

Temperature schedule adjusted:
```python
DEFAULT_TEMP_SCHEDULE = {
    0: 1.5,   # High exploration (was 1.0)
    20: 1.0,  # Stay exploratory longer (was move 10)
    40: 0.3   # Low temperature endgame (was 0.1 at move 20)
}
```

### 2. Longer Games

`max_moves` increased from 100 → 200 to avoid artificial draws.

### 3. Lower Arena Threshold

Early iterations use 50% threshold instead of 55% to ensure champion updates:
```python
threshold = 0.50 if iteration < 5 else 0.55
```

### 4. Diagnostic Logging

Pipeline now reports:
- Decisive position percentage
- Near-draw position percentage
- Value mean and standard deviation

Expected with shaped rewards:
- Decisive positions: 20-40%
- Value std: > 0.1

## Expected Training Behavior

### With Shaped Rewards

**Iteration 1:**
- Self-play values: diverse (std > 0.1)
- Arena: may still have draws, but champion should update if win_rate ≥ 50%

**Iterations 2-3:**
- Arena draws should decrease below 80%
- Some wins/losses appear
- Champion updates regularly

**Iterations 4+:**
- Decisive games increase
- Arena draws drop to 30-50%
- Clear improvement trajectory

### Warning Signs

**Red flags** (abort training):
- All self-play values clustered near 0.0 (std < 0.01)
- Arena 100% draws for 3+ iterations
- Training loss plateaus immediately

**Yellow flags** (watch closely):
- Arena draws > 80% at iteration 3
- No champion replacement by iteration 4
- Self-play value std decreasing over time

## Rollback Plan

If shaped rewards cause issues:

1. Keep exploration fixes (temperature, max_moves, threshold)
2. Disable shaped rewards:
   ```python
   # In pipeline.py, after line 116
   experiences = worker.generate_batch(
       games_per_iter,
       max_moves=200,
       num_workers=1,
       use_shaped_rewards=False  # Add this parameter
   )
   ```
3. Or adjust reward weights in `training/rewards.py`:
   ```python
   REWARD_WEIGHTS = {
       'material': 0.5,          # Reduce intermediate signal
       'pawn_advancement': 0.1,
       'piece_activity': 0.1,
       'outcome': 1.0            # Increase outcome weight
   }
   ```

## Testing

Run local tests before deploying:
```bash
source .venv/bin/activate
python test_shaped_rewards.py
```

All tests should pass:
- Test 1: Reward computation
- Test 2: Self-play generation
- Test 3: Training

## Performance Impact

**Compute cost:** ~5-10% slower due to reward computation
**Memory:** Minimal (stores board copies during self-play)
**Training stability:** Significantly improved (higher chance of convergence)

## References

- Original approach: `docs/phase2_completion_summary.md` (pure AlphaZero)
- Shaped rewards: Standard in RL literature (e.g., OpenAI gym environments)
- AlphaZero paper: Used only sparse rewards but with massive compute (5000 TPUs)

## Next Steps

1. Deploy to Colab with shaped rewards enabled
2. Monitor iteration 2 results:
   - Check self-play value distribution
   - Check arena draw rate
   - Verify champion updates
3. If iteration 2 still shows 100% draws, consider:
   - Increasing intermediate reward weights
   - Further increasing exploration
   - Adding Dirichlet noise to root MCTS

---

**Last updated:** October 11, 2025
**Contact:** See repository issues for questions
