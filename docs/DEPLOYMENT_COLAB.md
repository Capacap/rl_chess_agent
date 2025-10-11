# Colab Deployment Guide - Shaped Rewards v2.0

**Date:** October 11, 2025
**Target:** Resume training with shaped rewards after iteration 1

## Current Situation

- Iteration 1 completed with all arena games as draws
- Champion not updated (stuck on iteration_0)
- Need to deploy shaped rewards fixes

## What Changed

1. **Shaped rewards:** Blends game outcome with position features (material, pawn advancement, activity)
2. **Increased exploration:** Temperature schedule raised (1.0 → 1.5 early game)
3. **Longer games:** max_moves increased from 100 → 200
4. **Lower threshold:** Arena threshold 50% for iterations 1-5 (was 55%)
5. **Better diagnostics:** Logs value distribution stats

## Deployment Steps

### Step 1: Pull Latest Code

In your Colab notebook:

```python
# Navigate to repo
%cd /content/rl_chess_agent

# Stash any local changes
!git stash

# Pull latest changes
!git pull origin master

# Verify shaped rewards module exists
!ls -la training/rewards.py
```

Expected output: `training/rewards.py` should exist

### Step 2: Run Tests

Verify everything works:

```python
# Run test suite
!python test_shaped_rewards.py
```

Expected output: "ALL TESTS PASSED ✓"

If tests fail, do NOT proceed - report the error.

### Step 3: Resume Training

**Option A: Continue from iteration 1 challenger** (recommended)

```python
!python train.py \
  --resume checkpoints/YOUR_TIMESTAMP/iteration_1_challenger.pt \
  --iterations 9 \
  --games-per-iter 50 \
  --simulations 20 \
  --arena-games 20 \
  --checkpoint-dir checkpoints/YOUR_TIMESTAMP \
  --gdrive-backup-dir /content/drive/MyDrive/chess_training
```

**Option B: Start fresh iteration 2** (if challenger doesn't exist)

```python
!python train.py \
  --resume checkpoints/YOUR_TIMESTAMP/iteration_1.pt \
  --iterations 9 \
  --games-per-iter 50 \
  --simulations 20 \
  --arena-games 20 \
  --checkpoint-dir checkpoints/YOUR_TIMESTAMP \
  --gdrive-backup-dir /content/drive/MyDrive/chess_training
```

Replace `YOUR_TIMESTAMP` with your actual checkpoint directory (e.g., `20251011_143022`).

### Step 4: Monitor Iteration 2

Check the training log after ~3-4 hours:

```python
!tail -100 checkpoints/YOUR_TIMESTAMP/training.log
```

**Look for these lines:**

```
Self-play data quality:
  Decisive positions: X%
  Near-draw positions: Y%
  Value mean: Z
  Value std: W
```

**Good signs:**
- Decisive positions: 20-50%
- Value std: > 0.1
- Arena draws: < 90%
- Champion updated at least once by iteration 3

**Bad signs:**
- All values near 0.0 (std < 0.01)
- Arena still 100% draws
- No champion updates

## Decision Points

### After Iteration 2

**If arena draws < 90%:**
→ Continue training, working as expected

**If arena still has 90%+ draws but value std > 0.1:**
→ Continue for 1 more iteration, shaped rewards need time to bootstrap

**If arena 100% draws AND value std < 0.05:**
→ Abort, something is broken. Check:
1. Did shaped rewards actually load? (check training.log for "Self-play data quality")
2. Are reward weights reasonable? (check `training/rewards.py`)
3. Is MCTS working? (check if games reach max_moves)

### After Iteration 3

**If champion updated at least once:**
→ Great! Training is working, continue to completion

**If champion never updated AND arena still >80% draws:**
→ Consider increasing intermediate reward weights:
```python
# Edit training/rewards.py
REWARD_WEIGHTS = {
    'material': 1.5,        # Increase from 1.0
    'pawn_advancement': 0.5,  # Increase from 0.3
    'piece_activity': 0.3,    # Increase from 0.2
    'outcome': 0.3            # Decrease from 0.5
}
```

Then restart training with adjusted weights.

## Monitoring Commands

```python
# Check current progress
!grep "Iteration.*complete" checkpoints/YOUR_TIMESTAMP/training.log | tail -5

# Check arena results
!grep -A 5 "Arena results" checkpoints/YOUR_TIMESTAMP/training.log | tail -20

# Check champion updates
!grep "promoted\|retained" checkpoints/YOUR_TIMESTAMP/training.log | tail -10

# Check value statistics
!grep -A 4 "Self-play data quality" checkpoints/YOUR_TIMESTAMP/training.log | tail -15

# List checkpoints
!ls -lh checkpoints/YOUR_TIMESTAMP/*.pt
```

## Backup Verification

Ensure Google Drive backup is working:

```python
!ls -lh /content/drive/MyDrive/chess_training/YOUR_TIMESTAMP/
```

Should see:
- `iteration_*.pt` files
- `training.log`

## Troubleshooting

### Error: "No module named 'training.rewards'"

→ Git pull didn't work. Manually verify:
```python
!cat training/rewards.py | head -20
```

Should show reward functions. If not, clone repo again.

### Error: "shaped_rewards parameter not found"

→ Old version of selfplay.py. Verify:
```python
!grep "use_shaped_rewards" training/selfplay.py
```

Should show the parameter. If not, git pull failed.

### Warning: "All arena games are draws"

→ Expected for iteration 2 if shaped rewards just deployed. Check value std:
- If std > 0.1: wait for iteration 3
- If std < 0.05: shaped rewards not working, investigate

### GPU out of memory

→ Reduce batch size:
```python
!python train.py ... --batch-size 128  # was 256
```

## Expected Timeline

With shaped rewards:

- **Iteration 2:** ~3-4 hours, may still have many draws but values should be diverse
- **Iteration 3:** ~3-4 hours, draws should decrease below 80%, champion should update
- **Iterations 4-10:** ~20-30 hours total, progressive improvement

Total: ~30-40 hours for 10 iterations

## Success Metrics

By iteration 5, you should see:
- Arena draw rate < 60%
- Champion updated 2-3 times
- Value std > 0.15
- Clear wins/losses in arena

By iteration 10:
- Arena draw rate < 40%
- Consistent champion updates
- Model plays sensible chess

## Rollback Plan

If shaped rewards make things worse:

1. Edit `training/selfplay.py` line 69:
```python
def play_game(self, max_moves: int = 200, use_shaped_rewards: bool = False):  # Change True → False
```

2. Restart training with pure outcome-based rewards

3. Keep the exploration and threshold fixes (those help regardless)

## Questions?

Check:
- [Shaped Rewards Implementation](shaped_rewards_implementation.md)
- [Training Guide](colab_training_guide.md)
- Repository issues

---

**Ready?** Proceed to Step 1 and deploy.
