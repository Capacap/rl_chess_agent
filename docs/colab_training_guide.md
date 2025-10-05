# Colab Training Guide

**Purpose:** Step-by-step guide for training the chess RL agent on Google Colab

**Last Updated:** October 5, 2025

**Current Phase:** Phase 0 → Phase 2 transition

---

## Prerequisites

✓ Repository is public (or you have authentication configured)
✓ `colab/setup_test.ipynb` passed all 8 checks
✓ Google account with Drive access

---

## Step 1: Subscribe to Colab Pro

**Link:** https://colab.research.google.com/signup

**Cost:** $9.99/month (cancel anytime after October 22)

**Why Colab Pro is required:**
- ✓ **Background execution** - Training continues when browser is closed (critical for 12-30hr runs)
- ✓ **24hr runtime** - vs 12hr on free tier (not enough for full training)
- ✓ **Priority GPU access** - V100/A100 instead of T4 (2-3x faster)
- ✓ **Longer idle timeout** - Less risk of session disconnect

**Without Pro:** Training will fail mid-run when session times out

---

## Step 2: Launch First Training Run (Phase 2)

### Open Training Notebook

1. Go to https://colab.research.google.com/
2. File → Open notebook → GitHub tab
3. Enter: `Capacap/rl_chess_agent`
4. Select: `colab/train.ipynb`

### Configure Runtime

1. Runtime → Change runtime type
2. Hardware accelerator: **GPU**
3. GPU type: **Premium** (if available with Pro)
4. Save

### Set Training Parameters

**For Phase 2 (first real training run):**

```python
# In Step 5 of train.ipynb
ITERATIONS = 10
GAMES_PER_ITER = 50
SIMULATIONS = 20
ARENA_GAMES = 20
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 1e-3
```

**Expected runtime:** ~12-15 hours

**What this will produce:**
- 10 checkpoint files (`iteration_0.pt` through `iteration_10_challenger.pt`)
- Training log with metrics
- First trained agent for evaluation

### Enable Background Execution

**CRITICAL:** Before running training cell

1. Runtime → **Background execution** (Colab Pro feature)
2. Verify checkmark appears

This allows training to continue when you:
- Close browser tab
- Lose internet connection
- Computer goes to sleep

### Start Training

1. Run all cells in order (or Runtime → Run all)
2. Monitor first 30 minutes to verify:
   - GPU detected (should show V100 or A100)
   - Self-play games generating (~13s/game expected)
   - No errors in output
3. After verified stable, you can close browser

---

## Step 3: Monitor Training Progress

### Check Progress (every 4-6 hours)

**Reconnect to notebook:**
1. Open `train.ipynb` in Colab
2. Session should reconnect automatically
3. Run "Step 9: Monitor Training Progress" cell

**Expected output:**
```
Checkpoints saved: 5
Progress: 5/10 iterations

Latest checkpoint: checkpoints/20251005_123456/iteration_5_challenger.pt
Size: 33.7 MB

GPU utilization: 85%, Memory used: 4096 MiB
```

### View Detailed Logs

```python
!tail -50 checkpoints/YYYYMMDD_HHMMSS/training.log
```

**Look for:**
- Policy loss decreasing (3.5 → 1.0 range)
- Value loss decreasing (0.5 → 0.05 range)
- Arena win rates (should vary, not all draws)
- Champion promotions (should see some "✓ Challenger promoted")

### Troubleshooting

**Session disconnected:**
- With background execution: Training continues, just reconnect
- Verify by checking checkpoint count
- If stuck on same iteration for >2 hours: May have crashed, check logs

**Out of memory:**
- Reduce `BATCH_SIZE` to 128
- Resume from last checkpoint

**Too slow:**
- Reduce `SIMULATIONS` to 15
- Reduce `ARENA_GAMES` to 15

**Resume from checkpoint:**
```python
!python train.py \
  --resume checkpoints/YYYYMMDD_HHMMSS/iteration_5.pt \
  --iterations 10 \
  --checkpoint-dir checkpoints/YYYYMMDD_HHMMSS
```

---

## Step 4: Download Checkpoints (After Training)

### Backup to Google Drive

**Run in Colab (Step 7 of train.ipynb):**
```python
!cp -r checkpoints/YYYYMMDD_HHMMSS /content/drive/MyDrive/chess_checkpoints/
```

**Verify backup:**
```python
!ls -lh /content/drive/MyDrive/chess_checkpoints/YYYYMMDD_HHMMSS/
```

### Download to Local Machine

**Option A: Direct download from Drive**
1. Open Google Drive in browser
2. Navigate to `chess_checkpoints/YYYYMMDD_HHMMSS/`
3. Right-click folder → Download
4. Extract to `~/Projects/rl_chess_agent/checkpoints/YYYYMMDD_HHMMSS/`

**Option B: Use Drive desktop sync**
1. Install Google Drive for Desktop
2. Checkpoints auto-sync to local Drive folder
3. Copy to project: `cp -r ~/GoogleDrive/chess_checkpoints/YYYYMMDD_HHMMSS checkpoints/`

**Verify download:**
```bash
ls -lh checkpoints/YYYYMMDD_HHMMSS/
# Should see: iteration_*.pt files and training.log
```

---

## Step 5: Evaluate Locally (Phase 2 → Phase 3 Transition)

### Test Against Baselines

**Activate local environment:**
```bash
cd ~/Projects/rl_chess_agent
source .venv/bin/activate
```

**Evaluate vs GreedyAgent:**
```bash
python evaluate_agent.py \
  --checkpoint checkpoints/YYYYMMDD_HHMMSS/iteration_10_challenger.pt \
  --opponent greedy \
  --games 50 \
  --simulations 20 \
  --color both
```

**Expected output:**
```
======================================================================
OVERALL RESULTS
======================================================================
Total games: 100
Wins: 75
Losses: 20
Draws: 5
Win rate: 75.0%
======================================================================
```

**Evaluate vs MCTSAgent:**
```bash
python evaluate_agent.py \
  --checkpoint checkpoints/YYYYMMDD_HHMMSS/iteration_10_challenger.pt \
  --opponent mcts \
  --games 50 \
  --simulations 20 \
  --color both
```

### Success Criteria (Phase 2)

**Minimum requirements:**
- ✓ Win rate vs GreedyAgent: **>70%**
- ✓ Win rate vs MCTSAgent: **>50%**
- ✓ No illegal moves or crashes
- ✓ Average move time <2s on CPU

**If successful:** Proceed to Phase 3 (hyperparameter tuning)

**If failed (win rate <50% vs Greedy):**
- Check training.log for anomalies
- Verify policy/value loss decreased
- Check for arena stagnation (all draws)
- May need to adjust hyperparameters and rerun

---

## Integration with Roadmap

### Phase 0: Colab Setup (Oct 4-5) ✓ COMPLETE

- [x] Subscribe to Colab Pro
- [x] Run setup_test.ipynb (passed in 18 min)
- [x] Verify GPU training works
- [x] Checkpoint save/load tested

**Deliverable:** Working cloud training environment

### Phase 2: First Training Run (Oct 5-9) ← YOU ARE HERE

**Timeline:**
- **Oct 5 evening:** Launch training on Colab (10 iter, 50 games)
- **Oct 6-7:** Training runs (~12-15 hours)
- **Oct 7:** Download checkpoints, evaluate locally
- **Oct 8-9:** Analyze results, decision point

**Configuration:**
- 10 iterations
- 50 games per iteration (500 total self-play games)
- 20 MCTS simulations
- 20 arena games per iteration

**Expected results:**
- First trained agent
- Baseline performance metrics
- Validation that approach works

**Key decision:** Does agent beat baselines? If yes → Phase 3. If no → diagnose and adjust.

**Deliverable:** Trained model with >70% win rate vs GreedyAgent

### Phase 3: Hyperparameter Tuning (Oct 10-16)

**If Phase 2 successful, run 3 parallel experiments:**

**Experiment A - Deeper network:**
```python
ITERATIONS = 10
BLOCKS = 6  # vs 4 in Phase 2
CHANNELS = 64
SIMULATIONS = 20
```

**Experiment B - More MCTS:**
```python
ITERATIONS = 10
BLOCKS = 4
SIMULATIONS = 40  # vs 20 in Phase 2
```

**Experiment C - More data:**
```python
ITERATIONS = 10
GAMES_PER_ITER = 100  # vs 50 in Phase 2
SIMULATIONS = 30
```

**Process:**
1. Launch 3 separate Colab notebooks (can run simultaneously)
2. Each runs ~12-15 hours
3. Download all checkpoints
4. Compare performance locally
5. Select best configuration

**Deliverable:** Optimal hyperparameters identified

### Phase 3.5: Production Training (Oct 16-18)

**Use best config from Phase 3:**
```python
ITERATIONS = 15
GAMES_PER_ITER = 100
SIMULATIONS = [best_value]
BLOCKS = [best_value]
ARENA_GAMES = 30
```

**Expected runtime:** 24-30 hours

**Target performance:**
- Win rate vs GreedyAgent: >75%
- Win rate vs MCTSAgent: >60%
- Tournament competitive

**Deliverable:** Production-ready trained model

### Phase 4: Finalization (Oct 18-22)

**All local work, no Colab needed:**
1. Integrate best checkpoint into `my_agent.py`
2. Test CPU-only inference
3. Verify time limit compliance
4. Test against all baselines
5. Package for submission

**Deliverable:** Tournament-ready submission file

---

## Best Practices

### Before Starting Training

1. ✓ Push latest code to GitHub
2. ✓ Enable background execution in Colab
3. ✓ Verify GPU is V100 or A100 (not T4)
4. ✓ Set calendar reminder to check progress in 6 hours

### During Training

1. Check progress every 4-6 hours
2. Verify checkpoints are incrementing
3. Monitor GPU utilization (should be >80%)
4. Back up to Drive every few iterations (run Step 7)

### After Training

1. Download checkpoints immediately (before session expires)
2. Test locally within 24 hours
3. Document results (win rates, config used)
4. Tag successful runs in git: `git tag phase2-complete`

### Cost Management

- Cancel Colab Pro subscription after Oct 22 (or keep if useful)
- Total cost: ~$10 for entire project
- Alternative (RunPod) would cost similar but require more setup

---

## Troubleshooting Common Issues

### Training is too slow (>2 hours per iteration)

**Diagnosis:** GPU not being used, or T4 instead of V100

**Fix:**
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
# Should show: "Tesla V100" or "Tesla A100"
# If shows "Tesla T4": Restart runtime and request better GPU
```

### All arena games are draws

**Diagnosis:** Untrained network playing itself

**Expected:** First 1-2 iterations may have draws, should decrease
**Problem:** If all iterations have 100% draws, MCTS may not be working

**Fix:** Check training.log for actual move selection variety

### Out of memory errors

**Diagnosis:** Batch size too large for GPU memory

**Fix:**
```python
BATCH_SIZE = 128  # Reduce from 256
```

Or reduce network size:
```python
BLOCKS = 3  # Reduce from 4
```

### Session keeps disconnecting

**Diagnosis:** Not using background execution, or exceeded runtime

**Fix:**
1. Verify Colab Pro is active
2. Enable background execution
3. Keep one browser tab with Colab open (can be minimized)

---

## Support Resources

**Colab documentation:** https://research.google.com/colaboratory/faq.html

**Colab Pro features:** https://colab.research.google.com/signup

**Project roadmap:** `docs/roadmap.md`

**Colab setup:** `colab/README.md`

**Repository issues:** https://github.com/Capacap/rl_chess_agent/issues

---

## Quick Reference Commands

**Launch training:**
```python
# In colab/train.ipynb Step 6
!python train.py --iterations 10 --games-per-iter 50 --simulations 20 --arena-games 20
```

**Check progress:**
```python
!ls -lt checkpoints/*/iteration_*.pt | head -5
```

**View logs:**
```python
!tail -50 checkpoints/*/training.log
```

**Backup to Drive:**
```python
!cp -r checkpoints/* /content/drive/MyDrive/chess_checkpoints/
```

**Resume training:**
```python
!python train.py --resume checkpoints/YYYYMMDD_HHMMSS/iteration_5.pt --iterations 10
```

**Local evaluation:**
```bash
python evaluate_agent.py --checkpoint checkpoints/run1/iteration_10_challenger.pt --opponent greedy --games 50
```

---

**Timeline Status:** Phase 0 complete, entering Phase 2 (Oct 5-9)

**Next Action:** Subscribe to Colab Pro and launch first training run
