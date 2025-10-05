# Colab Training Setup

This directory contains Jupyter notebooks for training the chess RL agent on Google Colab.

## Files

- **`setup_test.ipynb`** - Verify Colab environment is configured correctly (5-10 min)
- **`train.ipynb`** - Main training notebook for full training runs (12-30 hours)

## Quick Start

### 1. Subscribe to Colab Pro

Required for:
- Background execution (sessions continue after closing browser)
- Longer runtimes (24hr vs 12hr)
- Priority GPU access (V100/A100 vs T4)

**Cost:** $9.99/month

**Subscribe:** https://colab.research.google.com/signup

### 2. Upload Notebooks to Colab

**Option A: From GitHub (recommended)**
1. Open Google Colab: https://colab.research.google.com/
2. File → Open notebook → GitHub tab
3. Enter: `Capacap/rl_chess_agent`
4. Select `colab/setup_test.ipynb`

**Option B: Upload directly**
1. Download notebooks from this directory
2. Open Google Colab
3. File → Upload notebook
4. Select `setup_test.ipynb`

### 3. Run Setup Verification

**Purpose:** Ensure environment is ready before starting long training run

1. Open `setup_test.ipynb` in Colab
2. Runtime → Change runtime type → GPU (T4 or better)
3. Runtime → Run all
4. Verify all cells show ✓

**Expected output:**
- ✓ GPU ready
- ✓ Google Drive mounted and writable
- ✓ Repository cloned successfully
- ✓ All dependencies installed
- ✓ GPU inference working
- ✓ Checkpoint save/load working
- ✓ Drive backup working

**If any tests fail:** See troubleshooting section in notebook

### 4. Launch Training

1. Open `train.ipynb` in Colab
2. **Enable background execution:** Runtime → Background execution (Colab Pro only)
3. Configure training parameters (Step 5 in notebook)
   - Quick test: 5 iterations, 50 games (~2-3 hours)
   - Development: 10 iterations, 50 games (~12-15 hours)
   - Production: 15 iterations, 100 games (~24-30 hours)
4. Run all cells

**Monitor progress:**
- Check "Step 9: Monitor Training Progress" cell
- View training.log for detailed metrics
- Checkpoints auto-saved every iteration

### 5. Download Checkpoints

**During training:**
```python
# Run Step 7 in train.ipynb
!cp -r checkpoints/* /content/drive/MyDrive/chess_checkpoints/
```

**After training:**
1. Open Google Drive in browser
2. Navigate to `chess_checkpoints/YYYYMMDD_HHMMSS/`
3. Download entire folder
4. Extract to local project: `checkpoints/YYYYMMDD_HHMMSS/`

**Or use Drive desktop sync:**
- Install Google Drive desktop app
- Checkpoints auto-sync to local machine

## Workflow

```
┌─────────────────────────────────────────────────────────┐
│  Local Development                                      │
│  - Write code                                           │
│  - Test locally                                         │
│  - Commit & push to GitHub                              │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Google Colab                                           │
│  - Clone from GitHub                                    │
│  - Run training (12-30 hours)                           │
│  - Save checkpoints to Google Drive                     │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│  Local Evaluation                                       │
│  - Download checkpoints from Drive                      │
│  - Test against baselines                               │
│  - Integrate best model into my_agent.py                │
└─────────────────────────────────────────────────────────┘
```

## Training Configurations

### Quick Test (2-3 hours)
**Use case:** Verify pipeline works end-to-end

```python
ITERATIONS = 5
GAMES_PER_ITER = 50
SIMULATIONS = 20
ARENA_GAMES = 20
```

**Expected:** Basic agent, may not beat baselines

### Development Run (12-15 hours)
**Use case:** First real training, evaluate if approach is working

```python
ITERATIONS = 10
GAMES_PER_ITER = 50
SIMULATIONS = 20
ARENA_GAMES = 20
```

**Target:** Beat GreedyAgent (>70%), competitive vs MCTS (>50%)

### Production Run (24-30 hours)
**Use case:** Final agent for tournament submission

```python
ITERATIONS = 15
GAMES_PER_ITER = 100
SIMULATIONS = 40
ARENA_GAMES = 30
```

**Target:** Beat MCTS (>60%), tournament competitive

## Cost Analysis

**Colab Pro:** $9.99/month
- Cancel after training completes (Oct 22)
- Total cost: ~$10

**Alternative (RunPod):** ~$0.34/hr × 30hr = $10.20
- More setup complexity
- Less convenient

**Recommendation:** Colab Pro for integrated workflow

## Troubleshooting

### Session Disconnected
- **With Colab Pro background execution:** Training continues
- **Without Pro:** Training stops, must resume from last checkpoint
- **Solution:** Enable background execution (Runtime menu)

### Out of Memory
- Reduce `BATCH_SIZE` (try 128)
- Reduce `GAMES_PER_ITER` (try 25)
- Request smaller GPU if V100/A100 causes issues

### Training Too Slow
- Reduce `SIMULATIONS` (try 10-15)
- Reduce `ARENA_GAMES` (try 10-15)
- Use quick test config first

### Resume from Crash
```python
!python train.py --resume checkpoints/YYYYMMDD_HHMMSS/iteration_5.pt --iterations 10
```

### GitHub Authentication
If repo is private:
```python
# Use personal access token
!git clone https://YOUR_TOKEN@github.com/Capacap/rl_chess_agent.git
```

Or use SSH (requires key upload to Colab)

## Best Practices

**Before starting training:**
1. ✓ Run `setup_test.ipynb` and verify all tests pass
2. ✓ Push latest code to GitHub
3. ✓ Enable background execution
4. ✓ Monitor first 2-3 iterations to catch errors early

**During training:**
1. Check progress every 4-6 hours
2. Verify checkpoints are saving
3. Monitor GPU utilization (should be >80%)
4. Back up to Drive every few iterations

**After training:**
1. Download all checkpoints immediately
2. Test locally against baselines
3. Tag successful runs in git
4. Document hyperparameters and results

## Timeline (per roadmap.md)

- **Oct 4-5:** Colab setup and verification (Phase 0)
- **Oct 5-9:** First training run (Phase 2)
- **Oct 10-16:** Hyperparameter tuning (Phase 3)
- **Oct 16-18:** Production training (Phase 3.5)
- **Oct 18-22:** Local evaluation and submission prep (Phase 4)

## Support

**Colab issues:** https://research.google.com/colaboratory/faq.html

**Project issues:** https://github.com/Capacap/rl_chess_agent/issues

**Roadmap:** See `docs/roadmap.md` for full training plan
