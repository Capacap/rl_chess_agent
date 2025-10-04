# Training Performance Analysis

**Test Run:** `checkpoints/20251004_212212`
**Config:** 1 iteration, 5 games, 20 MCTS simulations, 50 arena games

## Results

**Total Time:** 58.8 minutes

**Breakdown:**
- Self-play: 9.2 min (15.7%)
- Training: 1.9s (0.1%)
- Arena: 49.5 min (84.3%)

## Key Findings

### 1. Arena Evaluation is the Bottleneck
- 50 arena games took **49.5 minutes** (~1 min/game)
- This is **5.3x longer** than the self-play phase
- Arena uses same MCTS as self-play but plays 10x more games

### 2. Self-Play Performance
- 5 games in 9.2 min = **1.84 min/game**
- 500 experiences collected (avg 100 exp/game)
- Reasonable performance for untrained network

### 3. Training is Fast
- GPU-accelerated training: **1.9 seconds**
- Negligible compared to inference time
- Not a bottleneck

## Implications

**For 10-iteration training:**
- Self-play: 10 iter × 100 games × 1.84 min = **~30 hours**
- Arena: 10 iter × 50 games × 1 min = **~8.3 hours**
- Training: 10 iter × 2s = **~20 seconds**
- **Total: ~38 hours**

## Optimization Options

### Option 1: Reduce Arena Games (Recommended)
```bash
python train.py --arena-games 20
```
- Reduces arena from 49.5 min → 20 min per iteration
- Total training: ~30 + 3.3 = **~33 hours** (13% improvement)
- Trade-off: Less statistical confidence in replacement decisions

### Option 2: Reduce MCTS Simulations
```bash
python train.py --simulations 20 --arena-games 20
```
- Faster MCTS = faster games
- Expected: ~50% time reduction
- Total training: **~17 hours**
- Trade-off: Lower quality play

### Option 3: Reduce Games Per Iteration
```bash
python train.py --games-per-iter 50 --arena-games 20 --simulations 20
```
- Faster iterations, more iterations for same time budget
- Total for 10 iterations: **~10 hours**
- Trade-off: Less training data per iteration

## Recommended Settings

**For development/testing:**
```bash
python train.py --iterations 5 --games-per-iter 25 --simulations 20 --arena-games 20
```
Expected: ~5 hours

**For production/competition:**
```bash
python train.py --iterations 10 --games-per-iter 100 --simulations 40 --arena-games 30
```
Expected: ~30 hours

## Future Optimizations

1. **Virtual Loss Batching** - Batch MCTS node expansions for GPU
2. **Arena Parallelization** - Run arena games in parallel (independent)
3. **Early Stopping** - Stop arena early if challenger clearly winning/losing
4. **Adaptive Simulations** - Use fewer simulations for obvious positions

These would require significant architectural changes and are deferred to future phases.
