# Chess RL Agent - Development Roadmap

**Deadline:** October 22, 2025
**Current Phase:** Cloud Training Setup
**Status:** In Progress
**Training Platform:** Google Colab Pro

---

## Training Architecture

**Division of Labor:**
- **Colab (Cloud):** GPU-accelerated training, self-play generation, arena evaluation
- **Local (Development PC):** Code development, testing, agent evaluation, submission prep

**Workflow:**
```
Local (code) → GitHub → Colab (training) → Google Drive (checkpoints) → Local (evaluation)
```

---

## Phase 0: Colab Infrastructure Setup (Oct 4-5)

### 0.1 Google Colab Pro Setup
- **Subscribe:** Colab Pro ($9.99/month)
  - 24hr runtime sessions (vs 12hr free tier)
  - Background execution enabled
  - Priority GPU access (V100/A100)
- **Connect Google Drive:** For checkpoint persistence
- **GitHub authentication:** SSH key or personal access token

### 0.2 Colab Notebook Creation
- **Create `colab/train.ipynb`:**
  - Drive mounting automation
  - Repo cloning from GitHub
  - Dependency installation
  - Training execution wrapper
  - Checkpoint backup to Drive
  - Real-time log monitoring
- **Create `colab/setup_test.ipynb`:**
  - GPU availability verification
  - PyTorch CUDA test
  - Dependency validation

### 0.3 Workflow Verification
- **Test end-to-end:**
  - Push code from local → GitHub
  - Pull in Colab
  - Run 1-iteration test training
  - Download checkpoint to local
  - Load weights in `my_agent.py`
- **Verify Google Drive sync:** Auto-backup checkpoints

**Deliverable:** Functional cloud training pipeline with checkpoint sync

**Time estimate:** 2-3 hours

---

## Phase 1: Neural Network Foundation (Oct 2-5, COMPLETED)

### 1.1 State Representation
- **Board Encoder:** `chess.Board` → tensor representation
  - 8×8×12 planes for piece positions (6 piece types × 2 colors)
  - Metadata planes: castling rights, en passant, side to move
  - Move history (optional: last N positions for repetition detection)
- **Move Encoder:** Legal moves → action space indexing
  - From-to square encoding (64×64 = 4096 possible moves)
  - Underpromotions handling
  - Masking illegal moves

### 1.2 Network Architecture
- **Design:** Small ResNet or CNN (must run <2s on CPU, <2GB RAM)
  - Input: State tensor from encoder
  - Backbone: 3-5 residual blocks
  - Dual heads:
    - **Policy head:** Outputs move probabilities (4096-dim softmax)
    - **Value head:** Outputs position evaluation (-1 to +1)
- **Implementation:** PyTorch, explicit typing
- **Test:** Forward pass on sample positions, verify output shapes

### 1.3 Data Pipeline
- **Experience tuple:** (board_state, move_probabilities, game_outcome)
- **Storage:** Simple pickle/JSON for now, upgrade if needed
- **Loader:** Batch sampling for training

**Deliverable:** Functional NN that can evaluate positions and suggest moves (untrained)

**Status:** ✓ Complete (model/network.py, training/pipeline.py exist)

---

## Phase 2: First Training Run (Oct 5-9)

**Location:** Google Colab
**Goal:** Validate training pipeline, produce first trained model

### 2.1 Launch Training on Colab

**Training configuration:**
```bash
python train.py \
  --iterations 10 \
  --games-per-iter 50 \
  --simulations 20 \
  --arena-games 20 \
  --batch-size 256 \
  --epochs 5
```

**Expected runtime:** 12-15 hours (overnight run)

**Monitoring:**
- Check Colab session every 4-6 hours
- View `training.log` for progress
- Monitor GPU utilization
- Verify checkpoints saving correctly

### 2.2 Checkpoint Management

**Colab (automatic backup):**
```python
# After each iteration
!cp -r checkpoints/* /content/drive/MyDrive/chess_checkpoints/
```

**Local (download):**
- Download from Google Drive to `checkpoints/`
- Or use Drive desktop sync

### 2.3 Evaluation (Local)

**Test against baselines:**
```bash
# Test trained agent
python tests/test_agent.py \
  --checkpoint checkpoints/YYYYMMDD_HHMMSS/iteration_10.pt \
  --opponent greedy \
  --games 50

python tests/test_agent.py \
  --checkpoint checkpoints/YYYYMMDD_HHMMSS/iteration_10.pt \
  --opponent mcts \
  --games 50
```

**Success criteria:**
- Win rate vs GreedyAgent: >70%
- Win rate vs MCTSAgent: >55%
- No crashes, valid moves only

**If failed:**
- Analyze training.log for issues
- Check policy/value loss convergence
- Verify MCTS integration working
- Adjust hyperparameters and rerun

**Deliverable:** First trained model with baseline evaluation results

**Time estimate:** 15 hours (mostly automated, ~2 hours active work)

---

## Phase 3: Hyperparameter Tuning (Oct 10-16)

**Location:** Google Colab (parallel experiments)
**Goal:** Identify optimal hyperparameters for final training run

### 3.1 Parallel Experiment Suite

**Run 3 experiments simultaneously on separate Colab notebooks:**

**Experiment A - Deeper Network:**
```bash
python train.py --iterations 10 --blocks 6 --channels 64 \
  --games-per-iter 50 --simulations 20 --arena-games 20
```

**Experiment B - More MCTS Simulations:**
```bash
python train.py --iterations 10 --blocks 4 --channels 64 \
  --games-per-iter 50 --simulations 40 --arena-games 20
```

**Experiment C - Temperature Schedule Variant:**
```bash
# Modify training/selfplay.py temperature decay
python train.py --iterations 10 --blocks 4 --channels 64 \
  --games-per-iter 50 --simulations 30 --arena-games 20
```

**Expected:** ~12 hours per experiment (run concurrently)

### 3.2 Evaluation & Selection (Local)

**Download all checkpoints, compare:**
```bash
# Create comparison matrix
python scripts/compare_agents.py \
  --agents exp_a/iteration_10.pt exp_b/iteration_10.pt exp_c/iteration_10.pt \
  --opponents greedy mcts \
  --games 50
```

**Metrics to compare:**
- Win rate vs baselines
- Average move time (CPU inference)
- Game length distribution
- Training loss convergence

**Select best configuration** for final training run

### 3.3 CPU Inference Optimization (Local)

**Profile inference speed:**
```bash
python scripts/profile_agent.py \
  --checkpoint best_experiment/iteration_10.pt \
  --positions 100
```

**Optimizations:**
- Verify model runs in <1.9s per move on CPU
- Test with `torch.set_num_threads(1)` if needed
- Ensure no GPU dependencies in inference code
- Test on clean environment (CPU-only)

**Robustness testing:**
- Edge cases: endgames, checkmates, stalemates
- Time pressure scenarios
- Memory profiling (<2GB)

**Deliverable:** Best hyperparameter configuration identified and validated

**Time estimate:** 15 hours (automated experiments) + 4 hours (evaluation)

---

## Phase 3.5: Final Production Training (Oct 16-18)

**Location:** Google Colab
**Goal:** Train final tournament agent with optimal hyperparameters

### 3.5.1 Production Training Run

**Configuration (using best params from Phase 3):**
```bash
python train.py \
  --iterations 15 \
  --games-per-iter 100 \
  --simulations [best_value] \
  --blocks [best_value] \
  --channels 64 \
  --arena-games 30 \
  --batch-size 256 \
  --epochs 5 \
  --checkpoint-dir checkpoints/production_run
```

**Expected runtime:** 24-30 hours (full day continuous run)

**Monitoring:**
- Enable Colab Pro background execution
- Check progress every 6-8 hours
- Monitor for any crashes or stalls
- Verify checkpoint backups to Drive

### 3.5.2 Final Validation (Local)

**Download production checkpoint:**
```bash
# From Google Drive
cp -r ~/GoogleDrive/chess_checkpoints/production_run ./checkpoints/
```

**Comprehensive testing:**
```bash
# Test against all baselines
python tests/test_agent.py --checkpoint checkpoints/production_run/iteration_15.pt

# CPU-only verification
CUDA_VISIBLE_DEVICES="" python tests/test_agent.py \
  --checkpoint checkpoints/production_run/iteration_15.pt

# Time limit compliance
python tests/test_timing.py --checkpoint checkpoints/production_run/iteration_15.pt
```

**Success criteria:**
- Win rate vs GreedyAgent: >75%
- Win rate vs MCTSAgent: >60%
- Average move time: <1.8s (on CPU)
- No illegal moves, no crashes

**Deliverable:** Production-ready trained model

**Time estimate:** 30 hours (training) + 3 hours (validation)

---

## Phase 4: Tournament Preparation (Oct 18-22)

**Location:** Local only
**Goal:** Package and test final submission

### 4.1 Agent Integration

**Update `my_agent.py`:**
```python
class MyAgent(Agent):
    def __init__(self):
        # Load production weights
        checkpoint = torch.load('checkpoints/production_run/iteration_15.pt',
                                map_location='cpu')
        self.network = ChessNet(...)
        self.network.load_state_dict(checkpoint['network'])
        self.network.eval()
        torch.set_num_threads(1)  # CPU optimization

    def make_move(self, board, time_limit):
        # MCTS + NN inference
        # Fallback to greedy if time pressure
        ...
```

**Error handling:**
- Timeout fallback (if MCTS exceeds time budget)
- Network failure fallback (random legal move)
- Logging disabled for submission

### 4.2 Testing & Validation

**Clean environment test:**
```bash
# Create fresh venv
python -m venv test_env
source test_env/bin/activate
pip install -r requirements.txt

# Test agent
python -c "
from my_agent import MyAgent
from agent_interface import Agent
agent = MyAgent()
# Run test game
"
```

**Edge case testing:**
```bash
python tests/test_edge_cases.py --agent my_agent.py
# Tests: checkmate detection, stalemate, repetition, 50-move rule
```

**Memory profiling:**
```bash
python -m memory_profiler tests/test_agent.py
# Verify <2GB usage
```

### 4.3 Submission Checklist

- [ ] Agent inherits from `Agent` class
- [ ] `make_move()` returns legal moves within 2s time limit
- [ ] All dependencies in `requirements.txt`
- [ ] Model weights included (verify file size)
- [ ] CPU-only inference verified (no CUDA calls)
- [ ] Tested on clean environment
- [ ] No external file I/O or network calls
- [ ] Deterministic behavior (seeded RNG if needed)
- [ ] Renamed to `[your_name].py`

### 4.4 Final Validation

**Tournament simulation:**
```bash
# Play 20 games against each baseline
python tests/tournament_sim.py --agent my_agent.py --games 20
```

**Expected performance:**
- GreedyAgent: >75% win rate
- MCTSAgent: >60% win rate
- RandomAgent: >95% win rate

**Deliverable:** Submission-ready `[your_name].py` file

**Time estimate:** 6 hours (integration + testing)

---

## Key Constraints

**Tournament limits (CPU-only submission):**
- 2s per move (reserve 0.2s safety margin)
- CPU-only execution (no GPU/CUDA in submitted agent)
- 2GB RAM maximum
- Dependencies must be in `requirements.txt`
- No external resources (network, files, databases)

**Training limits (Colab):**
- 24hr session limit (Colab Pro)
- Must handle session disconnects gracefully
- Checkpoint frequently (every iteration)
- GPU training allowed (not for submission)

**Success criteria:**
- Beat GreedyAgent consistently (>75% win rate)
- Beat baseline MCTS (>60% win rate)
- Competitive in tournament (top 50% = top 10-15 agents)

---

## Colab Workflow Best Practices

**Session management:**
- Enable background execution (Colab Pro feature)
- Use `nohup` or screen if running long commands
- Monitor via Google Drive checkpoint timestamps
- Keep browser tab open when possible (prevents idle disconnect)

**Checkpoint strategy:**
- Save after every iteration (not just at end)
- Backup to Google Drive immediately after save
- Name checkpoints with timestamps
- Keep last 3 iterations minimum (rollback capability)

**Debugging failures:**
- Always check `training.log` first
- Monitor GPU memory usage (nvidia-smi)
- Verify CUDA availability at start
- Test on 1 iteration before launching full run

**Code sync:**
- Always commit locally before pushing
- Pull in Colab before each run
- Use branches for experimental changes
- Tag successful training runs in git

---

## Timeline Summary

| Phase | Duration | Location | Key Deliverable |
|-------|----------|----------|-----------------|
| Phase 0 | Oct 4-5 (2-3h) | Local + Colab | Cloud training pipeline |
| Phase 1 | Oct 2-5 (DONE) | Local | Network architecture |
| Phase 2 | Oct 5-9 (15h) | Colab + Local | First trained model |
| Phase 3 | Oct 10-16 (19h) | Colab + Local | Optimal hyperparameters |
| Phase 3.5 | Oct 16-18 (33h) | Colab + Local | Production model |
| Phase 4 | Oct 18-22 (6h) | Local | Tournament submission |

**Total active work:** ~37 hours
**Total training time:** ~58 hours (automated on Colab)

**Critical path:** Phase 0 must complete by Oct 5 to stay on schedule

---

## Resources

**Cloud:**
- [Google Colab](https://colab.research.google.com/)
- [Colab Pro](https://colab.research.google.com/signup) ($9.99/month)
- [Google Drive](https://drive.google.com/) (checkpoint storage)

**Existing code:**
- [train.py](../train.py) - Training pipeline entrypoint
- [model/network.py](../model/network.py) - ChessNet architecture
- [training/pipeline.py](../training/pipeline.py) - Training loop
- [my_agent.py](../my_agent.py) - Tournament submission template
- [mcts_agent.py](../mcts_agent.py) - Pure MCTS baseline
- [agent_interface.py](../agent_interface.py) - Interface contract

**Documentation:**
- [project_description.md](./project_description.md) - Tournament rules
- [training_performance.md](./training_performance.md) - Performance analysis

**Key papers/references:**
- AlphaZero: MCTS + NN, self-play training
- Leela Chess Zero: Open-source chess engine using similar approach
- Policy-value network architecture patterns

---

## Risk Mitigation

**Risk: Colab session crashes during training**
- Mitigation: Checkpoint every iteration, can resume with `--resume`
- Fallback: Run multiple shorter training sessions

**Risk: Trained model too slow on CPU**
- Mitigation: Profile early (Phase 3.3), adjust network size
- Fallback: Reduce MCTS simulations, use faster forward pass

**Risk: Training doesn't improve over baseline**
- Mitigation: Evaluate after Phase 2, adjust hyperparameters
- Fallback: Enhanced MCTS without NN (proven baseline)

**Risk: Runs out of time before deadline**
- Mitigation: Phase 3.5 buffer (Oct 16-18), can skip Phase 3 if needed
- Fallback: Submit best Phase 2 checkpoint

---

## Notes

- **Philosophy:** Simplest solution that works (per .claude/CLAUDE.md)
- **Fail fast:** Evaluate after each phase, adjust if not improving
- **Time management:** Phase 4 is buffer time for bugs, not new features
- **Determinism:** Seed RNG for reproducible training (set in train.py)
- **Cost:** Colab Pro ($10) + time investment (~40 active hours)
