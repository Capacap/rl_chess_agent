# Chess RL Agent - Development Roadmap

**Deadline:** October 22, 2025
**Current Phase:** Neural Network Integration
**Status:** Planning

---

## Phase 1: Neural Network Foundation (Week 1: Oct 2-9)

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

---

## Phase 2: Training Infrastructure (Week 2: Oct 9-16)

### 2.1 Self-Play Generator
- **Agent:** MCTS + NN hybrid
  - Use policy network to guide expansion (prior probabilities)
  - Use value network instead of random rollouts
  - Temperature-based move selection (exploration vs exploitation)
- **Game loop:**
  - Play N games to completion
  - Record (state, MCTS policy, outcome) at each position
  - Save experience to buffer

### 2.2 Training Loop
- **Loss function:**
  - Policy loss: Cross-entropy between MCTS policy and network policy
  - Value loss: MSE between game outcome and value prediction
  - Combined: `loss = policy_loss + value_loss`
- **Optimizer:** Adam, learning rate ~1e-3
- **Batch size:** 256-1024 (tune based on memory)
- **Epochs:** Train on experience buffer until convergence

### 2.3 Checkpoint & Evaluation
- **Save/Load:** Best model weights
- **Arena:** Pit new model vs current best
  - Play M games (e.g., 50)
  - Replace if new_score > threshold (e.g., 55%)
- **Metrics:** Win rate, avg game length, move time

**Deliverable:** End-to-end training pipeline producing improving models

---

## Phase 3: Iteration & Tuning (Week 3: Oct 16-22)

### 3.1 Hyperparameter Tuning
- **MCTS params:**
  - Simulations per move (balance strength vs speed)
  - Exploration weight (UCB constant)
  - Temperature schedule (high early, low late-game)
- **Network params:**
  - Depth (residual blocks)
  - Width (channels per layer)
  - Learning rate, batch size

### 3.2 Advanced Features (if time permits)
- **Position evaluation refinements:**
  - King safety indicators
  - Piece mobility features
  - Pawn structure (doubled, isolated, passed)
- **MCTS improvements:**
  - Virtual loss for parallel simulations
  - Progressive widening
  - Time management (allocate more time for critical positions)

### 3.3 Final Optimization
- **Profiling:** Identify bottlenecks (likely MCTS or NN inference)
- **Speed optimizations:**
  - Batch NN inference if doing parallel MCTS
  - Reduce rollout depth
  - Prune unpromising branches early
- **Robustness:**
  - Test edge cases (endgames, tactics, time pressure)
  - Ensure CPU-only inference works
  - Verify 2s time limit compliance

**Deliverable:** Tournament-ready agent competitive against strong baselines

---

## Phase 4: Tournament Preparation (Oct 20-22)

### 4.1 Finalization
- **my_agent.py:** Implement production inference code
  - Load trained weights
  - Force CPU mode
  - Error handling, fallback to random if failure
- **Testing:** Run full game suite against all baselines
- **Documentation:** Inference-only (no training details needed)

### 4.2 Submission Checklist
- [ ] Agent inherits from `Agent` class
- [ ] `make_move()` returns legal moves within time_limit
- [ ] All dependencies in `requirements.txt`
- [ ] Model weights included (if file size allows)
- [ ] CPU-only inference verified
- [ ] Tested on clean environment

---

## Key Constraints

**Hard limits:**
- 2s per move (reserve 0.1s margin)
- CPU-only execution
- 2GB RAM maximum
- Dependencies must be in `requirements.txt`

**Success criteria:**
- Beat GreedyAgent consistently (>70% win rate)
- Beat baseline MCTS (>55% win rate)
- Competitive in tournament (top 50%)

---

## Open Questions

1. **State representation:** How many history planes? (affects network input size)
2. **Training data:** How many self-play games per iteration? (quality vs quantity)
3. **Network size:** Depth vs width tradeoff for CPU inference speed
4. **Move encoding:** Flat 4096 or structured (from_square, to_square, promotion)?

---

## Resources

**Existing code:**
- [mcts_agent.py](../mcts_agent.py) - Pure MCTS baseline
- [test_mcts.py](../test_mcts.py) - Benchmarking framework
- [agent_interface.py](../agent_interface.py) - Interface contract

**Key papers/references:**
- AlphaZero: MCTS + NN, self-play training
- Leela Chess Zero: Open-source chess engine using similar approach
- Policy-value network architecture patterns

---

## Notes

- **Philosophy:** Simplest solution that works (per AGENTS.md)
- **Fail fast:** If NN doesn't improve over MCTS baseline by Oct 16, revert to enhanced MCTS
- **Time management:** Reserve final 2 days for bug fixes, not new features
- **Determinism:** Seed RNG for reproducible training/testing
