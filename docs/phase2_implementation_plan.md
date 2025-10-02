# Phase 2 Implementation Plan: Training Infrastructure

**Target:** Week 2 (Oct 9-16, 2025)
**Status:** Ready for implementation
**Prerequisites:** Phase 1 complete (NN foundation operational)

---

## Problem Analysis

### Current State
- Phase 1 deliverables: `ChessNet`, `encode_board()`, `encode_move()`, `ChessDataset`
- Baseline MCTS (`mcts_agent.py`) uses random rollouts → replace with NN guidance
- Untrained network exists but has random weights
- Need self-play data generation + training loop to produce improving models

### Key Challenges

1. **MCTS-NN integration**: Replace random rollouts with value network
   - **Solution**: Hybrid approach—value head for leaf evaluation, policy head for move priors
   - Simulations reduced (100→40) due to NN inference cost (~50ms vs <1ms random)

2. **Experience quality vs quantity**: How many games per iteration?
   - **Decision**: 100 games/iteration initially (tune based on training signal)
   - Rationale: Balance diversity (need varied positions) vs time (2s move budget)

3. **Training stability**: Policy/value loss weighting, learning rate
   - **Approach**: Equal weighting (`loss = policy_loss + value_loss`) initially
   - Monitor loss curves, adjust if one head dominates

4. **Model improvement detection**: When to replace current best?
   - **Criterion**: Arena win rate >55% over 50 games (Wilson score >50% at 95% CI)
   - Avoid replacing on noise—require statistical significance

---

## Architecture Decisions

### 1. NN-Guided MCTS (`training/mcts_nn.py`)

**Modifications to baseline MCTS**:

```python
class NeuralMCTSNode:
    """MCTS node enhanced with neural network guidance."""

    def __init__(self, board: chess.Board, parent=None, move=None, prior: float = 0.0):
        self.board = board.copy()
        self.parent = parent
        self.move = move
        self.prior = prior  # Policy network prior probability

        # MCTS stats (same as baseline)
        self.visits = 0
        self.total_value = 0.0  # Sum of backpropagated values

        # Children
        self.children: dict[chess.Move, NeuralMCTSNode] = {}
        self.is_expanded = False
```

**UCB formula with priors** (PUCT algorithm from AlphaZero):
```
PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))

Where:
- Q(s,a) = total_value / visits (exploitation)
- P(s,a) = prior from policy network
- N(s) = parent visits
- N(s,a) = child visits
- c_puct = exploration constant (~1.0)
```

**Implementation**:
```python
def select_child(self, c_puct: float = 1.0) -> NeuralMCTSNode:
    """Select child using PUCT."""

    def puct_score(child: NeuralMCTSNode) -> float:
        if child.visits == 0:
            q_value = 0.0
        else:
            q_value = child.total_value / child.visits

        u_value = c_puct * child.prior * math.sqrt(self.visits) / (1 + child.visits)
        return q_value + u_value

    return max(self.children.values(), key=puct_score)

def expand(self, policy: torch.Tensor, legal_moves: list[chess.Move]) -> None:
    """
    Expand node with children, using policy network for priors.

    Args:
        policy: Policy distribution [4096], softmax over legal moves
        legal_moves: Legal moves from this position
    """
    for move in legal_moves:
        action_idx = encode_move(move)
        prior_prob = policy[action_idx].item()

        new_board = self.board.copy()
        new_board.push(move)

        self.children[move] = NeuralMCTSNode(
            new_board, parent=self, move=move, prior=prior_prob
        )

    self.is_expanded = True
```

**Leaf evaluation** (replaces random rollout):
```python
def evaluate_leaf(self, network: ChessNet) -> float:
    """
    Evaluate position using value network.

    Returns:
        Value in [-1, 1] from current player's perspective
    """
    state = encode_board(self.board).unsqueeze(0)

    # Get legal moves mask
    legal_moves = list(self.board.legal_moves)
    legal_mask = torch.zeros(1, 4096, dtype=torch.bool)
    for move in legal_moves:
        legal_mask[0, encode_move(move)] = True

    with torch.no_grad():
        policy, value = network(state, legal_mask)

    # Expand node if not terminal
    if not self.board.is_game_over():
        self.expand(policy[0], legal_moves)

    return value.item()
```

**Rationale**:
- PUCT balances NN priors (promising moves) with exploration (uncertainty)
- No random rollouts—value network directly estimates position
- Expansion at leaf (not at selection) reduces NN calls (1 per simulation vs 1 per node)

---

### 2. Self-Play Generator (`training/selfplay.py`)

**Game generation loop**:

```python
class SelfPlayWorker:
    """Generates training data via self-play."""

    def __init__(self, network: ChessNet, temperature_schedule: dict[int, float]):
        self.network = network
        self.temp_schedule = temperature_schedule  # {move_num: temperature}

    def play_game(self, num_simulations: int = 40) -> list[Experience]:
        """
        Play one game to completion, recording experiences.

        Returns:
            List of (fen, policy, value) tuples
        """
        board = chess.Board()
        experiences = []
        move_count = 0

        while not board.is_game_over():
            # Run MCTS from current position
            root = NeuralMCTSNode(board)
            for _ in range(num_simulations):
                self._mcts_iteration(root)

            # Extract MCTS policy (visit counts → probabilities)
            mcts_policy = self._get_policy_from_visits(root)

            # Sample move with temperature
            temp = self._get_temperature(move_count)
            move = self._sample_move(root, temperature=temp)

            # Record experience (outcome filled later)
            experiences.append({
                'fen': board.fen(),
                'policy': mcts_policy,  # [4096] array
                'value': None  # Placeholder
            })

            # Execute move
            board.push(move)
            move_count += 1

        # Backfill game outcome
        outcome = self._get_outcome(board)
        for i, exp in enumerate(experiences):
            # Flip outcome for alternating players
            player_outcome = outcome if i % 2 == 0 else -outcome
            exp['value'] = player_outcome

        return experiences

    def _get_temperature(self, move_count: int) -> float:
        """Get temperature for move selection (high early, low late)."""
        for threshold, temp in sorted(self.temp_schedule.items()):
            if move_count < threshold:
                return temp
        return 0.1  # Low temp endgame (deterministic)

    def _sample_move(self, root: NeuralMCTSNode, temperature: float) -> chess.Move:
        """
        Sample move from MCTS visit distribution with temperature.

        Args:
            temperature: 0 = deterministic (argmax), 1 = proportional, >1 = exploratory
        """
        moves = list(root.children.keys())
        visits = [root.children[m].visits for m in moves]

        if temperature < 1e-3:
            # Deterministic: pick most visited
            return moves[visits.index(max(visits))]

        # Apply temperature
        visits_temp = [v ** (1.0 / temperature) for v in visits]
        total = sum(visits_temp)
        probs = [v / total for v in visits_temp]

        return random.choices(moves, weights=probs, k=1)[0]

    def _get_policy_from_visits(self, root: NeuralMCTSNode) -> np.ndarray:
        """Convert MCTS visit counts to policy distribution."""
        policy = np.zeros(4096, dtype=np.float32)

        total_visits = sum(child.visits for child in root.children.values())
        if total_visits == 0:
            return policy  # All zeros (shouldn't happen)

        for move, child in root.children.items():
            action_idx = encode_move(move)
            policy[action_idx] = child.visits / total_visits

        return policy
```

**Temperature schedule** (exploration → exploitation):
```python
TEMP_SCHEDULE = {
    0: 1.0,   # Moves 0-10: High exploration
    10: 0.5,  # Moves 10-20: Moderate
    20: 0.1   # Moves 20+: Near-deterministic
}
```

**Parallel game generation** (if time permits):
- Generate N games in parallel using `multiprocessing`
- Share network weights (read-only), separate game states
- Aggregate experiences after batch completes

**Rationale**:
- MCTS visit counts = improved policy (better than raw NN)
- Temperature control balances diverse openings vs strong endgames
- Recording every position maximizes data per game

---

### 3. Training Loop (`training/train.py`)

**Loss function**:
```python
def compute_loss(
    policy_pred: torch.Tensor,  # [B, 4096]
    value_pred: torch.Tensor,   # [B, 1]
    policy_target: torch.Tensor,  # [B, 4096]
    value_target: torch.Tensor    # [B, 1]
) -> tuple[torch.Tensor, dict]:
    """
    Combined policy + value loss.

    Returns:
        total_loss, {policy_loss, value_loss} for logging
    """
    # Policy loss: Cross-entropy (KL divergence)
    policy_loss = -torch.sum(policy_target * torch.log(policy_pred + 1e-8), dim=1).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, value_target)

    # Combined (equal weighting)
    total_loss = policy_loss + value_loss

    return total_loss, {'policy': policy_loss.item(), 'value': value_loss.item()}
```

**Training iteration**:
```python
def train_iteration(
    network: ChessNet,
    experience_buffer: list[dict],
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-3
) -> None:
    """
    Train network on experience buffer.

    Args:
        network: ChessNet to train
        experience_buffer: List of {fen, policy, value} dicts
        epochs: Number of passes through data
    """
    # Create dataset
    dataset = ChessDataset.from_experiences(experience_buffer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Training loop
    network.train()
    for epoch in range(epochs):
        epoch_losses = {'policy': [], 'value': []}

        for states, policy_targets, value_targets in loader:
            # Forward pass
            legal_masks = get_legal_masks(states)  # From FEN strings
            policy_pred, value_pred = network(states, legal_masks)

            # Compute loss
            loss, metrics = compute_loss(
                policy_pred, value_pred,
                policy_targets, value_targets
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log
            epoch_losses['policy'].append(metrics['policy'])
            epoch_losses['value'].append(metrics['value'])

        # Epoch summary
        avg_policy = np.mean(epoch_losses['policy'])
        avg_value = np.mean(epoch_losses['value'])
        print(f"Epoch {epoch+1}/{epochs}: "
              f"policy_loss={avg_policy:.4f}, value_loss={avg_value:.4f}")
```

**Rationale**:
- Cross-entropy (not MSE) for policy—matches probability distribution
- Equal loss weighting (1:1) unless one head fails to learn
- Adam optimizer standard for NNs, lr=1e-3 safe starting point
- Multiple epochs over buffer extracts more signal per game

---

### 4. Arena Evaluation (`training/arena.py`)

**Model comparison protocol**:

```python
class Arena:
    """Pit two agents against each other."""

    def compete(
        self,
        challenger: ChessNet,
        champion: ChessNet,
        num_games: int = 50,
        time_per_move: float = 2.0
    ) -> dict:
        """
        Play num_games, alternating colors.

        Returns:
            {wins, losses, draws, win_rate}
        """
        results = {'wins': 0, 'losses': 0, 'draws': 0}

        for game_idx in range(num_games):
            # Alternate who plays white
            if game_idx % 2 == 0:
                white_net, black_net = challenger, champion
            else:
                white_net, black_net = champion, challenger

            # Play game
            outcome = self._play_game(white_net, black_net, time_per_move)

            # Record result from challenger's perspective
            if outcome == 'white' and game_idx % 2 == 0:
                results['wins'] += 1
            elif outcome == 'black' and game_idx % 2 == 1:
                results['wins'] += 1
            elif outcome == 'draw':
                results['draws'] += 1
            else:
                results['losses'] += 1

        results['win_rate'] = (results['wins'] + 0.5 * results['draws']) / num_games
        return results

    def _play_game(
        self,
        white_net: ChessNet,
        black_net: ChessNet,
        time_per_move: float
    ) -> str:
        """Play one game, return 'white', 'black', or 'draw'."""
        board = chess.Board()

        while not board.is_game_over():
            current_net = white_net if board.turn == chess.WHITE else black_net

            # MCTS move selection
            root = NeuralMCTSNode(board)
            start = time.time()
            sims = 0

            while time.time() - start < time_per_move - 0.1:
                self._mcts_iteration(root, current_net)
                sims += 1

            # Select most visited move
            if not root.children:
                move = random.choice(list(board.legal_moves))
            else:
                move = max(root.children.keys(),
                          key=lambda m: root.children[m].visits)

            board.push(move)

        # Return outcome
        outcome = board.outcome()
        if outcome.winner is None:
            return 'draw'
        return 'white' if outcome.winner == chess.WHITE else 'black'
```

**Replacement criterion**:
```python
def should_replace(challenger_score: float, num_games: int, threshold: float = 0.55) -> bool:
    """
    Decide if challenger should replace champion.

    Use Wilson score interval to avoid replacing on noise.
    """
    if challenger_score >= threshold:
        # Check statistical significance (95% CI lower bound > 0.5)
        z = 1.96  # 95% confidence
        p = challenger_score
        n = num_games

        lower_bound = (p + z*z/(2*n) - z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / (1 + z*z/n)

        return lower_bound > 0.5
    return False
```

**Rationale**:
- 50 games sufficient for significance test (±14% margin of error)
- Alternating colors eliminates first-move advantage bias
- Wilson score prevents replacing on lucky streaks
- Threshold 55% provides margin (not just >50%)

---

## Implementation Order

### Day 1-2: NN-MCTS Integration
1. `training/mcts_nn.py`
   - `NeuralMCTSNode` class (PUCT selection, NN evaluation)
   - Test: Verify NN integration doesn't crash, moves remain legal

**Acceptance**:
- NN-guided MCTS produces legal moves
- Inference time <100ms per simulation
- No memory leaks over 100 games

---

### Day 3-4: Self-Play Pipeline
1. `training/selfplay.py`
   - `SelfPlayWorker.play_game()` (MCTS + temp sampling)
   - `generate_batch(num_games)` → experience list

2. Test data generation:
```python
worker = SelfPlayWorker(network=untrained_net, temp_schedule=TEMP_SCHEDULE)
experiences = worker.generate_batch(num_games=10)

# Validate
assert len(experiences) > 0
assert all('fen' in exp and 'policy' in exp and 'value' in exp
           for exp in experiences)
assert all(exp['value'] in {-1.0, 0.0, 1.0} for exp in experiences)
```

**Acceptance**:
- 10 self-play games complete in <5min
- Experiences have correct format
- Policy distributions sum to 1.0

---

### Day 5: Training Loop
1. `training/train.py`
   - `compute_loss()` (policy + value)
   - `train_iteration()` (optimizer, epochs)
   - Loss logging (tensorboard optional, print minimum)

2. Smoke test:
```python
# Generate dummy data
experiences = generate_synthetic_experiences(1000)

# Train
train_iteration(network, experiences, epochs=3, batch_size=128)

# Check: loss should decrease
assert final_loss < initial_loss
```

**Acceptance**:
- Loss decreases over epochs
- No NaN/Inf in gradients
- Network weights update (not stuck)

---

### Day 6: Arena Evaluation
1. `training/arena.py`
   - `Arena.compete()` (play N games)
   - `should_replace()` (statistical test)

2. Test champion vs clone (should be ~50% win rate):
```python
arena = Arena()
results = arena.compete(network, network.copy(), num_games=20)
assert 0.4 <= results['win_rate'] <= 0.6  # Stochastic, allow margin
```

**Acceptance**:
- Arena runs to completion
- Win rate calculation correct
- Replacement logic sound

---

### Day 7: End-to-End Pipeline
1. `training/pipeline.py` (orchestrator):

```python
def training_pipeline(
    initial_network: ChessNet,
    num_iterations: int = 10,
    games_per_iter: int = 100
) -> ChessNet:
    """
    Full self-play → train → arena loop.

    Returns:
        Best network found
    """
    champion = initial_network

    for iteration in range(num_iterations):
        print(f"\n=== Iteration {iteration+1}/{num_iterations} ===")

        # 1. Self-play
        print(f"Generating {games_per_iter} self-play games...")
        worker = SelfPlayWorker(champion, TEMP_SCHEDULE)
        experiences = worker.generate_batch(games_per_iter)
        print(f"Collected {len(experiences)} experiences")

        # 2. Train challenger
        print("Training challenger network...")
        challenger = champion.copy()
        train_iteration(challenger, experiences, epochs=5)

        # 3. Arena
        print("Arena: Challenger vs Champion...")
        arena = Arena()
        results = arena.compete(challenger, champion, num_games=50)
        print(f"Challenger win rate: {results['win_rate']:.1%}")

        # 4. Replace if better
        if should_replace(results['win_rate'], num_games=50):
            print("Challenger promoted to champion!")
            champion = challenger
            champion.save(f"checkpoints/iter_{iteration}.pt")
        else:
            print("Champion retained.")

    return champion
```

2. Test with minimal settings (2 iterations, 10 games each):
```python
best_net = training_pipeline(
    initial_network=ChessNet(),
    num_iterations=2,
    games_per_iter=10
)
```

**Acceptance**:
- Pipeline completes without errors
- Champion model saved to disk
- Metrics logged at each stage

---

## File Structure (After Phase 2)

```
rl_chess_agent/
├── training/
│   ├── __init__.py
│   ├── dataset.py        # (Phase 1)
│   ├── mcts_nn.py        # NEW: Neural MCTS
│   ├── selfplay.py       # NEW: Self-play worker
│   ├── train.py          # NEW: Training loop
│   ├── arena.py          # NEW: Model evaluation
│   └── pipeline.py       # NEW: Orchestrator
├── checkpoints/          # NEW: Model saves
├── [existing Phase 1 files]
└── test_phase2_pipeline.py  # NEW: Integration test
```

---

## Validation Strategy

### Unit Tests
- `tests/test_mcts_nn.py`: PUCT selection, NN evaluation
- `tests/test_selfplay.py`: Game generation, temperature sampling
- `tests/test_training.py`: Loss computation, optimizer step
- `tests/test_arena.py`: Win rate calculation, replacement logic

### Integration Test
`test_phase2_pipeline.py`:
```python
def test_minimal_pipeline():
    """Run 1 training iteration end-to-end."""
    net = ChessNet(channels=32, blocks=2)  # Tiny for speed

    # Generate experiences
    worker = SelfPlayWorker(net, {0: 1.0})
    experiences = worker.generate_batch(num_games=5)

    # Train
    train_iteration(net, experiences, epochs=1, batch_size=32)

    # Arena
    arena = Arena()
    results = arena.compete(net, net, num_games=2)

    print("Phase 2 pipeline operational")
```

---

## Critical Decisions & Rationale

### 1. PUCT vs UCB1
- **Choice**: PUCT (AlphaZero variant)
- **Why**: Integrates NN priors naturally, proven for board games
- **Tradeoff**: Slightly more complex than UCB1 (acceptable)

### 2. Temperature schedule
- **Choice**: High early (1.0), low late (0.1)
- **Why**: Diverse openings improve generalization, strong endgames win games
- **Values**: Standard AlphaZero schedule

### 3. Equal loss weighting
- **Choice**: `loss = policy_loss + value_loss` (1:1)
- **Why**: No a priori reason to prefer one head
- **Tuning**: Monitor losses, adjust if one stagnates

### 4. 50-game arena
- **Choice**: 50 games for replacement decision
- **Why**: ±14% margin acceptable (vs 100 games = ±10% but 2× cost)
- **Statistical**: Wilson score ensures significance

### 5. Simulations: 100 → 40
- **Choice**: Reduce MCTS sims due to NN cost
- **Why**: 50ms NN × 40 = 2s (budget), vs 0.5ms random × 100 = 50ms
- **Tradeoff**: Fewer sims but better evaluation (NN > random)

---

## Risk Mitigation

### Risk: NN too slow, can't fit 40 simulations in 2s
- **Detection**: Profile actual game times in Day 3
- **Mitigation**: Reduce to 20 sims, or shrink network (3 blocks → 2)
- **Fallback**: Hybrid MCTS (NN for expansion only, random rollouts)

### Risk: Self-play games too slow
- **Detection**: <10 games in 5 minutes (Day 4)
- **Mitigation**: Reduce max game length (100 moves → 50), early resignation
- **Parallel**: Implement multiprocessing if sequential too slow

### Risk: Training unstable (NaN loss)
- **Detection**: Loss monitoring in training loop
- **Mitigation**: Gradient clipping, reduce LR to 1e-4
- **Fallback**: Batch norm → Layer norm (more stable)

### Risk: Network doesn't improve over random
- **Detection**: Arena shows <30% win vs random agent (Day 6)
- **Decision point**: Revert to enhanced MCTS (per roadmap) or debug
- **Deadline**: Oct 14 (2 days to pivot if needed)

---

## Success Metrics (Phase 2 Exit Criteria)

1. **Functional pipeline**: Self-play → train → arena runs to completion
2. **Model improvement**: Champion beats initial network (win rate >60%)
3. **Data quality**: MCTS policies show reasonable move preferences (not uniform)
4. **Performance**: Self-play games complete in <30s each
5. **Stability**: 10 training iterations without crashes/NaN

**Deliverable**: Trained network that defeats baseline MCTS (>55% win rate)

---

## Next Steps (Phase 3 Preview)

- Hyperparameter tuning (MCTS sims, LR, batch size)
- Advanced features (mobility, king safety planes)
- Profiling & optimization (batched inference, pruning)
- Endgame strength (tapered eval, tablebase hints)

**Phase 2 → 3 handoff**: Operational training pipeline producing improving models

---

## Open Questions → Resolutions

1. **How many games per iteration?**
   - **Answer**: 100 games (tune if training signal weak/noisy)

2. **Temperature schedule values?**
   - **Answer**: {0: 1.0, 10: 0.5, 20: 0.1} (AlphaZero standard)

3. **Loss weighting?**
   - **Answer**: Equal (1:1), monitor and adjust if needed

4. **Replacement threshold?**
   - **Answer**: 55% win rate with Wilson score >50% (statistical significance)

5. **MCTS simulations with NN?**
   - **Answer**: 40 (vs 100 for random, due to inference cost)

---

## Notes

- **Philosophy**: All decisions adhere to AGENTS.md (simple, explicit, fail fast)
- **Determinism**: Seed RNG for reproducible training (`random.seed()`, `torch.manual_seed()`)
- **Traceability**: Log iteration metrics (games, loss, win rate) to CSV
- **Time budget**: 7 days, reserve Day 7 for integration issues
- **Fail fast clause**: If network doesn't beat random by Day 5, abort and enhance MCTS

**Estimated time**: ~60 hours (8-9 hours/day × 7 days)
