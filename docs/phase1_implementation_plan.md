# Phase 1 Implementation Plan: Neural Network Foundation

**Target:** Week 1 (Oct 2-9, 2025)
**Status:** Ready for implementation
**Philosophy:** Simplest solution that works, fail fast and loud

---

## Problem Analysis

### Current State
- Functional MCTS baseline (`mcts_agent.py`) using random rollouts
- CPU constraint: <2s per move, <2GB RAM
- Action space: 4096 possible moves (64×64 from-to encoding)
- Need NN to replace random rollouts with learned evaluation

### Key Challenges Identified

1. **Move encoding complexity**: 4096-dim action space includes illegal moves
   - **Solution**: Legal move masking via `board.legal_moves` at inference
   - Flat encoding simpler than structured (from, to, promotion) tuples

2. **State representation depth**: History planes vs single position
   - **Decision**: Start with single position (8×8×13 planes)
   - Rationale: Threefold repetition handled by `chess.Board`, history adds complexity
   - Can add later if needed (YAGNI)

3. **Network size vs inference speed**: Depth/width tradeoff for CPU
   - **Constraint**: ~50ms inference budget (2s move time / ~40 MCTS sims)
   - **Target**: 3-4 residual blocks, 64-128 channels
   - Profile early, tune if needed

4. **Training data format**: Experience tuple design
   - **Core tuple**: `(board_fen: str, mcts_policy: np.ndarray[4096], outcome: float)`
   - FEN string more robust than tensor serialization
   - Policy as visit counts normalized to probabilities

---

## Architecture Decisions

### 1. State Encoder (`encoding/state.py`)

**Input**: `chess.Board` → **Output**: `torch.Tensor[1, 13, 8, 8]`

**Plane layout** (13 channels):
```
0-5:   White pieces (P, N, B, R, Q, K)
6-11:  Black pieces (P, N, B, R, Q, K)
12:    Metadata (binary features tiled 8×8)
       - bit 0-3: castling rights (K, Q, k, q)
       - bit 4: side to move (1=white, 0=black)
       - bit 5-7: reserved (en passant if needed)
```

**Implementation**:
```python
def encode_board(board: chess.Board) -> torch.Tensor:
    """
    Encode board state as 13×8×8 tensor.

    Returns:
        Tensor shape [1, 13, 8, 8], dtype float32

    Invariant: Output is deterministic for same board state
    """
```

**Rationale**:
- Binary piece presence (0/1) over piece types avoids sparse encoding
- Single metadata plane sufficient for castling/turn (en passant rare, can ignore)
- No normalization needed (values already 0/1)

---

### 2. Move Encoder (`encoding/move.py`)

**Input**: `chess.Move` → **Output**: `int` (action index)
**Inverse**: `int` → `chess.Move` (with validation)

**Encoding scheme**:
```
action_index = from_square * 64 + to_square
from_square, to_square ∈ [0, 63]
```

**Underpromotion handling**:
- Promotions to queen: Standard encoding
- Underpromotions (N, B, R): Add offset `4096 + promotion_offset`
- Total action space: 4096 + 3*64 = 4288 (reserve 4096 for simplicity, mask rest)

**Implementation**:
```python
def encode_move(move: chess.Move) -> int:
    """Map chess.Move to action index [0, 4095]."""

def decode_move(action: int, board: chess.Board) -> chess.Move | None:
    """
    Map action index to chess.Move. Returns None if illegal.

    Args:
        action: Index in [0, 4095]
        board: Board state for legality check

    Returns:
        chess.Move if legal, None otherwise
    """
```

**Rationale**:
- Flat encoding simpler than structured tuples (from, to, promo)
- Validation at decode time (parse, don't validate)
- Underpromotions rare, handle separately if needed

---

### 3. Network Architecture (`model/network.py`)

**Design**: Residual CNN (inspired by AlphaZero but scaled down)

**Structure**:
```
Input: [B, 13, 8, 8]
  ↓
Conv 3×3, 64 channels, stride=1, padding=1
BatchNorm + ReLU
  ↓
ResBlock × 4  (conv → bn → relu → conv → bn → skip connection → relu)
  ↓
         ┌─────────────────────┐
         ↓                     ↓
   Policy Head           Value Head
   ↓                     ↓
Conv 1×1, 32 ch         Conv 1×1, 32 ch
BN + ReLU               BN + ReLU
Flatten                 GlobalAvgPool
FC → 4096               FC → 1
Softmax (masked)        Tanh
```

**ResBlock implementation**:
```python
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)
```

**Policy head with masking**:
```python
def forward_policy(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Features [B, C, 8, 8]
        legal_mask: Boolean [B, 4096], True=legal

    Returns:
        Policy logits [B, 4096], softmax over legal moves only
    """
    logits = self.policy_layers(x)  # [B, 4096]
    logits = logits.masked_fill(~legal_mask, float('-inf'))
    return F.softmax(logits, dim=1)
```

**Parameter count estimate**:
```
Conv initial: 13×64×3×3 ≈ 7K
ResBlocks (4): 64×64×3×3×2 ≈ 295K
Policy head: 64×32×8×8 + 32×8×8×4096 ≈ 8M
Value head: 64×32×8×8 + 32×64 ≈ 131K
Total: ~8.4M parameters ≈ 34MB (fp32)
```

**Inference time target**: <50ms on CPU

**Rationale**:
- 4 ResBlocks balances capacity vs speed (vs AlphaZero's 20)
- Dual heads share feature extraction (efficient)
- BatchNorm critical for stable training
- Masked softmax prevents illegal move selection

---

### 4. Data Pipeline (`training/dataset.py`)

**Experience storage format** (JSON Lines):
```json
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
 "policy": [0.0, 0.15, ..., 0.0],  # 4096 floats, sum=1.0
 "value": 1.0}  # {-1.0, 0.0, 1.0}
```

**Dataset class**:
```python
class ChessDataset(torch.utils.data.Dataset):
    def __init__(self, experience_file: str):
        """Load JSONL file of experiences."""
        self.data = []
        with open(experience_file) as f:
            for line in f:
                self.data.append(json.loads(line))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            state: [13, 8, 8] encoded board
            policy_target: [4096] MCTS visit distribution
            value_target: scalar in {-1, 0, 1}
        """
        exp = self.data[idx]
        board = chess.Board(exp['fen'])
        state = encode_board(board)
        policy = torch.tensor(exp['policy'], dtype=torch.float32)
        value = torch.tensor(exp['value'], dtype=torch.float32)
        return state, policy, value
```

**Rationale**:
- JSONL allows streaming/appending (vs pickle requires reload)
- FEN strings human-readable for debugging
- Lazy loading if dataset grows large (can optimize later)

---

## Implementation Order

### Day 1-2: Encoders (Critical path)
1. `encoding/__init__.py` (empty, marks package)
2. `encoding/state.py`
   - `encode_board(board) -> Tensor`
   - Test: Determinism, shape, value ranges
3. `encoding/move.py`
   - `encode_move(move) -> int`
   - `decode_move(action, board) -> Move | None`
   - Test: Round-trip encoding, illegal move handling

**Acceptance criteria**:
- Encoder tests pass (100% coverage on edge cases)
- Encodes starting position, mid-game, endgame correctly
- Handles promotions, castling, en passant

---

### Day 3-4: Network Architecture
1. `model/__init__.py`
2. `model/network.py`
   - `ChessNet(nn.Module)` class
   - `forward(state, legal_mask) -> (policy, value)`
   - Profile inference time on CPU

**Test cases**:
```python
# Dummy forward pass
net = ChessNet(channels=64, blocks=4)
state = torch.randn(1, 13, 8, 8)
legal_mask = torch.ones(1, 4096, dtype=torch.bool)
policy, value = net(state, legal_mask)

assert policy.shape == (1, 4096)
assert value.shape == (1, 1)
assert torch.allclose(policy.sum(dim=1), torch.tensor([1.0]))
assert -1 <= value.item() <= 1
```

**Profiling**:
```python
import time
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        policy, value = net(state, legal_mask)
elapsed = (time.perf_counter() - start) / 100
print(f"Avg inference: {elapsed*1000:.1f}ms")
```

**Acceptance criteria**:
- Forward pass outputs correct shapes
- Inference <50ms on CPU
- No NaN/Inf in outputs

---

### Day 5: Data Pipeline
1. `training/__init__.py`
2. `training/dataset.py`
   - `ChessDataset` class
   - `save_experience(experiences, file)` helper
   - `load_experience(file) -> Dataset` helper

**Test with synthetic data**:
```python
# Generate dummy experiences
experiences = [
    {'fen': chess.Board().fen(),
     'policy': [1.0/4096] * 4096,  # Uniform
     'value': 0.0}
    for _ in range(100)
]
save_experience(experiences, 'test.jsonl')

dataset = ChessDataset('test.jsonl')
loader = DataLoader(dataset, batch_size=32)
batch = next(iter(loader))
assert batch[0].shape == (32, 13, 8, 8)
```

**Acceptance criteria**:
- Can save/load experiences
- DataLoader batching works
- Handles empty/corrupted files gracefully

---

### Day 6-7: Integration Testing
1. `test_nn_integration.py`
   - End-to-end: Board → Encode → Network → Decode → Move
   - Verify legal move selection (mask works)
   - Compare NN policy vs random on sample positions

**Test script**:
```python
def test_nn_move_selection():
    board = chess.Board()
    net = ChessNet.load_from_checkpoint('untrained.pt')  # Random weights

    # Encode state
    state = encode_board(board).unsqueeze(0)

    # Get legal moves mask
    legal_moves = list(board.legal_moves)
    legal_mask = torch.zeros(1, 4096, dtype=torch.bool)
    for move in legal_moves:
        legal_mask[0, encode_move(move)] = True

    # Get policy
    policy, value = net(state, legal_mask)

    # Sample move
    action = torch.multinomial(policy, 1).item()
    move = decode_move(action, board)

    assert move is not None
    assert move in board.legal_moves
    print(f"Selected move: {move}, Value: {value.item():.3f}")
```

**Acceptance criteria**:
- Untrained network produces legal moves
- No crashes on edge cases (checkmate, stalemate)
- Output shapes/types match expectations

---

## Validation Strategy

### Unit Tests (per module)
- `tests/test_encoding.py`: State/move encoding round-trips
- `tests/test_network.py`: Architecture correctness, gradient flow
- `tests/test_dataset.py`: Data loading, batching

### Integration Test
- `tests/test_phase1_deliverable.py`: Full pipeline smoke test

### Manual Verification
```bash
# Quick sanity check
python -c "
from encoding.state import encode_board
from model.network import ChessNet
import chess

board = chess.Board()
state = encode_board(board)
net = ChessNet()
policy, value = net(state.unsqueeze(0))
print(f'Policy shape: {policy.shape}, Value: {value.item():.3f}')
"
```

---

## File Structure (After Phase 1)

```
rl_chess_agent/
├── encoding/
│   ├── __init__.py
│   ├── state.py       # Board → Tensor
│   └── move.py        # Move ↔ Action index
├── model/
│   ├── __init__.py
│   └── network.py     # ChessNet architecture
├── training/
│   ├── __init__.py
│   └── dataset.py     # Experience replay buffer
├── tests/
│   ├── test_encoding.py
│   ├── test_network.py
│   ├── test_dataset.py
│   └── test_phase1_deliverable.py
├── mcts_agent.py      # Existing (baseline)
├── agent_interface.py # Existing (contract)
└── requirements.txt   # Existing
```

---

## Critical Decisions & Rationale

### 1. Single position vs history planes
- **Choice**: Single position (13 planes)
- **Why**: `chess.Board` tracks repetitions, YAGNI for MVP
- **Revisit if**: Network can't learn positional understanding

### 2. Flat move encoding vs structured
- **Choice**: Flat 4096 action space
- **Why**: Simpler indexing, masking easier
- **Tradeoff**: Wastes capacity on illegal moves (acceptable for this scale)

### 3. ResNet vs plain CNN
- **Choice**: ResNet (4 blocks)
- **Why**: Proven for board games, gradient flow
- **Alternative considered**: Plain CNN (rejected: vanishing gradients)

### 4. JSON vs pickle for experiences
- **Choice**: JSONL
- **Why**: Human-readable, streamable, debuggable
- **Tradeoff**: Slower than pickle (premature optimization)

### 5. Network size (64 channels, 4 blocks)
- **Choice**: Conservative sizing
- **Why**: Must fit CPU inference budget
- **Tuning**: Profile first, then scale up if time allows

---

## Risk Mitigation

### Risk: NN inference too slow for MCTS
- **Mitigation**: Profile on Day 4, reduce blocks/channels if needed
- **Fallback**: Reduce MCTS simulations (quality vs quantity)

### Risk: Move encoding bugs cause illegal moves
- **Mitigation**: Extensive unit tests, fuzzing with random positions
- **Detection**: Integration test samples 1000 random positions

### Risk: Network doesn't learn (Phase 2 concern, but plan now)
- **Early signal**: Monitor policy entropy, value MSE on test set
- **Fallback**: Revert to enhanced MCTS per roadmap.md:170

---

## Open Questions → Resolutions

1. **How many history planes?**
   - **Answer**: Zero (start simple, add if needed)

2. **Move encoding for underpromotions?**
   - **Answer**: Extend action space to 4288, mask unused indices

3. **Network depth vs width?**
   - **Answer**: 4 blocks × 64 channels (profile-driven tuning later)

4. **Training data format?**
   - **Answer**: JSONL with FEN strings (simplicity over performance)

---

## Success Metrics (Phase 1 Exit Criteria)

1. **Functional encoder**: Board ↔ Tensor bijection for valid states
2. **Functional network**: Forward pass <50ms, outputs valid distributions
3. **Functional dataset**: Can load/batch synthetic experiences
4. **Integration test passes**: Untrained NN produces legal moves
5. **Code quality**: Type-checked, documented, tested

**Deliverable**: Import `ChessNet`, call `forward()` on encoded board, get policy/value.

---

## Next Steps (Phase 2 Preview)

- MCTS with NN guidance (replace random rollouts)
- Self-play loop (generate training data)
- Training loop (policy + value loss)
- Checkpoint/evaluation (arena testing)

**Phase 1 → 2 handoff**: Validated NN ready for integration into MCTS.

---

## Notes

- **Philosophy compliance**: All decisions follow AGENTS.md (simple, explicit, traceable)
- **Fail fast**: Assertions in encoders, early shape checks
- **Determinism**: Seed RNG in tests (`torch.manual_seed(42)`)
- **No premature optimization**: Profile before scaling network
- **Type safety**: All public functions have explicit signatures

**Time budget**: 7 days, reserve 1 day for bugs/integration issues.
