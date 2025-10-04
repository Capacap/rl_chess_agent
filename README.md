# Chess RL Agent

AlphaZero-style reinforcement learning agent for chess using MCTS and neural networks.

## Quick Start

```bash
# Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train a model
python train.py --iterations 5 --games-per-iter 25 --simulations 20

# Training logs saved to: checkpoints/<timestamp>/training.log
```

## Training Parameters

- `--iterations`: Number of training iterations (default: 10)
- `--games-per-iter`: Self-play games per iteration (default: 100)
- `--simulations`: MCTS simulations per move (default: 40)
- `--batch-size`: Training batch size (default: 256)
- `--epochs`: Training epochs per iteration (default: 5)
- `--arena-games`: Arena evaluation games (default: 50)

## Expected Training Time

Sequential self-play (current implementation):
- **~3 min/game** for untrained network
- **~30-60 min/iteration** (10-25 games)
- **~5-10 hours** for 10 iterations

Reduced settings for faster iteration:
```bash
python train.py --iterations 5 --games-per-iter 25 --simulations 20
# Expected: ~2 hours for baseline model
```

## Project Structure

```
├── model/          # Neural network architecture
├── encoding/       # Board/move encoding
├── training/       # Training pipeline (self-play, MCTS, arena)
├── tests/          # Test suite
├── docs/           # Documentation
├── train.py        # CLI entrypoint
└── checkpoints/    # Saved models and logs
```

## Documentation

- [Project Overview](docs/project_description.md)
- [Roadmap](docs/roadmap.md)
- Phase 1: [Plan](docs/phase1_implementation_plan.md) | [Summary](docs/phase1_completion_summary.md)
- Phase 2: [Plan](docs/phase2_implementation_plan.md) | [Summary](docs/phase2_completion_summary.md)
