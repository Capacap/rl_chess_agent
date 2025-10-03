"""Simplified Phase 2 test to identify issues"""

import torch
import chess
from model.network import ChessNet
from training.mcts_nn import mcts_search
from training.selfplay import SelfPlayWorker
from training.train import train_iteration, compute_loss
from training.arena import Arena
from encoding.state import encode_board
from encoding.move import create_legal_move_mask, encode_move
import numpy as np

print("="*60)
print("Phase 2 Simple Tests")
print("="*60)

# Test 1: MCTS
print("\n[1] Neural MCTS...")
network = ChessNet(channels=32, num_blocks=2)
board = chess.Board()
visits = mcts_search(board, network, num_simulations=5)
print(f"✓ Found {len(visits)} moves, {sum(visits.values())} total visits")

# Test 2: Self-play (1 game only)
print("\n[2] Self-play (1 game)...")
worker = SelfPlayWorker(network=network, num_simulations=3)
exps = worker.play_game()
print(f"✓ Generated {len(exps)} experiences")

# Test 3: Training
print("\n[3] Training...")
from training.selfplay import Experience

# Create minimal experiences
experiences = []
for _ in range(5):
    board = chess.Board()
    policy = np.zeros(4096, dtype=np.float32)
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        policy[encode_move(move)] = 1.0 / len(legal_moves)
    experiences.append(Experience(fen=board.fen(), policy=policy, value=1.0))

history = train_iteration(network=network, experiences=experiences,
                         batch_size=4, epochs=1, lr=1e-3)
print(f"✓ Final loss: {history['total'][-1]:.4f}")

# Test 4: Arena (minimal)
print("\n[4] Arena (2 games)...")
net1 = ChessNet(channels=32, num_blocks=2)
net2 = ChessNet(channels=32, num_blocks=2)
arena = Arena(num_simulations=3)
results = arena.compete(net1, net2, num_games=2)
print(f"✓ Results: W{results['wins']}-L{results['losses']}-D{results['draws']}")

print("\n" + "="*60)
print("All simple tests passed ✓")
print("="*60)
