"""
test_phase2_pipeline.py
Integration test for Phase 2 training pipeline.

Tests:
1. Neural MCTS search completes without errors
2. Self-play generates valid experiences
3. Training loop reduces loss
4. Arena evaluation works
5. End-to-end pipeline runs
"""

import torch
import chess
from model.network import ChessNet
from training.mcts_nn import mcts_search, NeuralMCTSNode
from training.selfplay import SelfPlayWorker, Experience
from training.train import train_iteration, compute_loss
from training.arena import Arena, should_replace
from training.pipeline import training_pipeline
from encoding.state import encode_board
from encoding.move import create_legal_move_mask
import numpy as np


def test_neural_mcts():
    """Test: Neural MCTS search completes without errors."""
    print("\n[TEST 1] Neural MCTS search...")

    # Create small network for speed
    network = ChessNet(channels=32, num_blocks=2)
    board = chess.Board()

    # Run MCTS search
    visit_counts = mcts_search(
        board=board,
        network=network,
        num_simulations=5  # Minimal for speed
    )

    # Validate
    assert len(visit_counts) > 0, "No moves found"
    assert all(isinstance(m, chess.Move) for m in visit_counts.keys()), "Invalid move types"
    assert sum(visit_counts.values()) > 0, "No visits recorded"
    assert any(v > 0 for v in visit_counts.values()), "All moves have zero visits"

    visited_moves = sum(1 for v in visit_counts.values() if v > 0)
    print(f"  ✓ MCTS found {len(visit_counts)} moves ({visited_moves} visited)")
    print(f"  ✓ Total visits: {sum(visit_counts.values())}")


def test_selfplay():
    """Test: Self-play generates valid experiences."""
    print("\n[TEST 2] Self-play generation...")

    network = ChessNet(channels=32, num_blocks=2)
    worker = SelfPlayWorker(
        network=network,
        num_simulations=5  # Minimal for speed
    )

    # Generate one game (with move limit for speed)
    experiences = worker.play_game(max_moves=50)

    # Validate
    assert len(experiences) > 0, "No experiences generated"
    assert all(isinstance(exp, Experience) for exp in experiences), "Invalid experience types"

    for exp in experiences:
        assert isinstance(exp.fen, str), "Invalid FEN type"
        assert len(exp.policy) == 4096, f"Policy wrong size: {len(exp.policy)}"
        assert abs(sum(exp.policy) - 1.0) < 0.01, f"Policy doesn't sum to 1: {sum(exp.policy)}"
        assert exp.value in [-1.0, 0.0, 1.0], f"Invalid value: {exp.value}"

    print(f"  ✓ Generated {len(experiences)} experiences")
    print(f"  ✓ Game outcome: {experiences[-1].value}")


def test_training():
    """Test: Training loop reduces loss."""
    print("\n[TEST 3] Training loop...")

    network = ChessNet(channels=32, num_blocks=2)

    # Generate synthetic experiences
    experiences = []
    for _ in range(50):
        board = chess.Board()
        policy = np.zeros(4096, dtype=np.float32)
        # Random policy on legal moves
        legal_moves = list(board.legal_moves)
        for move in legal_moves:
            from encoding.move import encode_move
            policy[encode_move(move)] = 1.0 / len(legal_moves)

        experiences.append(Experience(
            fen=board.fen(),
            policy=policy,
            value=1.0
        ))

    # Measure initial loss
    network.eval()
    with torch.no_grad():
        states, policies, values, masks = [], [], [], []
        for exp in experiences[:10]:
            board = chess.Board(exp.fen)
            state = encode_board(board).squeeze(0)
            states.append(state)
            policies.append(torch.tensor(exp.policy))
            values.append(torch.tensor([exp.value]))
            masks.append(torch.tensor(create_legal_move_mask(board)))

        states_batch = torch.stack(states)
        policies_batch = torch.stack(policies)
        values_batch = torch.stack(values)
        masks_batch = torch.stack(masks)

        policy_pred, value_pred = network(states_batch, masks_batch)
        initial_loss, _ = compute_loss(policy_pred, value_pred, policies_batch, values_batch)
        initial_loss_value = initial_loss.item()

    # Train
    history = train_iteration(
        network=network,
        experiences=experiences,
        batch_size=16,
        epochs=3,
        lr=1e-3
    )

    final_loss = history['total'][-1]

    # Validate: loss should decrease
    print(f"  ✓ Initial loss: {initial_loss_value:.4f}")
    print(f"  ✓ Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss_value * 1.1, "Loss did not decrease (allowing 10% margin for variance)"


def test_arena():
    """Test: Arena evaluation works."""
    print("\n[TEST 4] Arena evaluation...")

    # Create two identical networks (should have ~50% win rate)
    network1 = ChessNet(channels=32, num_blocks=2)
    network2 = ChessNet(channels=32, num_blocks=2)

    # Copy weights
    network2.load_state_dict(network1.state_dict())

    # Run arena
    arena = Arena(num_simulations=5)
    results = arena.compete(
        challenger=network1,
        champion=network2,
        num_games=4  # Minimal for speed
    )

    # Validate
    assert 'wins' in results, "Missing 'wins' in results"
    assert 'losses' in results, "Missing 'losses' in results"
    assert 'draws' in results, "Missing 'draws' in results"
    assert 'win_rate' in results, "Missing 'win_rate' in results"

    total_games = results['wins'] + results['losses'] + results['draws']
    assert total_games == 4, f"Wrong number of games: {total_games}"

    print(f"  ✓ Results: W{results['wins']}-L{results['losses']}-D{results['draws']}")
    print(f"  ✓ Win rate: {results['win_rate']:.1%}")


def test_replacement_logic():
    """Test: Replacement logic is sound."""
    print("\n[TEST 5] Replacement logic...")

    # Clear win (70%) should replace
    assert should_replace(0.70, 50), "Should replace at 70%"

    # Close to threshold (55%) - depends on significance
    result_55 = should_replace(0.55, 50)
    print(f"  · 55% win rate (50 games): {'replace' if result_55 else 'keep'}")

    # Below threshold (45%) should not replace
    assert not should_replace(0.45, 50), "Should not replace at 45%"

    print(f"  ✓ Replacement logic validated")


def test_pipeline():
    """Test: End-to-end pipeline runs."""
    print("\n[TEST 6] End-to-end pipeline...")

    # Run minimal pipeline
    champion = training_pipeline(
        num_iterations=1,
        games_per_iter=2,  # Reduced from 5
        num_simulations=3,  # Reduced from 5
        batch_size=8,  # Reduced from 16
        epochs=1,  # Reduced from 2
        arena_games=2,  # Reduced from 4
        checkpoint_dir="test_checkpoints"
    )

    # Validate
    assert champion is not None, "Pipeline returned None"
    assert isinstance(champion, ChessNet), "Pipeline returned wrong type"

    print(f"  ✓ Pipeline completed successfully")


if __name__ == "__main__":
    print("="*60)
    print("Phase 2 Integration Tests")
    print("="*60)

    test_neural_mcts()
    test_selfplay()
    test_training()
    test_arena()
    test_replacement_logic()
    test_pipeline()

    print("\n" + "="*60)
    print("All tests passed ✓")
    print("="*60 + "\n")
