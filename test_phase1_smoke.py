#!/usr/bin/env python3
"""
Quick smoke test for Phase 1 implementation.
Tests basic functionality without pytest infrastructure.
"""

import chess
import torch
from encoding.state import encode_board
from encoding.move import encode_move, decode_move, create_legal_move_mask
from model.network import ChessNet, count_parameters


def test_encoding():
    """Test board and move encoding."""
    print("Testing board encoding...")
    board = chess.Board()

    # Test board encoding
    state = encode_board(board)
    assert state.shape == (1, 13, 8, 8), f"Expected shape (1, 13, 8, 8), got {state.shape}"
    assert state.dtype == torch.float32, f"Expected dtype float32, got {state.dtype}"
    print("✓ Board encoding shape and dtype correct")

    # Test determinism
    state2 = encode_board(board)
    assert torch.equal(state, state2), "Board encoding not deterministic"
    print("✓ Board encoding is deterministic")

    # Test move encoding
    move = chess.Move.from_uci("e2e4")
    action = encode_move(move)
    assert 0 <= action < 4096, f"Action {action} out of range"
    print(f"✓ Move e2e4 encoded as action {action}")

    # Test move decoding
    decoded = decode_move(action, board)
    assert decoded == move, f"Expected {move}, got {decoded}"
    print("✓ Move decoding works")

    # Test legal move mask
    mask = create_legal_move_mask(board)
    assert len(mask) == 4096, f"Expected mask length 4096, got {len(mask)}"
    num_legal = sum(mask)
    assert num_legal == len(list(board.legal_moves)), f"Mask has {num_legal} moves, board has {len(list(board.legal_moves))}"
    print(f"✓ Legal move mask correct ({num_legal} legal moves)")


def test_network():
    """Test network architecture."""
    print("\nTesting network architecture...")

    # Create network
    net = ChessNet(channels=64, num_blocks=4)
    print("✓ Network created")

    # Count parameters
    params = count_parameters(net)
    print(f"✓ Network has {params:,} parameters ({params/1e6:.1f}M)")

    # Test forward pass
    board = chess.Board()
    state = encode_board(board)
    legal_mask = torch.tensor([create_legal_move_mask(board)])

    net.eval()
    with torch.no_grad():
        policy, value = net(state, legal_mask)

    assert policy.shape == (1, 4096), f"Expected policy shape (1, 4096), got {policy.shape}"
    assert value.shape == (1, 1), f"Expected value shape (1, 1), got {value.shape}"
    print("✓ Forward pass output shapes correct")

    # Check policy sums to 1
    policy_sum = policy.sum().item()
    assert 0.99 <= policy_sum <= 1.01, f"Policy sum {policy_sum} not close to 1.0"
    print(f"✓ Policy sums to {policy_sum:.6f}")

    # Check value range
    assert -1.0 <= value.item() <= 1.0, f"Value {value.item()} outside [-1, 1]"
    print(f"✓ Value {value.item():.3f} in correct range")

    # Check illegal moves have zero probability
    illegal_actions = [i for i, legal in enumerate(legal_mask[0]) if not legal]
    illegal_probs = policy[0, illegal_actions]
    assert (illegal_probs == 0).all(), "Illegal moves have non-zero probability"
    print("✓ Illegal moves have zero probability")


def test_integration():
    """Test full pipeline."""
    print("\nTesting full integration...")

    board = chess.Board()
    net = ChessNet(channels=64, num_blocks=4)
    net.eval()

    # Encode board
    state = encode_board(board)

    # Create legal move mask
    legal_mask = torch.tensor([create_legal_move_mask(board)])

    # Get policy
    with torch.no_grad():
        policy, value = net(state, legal_mask)

    # Sample a move
    action = torch.multinomial(policy, 1).item()
    selected_move = decode_move(action, board)

    assert selected_move is not None, "Selected move is None"
    assert selected_move in board.legal_moves, f"Selected move {selected_move} is illegal"
    print(f"✓ Network selected legal move: {selected_move}")

    # Test on different position
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")

    state = encode_board(board)
    legal_mask = torch.tensor([create_legal_move_mask(board)])

    with torch.no_grad():
        policy, value = net(state, legal_mask)

    action = torch.multinomial(policy, 1).item()
    selected_move = decode_move(action, board)

    assert selected_move in board.legal_moves, f"Selected move {selected_move} is illegal in mid-game"
    print(f"✓ Network works in mid-game position: {selected_move}")


def test_inference_speed():
    """Test inference speed requirement."""
    print("\nTesting inference speed...")
    import time

    net = ChessNet(channels=64, num_blocks=4)
    net.eval()

    board = chess.Board()
    state = encode_board(board)
    legal_mask = torch.tensor([create_legal_move_mask(board)])

    # Warmup
    with torch.no_grad():
        _ = net(state, legal_mask)

    # Measure
    times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            _ = net(state, legal_mask)
        times.append(time.perf_counter() - start)

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time*1000:.1f}ms")

    if avg_time < 0.050:
        print("✓ Inference time <50ms (meets requirement)")
    else:
        print(f"⚠ Inference time {avg_time*1000:.1f}ms exceeds 50ms target")


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 Smoke Test")
    print("=" * 60)

    test_encoding()
    test_network()
    test_integration()
    test_inference_speed()

    print("\n" + "=" * 60)
    print("✓ All smoke tests passed!")
    print("=" * 60)
