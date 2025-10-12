#!/usr/bin/env python3
"""
Debug reward computation to understand why values are clustering near 0.
"""

import chess
from training.rewards import compute_position_value, compute_material_balance, compute_pawn_advancement, compute_piece_activity

def test_positions():
    print("=" * 70)
    print("Testing Reward Components on Sample Positions")
    print("=" * 70)

    # Test 1: Starting position
    board = chess.Board()
    print("\n1. Starting position (should be ~0.0)")
    print(f"   Material:  {compute_material_balance(board):+.3f}")
    print(f"   Pawns:     {compute_pawn_advancement(board):+.3f}")
    print(f"   Activity:  {compute_piece_activity(board):+.3f}")
    print(f"   Overall:   {compute_position_value(board):+.3f}")

    # Test 2: White up a pawn
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPP1/RNBQKBNR w KQkq - 0 1")
    print("\n2. White missing a pawn (should be negative)")
    print(f"   Material:  {compute_material_balance(board):+.3f}")
    print(f"   Overall:   {compute_position_value(board):+.3f}")

    # Test 3: White up a queen
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRQ w KQkq - 0 1")
    print("\n3. White extra queen (should be strongly positive)")
    print(f"   Material:  {compute_material_balance(board):+.3f}")
    print(f"   Overall:   {compute_position_value(board):+.3f}")

    # Test 4: Pawn near promotion
    board = chess.Board("8/P7/8/8/8/8/8/8 w - - 0 1")
    print("\n4. White pawn on 7th rank (should be positive)")
    print(f"   Material:  {compute_material_balance(board):+.3f}")
    print(f"   Pawns:     {compute_pawn_advancement(board):+.3f}")
    print(f"   Overall:   {compute_position_value(board):+.3f}")

    # Test 5: After e4 e5
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
    print("\n5. After 1.e4 e5 (should be near 0)")
    print(f"   Material:  {compute_material_balance(board):+.3f}")
    print(f"   Pawns:     {compute_pawn_advancement(board):+.3f}")
    print(f"   Activity:  {compute_piece_activity(board):+.3f}")
    print(f"   Overall:   {compute_position_value(board):+.3f}")

    # Test 6: Scholar's mate
    board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    print("\n6. Scholar's mate (white winning, should be strongly positive)")
    print(f"   Material:  {compute_material_balance(board):+.3f}")
    print(f"   Overall:   {compute_position_value(board):+.3f}")

    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    print("If values are too small (all < 0.1), rewards won't provide gradient.")
    print("Expected: Material swings should produce values in 0.2-0.6 range.")
    print("=" * 70)

if __name__ == "__main__":
    test_positions()
