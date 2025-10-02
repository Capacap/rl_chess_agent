"""
test_mcts.py
Test and benchmark the MCTS agent against baseline agents.
"""

import chess
import time
from mcts_agent import MCTSAgent
from random_agent import RandomAgent
from greedy_agent import GreedyAgent

# Test configuration constants
DEFAULT_TIME_LIMIT = 2.0  # seconds per move
MAX_GAME_MOVES = 30  # maximum moves before declaring draw
DEFAULT_NUM_GAMES = 5  # number of games per benchmark
MOVE_PROGRESS_INTERVAL = 10  # print progress every N moves


def play_game(white_agent_class, black_agent_class, time_limit=DEFAULT_TIME_LIMIT, verbose=False):
    """
    Play a single game between two agents.
    
    Returns:
        result: '1-0' (white wins), '0-1' (black wins), or '1/2-1/2' (draw)
        move_count: Number of moves played
    """
    board = chess.Board()
    white_agent = white_agent_class(board, chess.WHITE)
    black_agent = black_agent_class(board, chess.BLACK)
    
    move_count = 0
    move_times = []
    
    while not board.is_game_over():
        current_agent = white_agent if board.turn == chess.WHITE else black_agent
        current_agent.set_board(board)
        
        # Time the move
        start = time.time()
        try:
            move = current_agent.make_move(board, time_limit)
        except Exception as e:
            print(f"Error: {current_agent} crashed: {e}")
            move = list(board.legal_moves)[0]
        elapsed = time.time() - start
        move_times.append(elapsed)
        
        # Check time limit violation
        if elapsed > time_limit:
            print(f"Warning: {current_agent} exceeded time limit: {elapsed:.2f}s")
        
        # Validate and make move
        if move not in board.legal_moves:
            print(f"Illegal move by {current_agent}: {move}")
            move = list(board.legal_moves)[0]
        
        board.push(move)
        move_count += 1
        
        if verbose and move_count % MOVE_PROGRESS_INTERVAL == 0:
            print(f"Move {move_count}, Last move time: {elapsed:.2f}s")

        # Safety: Stop very long games
        if move_count > MAX_GAME_MOVES:
            print(f"Game exceeded {MAX_GAME_MOVES} moves, declaring draw")
            break
    
    result = board.result()
    avg_time = sum(move_times) / len(move_times) if move_times else 0
    max_time = max(move_times) if move_times else 0
    
    if verbose:
        print(f"Game ended: {result}")
        print(f"Total moves: {move_count}")
        print(f"Avg move time: {avg_time:.3f}s, Max: {max_time:.3f}s")
    
    return result, move_count, avg_time, max_time


def benchmark_agent(test_agent_class, opponent_class, num_games=DEFAULT_NUM_GAMES,
                    test_plays_white=True, time_limit=DEFAULT_TIME_LIMIT):
    """
    Benchmark an agent against an opponent over multiple games.
    """
    print(f"\n{'='*60}")
    print(f"Testing {test_agent_class.__name__} vs {opponent_class.__name__}")
    print(f"Games: {num_games}, Time limit: {time_limit}s")
    print(f"{test_agent_class.__name__} plays: {'White' if test_plays_white else 'Black'}")
    print(f"{'='*60}\n")
    
    wins = 0
    losses = 0
    draws = 0
    total_moves = 0
    all_times = []
    
    for game_num in range(num_games):
        print(f"Game {game_num + 1}/{num_games}...", end=" ")
        
        if test_plays_white:
            result, moves, avg_time, max_time = play_game(
                test_agent_class, opponent_class, time_limit, verbose=False
            )
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1
        else:
            result, moves, avg_time, max_time = play_game(
                opponent_class, test_agent_class, time_limit, verbose=False
            )
            if result == "0-1":
                wins += 1
            elif result == "1-0":
                losses += 1
            else:
                draws += 1
        
        total_moves += moves
        all_times.append(avg_time)
        
        print(f"{result} ({moves} moves, avg {avg_time:.2f}s, max {max_time:.2f}s)")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {test_agent_class.__name__} vs {opponent_class.__name__}")
    print(f"{'='*60}")
    print(f"Wins:   {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"Losses: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    print(f"Draws:  {draws}/{num_games} ({draws/num_games*100:.1f}%)")
    print(f"Score:  {(wins + 0.5*draws)/num_games:.2f} / 1.0")
    print(f"\nAvg game length: {total_moves/num_games:.1f} moves")
    print(f"Avg move time: {sum(all_times)/len(all_times):.3f}s")
    print(f"{'='*60}\n")
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'score': (wins + 0.5*draws)/num_games
    }


def quick_test():
    """Quick sanity check that the agent works."""
    print("Running quick sanity check...")
    board = chess.Board()
    agent = MCTSAgent(board, chess.WHITE)
    
    # Test that it can make a move
    move = agent.make_move(board, DEFAULT_TIME_LIMIT)
    assert move in board.legal_moves, "Agent returned illegal move!"
    print(f"✓ Agent returned legal move: {move}")

    # Test from a mid-game position
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    agent.set_board(board)
    move = agent.make_move(board, DEFAULT_TIME_LIMIT)
    assert move in board.legal_moves, "Agent returned illegal move in mid-game!"
    print(f"✓ Agent works in mid-game position: {move}")
    
    print("✓ All sanity checks passed!\n")


if __name__ == "__main__":
    # Quick sanity check
    quick_test()
    
    # Benchmark against Random
    print("\n" + "="*60)
    print("BENCHMARK 1: MCTS vs RandomAgent")
    print("="*60)
    benchmark_agent(MCTSAgent, RandomAgent, test_plays_white=True)
    benchmark_agent(MCTSAgent, RandomAgent, test_plays_white=False)

    # Benchmark against Greedy
    print("\n" + "="*60)
    print("BENCHMARK 2: MCTS vs GreedyAgent")
    print("="*60)
    benchmark_agent(MCTSAgent, GreedyAgent, test_plays_white=True)
    benchmark_agent(MCTSAgent, GreedyAgent, test_plays_white=False)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
