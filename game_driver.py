"""
game_driver.py
A simple driver to run a game between two agents and output the result.
"""

import chess
import chess.pgn
from random_agent import RandomAgent
from greedy_agent import GreedyAgent

def run_game(white_agent_class, black_agent_class, time_limit_per_move=2.0):
    """
    Runs a single game between two agent classes.

    Args:
    white_agent_class: The class for the White player.
    black_agent_class: The class for the Black player.
    time_limit_per_move (float): Max seconds per move.

    Returns:
    chess.Board: The final board state.
    chess.pgn.Game: The game in PGN format.
    """
    board = chess.Board()
    # Instantiate the agents
    white_agent = white_agent_class(board, chess.WHITE)
    black_agent = black_agent_class(board, chess.BLACK)

    # Create a PGN game to record moves
    game = chess.pgn.Game()
    game.headers["White"] = str(white_agent)
    game.headers["Black"] = str(black_agent)
    node = game

    # Game loop
    while not board.is_game_over():
        # Get the current active agent
        current_agent = white_agent if board.turn == chess.WHITE else black_agent
        # Update the agent's board state
        current_agent.set_board(board)

        # Get the move from the agent
        try:
            move = current_agent.make_move(board, time_limit_per_move)
        except Exception as e:
            print(f"Agent {current_agent} crashed on move {board.fullmove_number}! Defaulting to random move. Error: {e}")
            move = list(board.legal_moves)[0] # Default to first legal move on error

        # Validate the move (crucial for catching agent bugs)
        if move not in board.legal_moves:
            print(f"Agent {current_agent} made illegal move {move}. Defaulting to random move.")
            move = list(board.legal_moves)[0]

        # Make the move on the board and add it to the PGN
        board.push(move)
        node = node.add_variation(move)

        # Optional: Print the board to the console to watch
        # print(board)
        # print("------")

    # Set the game result in the PGN headers
    game.headers["Result"] = board.result()
    print(f"Game over. Result: {board.result()}. Outcome: {board.outcome()}")

    return board, game

if __name__ == "__main__":
    # Run a game between RandomAgent (White) and GreedyAgent (Black)
    final_board, game_pgn = run_game(RandomAgent, GreedyAgent)

    # Print the PGN to see the whole game
    print("\n" + "="*50)
    print("PGN Output:")
    print(game_pgn)

    # Export the PGN to a file to view in a GUI like Lichess.org
    with open("game_random_vs_greedy.pgn", "w") as pgn_file:
        pgn_file.write(str(game_pgn))
    print("PGN file saved as 'game_random_vs_greedy.pgn'. Import this on Lichess to replay!")