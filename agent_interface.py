"""
agent_interface.py
Provided by Andreas.
DO NOT MODIFY THIS CLASS. Your agent must inherit from it and implement the make_move method.
"""

import chess
import time
from abc import ABC, abstractmethod

class Agent(ABC):
    """
    Abstract base class for all chess agents in the tournament.
    """

    def __init__(self, board: chess.Board, color: chess.Color):
        """
        Initialize the agent with the starting board and its color.
        This is called once at the beginning of a game.

        Args:
        board (chess.Board): The initial game board state.
        color (chess.Color): The color this agent is playing (chess.WHITE or chess.BLACK).
        """
        self.board = board
        self.color = color

    @abstractmethod
    def make_move(self, board: chess.Board, time_limit: float) -> chess.Move:
        """
        The tournament driver calls this method to get the agent's move.
        This is the main method that must be implemented by the student.

        Args:
        board (chess.Board): The current state of the game board.
        time_limit (float): The maximum time (in seconds) the agent has to choose a move.

        Returns:
        chess.Move: The move chosen by the agent.

        Raises:
        NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError("The 'make_move' method must be implemented by the student's agent.")

    def set_board(self, board: chess.Board) -> None:
        """
        Update the agent's internal board state. Useful if you cache the board.
        The tournament driver may call this to ensure the agent's board is synced.

        Args:
        board (chess.Board): The new, current board state.
        """
        self.board = board

    def __str__(self):
        """Returns a string identifier for the agent, using the class name."""
        return self.__class__.__name__

