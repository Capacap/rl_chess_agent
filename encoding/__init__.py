"""
encoding package
Board and move encoding utilities for chess neural network.
"""

from encoding.state import encode_board
from encoding.move import encode_move, decode_move

__all__ = ['encode_board', 'encode_move', 'decode_move']
