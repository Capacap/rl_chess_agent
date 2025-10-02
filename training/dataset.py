"""
dataset.py
Experience replay dataset for chess neural network training.

Data format (JSONL):
{
  "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "policy": [0.0, 0.15, ..., 0.0],  # 4096 floats, sum=1.0
  "value": 1.0  # {-1.0, 0.0, 1.0}
}
"""

import json
import chess
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from pathlib import Path
from encoding.state import encode_board


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess training experiences.

    Loads experiences from JSONL file and provides batched access.
    """

    def __init__(self, experience_file: str):
        """
        Load experiences from JSONL file.

        Args:
            experience_file: Path to .jsonl file with experiences

        Raises:
            FileNotFoundError: If experience file doesn't exist
            ValueError: If file is empty or malformed
        """
        self.data: List[Dict] = []

        # Load experiences from file
        with open(experience_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    self.data.append(json.loads(line))

        # Validation
        if len(self.data) == 0:
            raise ValueError(f"No experiences loaded from {experience_file}")

    def __len__(self) -> int:
        """Return number of experiences."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get single experience by index.

        Args:
            idx: Index in [0, len(dataset))

        Returns:
            state: [13, 8, 8] encoded board
            policy_target: [4096] MCTS visit distribution
            value_target: scalar in {-1, 0, 1}
        """
        exp = self.data[idx]

        # Parse FEN to board
        board = chess.Board(exp['fen'])

        # Encode board state
        state = encode_board(board).squeeze(0)  # Remove batch dimension

        # Convert policy and value to tensors
        policy = torch.tensor(exp['policy'], dtype=torch.float32)
        value = torch.tensor(exp['value'], dtype=torch.float32)

        return state, policy, value


def save_experience(experiences: List[Dict], file_path: str, append: bool = True) -> None:
    """
    Save experiences to JSONL file.

    Args:
        experiences: List of experience dicts with keys: fen, policy, value
        file_path: Output file path
        append: If True, append to existing file; if False, overwrite

    Format:
        One JSON object per line (JSONL format)
    """
    mode = 'a' if append else 'w'

    with open(file_path, mode) as f:
        for exp in experiences:
            f.write(json.dumps(exp) + '\n')


def load_experience(file_path: str) -> ChessDataset:
    """
    Load experiences from JSONL file into Dataset.

    Args:
        file_path: Path to experience file

    Returns:
        ChessDataset instance
    """
    return ChessDataset(file_path)


def validate_experience(exp: Dict) -> bool:
    """
    Validate experience dict has correct format.

    Args:
        exp: Experience dictionary

    Returns:
        True if valid, False otherwise

    Checks:
        - Has keys: fen, policy, value
        - FEN is valid chess position
        - Policy is list of 4096 floats summing to ~1.0
        - Value is in {-1.0, 0.0, 1.0}
    """
    required_keys = {'fen', 'policy', 'value'}
    if not all(k in exp for k in required_keys):
        return False

    # Validate FEN string
    try:
        chess.Board(exp['fen'])
    except:
        return False

    # Validate policy length and sum
    if not isinstance(exp['policy'], list) or len(exp['policy']) != 4096:
        return False

    policy_sum = sum(exp['policy'])
    if not (0.99 <= policy_sum <= 1.01):  # Allow small floating point error
        return False

    # Validate value range
    if not isinstance(exp['value'], (int, float)):
        return False
    if exp['value'] not in [-1.0, 0.0, 1.0]:
        return False

    return True
