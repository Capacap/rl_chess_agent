"""
test_dataset.py
Unit tests for dataset and experience storage.
"""

import json
import tempfile
import os
import chess
import torch
import pytest
from training.dataset import ChessDataset, save_experience, validate_experience
from torch.utils.data import DataLoader


class TestChessDataset:
    """Test ChessDataset class."""

    def test_dataset_load(self):
        """Test loading experiences from file."""
        # TODO: Create temporary JSONL file
        # TODO: Test dataset loading
        pass

    def test_dataset_length(self):
        """Dataset length should match number of experiences."""
        # TODO: Create dataset with known size
        # TODO: Assert len(dataset) is correct
        pass

    def test_dataset_getitem(self):
        """Test retrieving single experience."""
        # TODO: Create dataset
        # TODO: Get item by index
        # TODO: Verify shapes and types
        pass

    def test_dataset_batching(self):
        """Test DataLoader batching."""
        # TODO: Create dataset
        # TODO: Create DataLoader with batch_size=32
        # TODO: Verify batch shapes
        pass

    def test_empty_file_raises_error(self):
        """Empty experience file should raise ValueError."""
        # TODO: Create empty file
        # TODO: Assert ChessDataset raises ValueError
        pass


class TestExperienceSaving:
    """Test experience save/load functions."""

    def test_save_experience(self):
        """Test saving experiences to JSONL."""
        # TODO: Create temporary file
        # TODO: Save experiences
        # TODO: Read back and verify format
        pass

    def test_save_experience_append(self):
        """Test appending to existing file."""
        # TODO: Save initial experiences
        # TODO: Append more experiences
        # TODO: Verify both sets are present
        pass

    def test_save_experience_overwrite(self):
        """Test overwriting existing file."""
        # TODO: Save initial experiences
        # TODO: Save with append=False
        # TODO: Verify only new experiences present
        pass


class TestExperienceValidation:
    """Test experience validation."""

    def test_validate_correct_experience(self):
        """Valid experience should pass validation."""
        exp = {
            'fen': chess.Board().fen(),
            'policy': [1.0/4096] * 4096,
            'value': 0.0
        }
        # TODO: Uncomment when implemented
        # assert validate_experience(exp)

    def test_validate_missing_keys(self):
        """Experience missing keys should fail validation."""
        exp = {'fen': chess.Board().fen()}
        # TODO: Uncomment when implemented
        # assert not validate_experience(exp)

    def test_validate_invalid_fen(self):
        """Invalid FEN should fail validation."""
        exp = {
            'fen': 'invalid_fen_string',
            'policy': [1.0/4096] * 4096,
            'value': 0.0
        }
        # TODO: Uncomment when implemented
        # assert not validate_experience(exp)

    def test_validate_wrong_policy_length(self):
        """Policy with wrong length should fail."""
        exp = {
            'fen': chess.Board().fen(),
            'policy': [0.1] * 100,  # Wrong length
            'value': 0.0
        }
        # TODO: Uncomment when implemented
        # assert not validate_experience(exp)

    def test_validate_policy_sum(self):
        """Policy not summing to 1.0 should fail."""
        exp = {
            'fen': chess.Board().fen(),
            'policy': [0.0] * 4096,  # Sums to 0, not 1
            'value': 0.0
        }
        # TODO: Uncomment when implemented
        # assert not validate_experience(exp)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
