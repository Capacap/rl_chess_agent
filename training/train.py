"""
train.py
Training loop for chess neural network.

Loss function: policy_loss + value_loss
- Policy: Cross-entropy between MCTS policy and network policy
- Value: MSE between game outcome and value prediction
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
from model.network import ChessNet
from training.selfplay import Experience
from encoding.state import encode_board
from encoding.move import create_legal_move_mask
import chess
import numpy as np


def compute_loss(
    policy_pred: torch.Tensor,
    value_pred: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined policy + value loss.

    Args:
        policy_pred: Network policy output [B, 4096]
        value_pred: Network value output [B, 1]
        policy_target: MCTS policy distribution [B, 4096]
        value_target: Game outcome [B, 1]

    Returns:
        total_loss: Combined loss for backprop
        metrics: Dict with individual loss values for logging
    """
    # Policy loss: Cross-entropy (KL divergence)
    # -sum(target * log(pred))
    policy_loss = -torch.sum(
        policy_target * torch.log(policy_pred + 1e-8),
        dim=1
    ).mean()

    # Value loss: MSE
    value_loss = F.mse_loss(value_pred, value_target)

    # Combined loss (equal weighting)
    total_loss = policy_loss + value_loss

    metrics = {
        'policy': policy_loss.item(),
        'value': value_loss.item(),
        'total': total_loss.item()
    }

    return total_loss, metrics


def experiences_to_tensors(
    experiences: List[Experience]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert experience list to batched tensors.

    Args:
        experiences: List of Experience objects

    Returns:
        states: [B, 13, 8, 8]
        policy_targets: [B, 4096]
        value_targets: [B, 1]
        legal_masks: [B, 4096]
    """
    states = []
    policies = []
    values = []
    masks = []

    for exp in experiences:
        # Parse FEN
        board = chess.Board(exp.fen)

        # Encode state
        state = encode_board(board).squeeze(0)  # [13, 8, 8]
        states.append(state)

        # Policy and value
        policy = torch.tensor(exp.policy, dtype=torch.float32)
        value = torch.tensor([exp.value], dtype=torch.float32)
        policies.append(policy)
        values.append(value)

        # Legal mask
        mask = create_legal_move_mask(board)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        masks.append(mask_tensor)

    # Stack into batches
    states_batch = torch.stack(states)
    policies_batch = torch.stack(policies)
    values_batch = torch.stack(values)
    masks_batch = torch.stack(masks)

    return states_batch, policies_batch, values_batch, masks_batch


def train_iteration(
    network: ChessNet,
    experiences: List[Experience],
    batch_size: int = 256,
    epochs: int = 5,
    lr: float = 1e-3
) -> Dict[str, List[float]]:
    """
    Train network on experience buffer.

    Args:
        network: ChessNet to train (modified in-place)
        experiences: List of Experience objects
        batch_size: Mini-batch size for SGD
        epochs: Number of passes through data
        lr: Learning rate

    Returns:
        history: Dict with loss history per epoch
    """
    # Convert experiences to tensors
    states, policy_targets, value_targets, legal_masks = experiences_to_tensors(experiences)

    # Create data indices for batching
    num_samples = len(experiences)
    indices = list(range(num_samples))

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    # Training loop
    network.train()
    history = {'policy': [], 'value': [], 'total': []}

    for epoch in range(epochs):
        # Shuffle data
        np.random.shuffle(indices)

        epoch_losses = {'policy': [], 'value': [], 'total': []}

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            # Get batch indices
            batch_indices = indices[i:i + batch_size]

            # Extract batch
            batch_states = states[batch_indices]
            batch_policy_targets = policy_targets[batch_indices]
            batch_value_targets = value_targets[batch_indices]
            batch_masks = legal_masks[batch_indices]

            # Forward pass
            policy_pred, value_pred = network(batch_states, batch_masks)

            # Compute loss
            loss, metrics = compute_loss(
                policy_pred,
                value_pred,
                batch_policy_targets,
                batch_value_targets
            )

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record metrics
            epoch_losses['policy'].append(metrics['policy'])
            epoch_losses['value'].append(metrics['value'])
            epoch_losses['total'].append(metrics['total'])

        # Epoch summary
        avg_policy = np.mean(epoch_losses['policy'])
        avg_value = np.mean(epoch_losses['value'])
        avg_total = np.mean(epoch_losses['total'])

        history['policy'].append(avg_policy)
        history['value'].append(avg_value)
        history['total'].append(avg_total)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"policy_loss={avg_policy:.4f}, "
              f"value_loss={avg_value:.4f}, "
              f"total_loss={avg_total:.4f}")

    return history


def save_checkpoint(network: ChessNet, file_path: str) -> None:
    """
    Save network weights to file.

    Args:
        network: ChessNet to save
        file_path: Output file path (.pt extension)
    """
    torch.save(network.state_dict(), file_path)
    print(f"Checkpoint saved: {file_path}")


def load_checkpoint(network: ChessNet, file_path: str) -> None:
    """
    Load network weights from file.

    Args:
        network: ChessNet to load into (modified in-place)
        file_path: Checkpoint file path
    """
    network.load_state_dict(torch.load(file_path))
    network.eval()
    print(f"Checkpoint loaded: {file_path}")
