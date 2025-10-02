"""
network.py
Chess neural network architecture: ResNet with policy and value heads.

Architecture:
    Input [B, 13, 8, 8]
    → Conv 3×3, 64 channels
    → BatchNorm + ReLU
    → ResBlock × 4
    → Policy Head (4096-dim softmax) + Value Head (scalar tanh)

Target inference time: <50ms on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """
    Residual block with two conv layers and skip connection.

    Structure:
        x → conv1 → bn1 → relu → conv2 → bn2 → (+x) → relu
    """

    def __init__(self, channels: int):
        """
        Args:
            channels: Number of input/output channels
        """
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connection.

        Args:
            x: Input tensor [B, C, 8, 8]

        Returns:
            Output tensor [B, C, 8, 8]
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """
    Chess neural network with policy and value heads.

    Input: [B, 13, 8, 8] board encoding
    Output: (policy [B, 4096], value [B, 1])
    """

    def __init__(self, channels: int = 64, num_blocks: int = 4):
        """
        Args:
            channels: Number of channels in residual blocks
            num_blocks: Number of residual blocks
        """
        super(ChessNet, self).__init__()

        self.channels = channels
        self.num_blocks = num_blocks

        # Initial convolution: 13 → channels
        self.initial_conv = nn.Conv2d(13, channels, 3, padding=1)
        self.initial_bn = nn.BatchNorm2d(channels)

        # Residual blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        # Value head
        self.value_conv = nn.Conv2d(channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            x: Board state [B, 13, 8, 8]
            legal_mask: Boolean tensor [B, 4096], True = legal move

        Returns:
            policy: Move probabilities [B, 4096], sums to 1.0
            value: Position evaluation [B, 1], range [-1, 1]
        """
        # Initial convolution
        x = F.relu(self.initial_bn(self.initial_conv(x)))

        # Residual blocks
        for block in self.res_blocks:
            x = block(x)

        # Policy and value heads
        policy = self.forward_policy(x, legal_mask)
        value = self.forward_value(x)

        return policy, value

    def forward_policy(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """
        Policy head with legal move masking.

        Args:
            x: Features [B, C, 8, 8]
            legal_mask: Boolean [B, 4096], True=legal

        Returns:
            Policy logits [B, 4096], softmax over legal moves only
        """
        # Conv 1x1 + bn + relu
        x = F.relu(self.policy_bn(self.policy_conv(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC to 4096
        logits = self.policy_fc(x)

        # Mask illegal moves (set to -inf)
        logits = logits.masked_fill(~legal_mask, float('-inf'))

        # Softmax
        return F.softmax(logits, dim=1)

    def forward_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Value head for position evaluation.

        Args:
            x: Features [B, C, 8, 8]

        Returns:
            Value scalar [B, 1], range [-1, 1]
        """
        # Conv 1x1 + bn + relu
        x = F.relu(self.value_bn(self.value_conv(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # FC → 64 → relu
        x = F.relu(self.value_fc1(x))

        # FC → 1 → tanh
        x = torch.tanh(self.value_fc2(x))

        return x


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
