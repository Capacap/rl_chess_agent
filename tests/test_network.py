"""
test_network.py
Unit tests for neural network architecture.
"""

import torch
import pytest
from model.network import ChessNet, ResidualBlock, count_parameters


class TestResidualBlock:
    """Test ResidualBlock module."""

    def test_residual_block_shape(self):
        """Output shape should match input shape."""
        block = ResidualBlock(64)
        x = torch.randn(1, 64, 8, 8)
        # TODO: Uncomment when implemented
        # output = block(x)
        # assert output.shape == x.shape

    def test_residual_block_gradient_flow(self):
        """Verify gradients flow through skip connection."""
        # TODO: Test gradient computation
        pass


class TestChessNet:
    """Test ChessNet architecture."""

    def test_forward_pass_shape(self):
        """Forward pass should return correct shapes."""
        net = ChessNet(channels=64, num_blocks=4)
        state = torch.randn(2, 13, 8, 8)  # Batch of 2
        legal_mask = torch.ones(2, 4096, dtype=torch.bool)

        # TODO: Uncomment when implemented
        # policy, value = net(state, legal_mask)
        # assert policy.shape == (2, 4096)
        # assert value.shape == (2, 1)

    def test_policy_probabilities_sum(self):
        """Policy output should sum to 1.0."""
        net = ChessNet()
        state = torch.randn(1, 13, 8, 8)
        legal_mask = torch.ones(1, 4096, dtype=torch.bool)

        # TODO: Uncomment when implemented
        # policy, _ = net(state, legal_mask)
        # assert torch.allclose(policy.sum(dim=1), torch.tensor([1.0]))

    def test_value_range(self):
        """Value output should be in [-1, 1]."""
        net = ChessNet()
        state = torch.randn(1, 13, 8, 8)
        legal_mask = torch.ones(1, 4096, dtype=torch.bool)

        # TODO: Uncomment when implemented
        # _, value = net(state, legal_mask)
        # assert value.min() >= -1.0
        # assert value.max() <= 1.0

    def test_legal_move_masking(self):
        """Illegal moves should have zero probability."""
        net = ChessNet()
        state = torch.randn(1, 13, 8, 8)
        legal_mask = torch.zeros(1, 4096, dtype=torch.bool)
        legal_mask[0, [0, 10, 100]] = True  # Only 3 legal moves

        # TODO: Uncomment when implemented
        # policy, _ = net(state, legal_mask)
        # # Check only legal moves have non-zero probability
        # assert policy[0, 0] > 0
        # assert policy[0, 10] > 0
        # assert policy[0, 100] > 0
        # assert policy[0, 1] == 0  # Illegal move

    def test_parameter_count(self):
        """Verify network has expected number of parameters."""
        net = ChessNet(channels=64, num_blocks=4)
        # TODO: Uncomment when implemented
        # param_count = count_parameters(net)
        # # Should be approximately 8.4M parameters
        # assert 8_000_000 < param_count < 9_000_000

    def test_inference_time(self):
        """Verify inference time is <50ms on CPU."""
        import time

        net = ChessNet(channels=64, num_blocks=4)
        net.eval()
        state = torch.randn(1, 13, 8, 8)
        legal_mask = torch.ones(1, 4096, dtype=torch.bool)

        # TODO: Uncomment when implemented
        # # Warmup
        # with torch.no_grad():
        #     _ = net(state, legal_mask)
        #
        # # Measure
        # times = []
        # for _ in range(100):
        #     start = time.perf_counter()
        #     with torch.no_grad():
        #         _ = net(state, legal_mask)
        #     times.append(time.perf_counter() - start)
        #
        # avg_time = sum(times) / len(times)
        # print(f"Avg inference time: {avg_time*1000:.1f}ms")
        # assert avg_time < 0.050  # <50ms

    def test_no_nan_in_output(self):
        """Network should not produce NaN values."""
        net = ChessNet()
        state = torch.randn(1, 13, 8, 8)
        legal_mask = torch.ones(1, 4096, dtype=torch.bool)

        # TODO: Uncomment when implemented
        # policy, value = net(state, legal_mask)
        # assert not torch.isnan(policy).any()
        # assert not torch.isnan(value).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
