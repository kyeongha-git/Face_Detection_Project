"""
tests/layers/test_eca_cbam.py
Unit tests for the ECA_CBAM attention layer.
"""

import pytest
import torch
from src.layers.eca_cbam import (
    ECA_CBAM,
    ChannelAttention,
    SpatialAttention,
    AdaptiveKernelSize,
)

# ──────────────────────────────────────────────
# AdaptiveKernelSize utility
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestAdaptiveKernelSize:
    def test_returns_odd_number(self):
        """Returned kernel size must always be an odd number."""
        for ch in [32, 64, 128, 256, 512]:
            k = AdaptiveKernelSize(ch)
            assert k % 2 == 1, f"KernelSize={k} is not odd for channels={ch}"

    def test_increases_with_channels(self):
        """Kernel size must be non-decreasing as channel count grows."""
        k32 = AdaptiveKernelSize(32)
        k128 = AdaptiveKernelSize(128)
        k512 = AdaptiveKernelSize(512)
        assert k32 <= k128 <= k512


# ──────────────────────────────────────────────
# ChannelAttention
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestChannelAttention:
    @pytest.mark.parametrize(
        "channels,H,W",
        [
            (32, 80, 80),
            (64, 40, 40),
            (128, 20, 20),
        ],
    )
    def test_output_shape(self, channels, H, W):
        """Output shape must be (B, C, 1, 1)."""
        ca = ChannelAttention(channels)
        x = torch.randn(2, channels, H, W)
        out = ca(x)
        assert out.shape == (2, channels, 1, 1)

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        ca = ChannelAttention(64)
        x = torch.randn(1, 64, 16, 16)
        out = ca(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


# ──────────────────────────────────────────────
# SpatialAttention
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestSpatialAttention:
    @pytest.mark.parametrize("kernel_size", [3, 7])
    def test_output_shape(self, kernel_size):
        """Output shape must be (B, 1, H, W) — single channel, same spatial size as input."""
        sa = SpatialAttention(kernel_size=kernel_size)
        x = torch.randn(2, 64, 40, 40)
        out = sa(x)
        assert out.shape == (2, 1, 40, 40)

    def test_output_range(self):
        """Sigmoid output must be in [0, 1]."""
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(1, 64, 20, 20)
        out = sa(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_invalid_kernel_raises(self):
        """Kernel size other than 3 or 7 must raise AssertionError."""
        with pytest.raises(AssertionError):
            SpatialAttention(kernel_size=5)


# ──────────────────────────────────────────────
# ECA_CBAM (full module)
# ──────────────────────────────────────────────


@pytest.mark.unit
class TestECA_CBAM:
    @pytest.mark.parametrize(
        "channels,H,W",
        [
            (32, 80, 80),
            (64, 40, 40),
            (128, 20, 20),
            (256, 10, 10),
        ],
    )
    def test_output_shape_preserved(self, channels, H, W):
        """ECA_CBAM must preserve the input shape via its residual connection."""
        module = ECA_CBAM(channel=channels)
        x = torch.randn(1, channels, H, W)
        out = module(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With all-zero input, output must also be zero (verifies skip connection)."""
        module = ECA_CBAM(channel=64)
        # zero input → attention weights neutral → output equals residual (zero)
        x = torch.zeros(1, 64, 8, 8)
        out = module(x)
        assert torch.allclose(out, x, atol=1e-5)

    def test_gradient_flows(self):
        """Gradients must flow back to the input during backpropagation."""
        module = ECA_CBAM(channel=64)
        x = torch.randn(1, 64, 8, 8, requires_grad=True)
        out = module(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_batch_consistency(self):
        """Each sample in a batch must be processed independently."""
        module = ECA_CBAM(channel=64)
        module.eval()
        x1 = torch.randn(1, 64, 16, 16)
        x2 = torch.randn(1, 64, 16, 16)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out1 = module(x1)
            out2 = module(x2)
            out_batch = module(x_batch)

        assert torch.allclose(out1, out_batch[:1], atol=1e-5)
        assert torch.allclose(out2, out_batch[1:], atol=1e-5)
