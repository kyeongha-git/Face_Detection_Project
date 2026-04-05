"""
tests/layers/test_scm.py
Unit tests for SCM (Shuffle Context Module).
"""

import pytest
import torch
from src.layers.scm import SCM


@pytest.mark.unit
class TestSCM:
    @pytest.mark.parametrize(
        "in_ch,out_ch,H,W",
        [
            (64, 64, 80, 80),
            (64, 64, 40, 40),
            (64, 64, 20, 20),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, H, W):
        """Output shape must be (B, out_channel, H, W)."""
        scm = SCM(in_channel=in_ch, out_channel=out_ch)
        x = torch.randn(1, in_ch, H, W)
        out = scm(x)
        assert out.shape == (1, out_ch, H, W)

    def test_out_channel_must_be_divisible_by_4(self):
        """out_channel not divisible by 4 must raise AssertionError."""
        with pytest.raises(AssertionError):
            SCM(in_channel=64, out_channel=62)

    def test_output_nonnegative(self):
        """All output values must be >= 0 after the final ReLU activation."""
        scm = SCM(in_channel=64, out_channel=64)
        x = torch.randn(2, 64, 20, 20)
        out = scm(x)
        assert (out >= 0).all()

    def test_gradient_flows(self):
        """Gradients must flow back to the input during backpropagation."""
        scm = SCM(in_channel=64, out_channel=64)
        x = torch.randn(1, 64, 20, 20, requires_grad=True)
        out = scm(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_depthwise_shuffle_conv_groups(self):
        """shuffle_conv must be configured as depthwise (groups == out_channel)."""
        out_ch = 64
        scm = SCM(in_channel=64, out_channel=out_ch)
        assert scm.shuffle_conv.groups == out_ch

    def test_batch_size_independence(self):
        """Output shape must scale with batch size while keeping channel and spatial dims fixed."""
        scm = SCM(in_channel=64, out_channel=64)
        scm.eval()
        for batch in [1, 2, 4]:
            x = torch.randn(batch, 64, 20, 20)
            with torch.no_grad():
                out = scm(x)
            assert out.shape == (batch, 64, 20, 20)
