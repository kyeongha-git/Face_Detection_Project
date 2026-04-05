"""
tests/layers/test_wfpn.py
Unit tests for WFPN (Weighted Feature Pyramid Network).
"""

import pytest
import torch
from src.layers.wfpn import WFPN

# WFPN takes 3-stage MobileNet backbone outputs as input.
# in_channels_list = [64, 128, 256], out_channels = 64  (cfg_mnet defaults)
IN_CHANNELS_LIST = [64, 128, 256]
OUT_CHANNELS = 64


@pytest.fixture
def wfpn():
    return WFPN(in_channels_list=IN_CHANNELS_LIST, out_channels=OUT_CHANNELS)


@pytest.fixture
def multiscale_fmaps():
    """MobileNet 3-stage feature maps: stride 8/16/32 -> spatial 80/40/20 for a 640-px image."""
    return [
        torch.randn(1, 64, 80, 80),  # stage1 output
        torch.randn(1, 128, 40, 40),  # stage2 output
        torch.randn(1, 256, 20, 20),  # stage3 output
    ]


@pytest.mark.unit
class TestWFPN:
    def test_output_is_list_of_three(self, wfpn, multiscale_fmaps):
        """Output must be a list of exactly three feature maps."""
        out = wfpn(multiscale_fmaps)
        assert isinstance(out, list)
        assert len(out) == 3

    def test_output_channels(self, wfpn, multiscale_fmaps):
        """All output feature maps must have out_channels (64) channels."""
        out = wfpn(multiscale_fmaps)
        for i, fmap in enumerate(out):
            assert (
                fmap.shape[1] == OUT_CHANNELS
            ), f"Output[{i}] channels={fmap.shape[1]}, expected {OUT_CHANNELS}"

    def test_output_spatial_sizes(self, wfpn, multiscale_fmaps):
        """Each output feature map must retain the spatial size of its corresponding input."""
        expected_sizes = [(80, 80), (40, 40), (20, 20)]
        out = wfpn(multiscale_fmaps)
        for i, (fmap, (H, W)) in enumerate(zip(out, expected_sizes)):
            assert (fmap.shape[2], fmap.shape[3]) == (
                H,
                W,
            ), f"Output[{i}] spatial={fmap.shape[2:]}, expected ({H},{W})"

    def test_output_shapes_full(self, wfpn, multiscale_fmaps):
        """Full shape check: (B, 64, H, W) for each output level."""
        batch = 2
        fmaps = [
            torch.randn(batch, 64, 80, 80),
            torch.randn(batch, 128, 40, 40),
            torch.randn(batch, 256, 20, 20),
        ]
        out = wfpn(fmaps)
        assert out[0].shape == (batch, OUT_CHANNELS, 80, 80)
        assert out[1].shape == (batch, OUT_CHANNELS, 40, 40)
        assert out[2].shape == (batch, OUT_CHANNELS, 20, 20)

    def test_gradient_flows(self, wfpn, multiscale_fmaps):
        """Gradients must flow back to all inputs during backpropagation."""
        fmaps = [f.requires_grad_(True) for f in multiscale_fmaps]
        out = wfpn(fmaps)
        loss = sum(o.sum() for o in out)
        loss.backward()
        for i, fmap in enumerate(fmaps):
            assert fmap.grad is not None, f"No gradient for input[{i}]"

    def test_learnable_alpha_weights(self, wfpn):
        """alpha_conv1 and alpha_conv2 must be learnable parameters."""
        param_names = [name for name, _ in wfpn.named_parameters()]
        assert any("alpha_conv1" in n for n in param_names)
        assert any("alpha_conv2" in n for n in param_names)

    @pytest.mark.parametrize("out_ch", [32, 64, 128])
    def test_leaky_relu_threshold(self, out_ch):
        """WFPN must build without error for any supported out_channels value."""
        wfpn = WFPN(in_channels_list=[64, 128, 256], out_channels=out_ch)
        assert wfpn is not None
