"""
tests/models/test_heads.py
Output shape tests for ClassHead, BboxHead, and LandmarkHead.
"""

import pytest
import torch
from src.models.our_model import ClassHead, BboxHead, LandmarkHead

# OurModel defaults: out_channel=64, anchor_num=2
IN_CH = 64
ANCHOR_NUM = 2
B, H, W = 2, 40, 40  # example batch, spatial size
NUM_ANCHORS_TOTAL = B * H * W * ANCHOR_NUM


@pytest.mark.unit
class TestClassHead:
    @pytest.mark.parametrize("B,H,W", [(1, 80, 80), (2, 40, 40), (1, 20, 20)])
    def test_output_shape(self, B, H, W):
        """Output shape: (B, H*W*num_anchors, 2) — 2 classes (foreground / background)."""
        head = ClassHead(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(B, IN_CH, H, W)
        out = head(x)
        assert out.shape == (B, H * W * ANCHOR_NUM, 2)

    def test_output_dtype(self):
        """Output must be float32."""
        head = ClassHead(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(1, IN_CH, 40, 40)
        out = head(x)
        assert out.dtype == torch.float32


@pytest.mark.unit
class TestBboxHead:
    @pytest.mark.parametrize("B,H,W", [(1, 80, 80), (2, 40, 40), (1, 20, 20)])
    def test_output_shape(self, B, H, W):
        """Output shape: (B, H*W*num_anchors, 4) — 4 bounding box coordinates."""
        head = BboxHead(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(B, IN_CH, H, W)
        out = head(x)
        assert out.shape == (B, H * W * ANCHOR_NUM, 4)

    def test_output_dtype(self):
        """Output must be float32."""
        head = BboxHead(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(1, IN_CH, 40, 40)
        out = head(x)
        assert out.dtype == torch.float32


@pytest.mark.unit
class TestLandmarkHead:
    @pytest.mark.parametrize("B,H,W", [(1, 80, 80), (2, 40, 40), (1, 20, 20)])
    def test_output_shape(self, B, H, W):
        """Output shape: (B, H*W*num_anchors, 10) — 5 landmarks x 2 coordinates."""
        head = LandmarkHead(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(B, IN_CH, H, W)
        out = head(x)
        assert out.shape == (B, H * W * ANCHOR_NUM, 10)

    def test_output_dtype(self):
        """Output must be float32."""
        head = LandmarkHead(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(1, IN_CH, 40, 40)
        out = head(x)
        assert out.dtype == torch.float32


@pytest.mark.unit
class TestHeadsGradient:
    """All three detection heads must support backpropagation."""

    @pytest.mark.parametrize(
        "HeadCls,out_dim",
        [
            (ClassHead, 2),
            (BboxHead, 4),
            (LandmarkHead, 10),
        ],
    )
    def test_gradient_flows(self, HeadCls, out_dim):
        """Gradients must flow back to the input for every head type."""
        head = HeadCls(inchannels=IN_CH, num_anchors=ANCHOR_NUM)
        x = torch.randn(1, IN_CH, 40, 40, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
