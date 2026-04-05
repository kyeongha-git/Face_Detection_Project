"""
tests/layers/test_conv_block.py
Unit tests for DeformableConv2d and conv_bn helper functions.
"""

import pytest
import torch
from src.layers.conv_block import (
    DeformableConv2d,
    conv_bn,
    conv_bn_no_relu,
    conv_bn1X1,
    conv_dw,
)


@pytest.mark.unit
class TestDeformableConv2d:
    @pytest.mark.parametrize(
        "in_ch,out_ch,H,W",
        [
            (32, 64, 40, 40),
            (64, 64, 20, 20),
            (128, 128, 10, 10),
        ],
    )
    def test_output_shape(self, in_ch, out_ch, H, W):
        """With stride=1 and padding=1, spatial size must be preserved."""
        dcn = DeformableConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        x = torch.randn(1, in_ch, H, W)
        out = dcn(x)
        assert out.shape == (1, out_ch, H, W)

    def test_offset_initialized_to_zero(self):
        """offset_conv weight and bias must be zero-initialized (standard conv at start)."""
        dcn = DeformableConv2d(64, 64)
        assert torch.all(dcn.offset_conv.weight == 0)
        assert torch.all(dcn.offset_conv.bias == 0)

    def test_modulator_initialized_to_zero(self):
        """modulator_conv weight and bias must be zero-initialized."""
        dcn = DeformableConv2d(64, 64)
        assert torch.all(dcn.modulator_conv.weight == 0)
        assert torch.all(dcn.modulator_conv.bias == 0)

    def test_gradient_flows(self):
        """Gradients must flow back to the input during backpropagation."""
        dcn = DeformableConv2d(64, 64)
        x = torch.randn(1, 64, 16, 16, requires_grad=True)
        out = dcn(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_stride_reduces_spatial_size(self):
        """With stride=2, spatial size must be halved."""
        dcn = DeformableConv2d(64, 128, kernel_size=3, stride=2, padding=1)
        x = torch.randn(1, 64, 40, 40)
        out = dcn(x)
        assert out.shape == (1, 128, 20, 20)


@pytest.mark.unit
class TestConvBnHelpers:
    def test_conv_bn_output_shape(self):
        """conv_bn: DeformableConv -> BN -> LeakyReLU. Spatial size must be preserved."""
        block = conv_bn(64, 64, stride=1, leaky=0.1)
        x = torch.randn(1, 64, 20, 20)
        out = block(x)
        assert out.shape == (1, 64, 20, 20)

    def test_conv_bn_no_relu_output_shape(self):
        """conv_bn_no_relu: DeformableConv -> BN. Spatial size must be preserved."""
        block = conv_bn_no_relu(64, 64, stride=1)
        x = torch.randn(1, 64, 20, 20)
        out = block(x)
        assert out.shape == (1, 64, 20, 20)

    def test_conv_bn1X1_output_shape(self):
        """conv_bn1X1: 1x1 Conv -> BN -> LeakyReLU. Only channel dimension changes."""
        block = conv_bn1X1(128, 64, stride=1, leaky=0.1)
        x = torch.randn(1, 128, 40, 40)
        out = block(x)
        assert out.shape == (1, 64, 40, 40)

    def test_conv_dw_output_shape(self):
        """conv_dw: Depthwise + Pointwise. Spatial size preserved, channel transformed."""
        block = conv_dw(64, 128, stride=1, leaky=0.1)
        x = torch.randn(1, 64, 20, 20)
        out = block(x)
        assert out.shape == (1, 128, 20, 20)
