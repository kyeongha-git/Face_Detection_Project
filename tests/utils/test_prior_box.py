"""
tests/utils/test_prior_box.py
Tests for PriorBox anchor generation (count, value range, clip behavior).
"""

import pytest
import torch
from src.utils.prior_box import PriorBox

# Anchor count for a 640x640 input:
# steps=[8,16,32] -> feature maps: [80x80, 40x40, 20x20]
# min_sizes per level: 2 anchors/cell
# total: (80*80 + 40*40 + 20*20) * 2 = (6400+1600+400)*2 = 16800
EXPECTED_NUM_ANCHORS_640 = 16800
IMAGE_SIZE_640 = (640, 640)


@pytest.fixture
def prior_box_640(cfg_mnet):
    return PriorBox(cfg_mnet, image_size=IMAGE_SIZE_640, phase="train")


@pytest.mark.unit
class TestPriorBoxCount:
    def test_anchor_count_640(self, prior_box_640):
        """Total anchor count for a 640x640 image must be 16800."""
        anchors = prior_box_640.forward()
        assert anchors.shape[0] == EXPECTED_NUM_ANCHORS_640

    def test_anchor_dimension(self, prior_box_640):
        """Each anchor must have 4 values: (cx, cy, w, h)."""
        anchors = prior_box_640.forward()
        assert anchors.shape[1] == 4

    def test_anchor_output_is_tensor(self, prior_box_640):
        """Output must be a torch.Tensor."""
        anchors = prior_box_640.forward()
        assert isinstance(anchors, torch.Tensor)


@pytest.mark.unit
class TestPriorBoxValueRange:
    def test_no_clip_values_may_exceed_range(self, cfg_mnet):
        """With clip=False, anchor values are unconstrained (w and h must still be positive)."""
        cfg = dict(cfg_mnet)
        cfg["clip"] = False
        pb = PriorBox(cfg, image_size=IMAGE_SIZE_640)
        anchors = pb.forward()
        assert (anchors[:, 2:] > 0).all()  # w, h > 0

    def test_clip_constrains_to_unit(self, cfg_mnet):
        """With clip=True, all anchor values must be in [0, 1]."""
        cfg = dict(cfg_mnet)
        cfg["clip"] = True
        pb = PriorBox(cfg, image_size=IMAGE_SIZE_640)
        anchors = pb.forward()
        assert anchors.min() >= 0.0
        assert anchors.max() <= 1.0

    def test_center_values_positive(self, prior_box_640):
        """Anchor center coordinates (cx, cy) must all be positive."""
        anchors = prior_box_640.forward()
        assert (anchors[:, 0] > 0).all()  # cx
        assert (anchors[:, 1] > 0).all()  # cy

    def test_size_values_positive(self, prior_box_640):
        """Anchor size values (w, h) must all be positive."""
        anchors = prior_box_640.forward()
        assert (anchors[:, 2] > 0).all()  # w
        assert (anchors[:, 3] > 0).all()  # h


@pytest.mark.unit
class TestPriorBoxDifferentSizes:
    @pytest.mark.parametrize("H,W", [(320, 320), (640, 640), (480, 640)])
    def test_anchor_count_scales_with_image(self, cfg_mnet, H, W):
        """Anchor count must scale correctly when image size changes."""
        import math

        steps = cfg_mnet["steps"]  # [8, 16, 32]
        n_sizes_per_level = [len(sz) for sz in cfg_mnet["min_sizes"]]
        expected = sum(
            math.ceil(H / s) * math.ceil(W / s) * n
            for s, n in zip(steps, n_sizes_per_level)
        )
        pb = PriorBox(cfg_mnet, image_size=(H, W))
        anchors = pb.forward()
        assert anchors.shape[0] == expected
