"""
tests/utils/test_box_utils.py
Tests for box_utils: encode/decode round-trip consistency and point_form conversion.
"""

import pytest
import torch
from src.utils.box_utils import point_form, encode, decode

VARIANCES = [0.1, 0.2]


@pytest.mark.unit
class TestPointForm:
    def test_output_shape(self):
        """point_form output shape must match the input shape."""
        boxes = torch.rand(100, 4)  # (cx, cy, w, h) format
        pf = point_form(boxes)
        assert pf.shape == boxes.shape

    def test_xmin_less_than_xmax(self):
        """After point_form conversion, xmin < xmax and ymin < ymax must hold."""
        # center-form boxes with positive w and h
        cx = torch.rand(50) * 0.5 + 0.25  # [0.25, 0.75]
        cy = torch.rand(50) * 0.5 + 0.25
        w = torch.rand(50) * 0.2 + 0.05  # [0.05, 0.25]
        h = torch.rand(50) * 0.2 + 0.05
        boxes = torch.stack([cx, cy, w, h], dim=1)
        pf = point_form(boxes)
        assert (pf[:, 0] < pf[:, 2]).all()  # xmin < xmax
        assert (pf[:, 1] < pf[:, 3]).all()  # ymin < ymax

    def test_area_preserved(self):
        """Box area after point_form must equal the original w * h."""
        cx = torch.tensor([0.5])
        cy = torch.tensor([0.5])
        w = torch.tensor([0.4])
        h = torch.tensor([0.3])
        boxes = torch.stack([cx, cy, w, h], dim=1)
        pf = point_form(boxes)
        area = (pf[:, 2] - pf[:, 0]) * (pf[:, 3] - pf[:, 1])
        assert torch.allclose(area, w * h, atol=1e-5)


@pytest.mark.unit
class TestEncodeDecode:
    """Round-trip consistency tests: encode -> decode must recover the original boxes."""

    @pytest.fixture
    def prior_boxes(self):
        """100 dummy prior boxes in center-size format (normalised coordinates)."""
        cx = torch.rand(100) * 0.8 + 0.1
        cy = torch.rand(100) * 0.8 + 0.1
        w = torch.rand(100) * 0.2 + 0.05
        h = torch.rand(100) * 0.2 + 0.05
        return torch.stack([cx, cy, w, h], dim=1)

    @pytest.fixture
    def matched_boxes(self, prior_boxes):
        """GT-like boxes in point-form derived from prior_boxes with slight perturbation."""
        pf = point_form(prior_boxes)
        noise = torch.randn_like(pf) * 0.01
        boxes = (pf + noise).clamp(0, 1)
        # guarantee xmin < xmax
        boxes = torch.stack(
            [
                boxes[:, 0].clamp(max=boxes[:, 2] - 1e-3),
                boxes[:, 1].clamp(max=boxes[:, 3] - 1e-3),
                boxes[:, 2],
                boxes[:, 3],
            ],
            dim=1,
        )
        return boxes

    def test_encode_output_shape(self, matched_boxes, prior_boxes):
        """encode output must have shape (N, 4)."""
        encoded = encode(matched_boxes, prior_boxes, VARIANCES)
        assert encoded.shape == (100, 4)

    def test_decode_output_shape(self, matched_boxes, prior_boxes):
        """decode output must have shape (N, 4)."""
        encoded = encode(matched_boxes, prior_boxes, VARIANCES)
        decoded = decode(encoded, prior_boxes, VARIANCES)
        assert decoded.shape == (100, 4)

    def test_encode_decode_round_trip(self, matched_boxes, prior_boxes):
        """encode followed by decode must recover the original point-form boxes.

        matched_boxes are in point-form (xmin, ymin, xmax, ymax).
        decode output is also in point-form, so the two must be close.
        """
        encoded = encode(matched_boxes, prior_boxes, VARIANCES)
        decoded = decode(encoded, prior_boxes, VARIANCES)
        assert torch.allclose(
            decoded, matched_boxes, atol=1e-4
        ), f"Max diff: {(decoded - matched_boxes).abs().max():.6f}"


@pytest.mark.unit
class TestPointFormCenterSizeInverse:
    def test_center_size_has_known_bug(self):
        """center_size() in box_utils.py has a known torch.cat argument-order bug.

        Current source (box_utils.py:25):
            return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,   # <-- tensor, not a tuple
                             boxes[:, 2:] - boxes[:, :2], 1)

        Correct implementation:
            return torch.cat([(boxes[:, 2:] + boxes[:, :2])/2,
                               boxes[:, 2:] - boxes[:, :2]], dim=1)

        This test explicitly captures the bug so that any future fix can be verified here.
        """
        from src.utils.box_utils import center_size, point_form

        cx = torch.tensor([[0.5, 0.5, 0.4, 0.3]])  # a single (cx, cy, w, h) box
        pf = point_form(cx)
        with pytest.raises(TypeError):
            center_size(pf)  # must raise TypeError until the bug is fixed
