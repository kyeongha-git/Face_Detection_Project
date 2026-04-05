"""
tests/loss/test_multibox_loss.py
Tests for MultiBoxLoss with dummy data.

Note: multibox_loss.py calls .cuda() at module-level via:
  GPU = cfg_mnet['gpu_train']  (True)
  loc_t.cuda(), etc.
Importing this module on a CPU-only machine raises RuntimeError, so
the entire test class is conditionally skipped when CUDA is unavailable.
"""

import pytest
import torch

# Check CUDA availability upfront
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.unit
@pytest.mark.skipif(
    not CUDA_AVAILABLE, reason="MultiBoxLoss requires CUDA (module-level .cuda() call)"
)
class TestMultiBoxLoss:

    @pytest.fixture
    def loss_fn(self):
        from src.loss.multibox_loss import MultiBoxLoss

        return MultiBoxLoss(
            num_classes=2,
            overlap_thresh=0.35,
            prior_for_matching=True,
            bkg_label=0,
            neg_mining=True,
            neg_pos=7,
            neg_overlap=0.35,
            encode_target=False,
        )

    @pytest.fixture
    def prior_boxes(self):
        """Dummy prior boxes for a 640x640 image: shape (16800, 4)."""
        from src.utils.prior_box import PriorBox
        from src.utils.config import cfg_mnet

        pb = PriorBox(cfg_mnet, image_size=(640, 640))
        return pb.forward().cuda()

    @pytest.fixture
    def dummy_predictions(self, prior_boxes):
        """Dummy model output (B=2, N=16800) on GPU."""
        B = 2
        N = prior_boxes.shape[0]  # 16800
        bbox = torch.randn(B, N, 4).cuda()
        cls = torch.randn(B, N, 2).cuda()
        ldm = torch.randn(B, N, 10).cuda()
        return (bbox, cls, ldm)

    @pytest.fixture
    def dummy_targets(self):
        """Dummy GT: 3 objects per image, each row is (x1,y1,x2,y2, lm*10, label)."""
        B = 2
        targets = []
        for _ in range(B):
            n_obj = 3
            # (x1, y1, x2, y2): normalised to [0,1], x2>x1, y2>y1
            boxes = torch.zeros(n_obj, 15).cuda()
            boxes[:, 0] = torch.rand(n_obj) * 0.4  # x1
            boxes[:, 1] = torch.rand(n_obj) * 0.4  # y1
            boxes[:, 2] = boxes[:, 0] + 0.1  # x2
            boxes[:, 3] = boxes[:, 1] + 0.1  # y2
            boxes[:, 4:14] = torch.rand(n_obj, 10) * 0.5 + 0.25  # landmarks
            boxes[:, 14] = 1.0  # label (foreground)
            targets.append(boxes)
        return targets

    def test_loss_returns_three_values(
        self, loss_fn, dummy_predictions, prior_boxes, dummy_targets
    ):
        """Forward pass must return (loss_l, loss_c, loss_landm)."""
        result = loss_fn(dummy_predictions, prior_boxes, dummy_targets)
        assert len(result) == 3

    def test_losses_are_scalars(
        self, loss_fn, dummy_predictions, prior_boxes, dummy_targets
    ):
        """Each individual loss must be a 0-dim scalar tensor."""
        loss_l, loss_c, loss_landm = loss_fn(
            dummy_predictions, prior_boxes, dummy_targets
        )
        assert loss_l.ndim == 0
        assert loss_c.ndim == 0
        assert loss_landm.ndim == 0

    def test_losses_are_nonnegative(
        self, loss_fn, dummy_predictions, prior_boxes, dummy_targets
    ):
        """All loss values must be >= 0."""
        loss_l, loss_c, loss_landm = loss_fn(
            dummy_predictions, prior_boxes, dummy_targets
        )
        assert loss_l >= 0
        assert loss_c >= 0
        assert loss_landm >= 0

    def test_losses_are_finite(
        self, loss_fn, dummy_predictions, prior_boxes, dummy_targets
    ):
        """No loss value must be NaN or Inf."""
        loss_l, loss_c, loss_landm = loss_fn(
            dummy_predictions, prior_boxes, dummy_targets
        )
        assert torch.isfinite(loss_l)
        assert torch.isfinite(loss_c)
        assert torch.isfinite(loss_landm)

    def test_backward_propagates(
        self, loss_fn, dummy_predictions, prior_boxes, dummy_targets
    ):
        """Backward pass on the total loss must succeed and produce gradients."""
        bbox, cls, ldm = dummy_predictions
        bbox = bbox.requires_grad_(True)
        cls = cls.requires_grad_(True)
        ldm = ldm.requires_grad_(True)

        loss_l, loss_c, loss_landm = loss_fn(
            (bbox, cls, ldm), prior_boxes, dummy_targets
        )
        total_loss = loss_l + loss_c + loss_landm
        total_loss.backward()

        # At least one input must have received a gradient
        assert bbox.grad is not None or cls.grad is not None
