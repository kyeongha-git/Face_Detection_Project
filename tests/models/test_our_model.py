"""
tests/models/test_our_model.py
Full forward-pass tests for OurModel (train mode, test mode, output shapes).
"""

import pytest
import torch
from src.models.our_model import OurModel

# OurModel loads pretrain weights when cfg['pretrain']=True.
# cfg_mnet from conftest sets pretrain=False, so tests run structure-only with no file I/O.


@pytest.fixture(scope="module")
def model_train(cfg_mnet):
    """OurModel in train phase (eval() called for deterministic BN)."""
    return OurModel(cfg=cfg_mnet, phase="train").eval()


@pytest.fixture(scope="module")
def model_test(cfg_mnet):
    """OurModel in test phase."""
    return OurModel(cfg=cfg_mnet, phase="test").eval()


@pytest.mark.unit
class TestOurModelForwardTrain:
    def test_output_is_tuple_of_three(self, model_train, dummy_image_640):
        """Train mode output must be a 3-tuple: (bbox_regressions, classifications, ldm_regressions)."""
        with torch.no_grad():
            out = model_train(dummy_image_640)
        assert isinstance(out, tuple)
        assert len(out) == 3

    def test_bbox_shape(self, model_train, dummy_image_640):
        """bbox_regressions shape: (B, N_anchors, 4)."""
        with torch.no_grad():
            bbox, _, _ = model_train(dummy_image_640)
        assert bbox.ndim == 3
        assert bbox.shape[-1] == 4

    def test_cls_shape(self, model_train, dummy_image_640):
        """classifications shape: (B, N_anchors, 2)."""
        with torch.no_grad():
            _, cls, _ = model_train(dummy_image_640)
        assert cls.ndim == 3
        assert cls.shape[-1] == 2

    def test_ldm_shape(self, model_train, dummy_image_640):
        """ldm_regressions shape: (B, N_anchors, 10)."""
        with torch.no_grad():
            _, _, ldm = model_train(dummy_image_640)
        assert ldm.ndim == 3
        assert ldm.shape[-1] == 10

    def test_anchor_count_consistency(self, model_train, dummy_image_640):
        """bbox, cls, and ldm must share the same anchor count (dim=1)."""
        with torch.no_grad():
            bbox, cls, ldm = model_train(dummy_image_640)
        assert bbox.shape[1] == cls.shape[1] == ldm.shape[1]

    def test_batch_size_propagation(self, model_train):
        """Batch size must be correctly reflected in all output tensors."""
        for batch in [1, 2]:
            x = torch.randn(batch, 3, 640, 640)
            with torch.no_grad():
                bbox, cls, ldm = model_train(x)
            assert bbox.shape[0] == batch
            assert cls.shape[0] == batch
            assert ldm.shape[0] == batch


@pytest.mark.unit
class TestOurModelForwardTest:
    def test_cls_is_softmax_in_test_mode(self, model_test, dummy_image_640):
        """In test mode, softmax must be applied to classifications so each row sums to 1."""
        with torch.no_grad():
            _, cls, _ = model_test(dummy_image_640)
        sums = cls.sum(dim=-1)  # (B, N_anchors)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_output_shapes_same_as_train(
        self, model_train, model_test, dummy_image_640
    ):
        """Test mode output shapes must match train mode output shapes."""
        with torch.no_grad():
            train_out = model_train(dummy_image_640)
            test_out = model_test(dummy_image_640)
        for t, r in zip(train_out, test_out):
            assert t.shape == r.shape


@pytest.mark.unit
class TestOurModelGradient:
    def test_gradient_flows(self, cfg_mnet, dummy_image_640):
        """Gradients must flow back to the input image during backpropagation (train mode)."""
        model = OurModel(cfg=cfg_mnet, phase="train")
        x = dummy_image_640.requires_grad_(True)
        bbox, cls, ldm = model(x)
        loss = bbox.sum() + cls.sum() + ldm.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
