"""
tests/conftest.py
Shared pytest fixtures for all test modules.
"""

import pytest
import torch

# ──────────────────────────────────────────────
# Config fixtures
# ──────────────────────────────────────────────


@pytest.fixture(scope="session")
def cfg_mnet():
    """MobileNet 0.25 config with pretrain=False — tests structure only, no weight I/O."""
    return {
        "name": "mobilenet0.25",
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
        "loc_weight": 2.0,
        "gpu_train": False,
        "batch_size": 2,
        "ngpu": 1,
        "epoch": 250,
        "decay1": 190,
        "decay2": 220,
        "image_size": 640,
        "pretrain": False,  # skip pretrain weight loading during tests
        "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
        "in_channel": 32,
        "out_channel": 64,
    }


# ──────────────────────────────────────────────
# Dummy input fixtures
# ──────────────────────────────────────────────


@pytest.fixture
def dummy_image_640():
    """640x640 random image tensor (batch=1, RGB)."""
    return torch.randn(1, 3, 640, 640)


@pytest.fixture
def dummy_image_batch():
    """640x640 random image batch (batch=2, RGB)."""
    return torch.randn(2, 3, 640, 640)


# ──────────────────────────────────────────────
# Feature map fixtures (for layer-level tests)
# ──────────────────────────────────────────────


@pytest.fixture
def dummy_fmap_64ch():
    """64-channel 8x8 dummy feature map."""
    return torch.randn(1, 64, 8, 8)


@pytest.fixture
def dummy_fmap_128ch():
    """128-channel 8x8 dummy feature map."""
    return torch.randn(1, 128, 8, 8)
