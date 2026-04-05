"""
tests/utils/test_nms.py
Tests for the py_cpu_nms Non-Maximum Suppression implementation.
"""

import pytest
import numpy as np
from src.utils.nms.py_cpu_nms import py_cpu_nms


@pytest.mark.unit
class TestPyCpuNMS:
    def test_returns_list(self):
        """Return type must be a list."""
        dets = np.array([[0.0, 0.0, 1.0, 1.0, 0.9]])
        keep = py_cpu_nms(dets, thresh=0.5)
        assert isinstance(keep, list)

    def test_single_box_kept(self):
        """A single detection must always be kept."""
        dets = np.array([[10.0, 10.0, 50.0, 50.0, 0.9]])
        keep = py_cpu_nms(dets, thresh=0.5)
        assert len(keep) == 1
        assert keep[0] == 0

    def test_identical_boxes_suppress(self):
        """Perfectly overlapping boxes must be suppressed down to the highest-score one."""
        dets = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9],
                [0.0, 0.0, 10.0, 10.0, 0.8],
                [0.0, 0.0, 10.0, 10.0, 0.7],
            ]
        )
        keep = py_cpu_nms(dets, thresh=0.5)
        assert len(keep) == 1

    def test_nonoverlapping_boxes_all_kept(self):
        """Non-overlapping boxes must all be kept."""
        dets = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9],
                [20.0, 0.0, 30.0, 10.0, 0.85],
                [40.0, 0.0, 50.0, 10.0, 0.8],
            ]
        )
        keep = py_cpu_nms(dets, thresh=0.5)
        assert len(keep) == 3

    def test_high_threshold_keeps_more(self):
        """A higher threshold must keep at least as many boxes as a lower threshold."""
        dets = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.9],
                [2.0, 0.0, 12.0, 10.0, 0.8],  # partially overlapping
            ]
        )
        keep_strict = py_cpu_nms(dets, thresh=0.1)
        keep_loose = py_cpu_nms(dets, thresh=0.9)
        assert len(keep_strict) <= len(keep_loose)

    def test_highest_score_always_kept(self):
        """The box with the highest score must always be kept."""
        dets = np.array(
            [
                [0.0, 0.0, 10.0, 10.0, 0.95],  # <- highest score
                [0.0, 0.0, 10.0, 10.0, 0.80],
                [5.0, 5.0, 15.0, 15.0, 0.70],
            ]
        )
        keep = py_cpu_nms(dets, thresh=0.5)
        assert 0 in keep

    def test_score_ordering(self):
        """Returned indices must be in descending score order."""
        dets = np.array(
            [
                [0.0, 0.0, 5.0, 5.0, 0.6],  # idx 0
                [10.0, 10.0, 15.0, 15.0, 0.9],  # idx 1
                [20.0, 20.0, 25.0, 25.0, 0.8],  # idx 2
            ]
        )
        keep = py_cpu_nms(dets, thresh=0.5)
        scores_kept = dets[keep, 4]
        assert all(
            scores_kept[i] >= scores_kept[i + 1] for i in range(len(scores_kept) - 1)
        )

    def test_empty_input(self):
        """Empty input array must be handled without error and return an empty list."""
        dets = np.zeros((0, 5))
        keep = py_cpu_nms(dets, thresh=0.5)
        assert len(keep) == 0
