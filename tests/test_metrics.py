"""Unit tests for MAREA-Net evaluation metrics."""

import numpy as np
import pytest

from marea_net.metrics import compute_metrics


N_CLS = 8
CLASS_NAMES = [
    "Background", "Human Divers", "Aquatic Plants", "Wrecks/Ruins",
    "Robots/ROV", "Reefs/Inverts", "Fish/Vertebrates", "Sea-floor/Rocks",
]


def make_masks(n=10, h=32, w=32, n_cls=N_CLS):
    return np.random.randint(0, n_cls, (n, h, w), dtype=np.int32)


class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = make_masks()
        metrics = compute_metrics(y.copy(), y.copy(), CLASS_NAMES)
        assert abs(metrics["overall"]["miou"] - 1.0) < 1e-5

    def test_miou_range(self):
        y_true = make_masks()
        y_pred = make_masks()
        metrics = compute_metrics(y_pred, y_true, CLASS_NAMES)
        assert 0.0 <= metrics["overall"]["miou"] <= 1.0

    def test_mf1_range(self):
        y_true = make_masks()
        y_pred = make_masks()
        metrics = compute_metrics(y_pred, y_true, CLASS_NAMES)
        assert 0.0 <= metrics["overall"]["mf1"] <= 1.0

    def test_per_class_count(self):
        y_true = make_masks()
        y_pred = make_masks()
        metrics = compute_metrics(y_pred, y_true, CLASS_NAMES)
        assert len(metrics["per_class"]) == N_CLS

    def test_per_class_keys(self):
        y_true = make_masks()
        y_pred = make_masks()
        metrics = compute_metrics(y_pred, y_true, CLASS_NAMES)
        for entry in metrics["per_class"]:
            for key in ("class", "iou", "precision", "recall", "f1", "frequency"):
                assert key in entry

    def test_worst_case_all_wrong(self):
        """Predicting the wrong class for every pixel → IoU near 0."""
        y_true = np.zeros((5, 32, 32), dtype=np.int32)
        y_pred = np.ones((5, 32, 32), dtype=np.int32)
        metrics = compute_metrics(y_pred, y_true, CLASS_NAMES)
        # Background IoU should be 0, class-1 IoU should be 0
        assert metrics["per_class"][0]["iou"] < 1e-5
        assert metrics["per_class"][1]["iou"] < 1e-5
