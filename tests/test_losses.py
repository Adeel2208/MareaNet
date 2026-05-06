"""Unit tests for MAREA-Net loss functions."""

import numpy as np
import pytest
import tensorflow as tf

from marea_net.losses import (
    focal_ce_ohem,
    tversky_loss,
    dice_loss,
    plant_dice_loss,
    rare_class_presence_loss,
    create_loss_fn,
    compute_class_weights,
)
from marea_net.config import Config


N_CLS = 8
B, H, W = 2, 16, 16  # small spatial size for speed


def make_pred(n_cls=N_CLS):
    """Uniform softmax predictions."""
    logits = tf.random.normal([B, H, W, n_cls])
    return tf.nn.softmax(logits, axis=-1)


def make_true(n_cls=N_CLS):
    return tf.cast(tf.random.uniform([B, H, W], 0, n_cls, dtype=tf.int32), tf.uint8)


class TestFocalCEOHEM:
    def test_scalar_output(self):
        cw = tf.ones(N_CLS)
        gm = tf.ones(N_CLS) * 2.0
        loss = focal_ce_ohem(make_true(), make_pred(), cw, gm, 0.7, 0.05)
        assert loss.shape == ()

    def test_non_negative(self):
        cw = tf.ones(N_CLS)
        gm = tf.ones(N_CLS) * 2.0
        loss = focal_ce_ohem(make_true(), make_pred(), cw, gm, 0.7, 0.05)
        assert float(loss) >= 0.0

    def test_perfect_prediction_lower_than_random(self):
        """Perfect predictions should yield lower loss than random."""
        y_true = make_true()
        y_perfect = tf.one_hot(tf.cast(y_true, tf.int32), N_CLS)
        y_random = make_pred()
        cw = tf.ones(N_CLS)
        gm = tf.ones(N_CLS) * 2.0
        loss_perfect = focal_ce_ohem(y_true, y_perfect, cw, gm, 0.7, 0.0)
        loss_random = focal_ce_ohem(y_true, y_random, cw, gm, 0.7, 0.0)
        assert float(loss_perfect) < float(loss_random)


class TestTverskyLoss:
    def test_scalar_output(self):
        loss = tversky_loss(make_true(), make_pred(), 0.3, 0.7, N_CLS)
        assert loss.shape == ()

    def test_range(self):
        loss = tversky_loss(make_true(), make_pred(), 0.3, 0.7, N_CLS)
        assert 0.0 <= float(loss) <= 1.0 + 1e-5

    def test_perfect_prediction_near_zero(self):
        y_true = make_true()
        y_perfect = tf.one_hot(tf.cast(y_true, tf.int32), N_CLS)
        loss = tversky_loss(y_true, y_perfect, 0.3, 0.7, N_CLS)
        assert float(loss) < 0.05


class TestDiceLoss:
    def test_scalar_output(self):
        loss = dice_loss(make_true(), make_pred(), N_CLS)
        assert loss.shape == ()

    def test_range(self):
        loss = dice_loss(make_true(), make_pred(), N_CLS)
        assert 0.0 <= float(loss) <= 1.0 + 1e-5


class TestRareClassPresenceLoss:
    def test_scalar_output(self):
        y_true = make_true()
        y_pred = tf.random.uniform([B, 1])
        loss = rare_class_presence_loss(y_true, y_pred, rare_class_index=2)
        assert loss.shape == ()

    def test_non_negative(self):
        y_true = make_true()
        y_pred = tf.random.uniform([B, 1])
        loss = rare_class_presence_loss(y_true, y_pred, rare_class_index=2)
        assert float(loss) >= 0.0

    def test_plant_presence_alias(self):
        from marea_net.losses import plant_presence_loss
        assert plant_presence_loss is rare_class_presence_loss


class TestCreateLossFn:
    def test_returns_callable(self):
        cfg = Config()
        cw = np.ones(N_CLS, dtype=np.float32)
        fn = create_loss_fn(cw, cfg)
        assert callable(fn)

    def test_loss_is_scalar(self):
        cfg = Config()
        cw = np.ones(N_CLS, dtype=np.float32)
        fn = create_loss_fn(cw, cfg)
        y_true = make_true()
        preds = (make_pred(), make_pred(), tf.random.uniform([B, 1]))
        loss = fn(y_true, preds)
        assert loss.shape == ()

    def test_loss_non_negative(self):
        cfg = Config()
        cw = np.ones(N_CLS, dtype=np.float32)
        fn = create_loss_fn(cw, cfg)
        y_true = make_true()
        preds = (make_pred(), make_pred(), tf.random.uniform([B, 1]))
        assert float(fn(y_true, preds)) >= 0.0
