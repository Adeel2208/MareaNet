"""Unit tests for MAREA-Net model construction and forward pass."""

import numpy as np
import pytest
import tensorflow as tf

from marea_net import build_marea_net, Config


@pytest.fixture(scope="module")
def model():
    """Build model once for the whole module."""
    cfg = Config()
    # Use 2 classes to keep the test fast (avoids downloading full ImageNet weights
    # for every class variant); the architecture is class-count agnostic.
    cfg.config["model"]["num_classes"] = 2
    return build_marea_net(cfg)


@pytest.fixture(scope="module")
def default_model():
    """Build the default 8-class SUIM model."""
    return build_marea_net()


class TestModelBuild:
    def test_returns_keras_model(self, model):
        assert isinstance(model, tf.keras.Model)

    def test_three_outputs(self, model):
        assert len(model.outputs) == 3

    def test_output_names(self, model):
        names = [o.name.split("/")[0] for o in model.outputs]
        assert "seg_output" in names
        assert "aux_output" in names
        assert "presence_output" in names

    def test_param_count_reasonable(self, default_model):
        """Total params should be ~93M (88M encoder + ~5M decoder)."""
        n = default_model.count_params()
        assert 80_000_000 < n < 110_000_000, f"Unexpected param count: {n:,}"


class TestForwardPass:
    def test_seg_output_shape(self, default_model):
        x = tf.random.normal([1, 320, 320, 3])
        seg, aux, pres = default_model(x, training=False)
        assert seg.shape == (1, 320, 320, 8)

    def test_aux_output_shape(self, default_model):
        x = tf.random.normal([1, 320, 320, 3])
        seg, aux, pres = default_model(x, training=False)
        assert aux.shape == (1, 320, 320, 8)

    def test_presence_output_shape(self, default_model):
        x = tf.random.normal([1, 320, 320, 3])
        seg, aux, pres = default_model(x, training=False)
        assert pres.shape == (1, 1)

    def test_seg_probabilities_sum_to_one(self, default_model):
        x = tf.random.normal([2, 320, 320, 3])
        seg, _, _ = default_model(x, training=False)
        sums = tf.reduce_sum(seg, axis=-1).numpy()
        np.testing.assert_allclose(sums, np.ones_like(sums), atol=1e-5)

    def test_presence_in_zero_one(self, default_model):
        x = tf.random.normal([2, 320, 320, 3])
        _, _, pres = default_model(x, training=False)
        assert tf.reduce_all(pres >= 0.0) and tf.reduce_all(pres <= 1.0)

    def test_batch_size_two(self, default_model):
        x = tf.random.normal([2, 320, 320, 3])
        seg, aux, pres = default_model(x, training=False)
        assert seg.shape[0] == 2

    def test_training_mode_runs(self, default_model):
        x = tf.random.normal([1, 320, 320, 3])
        seg, aux, pres = default_model(x, training=True)
        assert seg.shape == (1, 320, 320, 8)
