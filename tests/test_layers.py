"""Unit tests for MAREA-Net custom layers."""

import numpy as np
import pytest
import tensorflow as tf

from marea_net.layers import (
    SimAM,
    StripPooling,
    SBCC,
    DSTS,
    CGAFusion,
    MAREADecoderBlock,
    ASPP_SP,
    WDTS,  # backward-compat alias
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rand(shape):
    return tf.random.normal(shape)


# ---------------------------------------------------------------------------
# SimAM
# ---------------------------------------------------------------------------

class TestSimAM:
    def test_output_shape(self):
        x = rand([2, 10, 10, 64])
        out = SimAM()(x)
        assert out.shape == x.shape

    def test_output_range(self):
        """SimAM multiplies by sigmoid — output should be ≤ input magnitude."""
        x = tf.ones([1, 8, 8, 32])
        out = SimAM()(x)
        assert tf.reduce_all(out <= x + 1e-5)

    def test_get_config(self):
        layer = SimAM(lam=1e-3)
        cfg = layer.get_config()
        assert cfg["lam"] == 1e-3


# ---------------------------------------------------------------------------
# StripPooling
# ---------------------------------------------------------------------------

class TestStripPooling:
    def test_output_shape(self):
        x = rand([2, 10, 10, 64])
        out = StripPooling(64)(x)
        assert out.shape == x.shape

    def test_get_config(self):
        layer = StripPooling(128)
        assert layer.get_config()["channels"] == 128


# ---------------------------------------------------------------------------
# SBCC
# ---------------------------------------------------------------------------

class TestSBCC:
    def test_output_shape(self):
        x = rand([2, 20, 20, 256])
        out = SBCC(256)(x)
        assert out.shape == x.shape

    def test_get_config(self):
        layer = SBCC(256)
        assert layer.get_config()["channels"] == 256


# ---------------------------------------------------------------------------
# DSTS (and WDTS alias)
# ---------------------------------------------------------------------------

class TestDSTS:
    def test_output_shape(self):
        x = rand([2, 40, 40, 128])
        out = DSTS(128)(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """With zero-initialised gate the output should equal the input."""
        layer = DSTS(16)
        x = rand([1, 8, 8, 16])
        # Build the layer
        _ = layer(x)
        # Zero out the gate weights so gate → 0 → output = x * 0 + x = x
        for w in layer.gate.weights:
            w.assign(tf.zeros_like(w))
        out = layer(x, training=False)
        np.testing.assert_allclose(out.numpy(), x.numpy(), atol=1e-5)

    def test_wdts_alias(self):
        assert WDTS is DSTS

    def test_get_config(self):
        layer = DSTS(128)
        assert layer.get_config()["channels"] == 128


# ---------------------------------------------------------------------------
# CGAFusion
# ---------------------------------------------------------------------------

class TestCGAFusion:
    def test_output_shape(self):
        dec = rand([2, 20, 20, 256])
        skip = rand([2, 20, 20, 512])
        out = CGAFusion(256)(dec, skip)
        assert out.shape == (2, 20, 20, 256)

    def test_get_config(self):
        layer = CGAFusion(64)
        assert layer.get_config()["channels"] == 64


# ---------------------------------------------------------------------------
# MAREADecoderBlock
# ---------------------------------------------------------------------------

class TestMAREADecoderBlock:
    def test_output_shape_plain(self):
        x = rand([2, 10, 10, 256])
        skip = rand([2, 20, 20, 512])
        out = MAREADecoderBlock(256)(x, skip)
        assert out.shape == (2, 20, 20, 256)

    def test_output_shape_with_sbcc(self):
        x = rand([2, 10, 10, 256])
        skip = rand([2, 20, 20, 512])
        out = MAREADecoderBlock(256, use_sbcc=True)(x, skip)
        assert out.shape == (2, 20, 20, 256)

    def test_output_shape_with_dsts(self):
        x = rand([2, 20, 20, 128])
        skip = rand([2, 40, 40, 256])
        out = MAREADecoderBlock(128, use_dsts=True)(x, skip)
        assert out.shape == (2, 40, 40, 128)

    def test_get_config(self):
        layer = MAREADecoderBlock(64, use_sbcc=True, use_dsts=False)
        cfg = layer.get_config()
        assert cfg["filters"] == 64
        assert cfg["use_sbcc"] is True
        assert cfg["use_dsts"] is False


# ---------------------------------------------------------------------------
# ASPP_SP
# ---------------------------------------------------------------------------

class TestASPP_SP:
    def test_output_shape(self):
        x = rand([2, 10, 10, 1024])
        out = ASPP_SP(out_channels=256)(x)
        assert out.shape == (2, 10, 10, 256)

    def test_get_config(self):
        layer = ASPP_SP(out_channels=128)
        assert layer.get_config()["out_channels"] == 128
