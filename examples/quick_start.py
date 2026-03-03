#!/usr/bin/env python3
"""Quick start example for MAREA-Net."""

import tensorflow as tf
from marea_net import build_marea_net, Config

# Create configuration
config = Config()

# Build model
print("Building MAREA-Net...")
model = build_marea_net(config)
model.summary()

# Example inference
print("\nRunning example inference...")
dummy_input = tf.random.normal([1, 320, 320, 3])
seg_output, aux_output, plant_output = model(dummy_input, training=False)

print(f"Segmentation output shape: {seg_output.shape}")
print(f"Auxiliary output shape: {aux_output.shape}")
print(f"Plant presence output shape: {plant_output.shape}")

print("\n✅ MAREA-Net is ready to use!")
