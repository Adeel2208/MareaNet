#!/usr/bin/env python3
"""Example of custom training loop with MAREA-Net."""

import tensorflow as tf
from marea_net import build_marea_net, Config
from marea_net.data import build_dataset, find_plant_samples
from marea_net.losses import compute_class_weights, create_loss_fn
from marea_net.utils import configure_gpu, make_optimizer, make_train_step

# Configuration
config = Config('config/config.yaml')
configure_gpu(config)

# Update data paths
config.config['data']['train_images'] = 'data/train/images'
config.config['data']['train_masks'] = 'data/train/masks'

# Find plant samples
plant_samples = find_plant_samples(
    config.get('data.train_images'),
    config.get('data.train_masks')
)

# Build dataset
train_ds, n_train = build_dataset(
    config.get('data.train_images'),
    config.get('data.train_masks'),
    config,
    training=True,
    plant_samples=plant_samples
)

# Compute class weights
class_weights = compute_class_weights(
    next(iter(train_ds.take(1)))[1].numpy(),
    config
)

# Build model
model = build_marea_net(config)

# Create loss and optimizer
loss_fn = create_loss_fn(class_weights, config)
optimizer = make_optimizer(1e-4, config)
train_step = make_train_step(model, optimizer, loss_fn)

# Training loop
print("Starting custom training...")
for epoch in range(5):  # Train for 5 epochs
    losses = []
    for xb, yb in train_ds.take(10):  # 10 steps per epoch
        loss = train_step(xb, yb)
        losses.append(float(loss))
    
    print(f"Epoch {epoch+1}/5 | Loss: {sum(losses)/len(losses):.4f}")

print("\n✅ Custom training complete!")
