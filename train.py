#!/usr/bin/env python3
"""Training script for MAREA-Net."""

import argparse
import time
import gc
import numpy as np
import tensorflow as tf

from marea_net.config import Config
from marea_net.model import build_marea_net, set_encoder_trainable
from marea_net.data import build_dataset, find_rare_class_samples
from marea_net.losses import compute_class_weights, create_loss_fn
from marea_net.metrics import evaluate_miou
from marea_net.utils import (configure_gpu, lr_schedule, make_optimizer,
                              make_train_step, save_model)


def parse_args():
    parser = argparse.ArgumentParser(description='Train MAREA-Net')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory containing train/test folders')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Directory to save trained models')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to model checkpoint to resume from')
    return parser.parse_args()


def train(config, args):
    """Main training loop."""
    print("=" * 80)
    print("  MAREA-Net v5.5-CNX  (ConvNeXtBase backbone)")
    print("=" * 80)
    
    # Override config with command-line args
    if args.epochs:
        config.config['training']['epochs'] = args.epochs
    if args.batch_size:
        config.config['training']['batch_size'] = args.batch_size
    
    # Update data paths
    config.config['data']['train_images'] = f"{args.data_dir}/train/images"
    config.config['data']['train_masks'] = f"{args.data_dir}/train/masks"
    config.config['data']['test_images'] = f"{args.data_dir}/test/images"
    config.config['data']['test_masks'] = f"{args.data_dir}/test/masks"
    
    # Configure GPU
    configure_gpu(config)
    
    # Find rare-class samples for oversampling
    rare_color = tuple(config.get('training.rare_class_color', [0, 255, 0]))
    print(f"\n📂 Scanning for rare-class images (colour {rare_color})...")
    rare_samples = find_rare_class_samples(
        config.get('data.train_images'),
        config.get('data.train_masks'),
        rare_color=rare_color,
    )
    oversample = config.get('training.oversample_plant', 2)
    print(f"  Rare-class donors: {len(rare_samples)} images (oversample ×{oversample - 1})")
    
    # Build datasets
    print("\n📂 Building datasets...")
    train_ds, n_train = build_dataset(
        config.get('data.train_images'),
        config.get('data.train_masks'),
        config,
        training=True,
        plant_samples=rare_samples,
    )
    test_ds, _ = build_dataset(
        config.get('data.test_images'),
        config.get('data.test_masks'),
        config,
        training=False
    )
    
    # Compute class weights
    print("\n⚖️  Computing class weights...")
    class_weights = compute_class_weights(
        next(iter(train_ds.take(1)))[1].numpy(),
        config
    )
    
    # Build or load model
    print("\n🏗️  Building model...")
    if args.resume:
        from marea_net.utils import load_model
        model = load_model(args.resume)
        print(f"  Loaded checkpoint from {args.resume}")
    else:
        model = build_marea_net(config)
    
    print(f"  Total params: {model.count_params():,}")
    
    # Create loss and optimizer
    loss_fn = create_loss_fn(class_weights, config)
    optimizer = make_optimizer(config.get('training.lr_warmup'), config)
    model.compile(optimizer=optimizer)
    
    # Training setup
    epochs = config.get('training.epochs', 100)
    unfreeze_epoch = config.get('training.unfreeze_epoch', 10)
    batch_size = config.get('training.batch_size', 8)
    steps_per_epoch = n_train // batch_size
    
    best_miou = 0.0
    best_weights = None
    history = {'loss': [], 'val_miou': [], 'lr': []}
    
    # Phase 1: Frozen encoder
    print(f"\n🧊 Phase 1: encoder frozen (epochs 0–{unfreeze_epoch-1})...")
    set_encoder_trainable(model, trainable=False)
    train_step = make_train_step(model, optimizer, loss_fn)
    
    # Training loop
    for epoch in range(epochs):
        # Phase 2: Unfreeze encoder
        if epoch == unfreeze_epoch:
            print(f"\n🔓 Phase 2: unfreezing full model at epoch {epoch}...")
            set_encoder_trainable(model, trainable=True)
            current_lr = float(optimizer.learning_rate)
            optimizer = make_optimizer(current_lr, config)
            model.compile(optimizer=optimizer)
            train_step = make_train_step(model, optimizer, loss_fn)
            print(f"  Fresh AdamW created (lr={current_lr:.2e})")
        
        # Update learning rate
        lr = lr_schedule(epoch, config)
        optimizer.learning_rate.assign(lr)
        history['lr'].append(lr)
        
        # Training epoch
        losses = []
        t0 = time.time()

        for step, (xb, yb) in enumerate(train_ds.take(steps_per_epoch)):
            losses.append(float(train_step(xb, yb)))
            
            if step % 20 == 0 and step > 0:
                print(f"    step {step}/{steps_per_epoch} | "
                      f"loss={np.mean(losses):.4f}", end='\r')
        
        avg_loss = float(np.mean(losses)) if losses else 0.0
        history['loss'].append(avg_loss)
        
        # Validation
        if epoch % 3 == 0 or epoch >= epochs - 5:
            val_miou = evaluate_miou(model, test_ds, config.get('model.num_classes'))
            history['val_miou'].append(val_miou)
            
            tag = ""
            if val_miou > best_miou:
                best_miou = val_miou
                best_weights = model.get_weights()
                tag = " ✨ BEST"
            
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"val_mIoU={val_miou:.4f} | lr={lr:.2e} | "
                  f"{time.time()-t0:.1f}s{tag}")
        else:
            print(f"  Epoch {epoch:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"lr={lr:.2e} | {time.time()-t0:.1f}s")
        
        # Periodic garbage collection
        if epoch % 10 == 0:
            gc.collect()
    
    # Restore best weights
    if best_weights:
        model.set_weights(best_weights)
        print(f"\n✅ Best weights restored (val mIoU = {best_miou:.4f})")
    
    # Save model
    save_path = f"{args.output_dir}/marea_net_best.keras"
    save_model(model, save_path, config)
    
    print(f"\n✅ Training complete. Best val mIoU = {best_miou:.4f}")
    
    return model, history


def main():
    args = parse_args()
    config = Config(args.config)
    
    model, history = train(config, args)
    
    print("\n🎉 Training finished successfully!")


if __name__ == "__main__":
    main()
