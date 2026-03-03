#!/usr/bin/env python3
"""Evaluation script for MAREA-Net."""

import argparse
import numpy as np

from marea_net.config import Config
from marea_net.data import build_dataset
from marea_net.metrics import tta_predict, compute_metrics
from marea_net.utils import load_model, configure_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MAREA-Net')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--test_dir', type=str, required=True,
                       help='Root directory containing test/images and test/masks')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--compare_baseline', type=str, default=None,
                       help='Path to baseline results for comparison')
    return parser.parse_args()


def print_comparison(label, baseline_ious, current_ious, class_names):
    """Print comparison with baseline results."""
    print(f"\n📊 vs {label} — avg {np.mean(baseline_ious):.4f}")
    print(f"  {'Class':20s} {'Current':>8s} {'Baseline':>8s} {'Δ':>8s}")
    
    for c in range(len(class_names)):
        d = current_ious[c] - baseline_ious[c]
        print(f"  {class_names[c]:20s} {current_ious[c]:8.4f} "
              f"{baseline_ious[c]:8.4f} {d:+8.4f} "
              f"{'✅' if d > 0 else '❌'}")
    
    avg_current = np.mean(current_ious)
    avg_baseline = np.mean(baseline_ious)
    print(f"  {'AVERAGE':20s} {avg_current:8.4f} {avg_baseline:8.4f} "
          f"{avg_current - avg_baseline:+8.4f}")


def evaluate(model, config, args):
    """Run full evaluation with metrics."""
    print("\n" + "═" * 85)
    print("  TEST — MAREA-Net v5.5-CNX (ConvNeXtBase)")
    if args.tta:
        n_aug = config.get('inference.tta_augmentations', 8)
        print(f"  Using {n_aug}-fold Test-Time Augmentation")
    print("═" * 85)
    
    # Update data paths
    config.config['data']['test_images'] = f"{args.test_dir}/images"
    config.config['data']['test_masks'] = f"{args.test_dir}/masks"
    
    # Build test dataset
    print("\n📂 Building test dataset...")
    test_ds, _ = build_dataset(
        config.get('data.test_images'),
        config.get('data.test_masks'),
        config,
        training=False
    )
    
    # Run prediction
    if args.tta:
        print("\n🔄 Running TTA prediction...")
        pred_cls, y_test = tta_predict(model, test_ds, config)
    else:
        print("\n🔍 Running standard prediction...")
        all_preds, all_gt = [], []
        for xb, yb in test_ds:
            pred = model(xb, training=False)[0]
            all_preds.append(np.argmax(pred.numpy(), axis=-1))
            all_gt.append(yb.numpy())
        pred_cls = np.concatenate(all_preds, axis=0)
        y_test = np.concatenate(all_gt, axis=0)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    class_names = config.get('classes.names')
    metrics = compute_metrics(pred_cls, y_test, class_names)
    
    # Compare with baselines
    if args.compare_baseline:
        baseline_data = np.load(args.compare_baseline, allow_pickle=True).item()
        baseline_ious = baseline_data['ious']
        baseline_name = baseline_data.get('name', 'Baseline')
        current_ious = [m['iou'] for m in metrics['per_class']]
        print_comparison(baseline_name, baseline_ious, current_ious, class_names)
    else:
        # Compare with known baselines
        print("\n📊 Comparison with published results:")
        
        # AquaSAM baseline
        aquasam_ious = [0.8234, 0.7782, 0.4292, 0.7885, 0.8408, 0.7582, 0.6815, 0.6808]
        current_ious = [m['iou'] for m in metrics['per_class']]
        print_comparison("AquaSAM (Xu et al. 2024)", aquasam_ious, current_ious, class_names)
        
        # MAREA-Net v5.5-GPU (ResNet50) expected
        marea_v55_ious = [0.8800, 0.7800, 0.3500, 0.7200, 0.7300, 0.6800, 0.7300, 0.6700]
        print_comparison("MAREA-Net v5.5-GPU (ResNet50)", marea_v55_ious, current_ious, class_names)
    
    return metrics


def main():
    args = parse_args()
    config = Config(args.config)
    
    # Configure GPU
    configure_gpu(config)
    
    # Load model
    print(f"\n📦 Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    print(f"✅ Model loaded successfully")
    
    # Run evaluation
    metrics = evaluate(model, config, args)
    
    print(f"\n✅ Evaluation complete!")
    print(f"   Final mIoU: {metrics['overall']['miou']:.4f}")
    print(f"   Final mF1:  {metrics['overall']['mf1']:.4f}")


if __name__ == "__main__":
    main()
