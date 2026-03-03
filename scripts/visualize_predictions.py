#!/usr/bin/env python3
"""Visualize model predictions."""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize predictions')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to original image')
    parser.add_argument('--prediction', type=str, required=True,
                       help='Path to prediction mask')
    parser.add_argument('--ground_truth', type=str, default=None,
                       help='Path to ground truth mask (optional)')
    parser.add_argument('--output', type=str, default='visualization.png',
                       help='Output path for visualization')
    return parser.parse_args()


def visualize(image_path, pred_path, gt_path=None, output_path='visualization.png'):
    """Create visualization of predictions."""
    
    # Load images
    image = np.array(Image.open(image_path).convert('RGB'))
    pred = np.array(Image.open(pred_path).convert('RGB'))
    
    # Create figure
    if gt_path:
        gt = np.array(Image.open(gt_path).convert('RGB'))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred)
        axes[2].set_title('Prediction')
        axes[2].axis('off')
        
        # Overlay
        overlay = (image * 0.5 + pred * 0.5).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Overlay
        overlay = (image * 0.5 + pred * 0.5).astype(np.uint8)
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    plt.close()


def main():
    args = parse_args()
    visualize(args.image, args.prediction, args.ground_truth, args.output)


if __name__ == "__main__":
    main()
