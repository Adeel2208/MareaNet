#!/usr/bin/env python3
"""Inference script for MAREA-Net."""

import argparse
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm

from marea_net.config import Config
from marea_net.utils import load_model, configure_gpu


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with MAREA-Net')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing input images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save predictions')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--tta', action='store_true',
                       help='Use test-time augmentation')
    parser.add_argument('--save_overlay', action='store_true',
                       help='Save overlay visualization')
    return parser.parse_args()


def preprocess_image(img_path, config):
    """Preprocess single image for inference."""
    img_h = config.get('model.input_height', 320)
    img_w = config.get('model.input_width', 320)
    
    img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, 
                                 dtype=tf.float32, expand_animations=False)
    img.set_shape([None, None, 3])
    original_shape = img.shape[:2]
    
    img = tf.image.resize(img, [img_h, img_w], method='bilinear', antialias=True)
    img = tf.keras.applications.convnext.preprocess_input(img * 255.0)
    
    return tf.expand_dims(img, 0), original_shape


def class_to_rgb(mask_cls, palette):
    """Convert class indices to RGB mask."""
    h, w = mask_cls.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_id, color in enumerate(palette):
        rgb[mask_cls == cls_id] = color
    
    return rgb


def create_overlay(image, mask_rgb, alpha=0.5):
    """Create overlay of mask on original image."""
    return (image * (1 - alpha) + mask_rgb * alpha).astype(np.uint8)


def inference(model, config, args):
    """Run inference on all images in input directory."""
    os.makedirs(args.output_dir, exist_ok=True)
    
    palette = config.get('classes.palette')
    img_files = [f for f in os.listdir(args.input_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    print(f"\n🔍 Found {len(img_files)} images in {args.input_dir}")
    print(f"💾 Saving predictions to {args.output_dir}")
    
    for img_file in tqdm(img_files, desc="Processing"):
        img_path = os.path.join(args.input_dir, img_file)
        
        # Preprocess
        img_tensor, original_shape = preprocess_image(img_path, config)
        
        # Predict
        if args.tta:
            # Simple TTA: original + horizontal flip
            pred1 = model(img_tensor, training=False)[0]
            pred2 = tf.image.flip_left_right(
                model(tf.image.flip_left_right(img_tensor), training=False)[0]
            )
            pred = (pred1 + pred2) / 2.0
        else:
            pred = model(img_tensor, training=False)[0]
        
        # Get class predictions
        pred_cls = tf.argmax(pred[0], axis=-1).numpy().astype(np.uint8)
        
        # Resize to original size
        pred_cls_resized = tf.image.resize(
            tf.expand_dims(pred_cls, -1),
            original_shape,
            method='nearest'
        ).numpy().squeeze().astype(np.uint8)
        
        # Convert to RGB
        mask_rgb = class_to_rgb(pred_cls_resized, palette)
        
        # Save prediction
        base_name = os.path.splitext(img_file)[0]
        mask_path = os.path.join(args.output_dir, f"{base_name}_pred.png")
        Image.fromarray(mask_rgb).save(mask_path)
        
        # Save overlay if requested
        if args.save_overlay:
            original_img = np.array(Image.open(img_path).convert('RGB'))
            overlay = create_overlay(original_img, mask_rgb)
            overlay_path = os.path.join(args.output_dir, f"{base_name}_overlay.png")
            Image.fromarray(overlay).save(overlay_path)
    
    print(f"\n✅ Inference complete! Processed {len(img_files)} images.")


def main():
    args = parse_args()
    config = Config(args.config)
    
    # Configure GPU
    configure_gpu(config)
    
    # Load model
    print(f"\n📦 Loading model from {args.model_path}...")
    model = load_model(args.model_path)
    print(f"✅ Model loaded successfully")
    
    # Run inference
    inference(model, config, args)


if __name__ == "__main__":
    main()
