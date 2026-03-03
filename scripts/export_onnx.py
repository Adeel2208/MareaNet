#!/usr/bin/env python3
"""Export MAREA-Net model to ONNX format."""

import argparse
import tensorflow as tf
import tf2onnx

from marea_net.utils import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained Keras model')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output path for ONNX model')
    parser.add_argument('--opset', type=int, default=13,
                       help='ONNX opset version')
    return parser.parse_args()


def export_to_onnx(model_path, output_path, opset=13):
    """Export Keras model to ONNX format."""
    
    print(f"📦 Loading model from {model_path}...")
    model = load_model(model_path)
    
    print(f"🔄 Converting to ONNX (opset {opset})...")
    
    # Get input signature
    input_signature = [tf.TensorSpec([None, 320, 320, 3], tf.float32, name='input')]
    
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=opset,
        output_path=output_path
    )
    
    print(f"✅ ONNX model saved to {output_path}")
    print(f"   Model inputs: {[i.name for i in onnx_model.graph.input]}")
    print(f"   Model outputs: {[o.name for o in onnx_model.graph.output]}")


def main():
    args = parse_args()
    
    try:
        export_to_onnx(args.model_path, args.output_path, args.opset)
    except ImportError:
        print("❌ tf2onnx not installed. Install with: pip install tf2onnx")
    except Exception as e:
        print(f"❌ Export failed: {e}")


if __name__ == "__main__":
    main()
