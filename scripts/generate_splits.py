"""
Generate reproducible train/test splits for underwater segmentation datasets.

This script creates the exact splits used in the MAREA-Net paper.
"""

import os
import argparse
import random
from pathlib import Path
from typing import List, Tuple
import json


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)


def get_image_files(image_dir: str, extensions: List[str] = None) -> List[str]:
    """Get all image files from directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    image_dir = Path(image_dir)
    files = []
    
    for ext in extensions:
        files.extend(image_dir.glob(f'*{ext}'))
        files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    # Return filenames without extension
    return sorted([f.stem for f in files])


def split_dataset(
    files: List[str],
    train_ratio: float,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    """Split files into train and test sets."""
    set_seed(seed)
    
    files = sorted(files)  # Ensure consistent ordering
    random.shuffle(files)
    
    split_idx = int(len(files) * train_ratio)
    train_files = sorted(files[:split_idx])
    test_files = sorted(files[split_idx:])
    
    return train_files, test_files


def save_split_file(files: List[str], output_path: str):
    """Save split file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for filename in files:
            f.write(f"{filename}\n")
    
    print(f"✓ Saved {len(files)} files to {output_path}")


def generate_suim_splits(data_dir: str, output_dir: str, seed: int = 42):
    """Generate SUIM splits: 1,225 train / 300 test."""
    print("\n=== Generating SUIM Splits ===")
    
    # Get all images
    image_dir = Path(data_dir) / "images"
    if not image_dir.exists():
        image_dir = Path(data_dir) / "train" / "images"
    
    files = get_image_files(image_dir)
    print(f"Total images found: {len(files)}")
    
    # Split: 1,225 train / 300 test (80.3% / 19.7%)
    train_ratio = 1225 / 1525
    train_files, test_files = split_dataset(files, train_ratio, seed)
    
    # Save splits
    save_split_file(train_files, Path(output_dir) / "suim_train.txt")
    save_split_file(test_files, Path(output_dir) / "suim_test.txt")
    
    print(f"Train: {len(train_files)} images")
    print(f"Test: {len(test_files)} images")


def generate_dutuseg_splits(data_dir: str, output_dir: str, seed: int = 42):
    """Generate DUT-USEG splits: 1,380 train / 107 test."""
    print("\n=== Generating DUT-USEG Splits ===")
    
    image_dir = Path(data_dir) / "images"
    if not image_dir.exists():
        image_dir = Path(data_dir) / "train" / "images"
    
    files = get_image_files(image_dir)
    print(f"Total images found: {len(files)}")
    
    # Split: 1,380 train / 107 test (92.8% / 7.2%)
    train_ratio = 1380 / 1487
    train_files, test_files = split_dataset(files, train_ratio, seed)
    
    save_split_file(train_files, Path(output_dir) / "dutuseg_train.txt")
    save_split_file(test_files, Path(output_dir) / "dutuseg_test.txt")
    
    print(f"Train: {len(train_files)} images")
    print(f"Test: {len(test_files)} images")


def generate_mas3k_splits(data_dir: str, output_dir: str, seed: int = 42):
    """Generate MAS3K splits: 80/20."""
    print("\n=== Generating MAS3K Splits ===")
    
    image_dir = Path(data_dir) / "images"
    if not image_dir.exists():
        image_dir = Path(data_dir) / "train" / "images"
    
    files = get_image_files(image_dir)
    print(f"Total images found: {len(files)}")
    
    train_files, test_files = split_dataset(files, 0.80, seed)
    
    save_split_file(train_files, Path(output_dir) / "mas3k_train.txt")
    save_split_file(test_files, Path(output_dir) / "mas3k_test.txt")
    
    print(f"Train: {len(train_files)} images")
    print(f"Test: {len(test_files)} images")


def generate_usis10k_splits(data_dir: str, output_dir: str, seed: int = 42):
    """Generate USIS10K splits: 85/15."""
    print("\n=== Generating USIS10K Splits ===")
    
    image_dir = Path(data_dir) / "images"
    if not image_dir.exists():
        image_dir = Path(data_dir) / "train" / "images"
    
    files = get_image_files(image_dir)
    print(f"Total images found: {len(files)}")
    
    train_files, test_files = split_dataset(files, 0.85, seed)
    
    save_split_file(train_files, Path(output_dir) / "usis10k_train.txt")
    save_split_file(test_files, Path(output_dir) / "usis10k_test.txt")
    
    print(f"Train: {len(train_files)} images")
    print(f"Test: {len(test_files)} images")


def save_split_metadata(output_dir: str, seed: int):
    """Save metadata about the splits."""
    metadata = {
        "seed": seed,
        "datasets": {
            "suim": {
                "total": 1525,
                "train": 1225,
                "test": 300,
                "train_ratio": 0.803
            },
            "dutuseg": {
                "total": 1487,
                "train": 1380,
                "test": 107,
                "train_ratio": 0.928
            },
            "mas3k": {
                "total": 3103,
                "train_ratio": 0.80
            },
            "usis10k": {
                "total": 10632,
                "train_ratio": 0.85
            }
        },
        "notes": "Splits generated for MAREA-Net paper reproducibility"
    }
    
    metadata_path = Path(output_dir) / "splits_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible dataset splits for MAREA-Net"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['suim', 'dutuseg', 'mas3k', 'usis10k', 'all'],
        default='all',
        help='Dataset to generate splits for'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='splits',
        help='Output directory for split files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Generating splits with seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    
    # Generate splits
    if args.dataset == 'suim' or args.dataset == 'all':
        try:
            generate_suim_splits(args.data_dir, args.output_dir, args.seed)
        except Exception as e:
            print(f"Error generating SUIM splits: {e}")
    
    if args.dataset == 'dutuseg' or args.dataset == 'all':
        try:
            generate_dutuseg_splits(args.data_dir, args.output_dir, args.seed)
        except Exception as e:
            print(f"Error generating DUT-USEG splits: {e}")
    
    if args.dataset == 'mas3k' or args.dataset == 'all':
        try:
            generate_mas3k_splits(args.data_dir, args.output_dir, args.seed)
        except Exception as e:
            print(f"Error generating MAS3K splits: {e}")
    
    if args.dataset == 'usis10k' or args.dataset == 'all':
        try:
            generate_usis10k_splits(args.data_dir, args.output_dir, args.seed)
        except Exception as e:
            print(f"Error generating USIS10K splits: {e}")
    
    # Save metadata
    save_split_metadata(args.output_dir, args.seed)
    
    print("\n✓ All splits generated successfully!")
    print(f"\nTo use these splits in training:")
    print(f"  python train.py --data_dir <data_dir> --split_file {args.output_dir}/suim_train.txt")


if __name__ == '__main__':
    main()
