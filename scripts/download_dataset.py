#!/usr/bin/env python3
"""Script to download and prepare SUIM dataset."""

import os
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Download SUIM dataset')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Directory to save dataset')
    return parser.parse_args()


def download_suim(output_dir):
    """
    Download SUIM dataset.
    
    Note: This is a placeholder. You need to:
    1. Visit: http://irvlab.cs.umn.edu/resources/suim-dataset
    2. Download the dataset manually
    3. Extract to the output_dir
    """
    print("=" * 80)
    print("SUIM Dataset Download Instructions")
    print("=" * 80)
    print("\nThe SUIM dataset must be downloaded manually:")
    print("\n1. Visit: http://irvlab.cs.umn.edu/resources/suim-dataset")
    print("2. Download 'SUIM.zip'")
    print("3. Extract to:", os.path.abspath(output_dir))
    print("\nExpected structure:")
    print(f"{output_dir}/")
    print("├── train/")
    print("│   ├── images/")
    print("│   └── masks/")
    print("└── test/")
    print("    ├── images/")
    print("    └── masks/")
    print("\n" + "=" * 80)


def verify_dataset(data_dir):
    """Verify dataset structure."""
    required_dirs = [
        'train/images',
        'train/masks',
        'test/images',
        'test/masks'
    ]
    
    print("\n🔍 Verifying dataset structure...")
    all_exist = True
    
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        exists = os.path.exists(full_path)
        status = "✅" if exists else "❌"
        print(f"  {status} {dir_path}")
        
        if exists:
            n_files = len(os.listdir(full_path))
            print(f"      ({n_files} files)")
        
        all_exist = all_exist and exists
    
    if all_exist:
        print("\n✅ Dataset structure verified!")
    else:
        print("\n❌ Dataset structure incomplete. Please check the paths.")
    
    return all_exist


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Show download instructions
    download_suim(args.output_dir)
    
    # Verify if dataset already exists
    if os.path.exists(os.path.join(args.output_dir, 'train')):
        verify_dataset(args.output_dir)


if __name__ == "__main__":
    main()
