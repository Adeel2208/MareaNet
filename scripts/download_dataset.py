#!/usr/bin/env python3
"""
Dataset setup instructions for MAREA-Net.

All four benchmarks used in the paper require manual download due to
licensing restrictions.  This script prints step-by-step instructions
and verifies the directory structure once the data is in place.

Usage
-----
    python scripts/download_dataset.py --dataset suim    --output_dir data/suim
    python scripts/download_dataset.py --dataset dutuseg --output_dir data/dutuseg
    python scripts/download_dataset.py --dataset mas3k   --output_dir data/mas3k
    python scripts/download_dataset.py --dataset usis10k --output_dir data/usis10k
    python scripts/download_dataset.py --dataset all     --output_dir data
"""

import argparse
import os


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------

DATASETS = {
    "suim": {
        "name": "SUIM",
        "citation": "Islam et al., IROS 2020",
        "url": "https://github.com/xahidbuffon/SUIM",
        "url_official": "http://irvlab.cs.umn.edu/resources/suim-dataset",
        "classes": 8,
        "train_images": 1525,
        "paper_split": "1,225 train / 300 test (our split)",
        "notes": (
            "Download 'SUIM.zip' from the official site or GitHub.  "
            "The archive contains a 'train_val' folder "
            "(1,525 images) and a separate 'TEST' folder (110 images).  "
            "For the paper split, randomly hold out 300 images from train_val "
            "as the test set and use the remaining 1,225 for training."
        ),
    },
    "dutuseg": {
        "name": "DUT-USEG",
        "citation": "Ma et al., JBUAA 2022",
        "url": "https://github.com/chongweiliu/DUT-USEG",
        "classes": 5,
        "train_images": 1380,
        "paper_split": "1,380 train / 107 test (UWSegFormer protocol)",
        "notes": (
            "Download the pixel-annotated subset (1,487 images) from GitHub.  "
            "Follow the UWSegFormer train/test split: 1,380 train, 107 test.  "
            "Rare class: scallop (class index 3, palette colour [0, 255, 0]).  "
            "Set rare_class_index=3 and rare_class_color=[0,255,0] in config."
        ),
    },
    "mas3k": {
        "name": "MAS3K",
        "citation": "Li et al., Bench 2021",
        "url": "https://github.com/LinLi-DL/MAS",
        "classes": 2,
        "train_images": 1769,
        "paper_split": "1,769 train / 1,141 test (default split, binary protocol)",
        "notes": (
            "Download the full MAS3K archive from GitHub.  "
            "Use the default 1,769/1,141 split.  "
            "Evaluate as binary marine-animal vs. background (num_classes=2).  "
            "Disable the rare-class presence head for this dataset."
        ),
    },
    "usis10k": {
        "name": "USIS10K",
        "citation": "Lian et al., ICML 2024",
        "url": "https://github.com/LiamLian0727/USIS10K",
        "classes": 7,
        "train_images": 8080,
        "paper_split": "8,080 train / 2,552 test (official semantic-mask protocol)",
        "notes": (
            "Download the semantic-mask split from the official GitHub repository.  "
            "Extend Phase-2 training to 60 epochs for the larger training set.  "
            "Reduce rare-class oversampling to 1.5× (set oversample_plant=1.5)."
        ),
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_SUBDIRS = ["train/images", "train/masks", "test/images", "test/masks"]


def print_instructions(key, output_dir):
    ds = DATASETS[key]
    print()
    print("=" * 72)
    print(f"  {ds['name']}  —  {ds['citation']}")
    print("=" * 72)
    print(f"  GitHub : {ds['url']}")
    if "url_official" in ds:
        print(f"  Official: {ds['url_official']}")
    print(f"  Classes: {ds['classes']}")
    print(f"  Split  : {ds['paper_split']}")
    print()
    print("  Notes:")
    for line in ds["notes"].split("  "):
        if line.strip():
            print(f"    {line.strip()}")
    print()
    print(f"  Target directory: {os.path.abspath(output_dir)}")
    print()
    print("  Expected layout after extraction:")
    print(f"    {output_dir}/")
    for sub in REQUIRED_SUBDIRS:
        print(f"      {sub}/")
    print()


def verify_dataset(output_dir):
    print(f"\n🔍 Verifying dataset structure in '{output_dir}' ...")
    all_ok = True
    for sub in REQUIRED_SUBDIRS:
        full = os.path.join(output_dir, sub)
        exists = os.path.isdir(full)
        n = len(os.listdir(full)) if exists else 0
        status = "✅" if exists and n > 0 else "❌"
        print(f"  {status}  {sub:<25s}  ({n} files)")
        all_ok = all_ok and exists and n > 0

    if all_ok:
        print("\n✅ Dataset structure looks correct.")
    else:
        print("\n❌ Some directories are missing or empty.  "
              "Please check the download and extraction steps.")
    return all_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print download instructions for MAREA-Net datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="suim",
        help="Which dataset to set up (default: suim).",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="Root directory where the dataset will be placed.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the directory structure (run after downloading).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    for key in keys:
        out = args.output_dir if args.dataset == "all" else args.output_dir
        if args.dataset == "all":
            out = os.path.join(args.output_dir, key)
        print_instructions(key, out)
        if args.verify:
            verify_dataset(out)


if __name__ == "__main__":
    main()
