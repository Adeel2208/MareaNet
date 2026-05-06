# Changes Made to Align Repository with CVPR 2026 Paper

This document summarizes all changes made to prepare the MAREA-Net repository for publication alongside the CVPR 2026 paper.

## Critical Fixes (Paper Alignment)

### 1. WDTS → DSTS Rename
**Files:** `layers.py`, `model.py`, `utils.py`, `__init__.py`

The paper consistently uses "**DSTS** (Dual-Scale Tone Scaling)" throughout. The code was using `WDTS` ("Wavelet-inspired Dual-scale Texture Sharpening") — a different name with a different description.

**Changes:**
- Renamed class `WDTS` → `DSTS` in `layers.py`
- Added `WDTS = DSTS` backward-compatibility alias for existing checkpoints
- Updated all references: `use_wdts` → `use_dsts` in `model.py`
- Updated docstrings to match paper description
- Added `DSTS` to `utils.py` custom_objects for model loading

### 2. CutMix Removed from Training
**Files:** `train.py`, `config.yaml`, `config.py`

The paper explicitly states CutMix was **dropped after M5** because it hurt performance (−1.25 mIoU, plant IoU collapsed to 0.0956). The code was still applying CutMix on every training step.

**Changes:**
- Removed `cutmix_batch` call from training loop in `train.py`
- Removed `cutmix_prob` from config files
- Function kept in `data.py` for reference but no longer invoked

### 3. Output Naming: plant_output → presence_output
**Files:** `model.py`

The paper describes a "rare-class presence head" that is dataset-agnostic (plants on SUIM, scallop on DUT-USEG).

**Changes:**
- Renamed model output from `plant_output` to `presence_output`
- Updated docstrings to reflect dataset-agnostic design

---

## Multi-Dataset Generalization

### 4. Dataset-Agnostic Rare-Class Handling
**Files:** `data.py`, `losses.py`, `train.py`, `config.yaml`

Made the rare-class detection and loss functions configurable for all four benchmarks.

**Changes:**
- `find_plant_samples` → `find_rare_class_samples` (takes `rare_color` parameter)
- `plant_presence_loss` → `rare_class_presence_loss` (takes `rare_class_index` parameter)
- `plant_dice_loss` now reads `rare_class_index` from config
- Added config keys: `rare_class_index`, `rare_class_color`, `rare_dice_weight`
- Backward-compatibility aliases kept for old function names

### 5. Multi-Dataset Config Files
**Files:** `config/config_dutuseg.yaml`, `config_mas3k.yaml`, `config_usis10k.yaml`

Created separate configs for each benchmark with correct:
- Number of classes (5, 2, 7)
- Palette colors
- Rare class index
- Training schedule adjustments (USIS10K: 70 epochs, MAS3K: no rare-class head)

---

## Repository Quality

### 6. requirements.txt — Pinned Versions
**File:** `requirements.txt`

Open ranges like `tensorflow>=2.10.0` span versions with breaking ConvNeXt preprocessing changes.

**Changes:**
- Pinned to `tensorflow>=2.12.0,<2.16.0`
- Pinned numpy, Pillow, opencv-python to compatible ranges

### 7. setup.py — Real Author Info
**File:** `setup.py`

**Changes:**
- `author="Your Name"` → `author="Adeel Mukhtar, Usman Ali"`
- `author_email` → real email
- `url` → `https://github.com/adeelmukhtar/marea-net`
- Added proper description referencing CVPR 2026

### 8. __init__.py — Complete Exports
**File:** `marea_net/__init__.py`

**Changes:**
- Added `__authors__` field
- Exported all public symbols (layers, losses, metrics, utils)
- Added proper module docstring with paper reference

### 9. README.md — Complete Rewrite
**File:** `README.md`

**Changes:**
- **Results tables** with actual numbers from the paper (SUIM, DUT-USEG, MAS3K, USIS10K)
- **Per-class IoU table** for SUIM (8 classes)
- **Real dataset links** with verified URLs:
  - SUIM: http://irvlab.cs.umn.edu/resources/suim-dataset + GitHub
  - DUT-USEG: https://github.com/chongweiliu/DUT-USEG
  - MAS3K: https://github.com/LinLi-DL/MAS
  - USIS10K: https://github.com/LiamLian0727/USIS10K
- **Correct citation** (Mukhtar & Ali, CVPR 2026)
- **Training/evaluation commands** for all datasets
- **Architecture diagram** with correct module names (DSTS not WDTS)

### 10. evaluate.py — Removed Fabricated Baselines
**File:** `evaluate.py`

The code had hardcoded "AquaSAM" and "ResNet50" baseline IoU numbers that didn't match the paper.

**Changes:**
- Removed fabricated baseline comparisons
- Replaced with correct SegFormer-B0 per-class IoUs from Table S1 in supplementary

### 11. scripts/download_dataset.py — All Four Benchmarks
**File:** `scripts/download_dataset.py`

Was a SUIM-only placeholder.

**Changes:**
- Expanded to cover all four benchmarks
- Accurate split instructions (1,225/300 for SUIM, 1,380/107 for DUT-USEG, etc.)
- Real GitHub URLs
- Dataset-specific notes (rare class indices, config adjustments)

### 12. tests/ — Created from Scratch
**Files:** `tests/test_layers.py`, `test_model.py`, `test_losses.py`, `test_metrics.py`

No tests existed. `CONTRIBUTING.md` referenced `pytest tests/` but the directory didn't exist.

**Changes:**
- Created comprehensive test suite covering:
  - All custom layers (SimAM, StripPooling, SBCC, DSTS, CGA, MAREADecoderBlock, ASPP_SP)
  - Model construction and forward pass
  - Loss functions (focal-OHEM, Tversky, Dice, rare-class presence)
  - Metrics (compute_metrics, IoU/F1 ranges)
  - Backward-compatibility aliases (WDTS, plant_presence_loss)

---

## Summary Statistics

| Category | Changes |
|----------|---------|
| Files modified | 15 |
| Files created | 8 (4 configs + 4 test files) |
| Critical naming fixes | 3 (WDTS→DSTS, plant→presence, CutMix removal) |
| Multi-dataset configs | 3 (DUT-USEG, MAS3K, USIS10K) |
| Test files added | 4 (layers, model, losses, metrics) |
| Lines of test code | ~400 |

---

## Verification Checklist

- [x] All paper terminology matches code (DSTS not WDTS)
- [x] CutMix removed from training (matches M5→M6 ablation)
- [x] Multi-dataset support (4 configs, dataset-agnostic rare-class handling)
- [x] Real dataset URLs verified via web search
- [x] README results match paper Tables 2-5
- [x] Per-class IoU matches Table S1 (supplementary)
- [x] Citation correct (Mukhtar & Ali, CVPR 2026)
- [x] No placeholder text ("Your Name", "TBD", "yourusername")
- [x] Tests pass (pytest tests/ -v)
- [x] Backward compatibility maintained (WDTS alias, plant_* aliases)

---

## Running the Tests

```bash
pip install pytest
pytest tests/ -v
```

All tests should pass. The test suite covers:
- Layer shape checks
- Model construction (3 outputs, correct shapes)
- Loss function ranges and perfect-prediction behavior
- Metrics computation (IoU/F1 in [0,1])
- Backward-compatibility aliases

---

## Training on Different Datasets

```bash
# SUIM (default, 8 classes)
python train.py --data_dir data/suim --epochs 100

# DUT-USEG (5 classes, scallop as rare class)
python train.py --data_dir data/dutuseg --config config/config_dutuseg.yaml --epochs 100

# MAS3K (binary, no rare-class head)
python train.py --data_dir data/mas3k --config config/config_mas3k.yaml --epochs 100

# USIS10K (7 classes, extended Phase-2)
python train.py --data_dir data/usis10k --config config/config_usis10k.yaml --epochs 70
```

---

## Contact

For questions about these changes:
- **Paper authors:** Adeel Mukhtar (2021bme123@student.uet.edu.pk), Usman Ali
- **Repository:** https://github.com/adeelmukhtar/marea-net
