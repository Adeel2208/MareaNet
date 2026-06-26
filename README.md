# MAREA-Net

**Marine-Aware Resilient Architecture for Underwater Semantic Segmentation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.12–2.15](https://img.shields.io/badge/tensorflow-2.12--2.15-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

MAREA-Net is an **enhancement-free** encoder-decoder for underwater semantic segmentation.
Instead of prepending a separate image-enhancement module, it handles underwater colour
degradation entirely inside the segmentation network through three targeted mechanisms:

| Module | Location | Role |
|--------|----------|------|
| **CGA** — Content-Guided Attention | All skip connections | Suppresses degraded encoder features |
| **SBCC** — Scale-Balanced Channel Calibration | Decoder L4 (20×20) | Recalibrates channel importance after colour-cast compression |
| **DSTS** — Dual-Scale Gated Feature Modulation | Decoder L3 (40×40) | Sigmoid-gated dual-kernel feature modulation |

An ASPP bottleneck with Strip Pooling and SimAM captures long-range context.
Auxiliary segmentation and rare-class presence heads provide deep supervision.

---

## Results

### SUIM (1,225 train / 300 test, 320×320, 8-fold TTA)

| Method | Backbone | Params | mIoU | mF1 |
|--------|----------|--------|------|-----|
| Imp. DLv3+ | MobOne-S0 | 5.1M | 59.05 | 62.04 |
| DeepLab-FN | RepVGG | 26M | 62.40 | 74.20 |
| AquaSAM | SAM ViT-B | 91M | 63.88 | 79.76 |
| SegFormer-B0 | MiT-B0 | 3.8M | 72.27 | 83.19 |
| SegFormer-B2 | MiT-B2 | 25M | 73.61 | **84.38** |
| SegFormer-B4 | MiT-B4 | 47M | **74.12** | **84.91** |
| **MAREA-Net (ours)** | **CNXBase** | **93M** | **73.85** | **84.04** |

Three-seed mean: **73.76 ± 0.25%** mIoU.

### DUT-USEG (1,380 train / 107 test, 320×320, 8-fold TTA)

| Method | Backbone | Params | mIoU |
|--------|----------|--------|------|
| DeepLabV3+ | MobileNetV2 | 5.8M | 66.68 |
| UWSegFormer | MiT | 13M | 71.41 |
| SegFormer-B4 | MiT-B4 | 47M | 73.19 |
| **MAREA-Net (ours)** | **CNXBase** | **93M** | **74.76** |

### MAS3K (binary mIoU) / USIS10K (semantic mIoU)

| Dataset | MAREA-Net | SegFormer-B0 | SegFormer-B4 |
|---------|-----------|--------------|--------------|
| MAS3K   | 76.00 | 74.20 | **77.41** |
| USIS10K | 68.00 | 64.80 | **71.05** |

### Per-class IoU on SUIM

| Class | Imp. DLv3+ | SegFormer-B0 | SegFormer-B4 | **MAREA-Net** |
|-------|-----------|--------------|--------------|---------------|
| Background | 86.12 | 87.52 | **89.32** | 88.96 |
| Human Divers | 58.24 | 78.14 | 81.57 | **82.24** |
| Aquatic Plants | 9.53 | 37.98 | 38.05 | **38.77** |
| Wrecks/Ruins | 56.71 | 75.62 | 75.03 | **75.70** |
| Robots/ROV | 62.87 | 74.46 | 78.95 | **81.59** |
| Reefs/Inverts | 65.39 | 72.18 | 73.51 | 73.30 |
| Fish/Vertebrates | 64.18 | 77.06 | **81.12** | 80.57 |
| Sea-floor/Rocks | 69.36 | 75.20 | **78.11** | 69.70 |

---

## Architecture

```
Input (320×320×3)
    ↓  ImageNet normalisation (no UIE preprocessing)
ConvNeXtBase Encoder  [ImageNet-22k pretrained, ~88M params]
    ├─ Stem:   80×80 @ 128ch  ──────────────────────────────── skip S1
    ├─ Stage0: 80×80 @ 128ch  ──────────────────────────────── skip S2
    ├─ Stage1: 40×40 @ 256ch  ──────────────────────────────── skip S3
    ├─ Stage2: 20×20 @ 512ch  ──────────────────────────────── skip S4
    └─ Stage3: 10×10 @ 1024ch
    ↓
ASPP + Strip Pooling + SimAM  →  10×10 @ 256ch
    ├─ Auxiliary head  →  320×320 @ n_cls  [training only]
    └─ Presence head   →  scalar sigmoid   [training only]
    ↓
4-Stage CGA-Gated Decoder
    ├─ L4: 20×20 @ 256ch  (CGA + SBCC)
    ├─ L3: 40×40 @ 128ch  (CGA + DSTS)
    ├─ L2: 80×80 @  64ch  (CGA)
    └─ L1: 80×80 @  32ch  (CGA)
    ↓
Two ×2 bilinear upsampling steps  →  320×320 @ n_cls
    ↓
Softmax  →  seg_output
```

Total parameters: ~93M (ConvNeXtBase 88M + decoder 5M).

---

## Installation

```bash
git clone https://github.com/adeelmukhtar/marea-net.git
cd marea-net

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## Dataset Preparation

MAREA-Net is evaluated on four benchmarks.  All require manual download.

| Dataset | Classes | Images | Paper | Download |
|---------|---------|--------|-------|----------|
| **SUIM** | 8 | 1,525 | [Islam et al., IROS 2020](https://ieeexplore.ieee.org/document/9196596) | [Official Site](http://irvlab.cs.umn.edu/resources/suim-dataset) · [GitHub](https://github.com/xahidbuffon/SUIM) |
| **DUT-USEG** | 5 | 1,487 | [Ma et al., JBUAA 2022](https://bhxb.buaa.edu.cn/bhzk/en/article/doi/10.13700/j.bh.1001-5965.2021.0464) | [GitHub](https://github.com/chongweiliu/DUT-USEG) |
| **MAS3K** | binary | 3,103 | [Li et al., Bench 2021](https://link.springer.com/chapter/10.1007/978-3-030-71058-3_12) | [GitHub](https://github.com/LinLi-DL/MAS) |
| **USIS10K** | 7 | 10,632 | [Lian et al., ICML 2024](https://proceedings.mlr.press/v235/lian24c.html) | [GitHub](https://github.com/LiamLian0727/USIS10K) |

Run `python scripts/download_dataset.py --dataset <name> --output_dir data` for detailed setup instructions.

Expected layout:

```
data/
├── train/
│   ├── images/   ← .jpg / .png
│   └── masks/    ← .bmp / .png  (RGB palette)
└── test/
    ├── images/
    └── masks/
```

---

## Training

```bash
# SUIM (default config)
python train.py --data_dir data/suim --epochs 100 --batch_size 8

# DUT-USEG (5 classes, scallop as rare class)
python train.py \
  --data_dir data/dutuseg \
  --config config/config_dutuseg.yaml \
  --epochs 100
```

Key training details (see `config/config.yaml`):

- Phase 1 (epochs 0–9): encoder frozen, LR warm-up 1e-5 → 1e-4
- Phase 2 (epochs 10–100): full joint training, cosine decay 1e-4 → 1e-6
- Augmentation: horizontal/vertical flips, 90° rotations, colour jitter
- Rare-class oversampling: 2× per epoch
- Loss: Focal-OHEM + Tversky + 0.5·Dice + 0.15·rare-Dice + 0.3·aux + 0.2·presence

---

## Evaluation

```bash
# Standard evaluation
python evaluate.py \
  --model_path models/marea_net_best.keras \
  --test_dir data/suim/test

# With 8-fold TTA (matches paper numbers)
python evaluate.py \
  --model_path models/marea_net_best.keras \
  --test_dir data/suim/test \
  --tta
```

---

## Inference

```bash
python inference.py \
  --model_path models/marea_net_best.keras \
  --input_dir /path/to/images \
  --output_dir results \
  --tta \
  --save_overlay
```

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Reproducibility

### Dataset Splits

The paper uses **custom splits** for reproducibility:

- **SUIM**: 1,225 train / 300 test (custom split, not default 1,525/110)
- **DUT-USEG**: 1,380 train / 107 test (UWSegFormer protocol)
- **MAS3K**: 1,769 train / 1,141 test (default split)
- **USIS10K**: 8,080 train / 2,552 test (official semantic protocol)

Generate splits with seed 42:

```bash
python scripts/generate_splits.py --dataset suim --data_dir data/suim --output_dir splits --seed 42
python scripts/verify_splits.py --dataset suim --data_dir data/suim --splits_dir splits
```

### Reproduce Paper Results

**Table 3 - SUIM (73.85% mIoU, 8-fold TTA)**
```bash
python train.py --data_dir data/suim --config config/config.yaml --epochs 100 --seed 42
python evaluate.py --model_path models/marea_net_best.keras --test_dir data/suim/test --tta
```

**Table 5 - DUT-USEG (74.76% mIoU, 8-fold TTA)**
```bash
python train.py --data_dir data/dutuseg --config config/config_dutuseg.yaml --epochs 100 --seed 42
python evaluate.py --model_path models/marea_net_best.keras --test_dir data/dutuseg/test --tta
```

**Table 6 - MAS3K (76.00% mIoU, 8-fold TTA)**
```bash
python train.py --data_dir data/mas3k --config config/config_mas3k.yaml --epochs 100 --seed 42
python evaluate.py --model_path models/marea_net_best.keras --test_dir data/mas3k/test --tta
```

**Table 7 - USIS10K (68.00% mIoU, 8-fold TTA)**
```bash
python train.py --data_dir data/usis10k --config config/config_usis10k.yaml --epochs 70 --seed 42
python evaluate.py --model_path models/marea_net_best.keras --test_dir data/usis10k/test --tta
```

**Multi-seed variance (Table 8)**: 73.76 ± 0.25% mIoU on SUIM (seeds: 42, 123, 456)

### Pretrained Checkpoints

Download from [GitHub Releases](https://github.com/Adeel2208/MareaNet/releases):
- `marea_net_suim_best.keras` (73.85% mIoU)
- `marea_net_dutuseg_best.keras` (74.76% mIoU)
- `marea_net_mas3k_best.keras` (76.00% mIoU)
- `marea_net_usis10k_best.keras` (68.00% mIoU)

---

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgements

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) — Facebook Research
- [SegFormer](https://github.com/NVlabs/SegFormer) — NVIDIA
- [SUIM](http://irvlab.cs.umn.edu/resources/suim-dataset) — Islam et al., IROS 2020
- [DEA-Net / CGA](https://github.com/cecret3350/DEA-Net) — Chen et al., IEEE TIP 2024
- [SimAM](https://github.com/ZjjConan/SimAM) — Yang et al., CVPR 2022
- [Strip Pooling](https://github.com/Andrew-Qibin/SPNet) — Hou et al., CVPR 2020
