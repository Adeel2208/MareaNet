# MAREA-Net v5.5-CNX

**Marine-Aware REsilient Architecture for Underwater Semantic Segmentation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MAREA-Net v5.5-CNX is a state-of-the-art deep learning architecture for underwater semantic segmentation, featuring a ConvNeXtBase backbone with custom attention mechanisms and multi-scale feature fusion.

## Features

- **ConvNeXtBase Backbone**: Modern CNN architecture replacing ResNet50
- **Custom Attention Modules**: SimAM, SBCC, WDTS for enhanced feature extraction
- **Advanced Decoder**: CGA-Fusion with strip pooling and ASPP
- **Robust Training**: Focal loss, Tversky loss, OHEM, CutMix augmentation
- **Plant-Aware Loss**: Specialized handling for aquatic vegetation
- **TTA Support**: 8-fold test-time augmentation for improved accuracy

## Architecture Overview

```
Input (320×320×3)
    ↓
ConvNeXtBase Encoder
    ├─ Stem:   80×80 @ 128ch
    ├─ Stage0: 80×80 @ 128ch
    ├─ Stage1: 40×40 @ 256ch
    ├─ Stage2: 20×20 @ 512ch
    └─ Stage3: 10×10 @ 1024ch
    ↓
ASPP + SimAM (10×10 @ 256ch)
    ↓
4-Stage Decoder (CGA-Fusion)
    ├─ L4: 20×20 @ 256ch (SBCC)
    ├─ L3: 40×40 @ 128ch (WDTS)
    ├─ L2: 80×80 @ 64ch
    └─ L1: 80×80 @ 32ch
    ↓
Output (320×320×8)
```

## Installation

### Quick Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/marea-net.git
cd marea-net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/train/images data/train/masks
mkdir -p data/test/images data/test/masks
mkdir -p models results
```

## Quick Start

### Training

```bash
python train.py --data_dir /path/to/dataset --epochs 100 --batch_size 8
```

### Inference

```bash
python inference.py --model_path models/best_model.keras --input_dir /path/to/images --output_dir results
```

### Evaluation

```bash
python evaluate.py --model_path models/best_model.keras --test_dir /path/to/test
```

## Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.bmp
│       └── ...
└── test/
    ├── images/
    └── masks/
```

## Classes

| ID | Class Name          | Color (RGB)     |
|----|---------------------|-----------------|
| 0  | Background          | (0, 0, 0)       |
| 1  | Human Divers        | (0, 0, 255)     |
| 2  | Aquatic Plants      | (0, 255, 0)     |
| 3  | Wrecks/Ruins        | (0, 255, 255)   |
| 4  | Robots/ROV          | (255, 0, 0)     |
| 5  | Reefs/Inverts       | (255, 0, 255)   |
| 6  | Fish/Vertebrates    | (255, 255, 0)   |
| 7  | Sea-floor/Rocks     | (255, 255, 255) |

## Configuration

Edit `config/config.yaml` to customize training parameters:

```yaml
model:
  input_size: [320, 320]
  num_classes: 8
  
training:
  batch_size: 8
  epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-4
```

## Results

| Model | mIoU | Background | Divers | Plants | Wrecks | ROV | Reefs | Fish | Seafloor |
|-------|------|------------|--------|--------|--------|-----|-------|------|----------|
| MAREA-Net v5.5-CNX | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{mareanet2024,
  title={MAREA-Net: Marine-Aware REsilient Architecture for Underwater Semantic Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ConvNeXt architecture from Facebook Research
- SUIM dataset for underwater imagery
- TensorFlow and Keras teams

## Project Structure

```
marea-net/
├── marea_net/          # Core package
│   ├── model.py       # Model architecture
│   ├── layers.py      # Custom layers
│   ├── data.py        # Data pipeline
│   ├── losses.py      # Loss functions
│   └── metrics.py     # Evaluation metrics
├── train.py           # Training script
├── inference.py       # Inference script
├── evaluate.py        # Evaluation script
├── config/            # Configuration files
├── docs/              # Documentation
├── examples/          # Usage examples
└── scripts/           # Utility scripts
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure.

## Documentation

- [Quick Start Guide](QUICK_START.md) - Get started in 5 minutes
- [Architecture Guide](docs/ARCHITECTURE.md) - Model architecture details
- [Training Guide](docs/TRAINING.md) - Comprehensive training guide
- [API Reference](docs/API.md) - Complete API documentation
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

## Contact

For questions and feedback, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).
