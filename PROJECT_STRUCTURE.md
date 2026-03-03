# MAREA-Net Project Structure

```
marea-net/
│
├── README.md                      # Project overview and quick start
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── PROJECT_STRUCTURE.md           # This file
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation script
├── .gitignore                     # Git ignore rules
│
├── config/                        # Configuration files
│   └── config.yaml               # Default training configuration
│
├── marea_net/                     # Main package
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Configuration management
│   ├── model.py                  # Model architecture
│   ├── layers.py                 # Custom layers (SimAM, SBCC, WDTS, etc.)
│   ├── data.py                   # Data loading and preprocessing
│   ├── losses.py                 # Loss functions
│   ├── metrics.py                # Evaluation metrics
│   └── utils.py                  # Utility functions
│
├── train.py                       # Training script
├── inference.py                   # Inference script
├── evaluate.py                    # Evaluation script
│
├── scripts/                       # Utility scripts
│   ├── download_dataset.py       # Dataset download helper
│   ├── visualize_predictions.py  # Visualization tool
│   └── export_onnx.py            # ONNX export tool
│
├── examples/                      # Example usage
│   ├── quick_start.py            # Quick start example
│   └── custom_training.py        # Custom training loop example
│
├── docs/                          # Documentation
│   ├── ARCHITECTURE.md           # Architecture details
│   ├── TRAINING.md               # Training guide
│   └── API.md                    # API reference
│
├── tests/                         # Unit tests (to be created)
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_losses.py
│   └── test_metrics.py
│
├── data/                          # Dataset directory (not in repo)
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       ├── images/
│       └── masks/
│
├── models/                        # Saved models (not in repo)
│   └── marea_net_best.keras
│
└── results/                       # Inference results (not in repo)
    ├── predictions/
    └── visualizations/
```

## Directory Descriptions

### Root Files

- `README.md`: Project overview, installation, and quick start guide
- `LICENSE`: MIT License for the project
- `CONTRIBUTING.md`: Guidelines for contributing to the project
- `requirements.txt`: Python package dependencies
- `setup.py`: Package installation configuration
- `.gitignore`: Files and directories to ignore in git

### config/

Configuration files for training and inference:
- `config.yaml`: Default configuration with all hyperparameters

### marea_net/

Core package containing all model code:

- `__init__.py`: Package initialization, exports main functions
- `config.py`: Configuration management class
- `model.py`: MAREA-Net architecture definition
- `layers.py`: Custom layers (SimAM, StripPooling, SBCC, WDTS, CGAFusion, etc.)
- `data.py`: Data loading, preprocessing, and augmentation
- `losses.py`: Loss functions (Focal, Tversky, Dice, etc.)
- `metrics.py`: Evaluation metrics (mIoU, TTA, etc.)
- `utils.py`: Utility functions (GPU config, LR schedule, etc.)

### Scripts

Main entry points for training, inference, and evaluation:

- `train.py`: Full training pipeline with command-line interface
- `inference.py`: Run inference on images
- `evaluate.py`: Evaluate model on test set

### scripts/

Utility scripts for common tasks:

- `download_dataset.py`: Helper for downloading SUIM dataset
- `visualize_predictions.py`: Create visualizations of predictions
- `export_onnx.py`: Export model to ONNX format

### examples/

Example code demonstrating usage:

- `quick_start.py`: Minimal example to get started
- `custom_training.py`: Example of custom training loop

### docs/

Detailed documentation:

- `ARCHITECTURE.md`: Architecture details and design decisions
- `TRAINING.md`: Comprehensive training guide
- `API.md`: API reference for all public functions

### tests/

Unit tests (to be implemented):

- `test_model.py`: Model architecture tests
- `test_data.py`: Data pipeline tests
- `test_losses.py`: Loss function tests
- `test_metrics.py`: Metrics tests

### data/

Dataset directory (not tracked in git):

```
data/
├── train/
│   ├── images/          # Training images
│   └── masks/           # Training masks (RGB)
└── test/
    ├── images/          # Test images
    └── masks/           # Test masks (RGB)
```

### models/

Saved model checkpoints (not tracked in git):

- `marea_net_best.keras`: Best model from training
- `marea_net_best_config.yaml`: Configuration used for training

### results/

Inference outputs (not tracked in git):

- `predictions/`: Predicted segmentation masks
- `visualizations/`: Overlay visualizations

## Usage Flow

### 1. Installation

```bash
git clone https://github.com/yourusername/marea-net.git
cd marea-net
pip install -r requirements.txt
```

### 2. Dataset Preparation

```bash
python scripts/download_dataset.py --output_dir data
```

### 3. Training

```bash
python train.py --data_dir data --output_dir models
```

### 4. Evaluation

```bash
python evaluate.py --model_path models/marea_net_best.keras --test_dir data/test
```

### 5. Inference

```bash
python inference.py --model_path models/marea_net_best.keras \
                    --input_dir data/test/images \
                    --output_dir results
```

## Key Components

### Model Architecture

- **Encoder**: ConvNeXtBase (pretrained on ImageNet)
- **Bottleneck**: ASPP with Strip Pooling
- **Decoder**: 4-stage with CGA-Fusion
- **Heads**: Segmentation + Auxiliary + Plant presence

### Custom Layers

- **SimAM**: Parameter-free attention
- **StripPooling**: Long-range dependency capture
- **SBCC**: Spatial-boundary channel calibration
- **WDTS**: Wavelet-inspired texture sharpening
- **CGAFusion**: Cross-scale gated attention fusion

### Loss Functions

- Focal Cross-Entropy with OHEM
- Tversky Loss
- Dice Loss
- Plant-specific Dice Loss
- Auxiliary supervision
- Plant presence classification

### Data Augmentation

- Horizontal/vertical flips
- 90° rotations
- Color jitter
- CutMix
- Plant oversampling

## Development Workflow

1. **Feature Development**: Create branch, implement feature
2. **Testing**: Add unit tests, run test suite
3. **Documentation**: Update relevant docs
4. **Code Review**: Submit PR, address feedback
5. **Merge**: Maintainer merges approved PR

## Configuration Management

All hyperparameters are managed through `config/config.yaml`:

- Model architecture settings
- Training hyperparameters
- Data augmentation parameters
- GPU configuration
- Inference settings

Override via command-line or custom config files.
