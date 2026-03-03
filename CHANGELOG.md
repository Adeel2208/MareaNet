# Changelog

All notable changes to MAREA-Net will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.5.0] - 2024-XX-XX

### Added
- Initial release of MAREA-Net v5.5-CNX
- ConvNeXtBase backbone replacing ResNet50
- Custom attention modules: SimAM, SBCC, WDTS
- CGA-Fusion decoder with 4 stages
- ASPP with Strip Pooling
- Multi-component loss function:
  - Focal Cross-Entropy with OHEM
  - Tversky Loss
  - Dice Loss
  - Plant-specific Dice Loss
  - Auxiliary supervision
  - Plant presence classification
- Comprehensive data augmentation:
  - Geometric transformations
  - Color jitter
  - CutMix
  - Plant oversampling
- Two-phase training strategy:
  - Phase 1: Frozen encoder
  - Phase 2: Full fine-tuning
- Test-time augmentation (8-fold)
- Command-line training script
- Command-line inference script
- Command-line evaluation script
- Configuration management via YAML
- Modular architecture for easy customization
- Comprehensive documentation:
  - Architecture guide
  - Training guide
  - API reference
- Example scripts:
  - Quick start
  - Custom training loop
- Utility scripts:
  - Dataset download helper
  - Visualization tool
  - ONNX export

### Features
- GPU memory growth configuration
- XLA JIT compilation support
- Mixed precision training support
- Learning rate warmup and cosine decay
- Automatic class weight computation
- Plant sample detection and oversampling
- Per-class IoU and F1 metrics
- Baseline comparison tools

### Documentation
- README with quick start guide
- Detailed architecture documentation
- Comprehensive training guide
- Complete API reference
- Contributing guidelines
- Project structure overview

## [Unreleased]

### Planned
- Unit tests for all modules
- Integration tests
- Continuous integration setup
- Pre-trained model weights
- Docker container
- Web demo
- Additional backbones (EfficientNet, Swin Transformer)
- Multi-GPU training support
- TensorBoard integration
- Weights & Biases integration
- Model quantization
- Mobile deployment support

## Version History

### v5.5-CNX (Current)
- ConvNeXtBase backbone
- Enhanced attention mechanisms
- Improved plant segmentation

### v5.5-GPU (Previous)
- ResNet50 backbone
- GPU optimizations
- Plant-aware loss

### v5.4-GPU
- Initial GPU-optimized version
- Basic attention modules

### v5.3-GPU
- Early development version

## Migration Guide

### From v5.5-GPU (ResNet50) to v5.5-CNX (ConvNeXt)

**Model Changes:**
- Backbone: ResNet50 → ConvNeXtBase
- Preprocessing: Channel-mean subtraction → ImageNet normalization
- Skip connections: Updated layer names
- Final head: Single ×2 upsample → Two ×2 upsamples

**Configuration:**
```yaml
# Old (v5.5-GPU)
model:
  backbone: "ResNet50"

# New (v5.5-CNX)
model:
  backbone: "ConvNeXtBase"
```

**Code Changes:**
```python
# Old
from keras.applications import ResNet50
encoder = ResNet50(...)

# New
from keras.applications import ConvNeXtBase
encoder = ConvNeXtBase(...)
```

**Performance:**
- Expected mIoU improvement: +2-3%
- Training time: Similar
- Inference time: Slightly slower (~10%)
- Model size: Larger (~88M vs ~25M params)

## Acknowledgments

- ConvNeXt architecture from Facebook Research
- SUIM dataset from University of Minnesota
- TensorFlow and Keras teams
- Open-source community

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
