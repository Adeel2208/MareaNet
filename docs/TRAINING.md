# Training Guide

## Prerequisites

- Python 3.8+
- TensorFlow 2.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- Dataset in proper format

## Dataset Preparation

### Directory Structure

```
data/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.bmp
│       ├── image_002.bmp
│       └── ...
└── test/
    ├── images/
    └── masks/
```

### Mask Format

Masks should be RGB images with the following color palette:

| Class | Color (RGB) |
|-------|-------------|
| Background | (0, 0, 0) |
| Human Divers | (0, 0, 255) |
| Aquatic Plants | (0, 255, 0) |
| Wrecks/Ruins | (0, 255, 255) |
| Robots/ROV | (255, 0, 0) |
| Reefs/Inverts | (255, 0, 255) |
| Fish/Vertebrates | (255, 255, 0) |
| Sea-floor/Rocks | (255, 255, 255) |

## Basic Training

```bash
python train.py --data_dir data --output_dir models
```

## Advanced Options

### Custom Configuration

```bash
python train.py \
    --config config/custom_config.yaml \
    --data_dir /path/to/dataset \
    --output_dir models \
    --epochs 150 \
    --batch_size 16
```

### Resume Training

```bash
python train.py \
    --data_dir data \
    --output_dir models \
    --resume models/marea_net_checkpoint.keras
```

## Configuration

Edit `config/config.yaml` to customize training:

### Model Settings

```yaml
model:
  input_height: 320
  input_width: 320
  num_classes: 8
  backbone: "ConvNeXtBase"
```

### Training Hyperparameters

```yaml
training:
  batch_size: 8          # Adjust based on GPU memory
  epochs: 100
  unfreeze_epoch: 10     # When to unfreeze encoder
  
  # Learning rate schedule
  lr_warmup: 1.0e-5
  lr_max: 1.0e-4
  lr_min: 1.0e-6
  warmup_epochs: 5
  weight_decay: 1.0e-4
```

### Loss Configuration

```yaml
training:
  tversky_alpha: 0.3     # FP weight
  tversky_beta: 0.7      # FN weight
  ohem_keep_ratio: 0.7   # Hard example ratio
  label_smoothing: 0.05
  plant_dice_weight: 0.15
```

### Augmentation

```yaml
training:
  cutmix_prob: 0.3       # CutMix probability
  oversample_plant: 2    # Plant oversampling factor
```

## Training Phases

### Phase 1: Frozen Encoder (Epochs 0-9)

- Only decoder and heads are trained
- Faster convergence
- Prevents catastrophic forgetting of ImageNet features

### Phase 2: Full Fine-tuning (Epochs 10-99)

- Entire model is trainable
- Fresh optimizer state
- Cosine learning rate decay

## Monitoring Training

The training script prints:

```
Epoch  50/100 | loss=0.3245 | val_mIoU=0.7234 | lr=5.23e-05 | 45.2s ✨ BEST
```

- `loss`: Training loss (lower is better)
- `val_mIoU`: Validation mean IoU (higher is better)
- `lr`: Current learning rate
- `✨ BEST`: Indicates new best validation score

## Memory Optimization

### Reduce Batch Size

```yaml
training:
  batch_size: 4  # For GPUs with <16GB VRAM
```

### Enable Mixed Precision

```yaml
gpu:
  mixed_precision: "mixed_float16"
```

### Reduce Image Size

```yaml
model:
  input_height: 256
  input_width: 256
```

## Common Issues

### Out of Memory

- Reduce batch size
- Reduce input resolution
- Enable mixed precision
- Close other GPU applications

### Slow Training

- Enable XLA JIT compilation (default)
- Use SSD for dataset storage
- Increase `num_parallel_calls` in data pipeline

### Poor Convergence

- Increase warmup epochs
- Adjust learning rate
- Check class weights
- Verify data augmentation

## Best Practices

1. **Start with default config**: Proven hyperparameters
2. **Monitor validation mIoU**: Primary metric
3. **Use plant oversampling**: Improves rare class performance
4. **Enable CutMix**: Better generalization
5. **Save checkpoints**: Resume if interrupted
6. **Validate regularly**: Every 3 epochs

## Output Files

After training:

```
models/
├── marea_net_best.keras        # Best model weights
└── marea_net_best_config.yaml  # Configuration used
```

## Next Steps

After training:

1. **Evaluate**: Run `evaluate.py` on test set
2. **Inference**: Use `inference.py` for predictions
3. **Fine-tune**: Adjust hyperparameters if needed
