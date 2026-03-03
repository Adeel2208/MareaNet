# MAREA-Net Architecture

## Overview

MAREA-Net v5.5-CNX is a semantic segmentation architecture designed specifically for underwater imagery. It combines a ConvNeXtBase encoder with custom attention mechanisms and multi-scale feature fusion.

## Architecture Components

### 1. Encoder: ConvNeXtBase

The backbone uses ConvNeXtBase pretrained on ImageNet:

```
Input: 320×320×3
  ↓
Stem (stride 4): 80×80 @ 128ch
  ↓
Stage 0 (3 blocks): 80×80 @ 128ch
  ↓
Stage 1 (3 blocks): 40×40 @ 256ch
  ↓
Stage 2 (27 blocks): 20×20 @ 512ch
  ↓
Stage 3 (3 blocks): 10×10 @ 1024ch
```

### 2. Bottleneck: ASPP with Strip Pooling

Atrous Spatial Pyramid Pooling captures multi-scale context:

- 1×1 convolution
- 3×3 atrous convolutions (rates: 6, 12, 18)
- Strip pooling for long-range dependencies
- SimAM attention module

Output: 10×10 @ 256ch

### 3. Decoder: 4-Stage with CGA-Fusion

Each decoder block uses Cross-scale Gated Attention Fusion:

```
Level 4: 256ch (SBCC attention) → 20×20
Level 3: 128ch (WDTS attention) → 40×40
Level 2: 64ch                   → 80×80
Level 1: 32ch                   → 80×80
```

### 4. Segmentation Head

Two 2× upsampling stages to reach 320×320:

```
80×80 → 160×160 → 320×320 → 8 classes
```

### 5. Auxiliary Heads

- **Deep Supervision**: Auxiliary segmentation head from bottleneck
- **Plant Presence**: Binary classification for aquatic plants

## Custom Modules

### SimAM (Simple Attention Module)

Parameter-free attention using spatial variance:

```python
e_inv = (x - μ)² / (4(σ² + λ) + λ)
output = x * sigmoid(e_inv)
```

### Strip Pooling

Captures long-range dependencies along horizontal and vertical axes:

```python
h_feat = pool_horizontal(x)
v_feat = pool_vertical(x)
output = (h_feat + v_feat) * sigmoid(fuse(h_feat + v_feat))
```

### SBCC (Spatial-Boundary Channel Calibration)

Combines channel attention with spatial attention for boundary refinement.

### WDTS (Wavelet-inspired Dual-scale Texture Sharpening)

Multi-scale texture enhancement using 3×3 and 5×5 depthwise convolutions.

### CGA-Fusion (Cross-scale Gated Attention Fusion)

Adaptive fusion of decoder and skip features:

```python
ca = channel_attention(combined)
sa = spatial_attention(combined)
w = sigmoid(refine(concat(combined, ca + sa)))
output = w * skip + (1 - w) * decoder
```

## Loss Function

Combined loss with multiple components:

1. **Focal Cross-Entropy** with OHEM (keep ratio: 0.7)
2. **Tversky Loss** (α=0.3, β=0.7) for class imbalance
3. **Dice Loss** (weight: 0.5)
4. **Plant Dice Loss** (weight: 0.15) for aquatic plants
5. **Auxiliary Loss** (weight: 0.3) for deep supervision
6. **Plant Presence Loss** (weight: 0.2) for binary classification

## Training Strategy

### Phase 1: Frozen Encoder (Epochs 0-9)

- Encoder weights frozen
- Train decoder and heads only
- Learning rate: warmup → max

### Phase 2: Full Fine-tuning (Epochs 10-99)

- Unfreeze entire model
- Fresh AdamW optimizer
- Learning rate: cosine decay

### Augmentation

- Horizontal/vertical flips
- 90° rotations
- Color jitter (brightness, contrast, saturation)
- CutMix (probability: 0.3)
- Plant oversampling (2×)

## Inference

### Test-Time Augmentation (TTA)

8-fold augmentation:
1. Original
2. Horizontal flip
3. Vertical flip
4. Both flips
5. Rotate 90°
6. Rotate 180°
7. Rotate 270°
8. Rotate 90° + horizontal flip

Predictions are averaged after inverse transformations.

## Model Size

- Total parameters: ~88M
- Encoder (ConvNeXtBase): ~88M
- Decoder + Heads: ~5M

## Performance

Expected mIoU on SUIM test set: ~0.72-0.75

Per-class IoU targets:
- Background: 0.88
- Human Divers: 0.78
- Aquatic Plants: 0.35-0.40
- Wrecks/Ruins: 0.72
- Robots/ROV: 0.73
- Reefs/Inverts: 0.68
- Fish/Vertebrates: 0.73
- Sea-floor/Rocks: 0.67
