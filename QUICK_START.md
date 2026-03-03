# Quick Start Guide

Get up and running with MAREA-Net in 5 minutes!

## 1. Installation

### Option A: Automated Setup (Recommended)

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

### Option B: Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/train/images data/train/masks
mkdir -p data/test/images data/test/masks
mkdir -p models results
```

## 2. Dataset Preparation

### Download SUIM Dataset

```bash
python scripts/download_dataset.py --output_dir data
```

Follow the instructions to download and extract the dataset.

### Verify Dataset Structure

```
data/
├── train/
│   ├── images/  (1525 images)
│   └── masks/   (1525 masks)
└── test/
    ├── images/  (110 images)
    └── masks/   (110 masks)
```

## 3. Training

### Basic Training

```bash
python train.py --data_dir data --output_dir models
```

This will:
- Train for 100 epochs
- Use batch size 8
- Save best model to `models/marea_net_best.keras`

### Custom Training

```bash
python train.py \
    --data_dir data \
    --output_dir models \
    --epochs 150 \
    --batch_size 16 \
    --config config/config.yaml
```

### Monitor Training

Watch for output like:
```
Epoch  50/100 | loss=0.3245 | val_mIoU=0.7234 | lr=5.23e-05 | 45.2s ✨ BEST
```

## 4. Evaluation

```bash
python evaluate.py \
    --model_path models/marea_net_best.keras \
    --test_dir data/test \
    --tta
```

Expected output:
```
Class                IoU    Prec   Rec    F1     Freq%
Background          0.8800 0.9234 0.9512 0.9371  65.2341
Human Divers        0.7800 0.8456 0.9123 0.8776   2.1234
...
mIoU (macro): 0.7234
```

## 5. Inference

### Single Directory

```bash
python inference.py \
    --model_path models/marea_net_best.keras \
    --input_dir data/test/images \
    --output_dir results \
    --save_overlay
```

### With TTA (Better Quality)

```bash
python inference.py \
    --model_path models/marea_net_best.keras \
    --input_dir data/test/images \
    --output_dir results \
    --tta \
    --save_overlay
```

## 6. Visualization

```bash
python scripts/visualize_predictions.py \
    --image data/test/images/image_001.jpg \
    --prediction results/image_001_pred.png \
    --ground_truth data/test/masks/image_001.bmp \
    --output visualization.png
```

## Python API Usage

### Simple Inference

```python
from marea_net import build_marea_net, Config
from marea_net.utils import load_model
import tensorflow as tf

# Load model
model = load_model('models/marea_net_best.keras')

# Preprocess image
img = tf.image.decode_image(tf.io.read_file('test.jpg'), channels=3, dtype=tf.float32)
img = tf.image.resize(img, [320, 320])
img = tf.keras.applications.convnext.preprocess_input(img * 255.0)
img = tf.expand_dims(img, 0)

# Predict
seg_pred, aux_pred, plant_pred = model(img, training=False)
pred_cls = tf.argmax(seg_pred[0], axis=-1).numpy()
```

### Custom Training Loop

```python
from marea_net import build_marea_net, Config
from marea_net.data import build_dataset, find_plant_samples
from marea_net.losses import compute_class_weights, create_loss_fn
from marea_net.utils import configure_gpu, make_optimizer, make_train_step

# Setup
config = Config('config/config.yaml')
configure_gpu(config)

# Data
plant_samples = find_plant_samples('data/train/images', 'data/train/masks')
train_ds, _ = build_dataset('data/train/images', 'data/train/masks', 
                             config, training=True, plant_samples=plant_samples)

# Model
model = build_marea_net(config)

# Loss and optimizer
class_weights = compute_class_weights(next(iter(train_ds))[1].numpy(), config)
loss_fn = create_loss_fn(class_weights, config)
optimizer = make_optimizer(1e-4, config)
train_step = make_train_step(model, optimizer, loss_fn)

# Training loop
for epoch in range(10):
    for x_batch, y_batch in train_ds:
        loss = train_step(x_batch, y_batch)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

## Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  input_height: 320
  input_width: 320
  num_classes: 8

training:
  batch_size: 8
  epochs: 100
  lr_max: 1.0e-4
```

## Common Issues

### Out of Memory

Reduce batch size:
```yaml
training:
  batch_size: 4
```

### Slow Training

Enable XLA (default):
```yaml
gpu:
  xla_jit: true
```

### Poor Plant Segmentation

Increase plant oversampling:
```yaml
training:
  oversample_plant: 3
  plant_dice_weight: 0.2
```

## Next Steps

- Read [TRAINING.md](docs/TRAINING.md) for detailed training guide
- Check [ARCHITECTURE.md](docs/ARCHITECTURE.md) for model details
- See [API.md](docs/API.md) for complete API reference
- Review [examples/](examples/) for more code examples

## Getting Help

- Check [README.md](README.md) for overview
- Read documentation in [docs/](docs/)
- Open an issue on GitHub
- Check existing issues for solutions

## Performance Tips

1. **Use TTA for inference**: +1-2% mIoU improvement
2. **Enable XLA JIT**: Faster training
3. **Use SSD for dataset**: Faster data loading
4. **Monitor GPU usage**: Adjust batch size accordingly
5. **Use plant oversampling**: Better rare class performance

Happy segmenting! 🌊
