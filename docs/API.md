# API Reference

## Core Functions

### build_marea_net

Build MAREA-Net model.

```python
from marea_net import build_marea_net, Config

config = Config('config/config.yaml')
model = build_marea_net(config)
```

**Parameters:**
- `config` (Config, optional): Configuration object. If None, uses default config.

**Returns:**
- `keras.Model`: MAREA-Net model with 3 outputs: [seg_output, aux_output, plant_output]

### Config

Configuration management class.

```python
from marea_net import Config

# Load from file
config = Config('config/config.yaml')

# Use default config
config = Config()

# Get values
batch_size = config.get('training.batch_size', default=8)
num_classes = config['model']['num_classes']

# Save config
config.save('output/config.yaml')
```

## Data Pipeline

### build_dataset

Build tf.data.Dataset for training or evaluation.

```python
from marea_net.data import build_dataset

dataset, n_samples = build_dataset(
    img_dir='data/train/images',
    msk_dir='data/train/masks',
    config=config,
    training=True,
    plant_samples=plant_samples
)
```

**Parameters:**
- `img_dir` (str): Directory containing images
- `msk_dir` (str): Directory containing masks
- `config` (Config): Configuration object
- `training` (bool): If True, apply augmentation and shuffling
- `plant_samples` (list, optional): List of (img_path, mask_path) tuples for oversampling

**Returns:**
- `tuple`: (dataset, num_samples)

### find_plant_samples

Find images containing aquatic plants.

```python
from marea_net.data import find_plant_samples

plant_samples = find_plant_samples(
    img_dir='data/train/images',
    msk_dir='data/train/masks'
)
```

**Returns:**
- `list`: List of (img_path, mask_path) tuples

### cutmix_batch

Apply CutMix augmentation.

```python
from marea_net.data import cutmix_batch

imgs_aug, masks_aug = cutmix_batch(imgs, masks, config)
```

## Loss Functions

### create_loss_fn

Create combined loss function.

```python
from marea_net.losses import create_loss_fn, compute_class_weights

class_weights = compute_class_weights(masks, config)
loss_fn = create_loss_fn(class_weights, config)

# Use in training
loss = loss_fn(y_true, [seg_pred, aux_pred, plant_pred])
```

### compute_class_weights

Compute class weights from mask distribution.

```python
from marea_net.losses import compute_class_weights

weights = compute_class_weights(mask_array, config)
```

**Returns:**
- `np.ndarray`: Class weights [N_CLS]

## Metrics

### evaluate_miou

Evaluate mean Intersection over Union.

```python
from marea_net.metrics import evaluate_miou

miou = evaluate_miou(model, test_dataset, n_cls=8)
print(f"mIoU: {miou:.4f}")
```

### tta_predict

Test-time augmentation prediction.

```python
from marea_net.metrics import tta_predict

pred_cls, y_true = tta_predict(model, test_dataset, config)
```

**Returns:**
- `tuple`: (predictions, ground_truth) as numpy arrays

### compute_metrics

Compute per-class and overall metrics.

```python
from marea_net.metrics import compute_metrics

metrics = compute_metrics(pred_cls, y_true, class_names)

# Access results
miou = metrics['overall']['miou']
mf1 = metrics['overall']['mf1']

for cls_metrics in metrics['per_class']:
    print(f"{cls_metrics['class']}: IoU={cls_metrics['iou']:.4f}")
```

## Utilities

### configure_gpu

Configure GPU settings.

```python
from marea_net.utils import configure_gpu

gpus = configure_gpu(config)
```

### lr_schedule

Learning rate schedule with warmup and cosine decay.

```python
from marea_net.utils import lr_schedule

lr = lr_schedule(epoch=50, config=config)
```

### make_optimizer

Create AdamW optimizer.

```python
from marea_net.utils import make_optimizer

optimizer = make_optimizer(lr=1e-4, config=config)
```

### make_train_step

Create training step function.

```python
from marea_net.utils import make_train_step

train_step = make_train_step(model, optimizer, loss_fn)

# Use in training loop
loss = train_step(x_batch, y_batch)
```

### save_model / load_model

Save and load models with custom layers.

```python
from marea_net.utils import save_model, load_model

# Save
save_model(model, 'models/my_model.keras', config)

# Load
model = load_model('models/my_model.keras')
```

## Model Functions

### set_encoder_trainable

Freeze or unfreeze the encoder.

```python
from marea_net.model import set_encoder_trainable

# Freeze encoder
set_encoder_trainable(model, trainable=False)

# Unfreeze encoder
set_encoder_trainable(model, trainable=True)
```

## Custom Layers

All custom layers are available in `marea_net.layers`:

```python
from marea_net.layers import (
    SimAM,
    StripPooling,
    SBCC,
    WDTS,
    CGAFusion,
    MAREADecoderBlock,
    ASPP_SP
)
```

### Example: Using Custom Layers

```python
from tensorflow.keras import layers
from marea_net.layers import SimAM, SBCC

# Build custom model
x = layers.Input(shape=(320, 320, 3))
x = layers.Conv2D(64, 3, padding='same')(x)
x = SimAM()(x)
x = SBCC(64)(x)
```

## Complete Training Example

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

## Complete Inference Example

```python
from marea_net.utils import load_model, configure_gpu
from marea_net import Config
import tensorflow as tf
import numpy as np

# Setup
config = Config()
configure_gpu(config)

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

print(f"Predicted classes: {np.unique(pred_cls)}")
print(f"Plant presence: {plant_pred[0, 0]:.4f}")
```
