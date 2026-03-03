"""Utility functions for MAREA-Net."""

import os
import tensorflow as tf
from tensorflow import keras


def configure_gpu(config):
    """
    Configure GPU settings.
    
    Args:
        config: Configuration object
        
    Returns:
        List of available GPUs
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth
            if config.get('gpu.memory_growth', True):
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"⚠️  GPU config error: {e}")
    
    # Set precision policy
    from tensorflow.keras import mixed_precision
    precision = config.get('gpu.mixed_precision', 'float32')
    mixed_precision.set_global_policy(precision)
    print(f"✅ Precision: {precision}")
    
    # Enable XLA JIT compilation
    if config.get('gpu.xla_jit', True):
        tf.config.optimizer.set_jit(True)
        print("✅ XLA JIT enabled")
    
    return gpus


def lr_schedule(epoch, config):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Args:
        epoch: Current epoch
        config: Configuration object
        
    Returns:
        Learning rate for the epoch
    """
    warmup_epochs = config.get('training.warmup_epochs', 5)
    unfreeze_epoch = config.get('training.unfreeze_epoch', 10)
    total_epochs = config.get('training.epochs', 100)
    lr_warmup = config.get('training.lr_warmup', 1e-5)
    lr_max = config.get('training.lr_max', 1e-4)
    lr_min = config.get('training.lr_min', 1e-6)
    
    if epoch < warmup_epochs:
        return float(lr_warmup + (lr_max - lr_warmup) * epoch / max(warmup_epochs, 1))
    
    if epoch < unfreeze_epoch:
        return float(lr_max)
    
    # Cosine decay after unfreezing
    t = epoch - unfreeze_epoch
    T = total_epochs - unfreeze_epoch
    return float(lr_min + (lr_max - lr_min) * 0.5 * (1.0 + tf.cos(3.14159 * t / T)))


def make_optimizer(lr, config):
    """
    Create AdamW optimizer.
    
    Args:
        lr: Learning rate
        config: Configuration object
        
    Returns:
        Keras optimizer
    """
    weight_decay = config.get('training.weight_decay', 1e-4)
    return keras.optimizers.AdamW(
        learning_rate=lr,
        weight_decay=weight_decay,
        clipnorm=1.0
    )


def make_train_step(model, optimizer, loss_fn):
    """
    Create training step function.
    
    Args:
        model: Keras model
        optimizer: Keras optimizer
        loss_fn: Loss function
        
    Returns:
        Training step function
    """
    train_vars = model.trainable_variables
    
    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            preds = model(xb, training=True)
            loss = loss_fn(yb, preds)
        
        grads = tape.gradient(loss, train_vars)
        gv = [(g, v) for g, v in zip(grads, train_vars) if g is not None]
        
        if gv:
            optimizer.apply_gradients(gv)
        
        return loss
    
    return train_step


def save_model(model, save_path, config=None):
    """
    Save model with metadata.
    
    Args:
        model: Keras model
        save_path: Path to save model
        config: Configuration object (optional)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"💾 Model saved to {save_path}")
    
    if config:
        config_path = save_path.replace('.keras', '_config.yaml')
        config.save(config_path)
        print(f"💾 Config saved to {config_path}")


def load_model(model_path):
    """
    Load saved model with custom objects.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Loaded Keras model
    """
    from .layers import (SimAM, StripPooling, SBCC, WDTS, 
                         CGAFusion, MAREADecoderBlock, ASPP_SP)
    
    custom_objects = {
        'SimAM': SimAM,
        'StripPooling': StripPooling,
        'SBCC': SBCC,
        'WDTS': WDTS,
        'CGAFusion': CGAFusion,
        'MAREADecoderBlock': MAREADecoderBlock,
        'ASPP_SP': ASPP_SP,
    }
    
    return keras.models.load_model(model_path, custom_objects=custom_objects)
