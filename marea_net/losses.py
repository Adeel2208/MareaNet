"""Loss functions for MAREA-Net."""

import numpy as np
import tensorflow as tf

from .config import Config


def compute_class_weights(mask_np, config: Config):
    """
    Compute class weights from mask distribution.
    
    Args:
        mask_np: Numpy array of masks
        config: Configuration object
        
    Returns:
        Numpy array of class weights [N_CLS]
    """
    n_cls = config.get('model.num_classes', 8)
    weight_floors = config.get('classes.weight_floors', {})
    class_names = config.get('classes.names', [])
    
    unique, counts = np.unique(mask_np, return_counts=True)
    total = mask_np.size
    freqs = {int(u): c / total for u, c in zip(unique, counts)}
    median = np.median(list(freqs.values()))
    
    weights = np.ones(n_cls, dtype=np.float32)
    for cid, freq in freqs.items():
        if 0 <= cid < n_cls:
            weights[cid] = min(25.0, median / (freq + 1e-8))
    
    # Apply minimum weight floors
    for cid, floor in weight_floors.items():
        weights[cid] = max(weights[cid], floor)
    
    print("  Class weights:")
    for i in range(n_cls):
        name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"    {name:20s}: {weights[i]:.2f}")
    
    return weights


def focal_ce_ohem(y_true, y_pred, cw, gm, keep_ratio, smoothing):
    """
    Focal Cross-Entropy with Online Hard Example Mining.
    
    Args:
        y_true: Ground truth labels [B, H, W]
        y_pred: Predicted probabilities [B, H, W, N_CLS]
        cw: Class weights [N_CLS]
        gm: Gamma values per class [N_CLS]
        keep_ratio: Ratio of hard examples to keep
        smoothing: Label smoothing factor
        
    Returns:
        Scalar loss value
    """
    n_cls = tf.shape(y_pred)[-1]
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
    y_true_i = tf.cast(y_true, tf.int32)
    y_true_oh = tf.one_hot(y_true_i, n_cls)
    
    # Label smoothing
    y_smooth = y_true_oh * (1.0 - smoothing) + smoothing / tf.cast(n_cls, tf.float32)
    
    # Focal loss
    pt = tf.reduce_sum(y_smooth * y_pred, axis=-1)
    gamma_map = tf.gather(gm, y_true_i)
    cw_map = tf.gather(cw, y_true_i)
    loss_map = tf.pow(1.0 - pt, gamma_map) * (-tf.math.log(pt + 1e-8)) * cw_map
    
    # OHEM
    flat = tf.reshape(loss_map, [tf.shape(loss_map)[0], -1])
    k = tf.maximum(tf.cast(tf.cast(tf.shape(flat)[1], tf.float32) * keep_ratio, tf.int32), 1)
    top_k, _ = tf.math.top_k(flat, k=k, sorted=False)
    
    return tf.reduce_mean(top_k)


def tversky_loss(y_true, y_pred, alpha, beta, n_cls):
    """
    Tversky loss for handling class imbalance.
    
    Args:
        y_true: Ground truth labels [B, H, W]
        y_pred: Predicted probabilities [B, H, W, N_CLS]
        alpha: Weight for false positives
        beta: Weight for false negatives
        n_cls: Number of classes
        
    Returns:
        Scalar loss value
    """
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_cls)
    
    tp = tf.reduce_sum(y_true_oh * y_pred, axis=[1, 2])
    fp = tf.reduce_sum((1 - y_true_oh) * y_pred, axis=[1, 2])
    fn = tf.reduce_sum(y_true_oh * (1 - y_pred), axis=[1, 2])
    
    return 1.0 - tf.reduce_mean(tp / (tp + alpha * fp + beta * fn + 1e-7))


def dice_loss(y_true, y_pred, n_cls):
    """
    Dice loss for segmentation.
    
    Args:
        y_true: Ground truth labels [B, H, W]
        y_pred: Predicted probabilities [B, H, W, N_CLS]
        n_cls: Number of classes
        
    Returns:
        Scalar loss value
    """
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_cls)
    
    inter = tf.reduce_sum(y_true_oh * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true_oh + y_pred, axis=[1, 2])
    
    return 1.0 - tf.reduce_mean((2.0 * inter + 1e-7) / (union + 1e-7))


def plant_dice_loss(y_true, y_pred, n_cls, plant_class=2):
    """
    Dice loss specifically for plant class.
    
    Args:
        y_true: Ground truth labels [B, H, W]
        y_pred: Predicted probabilities [B, H, W, N_CLS]
        n_cls: Number of classes
        plant_class: Index of plant class
        
    Returns:
        Scalar loss value
    """
    y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
    y_true_oh = tf.one_hot(tf.cast(y_true, tf.int32), n_cls)
    
    plant_true = y_true_oh[..., plant_class]
    plant_pred = y_pred[..., plant_class]
    
    inter = tf.reduce_sum(plant_true * plant_pred, axis=[1, 2])
    union = tf.reduce_sum(plant_true + plant_pred, axis=[1, 2])
    
    return 1.0 - tf.reduce_mean((2.0 * inter + 1e-7) / (union + 1e-7))


def plant_presence_loss(y_true_mask, y_pred_binary):
    """
    Binary cross-entropy for plant presence prediction.
    
    Args:
        y_true_mask: Ground truth mask [B, H, W]
        y_pred_binary: Predicted plant presence [B, 1]
        
    Returns:
        Scalar loss value
    """
    has_plant = tf.cast(
        tf.reduce_any(tf.equal(tf.cast(y_true_mask, tf.int32), 2), axis=[1, 2]),
        tf.float32
    )
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(
            has_plant,
            tf.squeeze(tf.cast(y_pred_binary, tf.float32), axis=-1)
        )
    )


def create_loss_fn(class_weights, config: Config):
    """
    Create combined loss function for MAREA-Net.
    
    Args:
        class_weights: Numpy array of class weights
        config: Configuration object
        
    Returns:
        Loss function
    """
    n_cls = config.get('model.num_classes', 8)
    gammas = config.get('classes.gammas', [1.0] * n_cls)
    tversky_alpha = config.get('training.tversky_alpha', 0.3)
    tversky_beta = config.get('training.tversky_beta', 0.7)
    ohem_keep = config.get('training.ohem_keep_ratio', 0.7)
    label_smooth = config.get('training.label_smoothing', 0.05)
    plant_dice_weight = config.get('training.plant_dice_weight', 0.15)
    
    cw = tf.constant(class_weights, dtype=tf.float32)
    gm = tf.constant(gammas, dtype=tf.float32)
    
    def loss_fn(y_true, y_preds):
        seg_pred, aux_pred, plant_pred = y_preds
        
        return (
            focal_ce_ohem(y_true, seg_pred, cw, gm, ohem_keep, label_smooth)
            + tversky_loss(y_true, seg_pred, tversky_alpha, tversky_beta, n_cls)
            + 0.5 * dice_loss(y_true, seg_pred, n_cls)
            + plant_dice_weight * plant_dice_loss(y_true, seg_pred, n_cls)
            + 0.3 * focal_ce_ohem(y_true, aux_pred, cw, gm, 0.9, label_smooth)
            + 0.2 * plant_presence_loss(y_true, plant_pred)
        )
    
    return loss_fn
