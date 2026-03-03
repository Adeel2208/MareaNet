"""Evaluation metrics for MAREA-Net."""

import numpy as np
import tensorflow as tf


def evaluate_miou(model, dataset, n_cls=8):
    """
    Evaluate mean Intersection over Union.
    
    Args:
        model: Keras model
        dataset: tf.data.Dataset
        n_cls: Number of classes
        
    Returns:
        Mean IoU score
    """
    confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
    
    for xb, yb in dataset:
        pred_cls = tf.argmax(model(xb, training=False)[0], axis=-1, 
                            output_type=tf.int32).numpy()
        yb_np = yb.numpy().astype(np.int32)
        
        for c in range(n_cls):
            tm = (yb_np == c)
            for cp in range(n_cls):
                confusion[c, cp] += int(np.sum(tm & (pred_cls == cp)))
    
    ious = []
    for c in range(n_cls):
        tp = confusion[c, c]
        fp = confusion[:, c].sum() - tp
        fn = confusion[c, :].sum() - tp
        ious.append(float(tp) / (float(tp + fp + fn) + 1e-8))
    
    return float(np.mean(ious))


def tta_predict(model, dataset, config):
    """
    Test-time augmentation prediction.
    
    Args:
        model: Keras model
        dataset: tf.data.Dataset
        config: Configuration object
        
    Returns:
        Tuple of (predictions, ground_truth)
    """
    n_aug = config.get('inference.tta_augmentations', 8)
    temperature = config.get('inference.temperature', 1.0)
    n_cls = config.get('model.num_classes', 8)
    img_h = config.get('model.input_height', 320)
    img_w = config.get('model.input_width', 320)
    
    # TensorFlow transformations
    tf_fwd = [
        lambda x: x,
        lambda x: tf.image.flip_left_right(x),
        lambda x: tf.image.flip_up_down(x),
        lambda x: tf.image.flip_left_right(tf.image.flip_up_down(x)),
        lambda x: tf.image.rot90(x, 1),
        lambda x: tf.image.rot90(x, 2),
        lambda x: tf.image.rot90(x, 3),
        lambda x: tf.image.rot90(tf.image.flip_left_right(x), 1),
    ]
    
    # Inverse transformations (NumPy)
    np_inv = [
        lambda x: x,
        lambda x: x[:, :, ::-1, :],
        lambda x: x[:, ::-1, :, :],
        lambda x: x[:, ::-1, ::-1, :],
        lambda x: np.rot90(x, k=3, axes=(1, 2)),
        lambda x: np.rot90(x, k=2, axes=(1, 2)),
        lambda x: np.rot90(x, k=1, axes=(1, 2)),
        lambda x: np.rot90(x[:, :, ::-1, :], k=3, axes=(1, 2)),
    ]
    
    all_preds, all_gt = [], []
    n_t = min(n_aug, len(tf_fwd))
    
    for xb, yb in dataset:
        accum = np.zeros([xb.shape[0], img_h, img_w, n_cls], dtype=np.float64)
        
        for t in range(n_t):
            p = model(tf_fwd[t](xb), training=False)[0].numpy().astype(np.float64)
            
            # Temperature scaling
            if temperature != 1.0:
                p = np.power(p + 1e-10, 1.0 / temperature)
                p /= p.sum(axis=-1, keepdims=True) + 1e-10
            
            accum += np_inv[t](p)
        
        all_preds.append(np.argmax(accum / n_t, axis=-1))
        all_gt.append(yb.numpy())
    
    return np.concatenate(all_preds, axis=0), np.concatenate(all_gt, axis=0)


def compute_metrics(pred_cls, y_true, class_names):
    """
    Compute per-class and overall metrics.
    
    Args:
        pred_cls: Predicted class indices [N, H, W]
        y_true: Ground truth class indices [N, H, W]
        class_names: List of class names
        
    Returns:
        Dictionary of metrics
    """
    n_cls = len(class_names)
    y_true = y_true.astype(np.int32)
    
    metrics = {
        'per_class': [],
        'overall': {}
    }
    
    ious, f1s = [], []
    
    print(f"  {'Class':20s} {'IoU':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'Freq%':>7s}")
    
    for c in range(n_cls):
        tc = (y_true == c)
        pc = (pred_cls == c)
        
        tp = int(np.sum(tc & pc))
        fp = int(np.sum(~tc & pc))
        fn = int(np.sum(tc & ~pc))
        
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        freq = np.mean(tc) * 100
        
        ious.append(iou)
        f1s.append(f1)
        
        metrics['per_class'].append({
            'class': class_names[c],
            'iou': iou,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'frequency': freq
        })
        
        print(f"  {class_names[c]:20s} {iou:.4f} {prec:.4f} {rec:.4f} {f1:.4f} {freq:7.4f}")
    
    metrics['overall']['miou'] = float(np.mean(ious))
    metrics['overall']['mf1'] = float(np.mean(f1s))
    
    print("─" * 85)
    print(f"  mIoU (macro): {metrics['overall']['miou']:.4f}   "
          f"mF1 (macro): {metrics['overall']['mf1']:.4f}")
    print("─" * 85)
    
    return metrics
