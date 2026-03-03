"""Data loading and preprocessing for MAREA-Net."""

import os
import glob
import numpy as np
import tensorflow as tf

from .config import Config


def rgb_mask_to_class_tf(mask_rgb, palette, img_h, img_w):
    """
    Convert RGB mask to class indices using nearest color matching.
    
    Args:
        mask_rgb: RGB mask tensor
        palette: Color palette tensor [N_CLS, 3]
        img_h: Image height
        img_w: Image width
        
    Returns:
        Class index mask [H, W]
    """
    mask_f = tf.cast(mask_rgb, tf.float32)
    flat = tf.reshape(mask_f, [-1, 1, 3])
    dists = tf.reduce_sum(tf.square(flat - palette[tf.newaxis, :, :]), axis=-1)
    labels = tf.argmin(dists, axis=1, output_type=tf.int32)
    return tf.cast(tf.reshape(labels, [img_h, img_w]), tf.uint8)


def create_parse_function(config: Config):
    """Create parse function for tf.data pipeline."""
    img_h = config.get('model.input_height', 320)
    img_w = config.get('model.input_width', 320)
    palette = tf.constant(config.get('classes.palette'), dtype=tf.float32)
    
    def parse_function(img_path, mask_path):
        # Load and resize image
        img = tf.image.decode_image(tf.io.read_file(img_path), channels=3, 
                                     dtype=tf.float32, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, [img_h, img_w], method='bilinear', antialias=True)
        
        # ConvNeXt preprocessing
        img = tf.keras.applications.convnext.preprocess_input(img * 255.0)
        
        # Load and resize mask
        msk = tf.image.decode_image(tf.io.read_file(mask_path), channels=3, 
                                     dtype=tf.uint8, expand_animations=False)
        msk.set_shape([None, None, 3])
        msk = tf.cast(tf.image.resize(msk, [img_h, img_w], method='nearest'), tf.uint8)
        
        # Convert RGB to class indices
        msk_cls = rgb_mask_to_class_tf(msk, palette, img_h, img_w)
        msk_cls.set_shape([img_h, img_w])
        
        return img, msk_cls
    
    return parse_function


def create_augment_function(config: Config):
    """Create augmentation function for training."""
    
    def augment_function(img, mask):
        # Horizontal flip
        def hflip(i, m):
            return (tf.image.flip_left_right(i),
                   tf.squeeze(tf.image.flip_left_right(tf.expand_dims(m, -1)), -1))
        
        # Vertical flip
        def vflip(i, m):
            return (tf.image.flip_up_down(i),
                   tf.squeeze(tf.image.flip_up_down(tf.expand_dims(m, -1)), -1))
        
        img, mask = tf.cond(tf.random.uniform([]) > 0.5,
                           lambda: hflip(img, mask), lambda: (img, mask))
        img, mask = tf.cond(tf.random.uniform([]) > 0.5,
                           lambda: vflip(img, mask), lambda: (img, mask))
        
        # Random rotation (90° increments)
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        mask = tf.squeeze(tf.image.rot90(tf.expand_dims(mask, -1), k), -1)
        
        # Color jitter
        def colour_jitter(i):
            i = tf.image.adjust_brightness(i, tf.random.uniform([], -0.1, 0.1))
            i = tf.image.adjust_contrast(i, tf.random.uniform([], 0.9, 1.1))
            i = tf.image.adjust_saturation(i, tf.random.uniform([], 0.85, 1.15))
            return i
        
        img = tf.cond(tf.random.uniform([]) > 0.5, 
                     lambda: colour_jitter(img), lambda: img)
        
        return img, mask
    
    return augment_function


def find_image_mask_pairs(img_dir, msk_dir):
    """
    Find matching image-mask pairs.
    
    Args:
        img_dir: Directory containing images
        msk_dir: Directory containing masks
        
    Returns:
        List of (image_path, mask_path) tuples
    """
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    pairs = []
    
    for ip in img_paths:
        base = os.path.splitext(os.path.basename(ip))[0]
        for ext in [".bmp", ".png", ".jpg", ".jpeg"]:
            cand = os.path.join(msk_dir, base + ext)
            if os.path.exists(cand):
                pairs.append((ip, cand))
                break
    
    return pairs


def find_plant_samples(img_dir, msk_dir):
    """
    Find images containing aquatic plants (class 2, green color).
    
    Args:
        img_dir: Directory containing images
        msk_dir: Directory containing masks
        
    Returns:
        List of (image_path, mask_path) tuples for plant-containing images
    """
    results = []
    pairs = find_image_mask_pairs(img_dir, msk_dir)
    
    for ip, mp in pairs:
        try:
            msk = tf.image.decode_image(tf.io.read_file(mp), channels=3, 
                                        expand_animations=False).numpy()
            # Check for green pixels (aquatic plants)
            if ((msk[..., 0] == 0) & (msk[..., 1] == 255) & (msk[..., 2] == 0)).any():
                results.append((ip, mp))
        except Exception as e:
            print(f"  Warning: skipping {mp}: {e}")
    
    return results


def build_dataset(img_dir, msk_dir, config: Config, training=True, plant_samples=None):
    """
    Build tf.data.Dataset for training or evaluation.
    
    Args:
        img_dir: Directory containing images
        msk_dir: Directory containing masks
        config: Configuration object
        training: If True, apply augmentation and shuffling
        plant_samples: List of plant sample pairs for oversampling
        
    Returns:
        Tuple of (dataset, num_samples)
    """
    batch_size = config.get('training.batch_size' if training else 'inference.batch_size', 8)
    oversample_plant = config.get('training.oversample_plant', 2)
    
    pairs = find_image_mask_pairs(img_dir, msk_dir)
    print(f"  Found {len(pairs)} image-mask pairs in {img_dir}")
    
    if not pairs:
        raise ValueError(f"No image-mask pairs found in {img_dir}")
    
    parse_fn = create_parse_function(config)
    
    dataset = tf.data.Dataset.from_tensor_slices((
        [p[0] for p in pairs],
        [p[1] for p in pairs]
    ))
    
    if training:
        dataset = dataset.shuffle(min(len(pairs), 500), reshuffle_each_iteration=True)
    
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        dataset = dataset.cache()
        
        # Plant oversampling
        if plant_samples and oversample_plant > 1:
            plant_ds = (
                tf.data.Dataset.from_tensor_slices((
                    [p[0] for p in plant_samples],
                    [p[1] for p in plant_samples]
                ))
                .repeat(oversample_plant - 1)
                .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
                .cache()
            )
            dataset = dataset.concatenate(plant_ds)
            print(f"  Plant oversampling: {len(plant_samples)} donors ×{oversample_plant-1} extra")
        
        # Augmentation
        augment_fn = create_augment_function(config)
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size, drop_remainder=training)
    
    if not training:
        dataset = dataset.cache()
    
    return dataset.prefetch(tf.data.AUTOTUNE), len(pairs)


def cutmix_batch(imgs, masks, config: Config):
    """
    Apply CutMix augmentation to a batch.
    
    Args:
        imgs: Batch of images [B, H, W, 3]
        masks: Batch of masks [B, H, W]
        config: Configuration object
        
    Returns:
        Tuple of (augmented_imgs, augmented_masks)
    """
    img_h = config.get('model.input_height', 320)
    img_w = config.get('model.input_width', 320)
    cutmix_prob = config.get('training.cutmix_prob', 0.3)
    
    B = imgs.shape[0]
    imgs_out = imgs.copy()
    masks_out = masks.copy()
    
    for i in range(B):
        if np.random.rand() > cutmix_prob:
            continue
        
        j = np.random.randint(0, B)
        lam = np.random.beta(1.0, 1.0)
        
        cut_h = int(img_h * np.sqrt(1.0 - lam))
        cut_w = int(img_w * np.sqrt(1.0 - lam))
        cx = np.random.randint(0, img_w)
        cy = np.random.randint(0, img_h)
        
        x1 = max(0, cx - cut_w // 2)
        x2 = min(img_w, cx + cut_w // 2)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(img_h, cy + cut_h // 2)
        
        imgs_out[i, y1:y2, x1:x2, :] = imgs[j, y1:y2, x1:x2, :]
        masks_out[i, y1:y2, x1:x2] = masks[j, y1:y2, x1:x2]
    
    return imgs_out, masks_out
