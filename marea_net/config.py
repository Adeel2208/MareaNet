"""Configuration management for MAREA-Net."""

import os
import yaml
from typing import Dict, Any, List


class Config:
    """Configuration class for MAREA-Net."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file. If None, uses default config.
        """
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            'model': {
                'name': 'MAREA_Net_v5.5_ConvNeXtBase',
                'input_height': 320,
                'input_width': 320,
                'num_classes': 8,
                'backbone': 'ConvNeXtBase',
                'backbone_weights': 'imagenet'
            },
            'training': {
                'batch_size': 8,
                'epochs': 100,
                'unfreeze_epoch': 10,
                'lr_warmup': 1e-5,
                'lr_max': 1e-4,
                'lr_min': 1e-6,
                'warmup_epochs': 5,
                'weight_decay': 1e-4,
                'tversky_alpha': 0.3,
                'tversky_beta': 0.7,
                'ohem_keep_ratio': 0.7,
                'label_smoothing': 0.05,
                'plant_dice_weight': 0.15,
                'cutmix_prob': 0.3,
                'oversample_plant': 2
            },
            'classes': {
                'names': [
                    "Background", "Human Divers", "Aquatic Plants", "Wrecks/Ruins",
                    "Robots/ROV", "Reefs/Inverts", "Fish/Vertebrates", "Sea-floor/Rocks"
                ],
                'gammas': [1.0, 3.0, 5.0, 3.0, 5.0, 2.0, 3.0, 2.0],
                'weight_floors': {1: 2.0, 2: 1.5, 3: 4.0, 4: 3.5, 6: 2.0},
                'palette': [
                    [0, 0, 0], [0, 0, 255], [0, 255, 0], [0, 255, 255],
                    [255, 0, 0], [255, 0, 255], [255, 255, 0], [255, 255, 255]
                ]
            },
            'data': {
                'train_images': 'data/train/images',
                'train_masks': 'data/train/masks',
                'test_images': 'data/test/images',
                'test_masks': 'data/test/masks'
            },
            'gpu': {
                'memory_growth': True,
                'mixed_precision': 'float32',
                'xla_jit': True
            },
            'inference': {
                'tta_augmentations': 8,
                'temperature': 1.0,
                'batch_size': 4
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated key."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration section."""
        return self.config[key]
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
