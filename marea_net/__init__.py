"""MAREA-Net: Marine-Aware REsilient Architecture for Underwater Semantic Segmentation.

Reference
---------
Adeel Mukhtar, Usman Ali.
"Marine-Aware Resilient Architecture for Underwater Semantic Segmentation."
CVPR 2026.
"""

__version__ = "5.5.0"
__authors__ = ["Adeel Mukhtar", "Usman Ali"]

from .model import build_marea_net, set_encoder_trainable
from .config import Config
from .layers import SimAM, StripPooling, SBCC, DSTS, CGAFusion, MAREADecoderBlock, ASPP_SP
from .losses import create_loss_fn, compute_class_weights
from .metrics import evaluate_miou, tta_predict, compute_metrics
from .utils import load_model, save_model

__all__ = [
    "build_marea_net",
    "set_encoder_trainable",
    "Config",
    "SimAM",
    "StripPooling",
    "SBCC",
    "DSTS",
    "CGAFusion",
    "MAREADecoderBlock",
    "ASPP_SP",
    "create_loss_fn",
    "compute_class_weights",
    "evaluate_miou",
    "tta_predict",
    "compute_metrics",
    "load_model",
    "save_model",
]
