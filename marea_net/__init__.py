"""MAREA-Net: Marine-Aware REsilient Architecture for Underwater Semantic Segmentation"""

__version__ = "5.5.0"
__author__ = "Your Name"

from .model import build_marea_net
from .config import Config

__all__ = ["build_marea_net", "Config"]
