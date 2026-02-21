"""
tessera.utils — Common utilities for the Tessera protocol.

Provides logging setup, device management, reproducibility helpers,
and parameter counting for PyTorch models.
"""

import logging
import random

import numpy as np
import torch


def setup_logging(name: str = "tessera", level=logging.INFO) -> logging.Logger:
    """
    Configure a logger with consistent formatting.

    Args:
        name: Logger name (appears in log output).
        level: Logging level (default: INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_device(device: str = "auto") -> str:
    """
    Resolve compute device.

    Args:
        device: One of "auto", "cuda", "cpu".

    Returns:
        "cuda" if available and requested, otherwise "cpu".
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_seed(seed: int = 42):
    """Set random seeds across all backends for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
