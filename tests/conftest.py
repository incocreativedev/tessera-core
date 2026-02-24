"""
Shared fixtures for Tessera test suite.
"""

import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Ensure tessera is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tessera.utils import set_seed


@pytest.fixture(autouse=True)
def reproducible():
    """Ensure reproducibility for every test."""
    set_seed(42)
    yield


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield d


class SmallTransformer(nn.Module):
    """Minimal transformer for testing."""

    def __init__(self, d_model=64, num_layers=2, vocab_size=128, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=max(1, d_model // 32),
                    dim_feedforward=d_model * 2,
                    batch_first=True,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))


@pytest.fixture
def small_tx():
    """Small transmitter model (64d, 2 layers)."""
    return SmallTransformer(d_model=64, num_layers=2)


@pytest.fixture
def small_rx():
    """Small receiver model (128d, 3 layers)."""
    return SmallTransformer(d_model=128, num_layers=3)


@pytest.fixture
def train_loader():
    """Simple training data."""
    X = torch.randint(0, 128, (100, 8))
    y = (X.float().mean(dim=1) >= 64).long()
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)


@pytest.fixture
def val_loader():
    """Simple validation data."""
    X = torch.randint(0, 128, (50, 8))
    y = (X.float().mean(dim=1) >= 64).long()
    return DataLoader(TensorDataset(X, y), batch_size=16, shuffle=False)


class TinyEdgeModel(nn.Module):
    """Minimal model simulating an edge deployment target (d=32, 1 layer)."""

    def __init__(self, d_model=32, vocab_size=128, num_classes=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=max(1, d_model // 16),
                    dim_feedforward=d_model * 2,
                    batch_first=True,
                    dropout=0.0,
                )
            ]
        )
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x.mean(dim=1))


@pytest.fixture
def tiny_edge():
    """Tiny edge model (32d, 1 layer) for quantisation transfer tests."""
    return TinyEdgeModel(d_model=32)


@pytest.fixture
def sample_uhs_vector():
    """A sample 2048-dim UHS vector."""
    return np.random.randn(2048).astype(np.float32).tolist()
