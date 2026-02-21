"""Tests for tessera.uhs — Universal Hub Space encoder/decoder."""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tessera.uhs import EncoderMLP, DecoderMLP, UniversalHubSpace


class TestEncoderMLP:
    def test_output_shape(self):
        enc = EncoderMLP(d_model=64, hub_dim=2048)
        x = torch.randn(8, 64)
        out = enc(x)
        assert out.shape == (8, 2048)

    def test_l2_normalised(self):
        enc = EncoderMLP(d_model=128, hub_dim=2048)
        x = torch.randn(4, 128)
        out = enc(x)
        norms = torch.norm(out, dim=1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=0)

    def test_hub_dim_attribute(self):
        enc = EncoderMLP(d_model=64, hub_dim=1024)
        assert enc.hub_dim == 1024


class TestDecoderMLP:
    def test_output_shape(self):
        dec = DecoderMLP(d_model=64, hub_dim=2048)
        z = torch.randn(8, 2048)
        out = dec(z)
        assert out.shape == (8, 64)


class TestUniversalHubSpace:
    def test_train_reduces_loss(self):
        uhs = UniversalHubSpace(d_model=64, hub_dim=256)  # Small for speed

        # Generate random activations directly (100 samples, 64-dim)
        activations = torch.randn(100, 64)
        loader = DataLoader(TensorDataset(activations), batch_size=32, shuffle=True)

        result = uhs.train(loader, epochs=5, verbose=False)
        losses = result["train_loss"]
        assert len(losses) == 5
        # Training should reduce loss (use average of last 2 vs first 2 for robustness)
        assert np.mean(losses[-2:]) < np.mean(losses[:2])

    def test_encode_decode_shape(self):
        uhs = UniversalHubSpace(d_model=64, hub_dim=256)
        x = torch.randn(8, 64)
        encoded = uhs.encode(x)
        assert encoded.shape == (8, 256)
        decoded = uhs.decode(encoded)
        assert decoded.shape == (8, 64)
