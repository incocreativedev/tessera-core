"""Tests for tessera.fingerprint — Activation fingerprinting."""

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tessera.fingerprint import (
    ActivationFingerprint, LayerFingerprint, compute_fingerprints,
)
from tests.conftest import SmallTransformer


class TestLayerFingerprint:
    def test_fields(self):
        d = 64
        fp = LayerFingerprint(
            layer_name="layers.0",
            layer_idx=0,
            mean=np.zeros(d),
            variance=np.ones(d),
            pca_top5=np.eye(5, d),
            pca_top5_ev=np.ones(5),
            pca_top10=np.eye(10, d),
            pca_top10_ev=np.ones(10),
            intrinsic_dim=32.0,
            d_layer=d,
            token_count=100,
        )
        assert fp.layer_name == "layers.0"
        assert fp.d_layer == d
        assert fp.intrinsic_dim == 32.0

    def test_cosine_similarity_identical(self):
        d = 16
        fp = LayerFingerprint(
            layer_name="l", layer_idx=0,
            mean=np.ones(d), variance=np.ones(d),
            pca_top5=np.eye(5, d), pca_top5_ev=np.ones(5),
            pca_top10=np.eye(10, d), pca_top10_ev=np.ones(10),
            intrinsic_dim=8.0, d_layer=d, token_count=50,
        )
        assert abs(fp.cosine_similarity(fp) - 1.0) < 1e-6


class TestComputeFingerprints:
    def test_basic(self, train_loader):
        model = SmallTransformer(d_model=64, num_layers=2)
        fps = compute_fingerprints(model, train_loader)

        assert isinstance(fps, dict)
        assert len(fps) > 0

        for name, fp in fps.items():
            assert isinstance(fp, LayerFingerprint)

    def test_layer_stats_shape(self, train_loader):
        model = SmallTransformer(d_model=64, num_layers=2)
        fps = compute_fingerprints(model, train_loader)

        for name, fp in fps.items():
            assert fp.mean.shape[0] == fp.d_layer
            assert fp.variance.shape[0] == fp.d_layer
            assert np.all(fp.variance >= 0)

    def test_different_models_different_fingerprints(self, train_loader):
        model_a = SmallTransformer(d_model=64, num_layers=2)
        model_b = SmallTransformer(d_model=64, num_layers=2)

        for p in model_b.parameters():
            torch.nn.init.normal_(p, std=2.0)

        fps_a = compute_fingerprints(model_a, train_loader)
        fps_b = compute_fingerprints(model_b, train_loader)

        key_a = list(fps_a.keys())[0]
        key_b = list(fps_b.keys())[0]
        assert not np.allclose(fps_a[key_a].mean, fps_b[key_b].mean)
