"""Tests for tessera.drift — Transfer fidelity measurement."""

import torch
import numpy as np
from tessera.drift import DriftMeasure
from tests.conftest import SmallTransformer


class TestDriftMeasure:
    def test_identical_models_low_drift(self, val_loader):
        model = SmallTransformer(d_model=64, num_layers=2)
        dm = DriftMeasure(model, model)
        drift = dm.compute(val_loader)
        # Same model should have ~0 drift
        assert drift < 0.01

    def test_different_models_positive_drift(self, val_loader):
        model_a = SmallTransformer(d_model=64, num_layers=2)
        model_b = SmallTransformer(d_model=64, num_layers=2)

        # Reinitialise model_b differently
        for p in model_b.parameters():
            torch.nn.init.normal_(p, std=5.0)

        dm = DriftMeasure(model_a, model_b)
        drift = dm.compute(val_loader)
        assert drift > 0

    def test_drift_non_negative(self, val_loader):
        model_a = SmallTransformer(d_model=64, num_layers=2)
        model_b = SmallTransformer(d_model=64, num_layers=2)
        dm = DriftMeasure(model_a, model_b)
        drift = dm.compute(val_loader)
        assert drift >= 0.0

    def test_cross_architecture_drift(self, val_loader):
        """Drift should work across different d_model sizes."""
        model_a = SmallTransformer(d_model=64, num_layers=2)
        model_b = SmallTransformer(d_model=128, num_layers=3)
        dm = DriftMeasure(model_a, model_b)
        drift = dm.compute(val_loader)
        assert drift >= 0.0

    def test_kl_diagonal_gaussian(self):
        """Test the KL divergence formula directly."""
        mu_p = np.zeros(10)
        var_p = np.ones(10)
        mu_q = np.zeros(10)
        var_q = np.ones(10)

        kl = DriftMeasure._kl_diagonal_gaussian(mu_p, var_p, mu_q, var_q)
        assert abs(kl) < 1e-10  # Identical distributions → KL = 0
