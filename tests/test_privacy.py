"""Tests for tessera.privacy — Differential privacy."""

import math
import pytest
import numpy as np
import torch
from tessera.privacy import DifferentialPrivacy


class TestDifferentialPrivacy:
    def test_create_default(self):
        dp = DifferentialPrivacy()
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.sigma > 0

    def test_sigma_calibration(self):
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        expected = math.sqrt(2.0 * math.log(1.25 / 1e-5)) / 1.0
        assert abs(dp.sigma - expected) < 1e-10

    def test_higher_epsilon_less_noise(self):
        dp_private = DifferentialPrivacy(epsilon=0.1, delta=1e-5)
        dp_relaxed = DifferentialPrivacy(epsilon=10.0, delta=1e-5)
        assert dp_private.sigma > dp_relaxed.sigma

    def test_infinite_epsilon_no_noise(self):
        dp = DifferentialPrivacy(epsilon=float("inf"), delta=1e-5)
        assert dp.sigma == 0.0
        vec = np.array([1.0, 2.0, 3.0])
        noisy = dp.add_noise(vec)
        np.testing.assert_array_equal(vec, noisy)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="positive"):
            DifferentialPrivacy(epsilon=-1.0)
        with pytest.raises(ValueError, match="positive"):
            DifferentialPrivacy(epsilon=0.0)

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=1.0, delta=0.0)
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=1.0, delta=1.0)
        with pytest.raises(ValueError):
            DifferentialPrivacy(epsilon=1.0, delta=-0.5)

    def test_add_noise_shape_preserved(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        vec = np.random.randn(2048).astype(np.float32)
        noisy = dp.add_noise(vec)
        assert noisy.shape == vec.shape
        assert not np.array_equal(noisy, vec)  # Noise was added

    def test_add_noise_2d(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        vec = np.random.randn(10, 128).astype(np.float32)
        noisy = dp.add_noise(vec)
        assert noisy.shape == (10, 128)

    def test_add_noise_tensor(self):
        dp = DifferentialPrivacy(epsilon=1.0)
        t = torch.randn(2048)
        noisy = dp.add_noise_tensor(t)
        assert noisy.shape == t.shape
        assert not torch.equal(noisy, t)

    def test_add_noise_tensor_no_noise(self):
        dp = DifferentialPrivacy(epsilon=float("inf"))
        t = torch.tensor([1.0, 2.0, 3.0])
        noisy = dp.add_noise_tensor(t)
        torch.testing.assert_close(noisy, t)

    def test_repr(self):
        dp = DifferentialPrivacy(epsilon=8.0, delta=1e-5)
        s = repr(dp)
        assert "8.0" in s
        assert "1e-05" in s
