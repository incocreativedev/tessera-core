"""Tests for tessera.correspondence — CKA-based layer matching."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tessera.correspondence import (
    LayerCorrespondence,
    collect_layer_activations,
    compute_cka_matrix,
    linear_cka,
    match_layers,
)


# ---------------------------------------------------------------------------
# linear_cka unit tests
# ---------------------------------------------------------------------------


class TestLinearCKA:
    def test_identical_matrices_returns_one(self):
        """CKA of a matrix with itself must equal 1.0."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 32)).astype(np.float32)
        score = linear_cka(X, X)
        assert abs(score - 1.0) < 1e-5

    def test_scale_invariance(self):
        """CKA is invariant to positive scaling of inputs."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 16)).astype(np.float32)
        Y = rng.standard_normal((50, 16)).astype(np.float32)
        score_orig = linear_cka(X, Y)
        score_scaled = linear_cka(X * 42.0, Y * 0.001)
        assert abs(score_orig - score_scaled) < 1e-5

    def test_orthogonal_matrices_near_zero(self):
        """CKA of two independent random matrices should be close to zero."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((200, 64)).astype(np.float32)
        Y = rng.standard_normal((200, 64)).astype(np.float32)
        score = linear_cka(X, Y)
        assert score < 0.3  # Very unlikely to exceed 0.3 with 200 samples

    def test_output_in_zero_one(self):
        """CKA score must always be in [0, 1]."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            X = rng.standard_normal((30, 8)).astype(np.float32)
            Y = rng.standard_normal((30, 12)).astype(np.float32)
            score = linear_cka(X, Y)
            assert 0.0 <= score <= 1.0

    def test_different_d_dimensions_allowed(self):
        """CKA should work when X and Y have different feature dimensions."""
        rng = np.random.default_rng(4)
        X = rng.standard_normal((40, 32)).astype(np.float32)
        Y = rng.standard_normal((40, 128)).astype(np.float32)
        score = linear_cka(X, Y)
        assert 0.0 <= score <= 1.0

    def test_mismatched_n_raises(self):
        """CKA must raise if sample counts differ."""
        X = np.ones((10, 4), dtype=np.float32)
        Y = np.ones((20, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            linear_cka(X, Y)

    def test_zero_matrix_returns_zero(self):
        """CKA of zero matrix with anything returns 0 (denom is zero)."""
        X = np.zeros((20, 8), dtype=np.float32)
        Y = np.random.randn(20, 8).astype(np.float32)
        score = linear_cka(X, Y)
        assert score == 0.0


# ---------------------------------------------------------------------------
# compute_cka_matrix tests
# ---------------------------------------------------------------------------


class TestComputeCKAMatrix:
    def test_shape(self):
        """Matrix shape must be (n_tx, n_rx)."""
        rng = np.random.default_rng(10)
        tx_acts = {f"tx_{i}": rng.standard_normal((30, 16)).astype(np.float32) for i in range(3)}
        rx_acts = {f"rx_{j}": rng.standard_normal((30, 16)).astype(np.float32) for j in range(2)}
        matrix, tx_names, rx_names = compute_cka_matrix(tx_acts, rx_acts)
        assert matrix.shape == (3, 2)
        assert len(tx_names) == 3
        assert len(rx_names) == 2

    def test_diagonal_high_for_identical(self):
        """Diagonal elements should be ~1.0 when tx and rx have identical acts."""
        rng = np.random.default_rng(11)
        acts = {f"layer_{i}": rng.standard_normal((40, 16)).astype(np.float32) for i in range(2)}
        matrix, _, _ = compute_cka_matrix(acts, acts)
        for i in range(len(acts)):
            assert matrix[i, i] > 0.99


# ---------------------------------------------------------------------------
# match_layers tests
# ---------------------------------------------------------------------------


class TestMatchLayers:
    def _identity_matrix(self, n):
        m = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(m, 1.0)
        return m

    def test_greedy_same_arch(self):
        """Greedy on identity CKA matrix should match layers by index."""
        n = 4
        names = [f"layer_{i}" for i in range(n)]
        matrix = self._identity_matrix(n)
        matches = match_layers(matrix, names, names, strategy="greedy")
        for i, (tx, rx, score) in enumerate(matches):
            assert tx == rx == f"layer_{i}"
            assert abs(score - 1.0) < 1e-5

    def test_hungarian_same_arch(self):
        """Hungarian on identity matrix should give same result as greedy."""
        pytest.importorskip("scipy")
        n = 4
        names = [f"layer_{i}" for i in range(n)]
        matrix = self._identity_matrix(n)
        matches = match_layers(matrix, names, names, strategy="hungarian")
        matched_pairs = {(tx, rx) for tx, rx, _ in matches}
        for i in range(n):
            assert (f"layer_{i}", f"layer_{i}") in matched_pairs

    def test_greedy_cross_arch(self):
        """Greedy with rectangular matrix returns one match per tx layer."""
        rng = np.random.default_rng(20)
        tx_names = [f"tx_{i}" for i in range(3)]
        rx_names = [f"rx_{j}" for j in range(5)]
        matrix = rng.random((3, 5)).astype(np.float32)
        matches = match_layers(matrix, tx_names, rx_names, strategy="greedy")
        assert len(matches) == 3
        for tx, rx, score in matches:
            assert tx in tx_names
            assert rx in rx_names
            assert 0.0 <= score <= 1.0

    def test_returns_sorted_by_tx_order(self):
        """Output should be sorted in transmitter layer order."""
        n = 3
        tx_names = [f"tx_{i}" for i in range(n)]
        rx_names = [f"rx_{j}" for j in range(n)]
        matrix = np.eye(n, dtype=np.float32)
        matches = match_layers(matrix, tx_names, rx_names)
        assert [m[0] for m in matches] == tx_names


# ---------------------------------------------------------------------------
# collect_layer_activations tests
# ---------------------------------------------------------------------------


class TestCollectLayerActivations:
    def _make_loader(self):
        X = torch.randint(0, 128, (32, 8))
        y = torch.zeros(32).long()
        return DataLoader(TensorDataset(X, y), batch_size=16)

    def test_collects_correct_layers(self, small_tx):
        loader = self._make_loader()
        layer_names = ["layers.0"]
        acts = collect_layer_activations(small_tx, loader, layer_names)
        assert "layers.0" in acts
        assert acts["layers.0"].ndim == 2
        assert acts["layers.0"].shape[0] == 32  # 32 samples

    def test_returns_float32(self, small_tx):
        loader = self._make_loader()
        acts = collect_layer_activations(small_tx, loader, ["layers.0"])
        assert acts["layers.0"].dtype == np.float32


# ---------------------------------------------------------------------------
# LayerCorrespondence integration tests
# ---------------------------------------------------------------------------


class TestLayerCorrespondence:
    def test_same_arch_compute(self, small_tx, train_loader):
        """Same-architecture models: should return one match per tx layer."""
        corr = LayerCorrespondence(small_tx, small_tx, device="cpu")
        matches = corr.compute(train_loader)
        assert len(matches) > 0
        tx_names = [m[0] for m in matches]
        # All tx layer names should be present
        for name in tx_names:
            assert isinstance(name, str) and len(name) > 0
        # All scores in valid range
        for _, _, score in matches:
            assert 0.0 <= score <= 1.0

    def test_cross_arch_compute(self, small_tx, small_rx, train_loader):
        """Cross-architecture: should produce valid matches without crashing."""
        corr = LayerCorrespondence(small_tx, small_rx, device="cpu")
        matches = corr.compute(train_loader)
        assert len(matches) > 0
        for tx_name, rx_name, score in matches:
            assert isinstance(tx_name, str)
            assert isinstance(rx_name, str)
            assert 0.0 <= score <= 1.0

    def test_cka_matrix_shape(self, small_tx, small_rx, train_loader):
        """cka_matrix() should return a 2D array with correct axis sizes."""
        corr = LayerCorrespondence(small_tx, small_rx, device="cpu")
        matrix, tx_names, rx_names = corr.cka_matrix(train_loader)
        assert matrix.ndim == 2
        assert matrix.shape == (len(tx_names), len(rx_names))

    def test_hungarian_strategy(self, small_tx, train_loader):
        """Hungarian strategy should run without error (scipy required)."""
        pytest.importorskip("scipy")
        corr = LayerCorrespondence(small_tx, small_tx, device="cpu")
        matches = corr.compute(train_loader, strategy="hungarian")
        assert len(matches) > 0
