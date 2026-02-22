"""Tests for tessera.weight_ops — weight extraction, SVD, chunking, hub encoding."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from tessera.weight_ops import (
    WeightStats,
    _adapt_weight,
    chunk_for_hub,
    compute_weight_stats,
    extract_weights,
    initialize_receiver_weights,
    svd_compress,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_linear_model(in_f=32, out_f=64, hidden=16):
    return nn.Sequential(
        nn.Linear(in_f, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_f),
    )


# ---------------------------------------------------------------------------
# extract_weights
# ---------------------------------------------------------------------------


class TestExtractWeights:
    def test_extracts_linear_layers(self):
        model = _simple_linear_model()
        weights = extract_weights(model)
        # Should find both Linear layers
        assert len(weights) >= 2
        for name, W in weights.items():
            assert W.ndim == 2

    def test_correct_shapes(self):
        model = nn.Linear(32, 64)
        _ = extract_weights(model, layer_names=[""])
        # nn.Sequential wraps, so use named module directly
        model2 = nn.Sequential(nn.Linear(32, 64))
        weights2 = extract_weights(model2)
        assert len(weights2) == 1
        name, W = next(iter(weights2.items()))
        assert W.shape == (64, 32)

    def test_missing_layer_skipped(self):
        model = nn.Linear(8, 16)
        weights = extract_weights(model, layer_names=["nonexistent"])
        assert len(weights) == 0

    def test_conv2d_reshaped_to_2d(self):
        model = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3))
        weights = extract_weights(model)
        assert len(weights) == 1
        W = next(iter(weights.values()))
        assert W.ndim == 2  # (16, 3*3*3) = (16, 27)
        assert W.shape == (16, 27)


# ---------------------------------------------------------------------------
# svd_compress
# ---------------------------------------------------------------------------


class TestSVDCompress:
    def test_auto_rank_preserves_energy(self):
        rng = np.random.default_rng(0)
        W = rng.standard_normal((64, 128)).astype(np.float32)
        snap = svd_compress(W, energy_threshold=0.95)
        W_recon = snap.reconstruct()
        fro_orig = np.linalg.norm(W, "fro")
        fro_err = np.linalg.norm(W - W_recon, "fro")
        relative_err = fro_err / (fro_orig + 1e-8)
        # At 95% energy, reconstruction error should be small
        assert relative_err < 0.25

    def test_explicit_rank(self):
        rng = np.random.default_rng(1)
        W = rng.standard_normal((32, 64)).astype(np.float32)
        snap = svd_compress(W, rank=8)
        assert snap.rank == 8
        assert snap.U.shape == (32, 8)
        assert snap.S.shape == (8,)
        assert snap.Vt.shape == (8, 64)

    def test_rank_clamped_to_valid_range(self):
        W = np.ones((4, 4), dtype=np.float32)
        snap = svd_compress(W, rank=100)  # clamp to min(4, 4)
        assert snap.rank <= 4

    def test_reconstruct_shape(self):
        rng = np.random.default_rng(2)
        W = rng.standard_normal((16, 32)).astype(np.float32)
        snap = svd_compress(W, rank=4)
        W_recon = snap.reconstruct()
        assert W_recon.shape == (16, 32)

    def test_snapshot_dataclass_properties(self):
        W = np.eye(8, dtype=np.float32)
        snap = svd_compress(W, rank=3, layer_name="test_layer")
        assert snap.layer_name == "test_layer"
        assert snap.original_shape == (8, 8)
        assert snap.compressed_size == snap.U.size + snap.S.size + snap.Vt.size

    def test_1d_array_raises(self):
        W = np.ones(10, dtype=np.float32)
        with pytest.raises(AssertionError):
            svd_compress(W)


# ---------------------------------------------------------------------------
# chunk_for_hub
# ---------------------------------------------------------------------------


class TestChunkForHub:
    def test_all_chunks_correct_size(self):
        W = np.random.randn(32, 64).astype(np.float32)
        snap = svd_compress(W, rank=4)
        chunks, metas = chunk_for_hub(snap, hub_dim=128)
        for chunk in chunks:
            assert chunk.shape == (128,)
            assert chunk.dtype == np.float32

    def test_round_trip_flatness(self):
        """Chunks concatenated should reproduce the original flat array (approx)."""
        W = np.random.randn(16, 32).astype(np.float32)
        snap = svd_compress(W, rank=4)
        flat_orig = np.concatenate([snap.U.flatten(), snap.S, snap.Vt.flatten()])
        chunks, _ = chunk_for_hub(snap, hub_dim=64)
        flat_recovered = np.concatenate(chunks)[: len(flat_orig)]
        np.testing.assert_allclose(flat_orig, flat_recovered, atol=1e-5)

    def test_meta_counts_correct(self):
        W = np.random.randn(32, 32).astype(np.float32)
        snap = svd_compress(W, rank=8)
        hub_dim = 256
        chunks, metas = chunk_for_hub(snap, hub_dim=hub_dim)
        assert len(chunks) == len(metas)
        for i, meta in enumerate(metas):
            assert meta.chunk_idx == i
            assert meta.total_chunks == len(chunks)
            assert meta.rank == snap.rank

    def test_single_chunk_if_small_enough(self):
        W = np.random.randn(4, 4).astype(np.float32)
        snap = svd_compress(W, rank=2)
        # flat size = 4*2 + 2 + 2*4 = 8+2+8 = 18 floats → fits in hub_dim=64
        chunks, metas = chunk_for_hub(snap, hub_dim=64)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# _adapt_weight
# ---------------------------------------------------------------------------


class TestAdaptWeight:
    def test_no_change_when_same_shape(self):
        W = np.ones((4, 8), dtype=np.float32)
        W_out = _adapt_weight(W, (4, 8))
        assert W_out.shape == (4, 8)
        np.testing.assert_array_equal(W, W_out)

    def test_crops_rows_and_cols(self):
        W = np.ones((8, 16), dtype=np.float32)
        W_out = _adapt_weight(W, (4, 8))
        assert W_out.shape == (4, 8)

    def test_pads_rows_and_cols(self):
        W = np.ones((2, 4), dtype=np.float32)
        W_out = _adapt_weight(W, (6, 10))
        assert W_out.shape == (6, 10)
        # Original values preserved
        np.testing.assert_array_equal(W_out[:2, :4], W)
        # Padding is zero
        assert W_out[2:, :].sum() == 0.0
        assert W_out[:, 4:].sum() == 0.0


# ---------------------------------------------------------------------------
# initialize_receiver_weights
# ---------------------------------------------------------------------------


class TestInitializeReceiverWeights:
    def test_weights_change_after_init(self):
        model = nn.Linear(16, 32)
        orig = model.weight.data.clone()
        W_new = np.ones((32, 16), dtype=np.float32) * 99.0
        initialize_receiver_weights(model, {"": W_new})
        # Weight was a top-level module with name ""
        assert not torch.allclose(model.weight.data, orig)

    def test_sequential_layers_updated(self):
        model = _simple_linear_model(in_f=8, out_f=16, hidden=4)
        weights = extract_weights(model)
        # Overwrite with zeros
        zero_weights = {name: np.zeros_like(W) for name, W in weights.items()}
        initialize_receiver_weights(model, zero_weights)
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                assert module.weight.abs().sum().item() == 0.0

    def test_dimension_mismatch_handled(self):
        """initialize_receiver_weights must not raise on shape mismatch."""
        model = nn.Linear(16, 32)
        # Decoded weight has wrong shape
        W_wrong = np.ones((10, 20), dtype=np.float32)
        initialize_receiver_weights(model, {"": W_wrong})  # Should not raise

    def test_missing_layer_skipped(self):
        model = nn.Linear(8, 16)
        # Should not raise even if layer name doesn't exist
        initialize_receiver_weights(model, {"nonexistent_layer": np.ones((4, 4))})


# ---------------------------------------------------------------------------
# compute_weight_stats
# ---------------------------------------------------------------------------


class TestComputeWeightStats:
    def test_returns_stats_for_each_layer(self):
        model = _simple_linear_model()
        stats = compute_weight_stats(model)
        assert len(stats) >= 2
        for name, ws in stats.items():
            assert isinstance(ws, WeightStats)
            assert ws.frobenius_norm >= 0.0
            assert ws.spectral_norm >= 0.0
            assert ws.effective_rank >= 1.0

    def test_stat_fields_are_finite(self):
        model = nn.Linear(32, 64)
        stats = compute_weight_stats(model)
        for ws in stats.values():
            assert np.isfinite(ws.mean)
            assert np.isfinite(ws.std)
            assert np.isfinite(ws.frobenius_norm)
            assert np.isfinite(ws.spectral_norm)
            assert np.isfinite(ws.effective_rank)

    def test_shape_recorded_correctly(self):
        # Use Sequential so the Linear layer gets a proper named path
        model = nn.Sequential(nn.Linear(16, 32))
        stats = compute_weight_stats(model)
        assert len(stats) == 1
        ws = next(iter(stats.values()))
        assert ws.shape == (32, 16)
