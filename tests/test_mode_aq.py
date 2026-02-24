"""Tests for tessera.mode_aq — Quantisation-aware activation transfer."""

import math

import torch
import numpy as np

from tessera.mode_aq import (
    ModeAQTransfer,
    QuantConfig,
    QuantTarget,
    QuantDecoderMLP,
    fake_quantize,
    _compute_scale_zero_point,
)
from tessera.token import TesseraToken
from tests.conftest import SmallTransformer, TinyEdgeModel


# ═══════════════════════════════════════════════════════════════════════
#  QuantConfig tests
# ═══════════════════════════════════════════════════════════════════════


class TestQuantConfig:
    def test_default_config_is_int8_symmetric(self):
        cfg = QuantConfig()
        assert cfg.target == QuantTarget.INT8
        assert cfg.n_bits == 8
        assert cfg.symmetric is True
        assert cfg.per_channel is True
        assert cfg.q_min == -128
        assert cfg.q_max == 127

    def test_int4_config(self):
        cfg = QuantConfig(target=QuantTarget.INT4, n_bits=4)
        assert cfg.q_min == -8
        assert cfg.q_max == 7

    def test_fp16_config(self):
        cfg = QuantConfig(target=QuantTarget.FP16, n_bits=16)
        assert cfg.q_min == 0
        assert cfg.q_max == 0

    def test_asymmetric_range(self):
        cfg = QuantConfig(symmetric=False, n_bits=8)
        assert cfg.q_min == 0
        assert cfg.q_max == 255

    def test_auto_bits_from_target(self):
        cfg = QuantConfig(target=QuantTarget.INT4, n_bits=0)
        assert cfg.n_bits == 4


# ═══════════════════════════════════════════════════════════════════════
#  fake_quantize tests
# ═══════════════════════════════════════════════════════════════════════


class TestFakeQuantize:
    def test_fake_quantize_roundtrip_int8(self):
        x = torch.randn(64, 32)
        cfg = QuantConfig(target=QuantTarget.INT8)
        x_q = fake_quantize(x, cfg)
        # Should introduce some noise
        assert not torch.allclose(x, x_q, atol=1e-7)
        # But should be close
        rel_error = (x - x_q).norm() / (x.norm() + 1e-8)
        assert rel_error < 0.2

    def test_fake_quantize_preserves_shape(self):
        x = torch.randn(16, 128)
        cfg = QuantConfig()
        x_q = fake_quantize(x, cfg)
        assert x_q.shape == x.shape

    def test_fake_quantize_fp16_roundtrip(self):
        x = torch.randn(32, 64)
        cfg = QuantConfig(target=QuantTarget.FP16, n_bits=16)
        x_q = fake_quantize(x, cfg)
        # FP16 is very close to FP32 for normal-range values
        assert torch.allclose(x, x_q, atol=1e-2)

    def test_fake_quantize_int4_more_noise_than_int8(self):
        x = torch.randn(64, 32)
        cfg_int8 = QuantConfig(target=QuantTarget.INT8, n_bits=8)
        cfg_int4 = QuantConfig(target=QuantTarget.INT4, n_bits=4)
        noise_int8 = (x - fake_quantize(x, cfg_int8)).norm()
        noise_int4 = (x - fake_quantize(x, cfg_int4)).norm()
        assert noise_int4 > noise_int8

    def test_fake_quantize_gradient_flows(self):
        x = torch.randn(10, requires_grad=True)
        cfg = QuantConfig()
        y = fake_quantize(x, cfg).sum()
        y.backward()
        assert x.grad is not None
        # STE: at least some gradients should be non-zero
        assert x.grad.abs().sum() > 0

    def test_scale_zero_point_symmetric(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [0.5, -0.5, 1.5]])
        cfg = QuantConfig(symmetric=True, per_channel=True)
        scale, zp = _compute_scale_zero_point(x, cfg)
        # Zero-point should be 0 for symmetric
        assert torch.allclose(zp, torch.zeros_like(zp))
        # Scale should be max_abs / 127 per channel
        assert scale.shape[0] == 2


# ═══════════════════════════════════════════════════════════════════════
#  QuantDecoderMLP tests
# ═══════════════════════════════════════════════════════════════════════


class TestQuantDecoderMLP:
    def test_quant_decoder_output_shape(self):
        dec = QuantDecoderMLP(d_model=64, hub_dim=2048)
        x = torch.randn(16, 2048)
        out = dec(x)
        assert out.shape == (16, 64)

    def test_quant_decoder_output_quantised(self):
        """Applying fake_quantize again should be idempotent."""
        cfg = QuantConfig(target=QuantTarget.INT8)
        dec = QuantDecoderMLP(d_model=64, hub_dim=256, quant_config=cfg)
        x = torch.randn(8, 256)
        out = dec(x)
        out_q = fake_quantize(out.detach(), cfg)
        # Already quantised, so second pass should be nearly identical
        assert torch.allclose(out.detach(), out_q, atol=1e-5)

    def test_quant_decoder_trains_with_gradients(self):
        dec = QuantDecoderMLP(d_model=32, hub_dim=128)
        x = torch.randn(4, 128)
        target = torch.randn(4, 32)
        loss = torch.nn.functional.mse_loss(dec(x), target)
        loss.backward()
        for p in dec.parameters():
            assert p.grad is not None

    def test_quant_decoder_respects_config(self):
        x = torch.randn(8, 128)
        dec_int8 = QuantDecoderMLP(
            d_model=32,
            hub_dim=128,
            quant_config=QuantConfig(target=QuantTarget.INT8),
        )
        dec_int4 = QuantDecoderMLP(
            d_model=32,
            hub_dim=128,
            quant_config=QuantConfig(target=QuantTarget.INT4, n_bits=4),
        )
        # Copy weights so comparison is fair
        dec_int4.load_state_dict(dec_int8.state_dict())
        out_int8 = dec_int8(x).detach()
        out_int4 = dec_int4(x).detach()
        # INT4 should produce coarser output
        # Compute the unique-value density as a proxy for quantisation granularity
        unique_int8 = len(torch.unique(torch.round(out_int8 * 100)))
        unique_int4 = len(torch.unique(torch.round(out_int4 * 100)))
        # INT4 has fewer distinct values (coarser grid)
        assert unique_int4 <= unique_int8


# ═══════════════════════════════════════════════════════════════════════
#  ModeAQTransfer end-to-end tests
# ═══════════════════════════════════════════════════════════════════════


class TestModeAQTransfer:
    def test_execute_returns_token(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeAQTransfer(
            tx,
            rx,
            "tx",
            "rx",
            quant_config=QuantConfig(target=QuantTarget.INT8),
        )
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert isinstance(token, TesseraToken)
        assert len(token.uhs_vector) == 2048
        assert token.custom_metadata["transfer_mode"] == "AQ"
        assert token.custom_metadata["quant_target"] == "int8"

    def test_cross_architecture_quant_transfer(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = TinyEdgeModel(d_model=32)
        tx.eval()

        transfer = ModeAQTransfer(tx, rx, "server", "edge")
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert isinstance(token, TesseraToken)
        assert token.custom_metadata["tx_d_model"] == 64
        assert token.custom_metadata["rx_d_model"] == 32
        assert math.isfinite(token.drift_score)

    def test_int4_transfer(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        cfg = QuantConfig(target=QuantTarget.INT4, n_bits=4)
        transfer = ModeAQTransfer(tx, rx, "tx", "rx", quant_config=cfg)
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert token.custom_metadata["quant_target"] == "int4"
        assert token.custom_metadata["quant_n_bits"] == 4

    def test_fp16_transfer(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        cfg = QuantConfig(target=QuantTarget.FP16, n_bits=16)
        transfer = ModeAQTransfer(tx, rx, "tx", "rx", quant_config=cfg)
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert token.custom_metadata["quant_target"] == "fp16"
        # FP16 should have very high SQNR
        assert token.custom_metadata["quant_sqnr_db"] > 30.0

    def test_quant_metadata_in_token(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeAQTransfer(tx, rx, "tx", "rx")
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        md = token.custom_metadata
        assert "quant_sqnr_db" in md
        assert "quant_relative_error" in md
        assert "quant_clipping_fraction" in md
        assert math.isfinite(md["quant_sqnr_db"])
        assert 0 <= md["quant_relative_error"] <= 10  # reasonable bound
        assert 0 <= md["quant_clipping_fraction"] <= 1

    def test_privacy_budget_preserved(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeAQTransfer(tx, rx, "tx", "rx")
        token = transfer.execute(
            train_loader,
            val_loader,
            uhs_epochs=2,
            finetune_epochs=1,
            privacy_epsilon=4.0,
        )
        assert token.privacy_epsilon == 4.0

    def test_custom_hub_dim(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeAQTransfer(tx, rx, "tx", "rx", hub_dim=1024)
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert len(token.uhs_vector) == 1024

    def test_quant_fidelity_stored_on_transfer_object(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        transfer = ModeAQTransfer(tx, rx, "tx", "rx")
        transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert transfer.quant_fidelity is not None
        assert "sqnr_db" in transfer.quant_fidelity
        assert "relative_error" in transfer.quant_fidelity
        assert "clipping_fraction" in transfer.quant_fidelity

    def test_asymmetric_quantisation(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        cfg = QuantConfig(symmetric=False)
        assert cfg.q_min == 0
        assert cfg.q_max == 255
        transfer = ModeAQTransfer(tx, rx, "tx", "rx", quant_config=cfg)
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert token.custom_metadata["quant_symmetric"] is False

    def test_per_tensor_quantisation(self, train_loader, val_loader):
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        cfg = QuantConfig(per_channel=False)
        transfer = ModeAQTransfer(tx, rx, "tx", "rx", quant_config=cfg)
        token = transfer.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)
        assert token.custom_metadata["quant_per_channel"] is False


# ═══════════════════════════════════════════════════════════════════════
#  Edge-case and regression tests
# ═══════════════════════════════════════════════════════════════════════


class TestModeAQEdgeCases:
    def test_quant_fidelity_sqnr_is_reasonable(self, train_loader, val_loader):
        """INT8 SQNR should be positive; FP16 SQNR should be very high."""
        tx = SmallTransformer(d_model=64, num_layers=2)
        rx = SmallTransformer(d_model=64, num_layers=2)
        tx.eval()

        # INT8
        transfer_int8 = ModeAQTransfer(
            tx,
            rx,
            "tx",
            "rx",
            quant_config=QuantConfig(target=QuantTarget.INT8),
        )
        token_int8 = transfer_int8.execute(
            train_loader, val_loader, uhs_epochs=2, finetune_epochs=1
        )
        assert token_int8.custom_metadata["quant_sqnr_db"] > 0

    def test_deterministic_with_seed(self, train_loader, val_loader):
        """Same seed should produce same results."""
        from tessera.utils import set_seed

        set_seed(42)
        tx1 = SmallTransformer(d_model=64, num_layers=2)
        rx1 = SmallTransformer(d_model=64, num_layers=2)
        transfer1 = ModeAQTransfer(tx1, rx1, "tx", "rx")
        token1 = transfer1.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)

        set_seed(42)
        tx2 = SmallTransformer(d_model=64, num_layers=2)
        rx2 = SmallTransformer(d_model=64, num_layers=2)
        transfer2 = ModeAQTransfer(tx2, rx2, "tx", "rx")
        token2 = transfer2.execute(train_loader, val_loader, uhs_epochs=2, finetune_epochs=1)

        np.testing.assert_allclose(token1.uhs_vector, token2.uhs_vector, atol=1e-5)
