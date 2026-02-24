"""
tessera.mode_aq — Mode AQ (Quantisation-Aware Activation Transfer).

Extends Mode A with quantisation-aware decoding and fine-tuning for
deployment to INT8/INT4 models on constrained edge hardware.

Pipeline (steps 4-6 overridden from ModeATransfer):
    1. Fingerprint transmitter model                    [inherited]
    2. Collect raw activations for UHS training          [inherited]
    3. Train transmitter UHS encoder/decoder             [inherited]
    4. Train receiver UHS with QuantDecoderMLP            [OVERRIDDEN]
    5. Encode transmitter → quantised-decode to receiver  [OVERRIDDEN]
    6. Fine-tune receiver with fake-quantised alignment   [OVERRIDDEN]
    7. Measure drift (adds quant-aware fidelity metric)   [OVERRIDDEN]

Supported targets:
    - INT8 (default): 256 levels, per-channel symmetric
    - INT4: 16 levels, per-channel symmetric
    - FP16: Half-precision (for GPU edge like Jetson)
    - Custom: Arbitrary (n_bits, symmetric) via QuantConfig

Hardware alignment:
    - ARM Cortex-A (NEON INT8): Primary target, tested
    - Qualcomm Hexagon DSP: INT8 symmetric quantisation
    - RISC-V Vector Extension: INT8 symmetric
    - Edge TPU (Coral): INT8 asymmetric (use symmetric=False)
    - NVIDIA Jetson (FP16): Use quant_target="fp16"
"""

import datetime
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .transfer import ModeATransfer
from .uhs import UniversalHubSpace
from .drift import DriftMeasure
from .privacy import DifferentialPrivacy
from .token import TesseraToken, KnowledgeType
from .utils import setup_logging

logger = setup_logging("tessera.mode_aq")


# ── Quantisation configuration ───────────────────────────────────────────


class QuantTarget(Enum):
    """Supported quantisation targets."""

    INT8 = "int8"
    INT4 = "int4"
    FP16 = "fp16"


@dataclass
class QuantConfig:
    """
    Quantisation configuration for the receiver model.

    Attributes:
        target: Quantisation target (INT8, INT4, FP16).
        n_bits: Number of bits (auto-set from target if 0).
        symmetric: Use symmetric quantisation (True for most ARM/RISC-V).
                   Asymmetric (False) for Edge TPU/Coral.
        per_channel: Per-channel quantisation (True) or per-tensor (False).
        calibration_batches: Number of batches for scale/zero-point calibration.
    """

    target: QuantTarget = QuantTarget.INT8
    n_bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    calibration_batches: int = 16

    def __post_init__(self):
        if self.n_bits == 0:
            bit_map = {QuantTarget.INT8: 8, QuantTarget.INT4: 4, QuantTarget.FP16: 16}
            self.n_bits = bit_map.get(self.target, 8)

    @property
    def q_min(self) -> int:
        if self.target == QuantTarget.FP16:
            return 0
        if self.symmetric:
            return -(2 ** (self.n_bits - 1))
        return 0

    @property
    def q_max(self) -> int:
        if self.target == QuantTarget.FP16:
            return 0
        if self.symmetric:
            return 2 ** (self.n_bits - 1) - 1
        return 2**self.n_bits - 1


# ── Fake quantisation primitives ─────────────────────────────────────────


def _compute_scale_zero_point(
    tensor: torch.Tensor,
    config: QuantConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel (or per-tensor) scale and zero_point.

    For symmetric: scale = max(|x|) / q_max, zero_point = 0
    For asymmetric: scale = (max - min) / (q_max - q_min),
                    zero_point = q_min - round(min / scale)
    """
    if config.target == QuantTarget.FP16:
        return torch.ones(1, device=tensor.device), torch.zeros(1, device=tensor.device)

    if config.per_channel and tensor.ndim >= 2:
        reduce_dims = tuple(range(1, tensor.ndim))
    else:
        reduce_dims = None

    if config.symmetric:
        if reduce_dims is not None:
            max_abs = tensor.abs().amax(dim=reduce_dims, keepdim=True).clamp(min=1e-8)
        else:
            max_abs = tensor.abs().max().clamp(min=1e-8)
        scale = max_abs / config.q_max
        zero_point = torch.zeros_like(scale)
    else:
        if reduce_dims is not None:
            x_min = tensor.amin(dim=reduce_dims, keepdim=True)
            x_max = tensor.amax(dim=reduce_dims, keepdim=True)
        else:
            x_min = tensor.min()
            x_max = tensor.max()
        x_range = (x_max - x_min).clamp(min=1e-8)
        scale = x_range / (config.q_max - config.q_min)
        zero_point = torch.round(config.q_min - x_min / scale)

    return scale, zero_point


class FakeQuantize(torch.autograd.Function):
    """
    Straight-through estimator for fake quantisation.

    Forward: quantise → dequantise (introduces quantisation error)
    Backward: identity (gradient flows through unchanged within clamp range)
    """

    @staticmethod
    def forward(ctx, x, scale, zero_point, q_min, q_max):
        x_q = torch.clamp(torch.round(x / scale) + zero_point, q_min, q_max)
        x_hat = (x_q - zero_point) * scale
        ctx.save_for_backward(
            x,
            scale,
            torch.tensor([q_min], device=x.device),
            torch.tensor([q_max], device=x.device),
        )
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):
        x, scale, q_min_t, q_max_t = ctx.saved_tensors
        q_min = q_min_t.item()
        q_max = q_max_t.item()
        x_q_float = x / scale
        mask = (x_q_float >= q_min) & (x_q_float <= q_max)
        return grad_output * mask.float(), None, None, None, None


def fake_quantize(x: torch.Tensor, config: QuantConfig) -> torch.Tensor:
    """Apply fake quantisation to a tensor using the given config."""
    if config.target == QuantTarget.FP16:
        return x.half().float()

    scale, zero_point = _compute_scale_zero_point(x.detach(), config)
    return FakeQuantize.apply(x, scale, zero_point, config.q_min, config.q_max)


# ── Quantisation-aware decoder ───────────────────────────────────────────


class QuantDecoderMLP(nn.Module):
    """
    UHS decoder that outputs values snapped to the target quantisation grid.

    During training: uses fake quantisation (STE) so gradients flow.
    During inference: outputs are genuinely quantised to the target grid.

    Same architecture as DecoderMLP (Linear→LayerNorm→GELU→Linear) with
    a FakeQuantize pass on the output.
    """

    def __init__(
        self,
        d_model: int,
        hub_dim: int = 2048,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.hub_dim = hub_dim
        self.quant_config = quant_config or QuantConfig()

        h = max(d_model, hub_dim)
        self.fc1 = nn.Linear(hub_dim, h)
        self.ln = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.ln(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = fake_quantize(x, self.quant_config)
        return x


# ── ModeAQ Transfer ──────────────────────────────────────────────────────


class ModeAQTransfer(ModeATransfer):
    """
    Quantisation-aware Mode A transfer for edge deployment.

    Extends ModeATransfer by:
        1. Using QuantDecoderMLP for the receiver's UHS decoder
        2. Applying fake quantisation during receiver fine-tuning
        3. Measuring quantisation-specific fidelity (SQNR, quant drift)
        4. Recording quant metadata in the output token

    Usage:
        transfer = ModeAQTransfer(
            transmitter=server_model,
            receiver=edge_model,
            transmitter_id="server_v3",
            receiver_id="cortex_a_device_001",
            quant_config=QuantConfig(target=QuantTarget.INT8),
        )
        token = transfer.execute(train_loader, val_loader)
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        transmitter_id: str = "transmitter",
        receiver_id: str = "receiver",
        device: str = "cpu",
        hub_dim: int = 2048,
        quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__(transmitter, receiver, transmitter_id, receiver_id, device, hub_dim)
        self.quant_config = quant_config or QuantConfig()
        self.quant_fidelity: Optional[Dict[str, float]] = None

    def execute(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        tx_layers: Optional[List[str]] = None,
        rx_layers: Optional[List[str]] = None,
        uhs_epochs: int = 10,
        finetune_epochs: int = 5,
        finetune_lr: float = 1e-3,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ) -> TesseraToken:
        """Execute quantisation-aware Mode A transfer."""
        logger.info("[ModeAQ] Starting quantisation-aware transfer")
        logger.info(
            f"  Target: {self.quant_config.target.value}, "
            f"{self.quant_config.n_bits}-bit, "
            f"{'symmetric' if self.quant_config.symmetric else 'asymmetric'}, "
            f"{'per-channel' if self.quant_config.per_channel else 'per-tensor'}"
        )

        # ── Steps 1-3: identical to ModeATransfer ────────────────────
        from .fingerprint import compute_fingerprints

        logger.info("[Step 1/7] Computing activation fingerprints...")
        self.tx_fingerprints = compute_fingerprints(
            self.transmitter, train_dataloader, tx_layers, self.device
        )
        self.rx_fingerprints = compute_fingerprints(
            self.receiver, train_dataloader, rx_layers, self.device
        )

        tx_layer_names = sorted(self.tx_fingerprints.keys())
        rx_layer_names = sorted(self.rx_fingerprints.keys())
        tx_d = self.tx_fingerprints[tx_layer_names[0]].d_layer
        rx_d = self.rx_fingerprints[rx_layer_names[0]].d_layer
        logger.info(f"  TX d_model={tx_d}, RX d_model={rx_d}")

        logger.info("[Step 2/7] Collecting activations for UHS training...")
        tx_acts = self._collect_activations(self.transmitter, train_dataloader, tx_layer_names)
        rx_acts = self._collect_activations(self.receiver, train_dataloader, rx_layer_names)
        tx_pooled = self._pool_activations(tx_acts, tx_layer_names)
        rx_pooled = self._pool_activations(rx_acts, rx_layer_names)

        logger.info("[Step 3/7] Training transmitter UHS encoder/decoder...")
        self.tx_uhs = UniversalHubSpace(tx_d, hub_dim=self.hub_dim, device=self.device)
        tx_act_loader = DataLoader(
            TensorDataset(torch.tensor(tx_pooled, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
        )
        self.tx_uhs.train(tx_act_loader, epochs=uhs_epochs, verbose=True)
        rt_error_tx = self.tx_uhs.round_trip_error(
            torch.tensor(tx_pooled[:100], dtype=torch.float32)
        )

        # ── Step 4: Train receiver UHS with QuantDecoderMLP ──────────
        logger.info("[Step 4/7] Training receiver UHS with quant-aware decoder...")
        self.rx_uhs = UniversalHubSpace(rx_d, hub_dim=self.hub_dim, device=self.device)
        self.rx_uhs.decoder = QuantDecoderMLP(
            rx_d,
            hub_dim=self.hub_dim,
            quant_config=self.quant_config,
        ).to(self.device)
        rx_act_loader = DataLoader(
            TensorDataset(torch.tensor(rx_pooled, dtype=torch.float32)),
            batch_size=32,
            shuffle=True,
        )
        self.rx_uhs.train(rx_act_loader, epochs=uhs_epochs, verbose=True)
        rt_error_rx = self.rx_uhs.round_trip_error(
            torch.tensor(rx_pooled[:100], dtype=torch.float32)
        )
        logger.info(f"  Receiver UHS quant round-trip error: {rt_error_rx:.4f}")

        # ── Step 5: Encode TX → quantised-decode into RX ─────────────
        logger.info("[Step 5/7] Transferring through hub space (quant-aware decode)...")
        tx_tensor = torch.tensor(tx_pooled, dtype=torch.float32).to(self.device)
        hub_vectors = self.tx_uhs.encode(tx_tensor)
        decoded_targets = self.rx_uhs.decode(hub_vectors).detach()
        logger.info(f"  Hub vectors shape: {hub_vectors.shape}")
        logger.info(f"  Decoded targets shape: {decoded_targets.shape}")

        # ── Step 6: Fine-tune receiver with fake quantisation ────────
        logger.info("[Step 6/7] Fine-tuning receiver (quant-aware)...")
        self._finetune_receiver_quant(
            self.receiver,
            train_dataloader,
            decoded_targets,
            rx_layer_names,
            epochs=finetune_epochs,
            lr=finetune_lr,
        )

        # ── Step 7: Measure drift + quant fidelity ───────────────────
        logger.info("[Step 7/7] Measuring drift and quantisation fidelity...")
        drift = DriftMeasure(self.transmitter, self.receiver, self.device).compute(val_dataloader)

        self.quant_fidelity = self._measure_quant_fidelity(hub_vectors, decoded_targets)

        dp = DifferentialPrivacy(privacy_epsilon, privacy_delta)
        avg_hub = hub_vectors.mean(dim=0).cpu().numpy()
        private_hub = dp.add_noise(avg_hub)
        compat_score = self._compatibility_score()

        token = TesseraToken(
            knowledge_type=KnowledgeType.ACTIVATION,
            uhs_vector=private_hub.tolist(),
            modality_weights={"A": 0.90, "W": 0.05, "B": 0.05},
            correlation_map={},
            lineage_dag={
                "nodes": [
                    {"id": "n0", "type": "anchor", "ref": self.transmitter_id},
                ],
                "root": "n0",
            },
            generation=1,
            projection_hints=[],
            privacy_epsilon=privacy_epsilon,
            privacy_delta=privacy_delta,
            drift_score=drift,
            source_model_id=self.transmitter_id,
            target_model_id=self.receiver_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            version="1.0",
            custom_metadata={
                "transfer_mode": "AQ",
                "tx_d_model": tx_d,
                "rx_d_model": rx_d,
                "tx_layers": len(tx_layer_names),
                "rx_layers": len(rx_layer_names),
                "compatibility_score": compat_score,
                "tx_round_trip_error": rt_error_tx,
                "rx_round_trip_error": rt_error_rx,
                "quant_target": self.quant_config.target.value,
                "quant_n_bits": self.quant_config.n_bits,
                "quant_symmetric": self.quant_config.symmetric,
                "quant_per_channel": self.quant_config.per_channel,
                "quant_sqnr_db": self.quant_fidelity["sqnr_db"],
                "quant_relative_error": self.quant_fidelity["relative_error"],
                "quant_clipping_fraction": self.quant_fidelity["clipping_fraction"],
            },
        )

        logger.info("  ModeAQ transfer complete!")
        logger.info(f"  Drift:           {drift:.6f}")
        logger.info(f"  SQNR:            {self.quant_fidelity['sqnr_db']:.1f} dB")
        logger.info(f"  Quant rel error: {self.quant_fidelity['relative_error']:.4f}")
        logger.info(f"  Clipping:        {self.quant_fidelity['clipping_fraction']:.2%}")

        return token

    # ── Quant-aware fine-tuning ──────────────────────────────────────

    def _finetune_receiver_quant(
        self,
        receiver: nn.Module,
        dataloader: DataLoader,
        targets: torch.Tensor,
        rx_layer_names: List[str],
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        """
        Fine-tune the receiver with fake-quantised activations.

        Like the parent's _finetune_receiver, but applies fake_quantize to
        the captured activations before computing MSE loss. This teaches
        the receiver to produce activations that survive quantisation.
        """
        receiver.train()
        receiver.to(self.device)
        targets = targets.to(self.device)

        optimiser = torch.optim.Adam(receiver.parameters(), lr=lr)
        mid_layer = rx_layer_names[len(rx_layer_names) // 2]

        for epoch in range(epochs):
            epoch_loss = 0.0
            sample_idx = 0

            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                batch_size = x.shape[0]
                if sample_idx + batch_size > len(targets):
                    break
                batch_targets = targets[sample_idx : sample_idx + batch_size]
                sample_idx += batch_size

                captured = {}

                def capture_hook(module, inp, output):
                    out = output[0] if isinstance(output, tuple) else output
                    if out.ndim == 3:
                        out = out.mean(dim=1)
                    captured["act"] = out

                hook = None
                for name, module in receiver.named_modules():
                    if name == mid_layer:
                        hook = module.register_forward_hook(capture_hook)
                        break

                x = x.to(self.device)
                receiver(x)

                if hook is not None:
                    hook.remove()
                if "act" not in captured:
                    continue

                act = captured["act"]
                tgt = batch_targets

                if act.shape[-1] != tgt.shape[-1]:
                    tgt = F.adaptive_avg_pool1d(tgt.unsqueeze(1), act.shape[-1]).squeeze(1)

                # KEY DIFFERENCE: fake-quantise activations before loss
                act_q = fake_quantize(act, self.quant_config)
                loss = F.mse_loss(act_q, tgt)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            logger.info(f"  QAT epoch {epoch + 1}/{epochs}: " f"alignment_loss={epoch_loss:.4f}")

        receiver.eval()

    # ── Quantisation fidelity metrics ────────────────────────────────

    def _measure_quant_fidelity(
        self,
        hub_vectors: torch.Tensor,
        decoded_targets: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measure quantisation-specific fidelity metrics.

        Returns:
            sqnr_db: Signal-to-quantisation-noise ratio in dB.
                     >20 dB is excellent, >10 dB is acceptable for INT8.
            relative_error: ||x - x_q|| / ||x|| (Frobenius norm ratio).
            clipping_fraction: Fraction of values that hit the quant boundary.
        """
        x = decoded_targets.float()
        x_q = fake_quantize(x, self.quant_config)
        noise = x - x_q

        # SQNR
        signal_power = (x**2).mean().item()
        noise_power = (noise**2).mean().item()
        if noise_power < 1e-12:
            sqnr_db = 100.0  # Effectively perfect (FP16 or no quantisation)
        else:
            sqnr_db = 10.0 * math.log10(signal_power / noise_power)

        # Relative error
        rel_error = torch.norm(noise, p="fro") / (torch.norm(x, p="fro") + 1e-8)

        # Clipping fraction
        if self.quant_config.target != QuantTarget.FP16:
            scale, zp = _compute_scale_zero_point(x.detach(), self.quant_config)
            x_scaled = x / scale
            clipped = (
                ((x_scaled < self.quant_config.q_min) | (x_scaled > self.quant_config.q_max))
                .float()
                .mean()
                .item()
            )
        else:
            clipped = 0.0

        return {
            "sqnr_db": sqnr_db,
            "relative_error": rel_error.item(),
            "clipping_fraction": clipped,
        }
