"""
tessera.transfer — Mode A knowledge transfer orchestrator.

Mode A (Activation) is the primary transfer modality in the Tessera protocol.
It transfers how a model behaves (activation patterns) rather than what it
stores (weights), making it architecture-agnostic.

Pipeline:
    1. Fingerprint transmitter model (per-layer activation statistics)
    2. Train UHS encoder/decoder pairs for both models
    3. Extract transmitter activations and encode to hub space
    4. Decode hub vectors into receiver's representation space
    5. Fine-tune receiver to align activations with decoded targets
    6. Measure drift (KL-divergence fidelity)
    7. Package result as a TesseraToken
"""

import datetime
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fingerprint import ActivationFingerprint, LayerFingerprint, compute_fingerprints
from .uhs import UniversalHubSpace
from .drift import DriftMeasure
from .privacy import DifferentialPrivacy
from .token import TesseraToken, KnowledgeType
from .utils import setup_logging

logger = setup_logging("tessera.transfer")


class ModeATransfer:
    """
    Orchestrates a complete Mode A (activation-based) knowledge transfer
    between a transmitter and a receiver model.

    The two models can have different architectures, different depths,
    and different hidden dimensions. The UHS bridges the gap.

    Usage:
        transfer = ModeATransfer(
            transmitter=trained_model,
            receiver=untrained_model,
            transmitter_id="model_a",
            receiver_id="model_b",
        )
        token = transfer.execute(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        transmitter_id: str = "transmitter",
        receiver_id: str = "receiver",
        device: str = "cpu",
    ):
        self.transmitter = transmitter
        self.receiver = receiver
        self.transmitter_id = transmitter_id
        self.receiver_id = receiver_id
        self.device = device

        # Populated during transfer
        self.tx_fingerprints: Optional[Dict[str, LayerFingerprint]] = None
        self.rx_fingerprints: Optional[Dict[str, LayerFingerprint]] = None
        self.tx_uhs: Optional[UniversalHubSpace] = None
        self.rx_uhs: Optional[UniversalHubSpace] = None

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
        """
        Execute the full Mode A transfer pipeline.

        Args:
            train_dataloader: Data for fingerprinting, UHS training, and fine-tuning.
            val_dataloader: Data for drift measurement.
            tx_layers: Transmitter layers to transfer (auto-detect if None).
            rx_layers: Receiver layers to align (auto-detect if None).
            uhs_epochs: Epochs for UHS encoder/decoder training.
            finetune_epochs: Epochs for receiver fine-tuning.
            finetune_lr: Learning rate for receiver fine-tuning.
            privacy_epsilon: DP epsilon budget.
            privacy_delta: DP delta parameter.

        Returns:
            TesseraToken with transfer metadata and UHS vector.
        """

        # ── Step 1: Fingerprint both models ──────────────────────────────
        logger.info("[Step 1/7] Computing activation fingerprints...")

        self.tx_fingerprints = compute_fingerprints(
            self.transmitter, train_dataloader, tx_layers, self.device
        )
        self.rx_fingerprints = compute_fingerprints(
            self.receiver, train_dataloader, rx_layers, self.device
        )

        tx_layer_names = sorted(self.tx_fingerprints.keys())
        rx_layer_names = sorted(self.rx_fingerprints.keys())

        logger.info(f"  Transmitter: {len(tx_layer_names)} layers fingerprinted")
        logger.info(f"  Receiver:    {len(rx_layer_names)} layers fingerprinted")

        # Infer model dimensions from fingerprints
        tx_d = self.tx_fingerprints[tx_layer_names[0]].d_layer
        rx_d = self.rx_fingerprints[rx_layer_names[0]].d_layer
        logger.info(f"  Transmitter d_model={tx_d}, Receiver d_model={rx_d}")

        # ── Step 2: Collect raw activations for UHS training ─────────────
        logger.info("[Step 2/7] Collecting activations for UHS training...")

        tx_acts = self._collect_activations(
            self.transmitter, train_dataloader, tx_layer_names
        )
        rx_acts = self._collect_activations(
            self.receiver, train_dataloader, rx_layer_names
        )

        # Aggregate across layers: average activation per sample
        tx_pooled = self._pool_activations(tx_acts, tx_layer_names)
        rx_pooled = self._pool_activations(rx_acts, rx_layer_names)

        # ── Step 3: Train UHS encoder/decoder for transmitter ────────────
        logger.info("[Step 3/7] Training transmitter UHS encoder/decoder...")

        self.tx_uhs = UniversalHubSpace(tx_d, device=self.device)
        tx_act_loader = DataLoader(
            TensorDataset(torch.tensor(tx_pooled, dtype=torch.float32)),
            batch_size=32, shuffle=True,
        )
        self.tx_uhs.train(tx_act_loader, epochs=uhs_epochs, verbose=True)

        rt_error_tx = self.tx_uhs.round_trip_error(
            torch.tensor(tx_pooled[:100], dtype=torch.float32)
        )
        logger.info(f"  Transmitter UHS round-trip error: {rt_error_tx:.4f}")

        # ── Step 4: Train UHS encoder/decoder for receiver ───────────────
        logger.info("[Step 4/7] Training receiver UHS encoder/decoder...")

        self.rx_uhs = UniversalHubSpace(rx_d, device=self.device)
        rx_act_loader = DataLoader(
            TensorDataset(torch.tensor(rx_pooled, dtype=torch.float32)),
            batch_size=32, shuffle=True,
        )
        self.rx_uhs.train(rx_act_loader, epochs=uhs_epochs, verbose=True)

        rt_error_rx = self.rx_uhs.round_trip_error(
            torch.tensor(rx_pooled[:100], dtype=torch.float32)
        )
        logger.info(f"  Receiver UHS round-trip error: {rt_error_rx:.4f}")

        # ── Step 5: Encode transmitter → decode into receiver space ──────
        logger.info("[Step 5/7] Transferring through hub space...")

        tx_tensor = torch.tensor(tx_pooled, dtype=torch.float32).to(self.device)
        hub_vectors = self.tx_uhs.encode(tx_tensor)
        decoded_targets = self.rx_uhs.decode(hub_vectors).detach()

        logger.info(f"  Hub vectors shape: {hub_vectors.shape}")
        logger.info(f"  Decoded targets shape: {decoded_targets.shape}")

        # ── Step 6: Fine-tune receiver ───────────────────────────────────
        logger.info("[Step 6/7] Fine-tuning receiver model...")

        self._finetune_receiver(
            self.receiver,
            train_dataloader,
            decoded_targets,
            rx_layer_names,
            epochs=finetune_epochs,
            lr=finetune_lr,
        )

        # ── Step 7: Measure drift and package token ────────────────
        logger.info("[Step 7/7] Measuring drift and creating token...")

        drift = DriftMeasure(
            self.transmitter, self.receiver, self.device
        ).compute(val_dataloader)

        # Apply differential privacy to the hub vector
        dp = DifferentialPrivacy(privacy_epsilon, privacy_delta)
        avg_hub = hub_vectors.mean(dim=0).cpu().numpy()
        private_hub = dp.add_noise(avg_hub)

        # Compute compatibility score (cosine similarity of mean fingerprints)
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
                "tx_d_model": tx_d,
                "rx_d_model": rx_d,
                "tx_layers": len(tx_layer_names),
                "rx_layers": len(rx_layer_names),
                "compatibility_score": compat_score,
                "tx_round_trip_error": rt_error_tx,
                "rx_round_trip_error": rt_error_rx,
            },
        )

        logger.info(f"  Transfer complete!")
        logger.info(f"  Drift:    {drift:.6f}")
        logger.info(f"  Compatibility:  {compat_score:.4f}")
        logger.info(f"  Privacy:        ε={privacy_epsilon}, δ={privacy_delta}")

        return token

    # ── Internal helpers ──────────────────────────────────────────────────

    def _collect_activations(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        layer_names: List[str],
    ) -> Dict[str, list]:
        """Collect raw activations via hooks (reused from fingerprint logic)."""
        activations: Dict[str, list] = defaultdict(list)
        hooks = []

        def make_hook(name):
            def fn(module, inp, output):
                out = output[0] if isinstance(output, tuple) else output
                out = out.detach().cpu()
                if out.ndim == 3:
                    out = out.mean(dim=1)  # Pool over sequence
                activations[name].append(out)
            return fn

        for name, module in model.named_modules():
            if name in layer_names:
                hooks.append(module.register_forward_hook(make_hook(name)))

        model.eval()
        model.to(self.device)
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x.to(self.device))

        for h in hooks:
            h.remove()
        return activations

    def _pool_activations(
        self, acts: Dict[str, list], layer_names: List[str]
    ) -> np.ndarray:
        """
        Pool activations across layers into a single matrix.

        Strategy: concatenate the mean activation across all target layers,
        then take the mean across layers to get (N_samples, d_model).
        """
        # Use the middle layer (most informative, per the spec)
        mid_idx = len(layer_names) // 2
        mid_layer = layer_names[mid_idx]

        if mid_layer in acts and len(acts[mid_layer]) > 0:
            pooled = torch.cat(acts[mid_layer], dim=0).numpy()
        else:
            # Fallback: first available layer
            for name in layer_names:
                if name in acts and len(acts[name]) > 0:
                    pooled = torch.cat(acts[name], dim=0).numpy()
                    break
            else:
                raise RuntimeError("No activations collected from any target layer")

        return pooled

    def _finetune_receiver(
        self,
        receiver: nn.Module,
        dataloader: DataLoader,
        targets: torch.Tensor,
        rx_layer_names: List[str],
        epochs: int = 5,
        lr: float = 1e-3,
    ):
        """
        Fine-tune the receiver to align its activations with decoded targets.

        This is the "reconstruction" step of Mode A: we adjust receiver
        parameters so its middle-layer activations match the knowledge
        decoded from the transmitter via UHS.
        """
        receiver.train()
        receiver.to(self.device)
        targets = targets.to(self.device)

        # We'll optimise a subset of parameters (last few layers) to avoid
        # catastrophic forgetting. For the demo, we optimise all parameters.
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

                batch_targets = targets[sample_idx: sample_idx + batch_size]
                sample_idx += batch_size

                # Forward pass with hook to capture activations
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

                # Alignment loss: MSE between receiver activations and targets
                # Handle dimension mismatch by projecting targets if needed
                act = captured["act"]
                tgt = batch_targets

                if act.shape[-1] != tgt.shape[-1]:
                    # Simple linear projection for dimension mismatch
                    tgt = F.adaptive_avg_pool1d(
                        tgt.unsqueeze(1), act.shape[-1]
                    ).squeeze(1)

                loss = F.mse_loss(act, tgt)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()

            logger.info(f"  Fine-tune epoch {epoch+1}/{epochs}: alignment_loss={epoch_loss:.4f}")

        receiver.eval()

    def _compatibility_score(self) -> float:
        """
        Compute compatibility score between transmitter and receiver.

        Uses cosine similarity of mean activation vectors from the middle layer.
        """
        if not self.tx_fingerprints or not self.rx_fingerprints:
            return 0.0

        tx_layers = sorted(self.tx_fingerprints.keys())
        rx_layers = sorted(self.rx_fingerprints.keys())

        tx_mid = self.tx_fingerprints[tx_layers[len(tx_layers) // 2]]
        rx_mid = self.rx_fingerprints[rx_layers[len(rx_layers) // 2]]

        # Truncate to smaller dimension for comparison
        d = min(tx_mid.d_layer, rx_mid.d_layer)
        tx_mean = tx_mid.mean[:d]
        rx_mean = rx_mid.mean[:d]

        dot = np.dot(tx_mean, rx_mean)
        norm_a = np.linalg.norm(tx_mean)
        norm_b = np.linalg.norm(rx_mean)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
