"""
tessera.mode_w — Mode W (Weight) knowledge transfer orchestrator.

Mode W transfers raw weight parameters between architectures through the
Universal Hub Space. Unlike Mode A which transfers activation patterns
(how a model behaves), Mode W directly initialises the receiver's weight
matrices from SVD-compressed transmitter weights decoded through hub space.

Pipeline:
    1. Extract transmitter weight matrices (Linear + Conv2d)
    2. Compute CKA-based layer correspondence (needs reference data)
    3. SVD-compress transmitter weights
    4. Train UHS encoder/decoder on flattened weight chunks
    5. Encode tx chunks → hub space → decode to receiver dimensions
    6. Initialise receiver weights from decoded matrices
    7. Measure weight-space drift and package TesseraToken
"""

import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .correspondence import LayerCorrespondence, _auto_detect_weight_layers
from .drift import WeightDriftMeasure
from .privacy import DifferentialPrivacy
from .token import KnowledgeType, TesseraToken
from .uhs import UniversalHubSpace
from .utils import setup_logging
from .weight_ops import (
    ChunkMeta,
    WeightSnapshot,
    chunk_for_hub,
    compute_weight_stats,
    decode_and_reassemble,
    encode_weight_chunks,
    extract_weights,
    initialize_receiver_weights,
    svd_compress,
)

logger = setup_logging("tessera.mode_w")


class ModeWTransfer:
    """
    Orchestrates a complete Mode W (weight-based) knowledge transfer
    between a transmitter and a receiver model.

    Mode W encodes the transmitter's learned weight matrices through the
    Universal Hub Space and decodes them into the receiver's architecture,
    giving the receiver a weight-space initialisation informed by the
    transmitter's training.

    Usage:
        transfer = ModeWTransfer(
            transmitter=trained_model,
            receiver=untrained_model,
            transmitter_id="model_a",
            receiver_id="model_b",
        )
        token = transfer.execute(reference_dataloader=data_loader)
    """

    def __init__(
        self,
        transmitter: nn.Module,
        receiver: nn.Module,
        transmitter_id: str = "transmitter",
        receiver_id: str = "receiver",
        device: str = "cpu",
        hub_dim: int = 2048,
    ):
        """
        Args:
            transmitter: Trained source model.
            receiver: Target model to receive knowledge.
            transmitter_id: Unique identifier for the transmitter.
            receiver_id: Unique identifier for the receiver.
            device: Torch device ("cpu", "cuda", etc.).
            hub_dim: Dimensionality of the Universal Hub Space. Default 2048.
        """
        self.transmitter = transmitter
        self.receiver = receiver
        self.transmitter_id = transmitter_id
        self.receiver_id = receiver_id
        self.device = device
        self.hub_dim = hub_dim

        # Populated during transfer
        self.correspondences: Optional[List[Tuple[str, str, float]]] = None
        self.tx_snapshots: Optional[Dict[str, WeightSnapshot]] = None
        self.tx_uhs: Optional[UniversalHubSpace] = None
        self.rx_uhs: Optional[UniversalHubSpace] = None

    def execute(
        self,
        reference_dataloader: DataLoader,
        tx_layers: Optional[List[str]] = None,
        rx_layers: Optional[List[str]] = None,
        svd_rank: Optional[int] = None,
        svd_energy: float = 0.95,
        uhs_epochs: int = 10,
        correspondence_strategy: str = "greedy",
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
    ) -> TesseraToken:
        """
        Execute the full Mode W transfer pipeline.

        Args:
            reference_dataloader: Shared data for CKA layer matching. Both
                                   models must accept the same inputs.
            tx_layers: Transmitter weight layers (auto-detect if None).
            rx_layers: Receiver weight layers (auto-detect if None).
            svd_rank: Fixed SVD rank (auto-select via svd_energy if None).
            svd_energy: Energy threshold for auto SVD rank (0.0–1.0).
            uhs_epochs: Training epochs for the UHS on weight chunks.
            correspondence_strategy: "greedy" or "hungarian" layer matching.
            privacy_epsilon: Differential privacy epsilon budget.
            privacy_delta: Differential privacy delta parameter.

        Returns:
            TesseraToken with KnowledgeType.WEIGHT.
        """

        # ── Step 1: Extract transmitter weights ──────────────────────────
        logger.info("[Step 1/7] Extracting transmitter weight matrices...")

        tx_layer_names = tx_layers or _auto_detect_weight_layers(self.transmitter)
        rx_layer_names = rx_layers or _auto_detect_weight_layers(self.receiver)

        tx_weights = extract_weights(self.transmitter, tx_layer_names)
        if not tx_weights:
            raise RuntimeError("No weight matrices found in transmitter. Check tx_layers.")

        logger.info(f"  Extracted {len(tx_weights)} layers from transmitter.")

        # ── Step 2: CKA-based layer correspondence ────────────────────────
        logger.info("[Step 2/7] Computing CKA layer correspondence...")

        corr = LayerCorrespondence(self.transmitter, self.receiver, device=self.device)
        # CKA operates on activation layers; map weight layers to activation names
        tx_act_layers = self._weight_to_activation_layers(self.transmitter)
        rx_act_layers = self._weight_to_activation_layers(self.receiver)

        self.correspondences = corr.compute(
            reference_dataloader,
            tx_layers=tx_act_layers or None,
            rx_layers=rx_act_layers or None,
            strategy=correspondence_strategy,
        )

        # Build (tx_weight_layer, rx_weight_layer) pairs
        layer_pairs = self._build_weight_pairs(tx_layer_names, rx_layer_names)
        logger.info(f"  {len(layer_pairs)} weight layer pairs to transfer.")

        # ── Step 3: SVD-compress transmitter weights ──────────────────────
        logger.info("[Step 3/7] SVD-compressing transmitter weights...")

        self.tx_snapshots = {}
        for name, W in tx_weights.items():
            snap = svd_compress(W, layer_name=name, rank=svd_rank, energy_threshold=svd_energy)
            self.tx_snapshots[name] = snap

        # ── Step 4: Train UHS on weight chunks ───────────────────────────
        logger.info("[Step 4/7] Training UHS on weight chunks...")

        all_chunks = self._collect_all_chunks(self.tx_snapshots)
        if len(all_chunks) == 0:
            raise RuntimeError("No weight chunks produced; check weight extraction.")

        self.tx_uhs = UniversalHubSpace(self.hub_dim, hub_dim=self.hub_dim, device=self.device)
        chunk_loader = DataLoader(
            TensorDataset(torch.tensor(np.stack(all_chunks), dtype=torch.float32)),
            batch_size=min(32, len(all_chunks)),
            shuffle=True,
        )
        self.tx_uhs.train(chunk_loader, epochs=uhs_epochs, verbose=True)
        rt_error = self.tx_uhs.round_trip_error(
            torch.tensor(np.stack(all_chunks[:min(50, len(all_chunks))]), dtype=torch.float32)
        )
        logger.info(f"  UHS round-trip error: {rt_error:.4f}")

        # Receiver UHS (same dim since hub_dim == chunk size == hub_dim)
        self.rx_uhs = self.tx_uhs  # shared encoder space; receiver gets same decoder

        # ── Step 5: Encode tx chunks → hub → decode to rx dims ───────────
        logger.info("[Step 5/7] Transferring weights through hub space...")

        decoded_weights: Dict[str, np.ndarray] = {}
        hub_vectors_all: List[np.ndarray] = []

        for tx_name, rx_name in layer_pairs:
            if tx_name not in self.tx_snapshots:
                logger.warning(f"  No snapshot for '{tx_name}'; skipping.")
                continue

            snap = self.tx_snapshots[tx_name]
            chunks, metas = chunk_for_hub(snap, self.hub_dim)
            encoded, scales = encode_weight_chunks(chunks, self.tx_uhs)
            hub_vectors_all.extend(encoded)

            # Determine receiver layer shape
            rx_shape = self._get_layer_shape(self.receiver, rx_name)
            rx_rank = min(snap.rank, min(rx_shape) if rx_shape else snap.rank)

            W_decoded = decode_and_reassemble(
                encoded, scales, metas, self.rx_uhs, rx_shape or snap.original_shape, rx_rank
            )
            decoded_weights[rx_name] = W_decoded
            logger.info(f"  {tx_name} → {rx_name}: decoded shape {W_decoded.shape}")

        # ── Step 6: Initialise receiver weights ───────────────────────────
        logger.info("[Step 6/7] Initialising receiver weights...")
        initialize_receiver_weights(self.receiver, decoded_weights)

        # ── Step 7: Measure drift and package token ───────────────────────
        logger.info("[Step 7/7] Measuring weight drift and creating token...")

        drift = WeightDriftMeasure(
            self.transmitter, self.receiver, device=self.device
        ).compute(correspondences=layer_pairs)

        dp = DifferentialPrivacy(privacy_epsilon, privacy_delta)
        if hub_vectors_all:
            avg_hub = np.mean(np.stack(hub_vectors_all), axis=0)
        else:
            avg_hub = np.zeros(self.hub_dim, dtype=np.float32)
        private_hub = dp.add_noise(avg_hub)

        token = TesseraToken(
            knowledge_type=KnowledgeType.WEIGHT,
            uhs_vector=private_hub.tolist(),
            modality_weights={"W": 0.90, "A": 0.05, "B": 0.05},
            correlation_map={},
            lineage_dag={
                "nodes": [{"id": "n0", "type": "anchor", "ref": self.transmitter_id}],
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
                "mode": "W",
                "tx_layers": len(tx_weights),
                "rx_layers": len(rx_layer_names),
                "layer_pairs": len(layer_pairs),
                "svd_energy": svd_energy,
                "svd_rank": svd_rank,
                "uhs_round_trip_error": float(rt_error),
                "hub_dim": self.hub_dim,
                "correspondence_strategy": correspondence_strategy,
            },
        )

        logger.info("  Mode W transfer complete!")
        logger.info(f"  Weight drift: {drift:.6f}")
        logger.info(f"  Privacy:      ε={privacy_epsilon}, δ={privacy_delta}")

        return token

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _weight_to_activation_layers(self, model: nn.Module) -> List[str]:
        """Return TransformerEncoderLayer / ModuleList children for CKA."""
        layers = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                layers.append(name)
        if not layers:
            for name, module in model.named_children():
                if isinstance(module, (nn.ModuleList, nn.Sequential)):
                    for sub_name, _ in module.named_children():
                        layers.append(f"{name}.{sub_name}")
        return layers

    def _build_weight_pairs(
        self,
        tx_layer_names: List[str],
        rx_layer_names: List[str],
    ) -> List[Tuple[str, str]]:
        """
        Build (tx_weight_layer, rx_weight_layer) pairs.

        Uses index-based pairing (same position in sorted list) since
        CKA correspondence operates on activation layers, not weight layers.
        For same-architecture models this is exact; for cross-architecture
        models it provides a reasonable heuristic.
        """
        pairs = []
        n = min(len(tx_layer_names), len(rx_layer_names))
        for i in range(n):
            pairs.append((tx_layer_names[i], rx_layer_names[i]))
        return pairs

    def _collect_all_chunks(
        self, snapshots: Dict[str, WeightSnapshot]
    ) -> List[np.ndarray]:
        """Collect all hub-dim-sized chunks from all snapshots for UHS training."""
        all_chunks: List[np.ndarray] = []
        for snap in snapshots.values():
            chunks, _ = chunk_for_hub(snap, self.hub_dim)
            all_chunks.extend(chunks)
        return all_chunks

    def _get_layer_shape(
        self, model: nn.Module, layer_name: str
    ) -> Optional[Tuple[int, int]]:
        """Return (out, in*kH*kW) 2D weight shape for a named layer."""
        for name, module in model.named_modules():
            if name == layer_name:
                if hasattr(module, "weight") and module.weight is not None:
                    w = module.weight
                    if w.ndim >= 2:
                        return (w.shape[0], int(np.prod(w.shape[1:])))
                    return (1, w.numel())
        return None
