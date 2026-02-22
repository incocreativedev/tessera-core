"""
Tessera — An activation-based protocol for AI-to-AI knowledge transfer.

Tessera enables trained neural networks to transfer learned knowledge to
untrained models across different architectures, frameworks, and domains.

Quick start:
    from tessera import ModeATransfer, TesseraToken, TokenSerializer

    transfer = ModeATransfer(transmitter, receiver, "model_a", "model_b")
    token = transfer.execute(train_loader, val_loader)
    TokenSerializer.save_token(token, "transfer.safetensors")
"""

__version__ = "0.1.0"

from .fingerprint import ActivationFingerprint, LayerFingerprint, compute_fingerprints
from .uhs import UniversalHubSpace, EncoderMLP, DecoderMLP
from .token import TesseraToken, KnowledgeType, TokenSerializer
from .transfer import ModeATransfer
from .drift import DriftMeasure
from .privacy import DifferentialPrivacy
from .gates import ProjectionType, ProjectionHint
from .binary import TBFSerializer, QuantType
from .registry import AnchorRegistry
from .swarm import (
    AggregationStrategy,
    SwarmAggregator,
    aggregate_tokens,
    score_token,
    validate_for_swarm,
    swarm_metadata,
)

__all__ = [
    # Fingerprinting
    "ActivationFingerprint",
    "LayerFingerprint",
    "compute_fingerprints",
    # Universal Hub Space
    "UniversalHubSpace",
    "EncoderMLP",
    "DecoderMLP",
    # Tokens
    "TesseraToken",
    "KnowledgeType",
    "TokenSerializer",
    # Transfer
    "ModeATransfer",
    # Measurement
    "DriftMeasure",
    # Privacy
    "DifferentialPrivacy",
    # Projections
    "ProjectionType",
    "ProjectionHint",
    # Binary format (v1.1)
    "TBFSerializer",
    "QuantType",
    # Registry
    "AnchorRegistry",
    # Swarm round-trip
    "AggregationStrategy",
    "SwarmAggregator",
    "aggregate_tokens",
    "score_token",
    "validate_for_swarm",
    "swarm_metadata",
]
