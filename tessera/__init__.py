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
from .mode_w import ModeWTransfer
from .mode_b import ModeBTransfer
from .mode_c import ModeCTransfer
from .mode_d import ModeDTransfer
from .swap import SWAPProjection
from .swarm import (
    SwarmAggregator,
    AggregationStrategy,
    swarm_metadata,
    validate_for_swarm,
    aggregate_tokens,
    score_token,
    compute_credits,
)
from .policy import (
    accept_token,
    check_round_acceptance,
    RoundPolicy,
    MIN_ACCEPTED_CONTRIBUTORS,
    MAX_CONTRIBUTOR_WEIGHT_FRACTION,
)
from .credits import (
    CreditEntry,
    CreditsLedger,
    UTILITY_WEIGHTS,
    compute_quality_score,
    compute_novelty_score,
    compute_freshness_score,
    compute_reliability_score,
    compute_utility,
)
from .audit import (
    AuditEntry,
    AuditLog,
    AuditEventType,
    generate_ai_bom,
    export_compliance_package,
    MAX_COMPOSED_EPSILON,
    MAX_COMPOSED_DELTA,
    DEFAULT_RETENTION_DAYS,
)
from .correspondence import LayerCorrespondence, linear_cka
from .weight_ops import (
    WeightSnapshot,
    WeightStats,
    extract_weights,
    svd_compress,
    compute_weight_stats,
)
from .drift import DriftMeasure, WeightDriftMeasure
from .privacy import DifferentialPrivacy
from .gates import ProjectionType, ProjectionHint
from .binary import TBFSerializer, QuantType
from .registry import AnchorRegistry
from .signing import (
    generate_keypair,
    sign_token,
    verify_token_signature,
    is_signed,
    strip_signature,
    private_key_to_pem,
    private_key_from_pem,
    public_key_to_hex,
    public_key_from_hex,
    save_private_key,
    load_private_key,
    SIGNATURE_KEY,
    PUBLIC_KEY_HEX_KEY,
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
    "ModeWTransfer",
    "ModeBTransfer",
    "ModeCTransfer",
    "ModeDTransfer",
    "SWAPProjection",
    "SwarmAggregator",
    "AggregationStrategy",
    "swarm_metadata",
    "validate_for_swarm",
    "aggregate_tokens",
    "score_token",
    "compute_credits",
    # Policy & Governance
    "accept_token",
    "check_round_acceptance",
    "RoundPolicy",
    "MIN_ACCEPTED_CONTRIBUTORS",
    "MAX_CONTRIBUTOR_WEIGHT_FRACTION",
    # Credits & Utility
    "CreditEntry",
    "CreditsLedger",
    "UTILITY_WEIGHTS",
    "compute_quality_score",
    "compute_novelty_score",
    "compute_freshness_score",
    "compute_reliability_score",
    "compute_utility",
    # Audit & Compliance
    "AuditEntry",
    "AuditLog",
    "AuditEventType",
    "generate_ai_bom",
    "export_compliance_package",
    "MAX_COMPOSED_EPSILON",
    "MAX_COMPOSED_DELTA",
    "DEFAULT_RETENTION_DAYS",
    # Correspondence & weight ops
    "LayerCorrespondence",
    "linear_cka",
    "WeightSnapshot",
    "WeightStats",
    "extract_weights",
    "svd_compress",
    "compute_weight_stats",
    # Measurement
    "DriftMeasure",
    "WeightDriftMeasure",
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
    # Signing & Authentication
    "generate_keypair",
    "sign_token",
    "verify_token_signature",
    "is_signed",
    "strip_signature",
    "private_key_to_pem",
    "private_key_from_pem",
    "public_key_to_hex",
    "public_key_from_hex",
    "save_private_key",
    "load_private_key",
    "SIGNATURE_KEY",
    "PUBLIC_KEY_HEX_KEY",
]
