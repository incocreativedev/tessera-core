"""
tessera.policy — Governance and acceptance rules for swarm (central operator v1).

Checks: required metadata, privacy fields, contributor weight cap,
minimum diversity. See docs/swarm-roundtrip-architecture.md.
"""

from typing import Tuple

from .token import TesseraToken

# Swarm custom_metadata keys (must match tessera.swarm)
SWARM_ROUND_ID = "swarm_round_id"
CONTRIBUTOR_ID = "contributor_id"
LOCAL_DATA_FINGERPRINT = "local_data_fingerprint"
QUALITY_SIGNALS = "quality_signals"
AGGREGATION_WEIGHT = "aggregation_weight"


# v1 constants
MIN_ACCEPTED_CONTRIBUTORS = 5
MAX_CONTRIBUTOR_WEIGHT_FRACTION = 0.15  # 15% cap per contributor per round


def accept_token(token: TesseraToken) -> Tuple[bool, str]:
    """
    Run governance checks on a single token. Returns (accepted, reason).
    """
    meta = token.custom_metadata or {}
    if not meta.get(SWARM_ROUND_ID):
        return False, "missing swarm_round_id"
    if not meta.get(CONTRIBUTOR_ID):
        return False, "missing contributor_id"
    # Require non-PII fingerprint (can be empty but key present for v1 lightweight check)
    if LOCAL_DATA_FINGERPRINT not in meta:
        return False, "missing local_data_fingerprint in swarm metadata"
    # Privacy: require token to have declared privacy params (no raw data)
    if token.privacy_epsilon is None or token.privacy_delta is None:
        return False, "missing privacy_epsilon or privacy_delta"
    if token.privacy_epsilon < 0 or token.privacy_delta < 0:
        return False, "invalid privacy parameters"
    # Basic payload
    if not token.uhs_vector or len(token.uhs_vector) == 0:
        return False, "empty uhs_vector"
    return True, "ok"


def check_round_acceptance(
    accepted_tokens: list,
    contributor_weights: dict,
) -> Tuple[bool, str]:
    """
    Check if a round can proceed: min contributors and per-contributor weight cap.
    contributor_weights: contributor_id -> aggregation_weight (e.g. utility).
    """
    if len(accepted_tokens) < MIN_ACCEPTED_CONTRIBUTORS:
        return False, f"fewer than {MIN_ACCEPTED_CONTRIBUTORS} accepted contributors"
    total = sum(contributor_weights.values())
    if total <= 0:
        return False, "total round weight is zero"
    for cid, w in contributor_weights.items():
        if w / total > MAX_CONTRIBUTOR_WEIGHT_FRACTION:
            return False, f"contributor {cid} exceeds {MAX_CONTRIBUTOR_WEIGHT_FRACTION*100:.0f}% cap"
    return True, "ok"
