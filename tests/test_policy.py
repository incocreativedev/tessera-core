"""
Tests for tessera.policy — governance rules and token acceptance.
"""

import numpy as np

from tessera.token import TesseraToken, KnowledgeType
from tessera.policy import (
    accept_token,
    check_round_acceptance,
    RoundPolicy,
    MIN_ACCEPTED_CONTRIBUTORS,
    MAX_CONTRIBUTOR_WEIGHT_FRACTION,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_swarm_token(
    contributor_id: str = "site_a",
    swarm_round_id: str = "round_001",
    local_data_fingerprint: str = "sha256:abc123",
    privacy_epsilon: float = 1.0,
    privacy_delta: float = 1e-5,
    uhs_dim: int = 2048,
) -> TesseraToken:
    """Build a valid swarm-ready token for testing."""
    return TesseraToken(
        knowledge_type=KnowledgeType.SWARM,
        uhs_vector=np.random.randn(uhs_dim).tolist(),
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={"nodes": [], "root": contributor_id},
        privacy_epsilon=privacy_epsilon,
        privacy_delta=privacy_delta,
        source_model_id=contributor_id,
        custom_metadata={
            "swarm_round_id": swarm_round_id,
            "contributor_id": contributor_id,
            "local_data_fingerprint": local_data_fingerprint,
        },
    )


# ── Token-level acceptance ──────────────────────────────────────────────


class TestAcceptToken:
    """Tests for accept_token()."""

    def test_valid_token_accepted(self):
        token = _make_swarm_token()
        ok, reason = accept_token(token)
        assert ok is True
        assert reason == ""

    def test_missing_swarm_round_id(self):
        token = _make_swarm_token()
        del token.custom_metadata["swarm_round_id"]
        ok, reason = accept_token(token)
        assert ok is False
        assert "swarm_round_id" in reason

    def test_missing_contributor_id(self):
        token = _make_swarm_token()
        del token.custom_metadata["contributor_id"]
        ok, reason = accept_token(token)
        assert ok is False
        assert "contributor_id" in reason

    def test_missing_local_data_fingerprint(self):
        token = _make_swarm_token()
        del token.custom_metadata["local_data_fingerprint"]
        ok, reason = accept_token(token)
        assert ok is False
        assert "local_data_fingerprint" in reason

    def test_invalid_privacy_epsilon(self):
        token = _make_swarm_token(privacy_epsilon=0.0)
        ok, reason = accept_token(token)
        assert ok is False
        assert "privacy_epsilon" in reason

    def test_invalid_privacy_delta_zero(self):
        token = _make_swarm_token(privacy_delta=0.0)
        ok, reason = accept_token(token)
        assert ok is False
        assert "privacy_delta" in reason

    def test_invalid_privacy_delta_one(self):
        token = _make_swarm_token(privacy_delta=1.0)
        ok, reason = accept_token(token)
        assert ok is False
        assert "privacy_delta" in reason

    def test_empty_uhs_vector(self):
        token = _make_swarm_token()
        token.uhs_vector = []
        ok, reason = accept_token(token)
        assert ok is False
        assert "UHS" in reason


# ── Round-level acceptance ──────────────────────────────────────────────


class TestCheckRoundAcceptance:
    """Tests for check_round_acceptance()."""

    def test_valid_round_with_seven_contributors(self):
        # 7 contributors × 1 token each = 14.3% per contributor (< 15% cap)
        tokens = [_make_swarm_token(contributor_id=f"site_{i}") for i in range(7)]
        ok, reason = check_round_acceptance(tokens)
        assert ok is True
        assert reason == ""

    def test_rejected_fewer_than_min_contributors(self):
        tokens = [_make_swarm_token(contributor_id=f"site_{i}") for i in range(3)]
        ok, reason = check_round_acceptance(tokens)
        assert ok is False
        assert "unique contributors" in reason

    def test_rejected_single_contributor_exceeds_weight_cap(self):
        # 10 tokens total: 3 from site_0 (30% > 15%) + 1 each from 7 others
        tokens = [_make_swarm_token(contributor_id="site_0") for _ in range(3)]
        for i in range(1, 8):
            tokens.append(_make_swarm_token(contributor_id=f"site_{i}"))
        ok, reason = check_round_acceptance(tokens)
        assert ok is False
        assert "site_0" in reason
        assert "cap" in reason.lower() or "%" in reason

    def test_no_tokens_rejected(self):
        ok, reason = check_round_acceptance([])
        assert ok is False

    def test_custom_min_contributors(self):
        # 7 contributors with min_contributors=3 — passes both contributor
        # count and weight cap (each at 14.3%)
        tokens = [_make_swarm_token(contributor_id=f"site_{i}") for i in range(7)]
        ok, reason = check_round_acceptance(tokens, min_contributors=3)
        assert ok is True


# ── RoundPolicy ─────────────────────────────────────────────────────────


class TestRoundPolicy:
    """Tests for the RoundPolicy dataclass."""

    def test_defaults(self):
        policy = RoundPolicy(round_id="round_001")
        assert policy.min_contributors == MIN_ACCEPTED_CONTRIBUTORS
        assert policy.max_weight_per_contributor == MAX_CONTRIBUTOR_WEIGHT_FRACTION

    def test_custom_overrides(self):
        policy = RoundPolicy(
            round_id="pilot_001",
            min_contributors=3,
            max_weight_per_contributor=0.25,
        )
        assert policy.min_contributors == 3
        assert policy.max_weight_per_contributor == 0.25

    def test_validate_round_delegates(self):
        # 7 contributors, relaxed min to 2 — passes weight cap too
        policy = RoundPolicy(round_id="round_001", min_contributors=2)
        tokens = [_make_swarm_token(contributor_id=f"site_{i}") for i in range(7)]
        ok, reason = policy.validate_round(tokens)
        assert ok is True

    def test_allowed_contributors_rejects_unlisted(self):
        policy = RoundPolicy(
            round_id="round_001",
            allowed_contributors={"site_a", "site_b"},
        )
        token = _make_swarm_token(contributor_id="site_c")
        ok, reason = policy.can_accept_token(token)
        assert ok is False
        assert "not in allowed set" in reason

    def test_allowed_contributors_accepts_listed(self):
        policy = RoundPolicy(
            round_id="round_001",
            allowed_contributors={"site_a", "site_b"},
        )
        token = _make_swarm_token(contributor_id="site_a")
        ok, reason = policy.can_accept_token(token)
        assert ok is True
