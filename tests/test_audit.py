"""Tests for tessera.audit — immutable audit trail and compliance."""

import json
import math

import numpy as np

from tessera.token import TesseraToken, KnowledgeType
from tessera.audit import (
    AuditEntry,
    AuditLog,
    AuditEventType,
    generate_ai_bom,
    export_compliance_package,
    MAX_COMPOSED_EPSILON,
    MAX_COMPOSED_DELTA,
    DEFAULT_RETENTION_DAYS,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_token(
    contributor_id: str = "site_a",
    round_id: str = "round_001",
    epsilon: float = 1.0,
    delta: float = 1e-5,
) -> TesseraToken:
    return TesseraToken(
        knowledge_type=KnowledgeType.SWARM,
        uhs_vector=np.random.randn(2048).tolist(),
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={"nodes": [], "root": contributor_id},
        privacy_epsilon=epsilon,
        privacy_delta=delta,
        source_model_id=contributor_id,
        custom_metadata={
            "swarm_round_id": round_id,
            "contributor_id": contributor_id,
            "local_data_fingerprint": f"sha256:{contributor_id}",
        },
    )


# ── AuditEntry ──────────────────────────────────────────────────────────


class TestAuditEntry:
    def test_hash_is_deterministic(self):
        e1 = AuditEntry(
            event_type=AuditEventType.TOKEN_SUBMITTED,
            round_id="r1",
            contributor_id="c1",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        e2 = AuditEntry(
            event_type=AuditEventType.TOKEN_SUBMITTED,
            round_id="r1",
            contributor_id="c1",
            timestamp="2026-01-01T00:00:00+00:00",
        )
        assert e1.entry_hash == e2.entry_hash

    def test_verify_passes_on_clean_entry(self):
        entry = AuditEntry(
            event_type=AuditEventType.ROUND_STARTED,
            round_id="r1",
        )
        assert entry.verify() is True

    def test_verify_fails_on_tampered_entry(self):
        entry = AuditEntry(
            event_type=AuditEventType.ROUND_STARTED,
            round_id="r1",
        )
        entry.round_id = "r2"  # tamper
        assert entry.verify() is False

    def test_to_dict_roundtrip(self):
        entry = AuditEntry(
            event_type=AuditEventType.TOKEN_ACCEPTED,
            round_id="r1",
            contributor_id="c1",
            details={"drift": 0.01},
        )
        restored = AuditEntry.from_dict(entry.to_dict())
        assert restored.entry_hash == entry.entry_hash
        assert restored.event_type == entry.event_type

    def test_prev_hash_chaining(self):
        e1 = AuditEntry(
            event_type=AuditEventType.ROUND_STARTED,
            round_id="r1",
        )
        e2 = AuditEntry(
            event_type=AuditEventType.TOKEN_SUBMITTED,
            round_id="r1",
            prev_hash=e1.entry_hash,
        )
        assert e2.prev_hash == e1.entry_hash
        assert e2.entry_hash != e1.entry_hash


# ── AuditLog ────────────────────────────────────────────────────────────


class TestAuditLog:
    def test_record_and_len(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record(AuditEventType.TOKEN_SUBMITTED, "r1", "c1")
        assert len(log) == 2

    def test_chain_integrity(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record(AuditEventType.TOKEN_ACCEPTED, "r1", "c1")
        log.record(AuditEventType.ROUND_AGGREGATED, "r1")
        ok, reason = log.verify_chain()
        assert ok is True
        assert reason == ""

    def test_tamper_detection(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record(AuditEventType.TOKEN_ACCEPTED, "r1", "c1")
        # Tamper with first entry
        log._entries[0].round_id = "tampered"
        ok, reason = log.verify_chain()
        assert ok is False
        assert "tampered" in reason.lower() or "mismatch" in reason.lower()

    def test_merkle_root_changes_with_new_entry(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        root1 = log.merkle_root()
        log.record(AuditEventType.TOKEN_ACCEPTED, "r1", "c1")
        root2 = log.merkle_root()
        assert root1 != root2

    def test_merkle_root_deterministic(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        assert log.merkle_root() == log.merkle_root()

    def test_empty_log_merkle_root(self):
        log = AuditLog()
        root = log.merkle_root()
        assert isinstance(root, str)
        assert len(root) == 64  # SHA-256 hex

    def test_get_round_entries(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record(AuditEventType.ROUND_STARTED, "r2")
        log.record(AuditEventType.TOKEN_ACCEPTED, "r1", "c1")
        entries = log.get_round_entries("r1")
        assert len(entries) == 2

    def test_get_violations(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record(
            AuditEventType.TOKEN_REJECTED,
            "r1",
            "c1",
            {"reason": "missing fingerprint"},
        )
        log.record(AuditEventType.TOKEN_ACCEPTED, "r1", "c2")
        violations = log.get_violations()
        assert len(violations) == 1
        assert violations[0].event_type == AuditEventType.TOKEN_REJECTED

    def test_record_token_submission(self):
        log = AuditLog()
        token = _make_token()
        entry = log.record_token_submission("r1", token, accepted=True)
        assert entry.event_type == AuditEventType.TOKEN_ACCEPTED
        assert entry.details["accepted"] is True

    def test_record_token_rejection(self):
        log = AuditLog()
        token = _make_token()
        entry = log.record_token_submission("r1", token, accepted=False, reason="policy violation")
        assert entry.event_type == AuditEventType.TOKEN_REJECTED
        assert "policy violation" in entry.details["reason"]

    def test_privacy_update_within_budget(self):
        log = AuditLog()
        entry = log.record_privacy_update("r1", 2.0, 1e-5, 5)
        assert entry.event_type == AuditEventType.PRIVACY_BUDGET_UPDATE
        assert entry.details["exceeded"] is False

    def test_privacy_exceeded_flags(self):
        log = AuditLog()
        entry = log.record_privacy_update("r1", 15.0, 1e-5, 100)
        assert entry.event_type == AuditEventType.PRIVACY_BUDGET_EXCEEDED
        assert entry.details["exceeded"] is True

    def test_serialise_roundtrip(self):
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record(AuditEventType.TOKEN_ACCEPTED, "r1", "c1")
        json_str = log.to_json()
        restored = AuditLog.from_json(json_str)
        assert len(restored) == 2
        ok, _ = restored.verify_chain()
        assert ok is True


# ── AI-BOM ──────────────────────────────────────────────────────────────


class TestAIBOM:
    def test_bom_structure(self):
        tokens = [_make_token(f"site_{i}") for i in range(5)]
        bom = generate_ai_bom("r1", tokens)
        assert bom["round_id"] == "r1"
        assert bom["contributor_count"] == 5
        assert len(bom["contributors"]) == 5
        assert bom["regulatory_alignment"]["eu_ai_act"]["article_12_record_keeping"]

    def test_bom_with_aggregation(self):
        tokens = [_make_token(f"site_{i}") for i in range(3)]
        agg_token = _make_token("aggregator")
        agg_token.custom_metadata["aggregation_method"] = "robust_weighted_mean"
        bom = generate_ai_bom("r1", tokens, aggregated_token=agg_token)
        assert bom["aggregation"] is not None
        assert bom["aggregation"]["method"] == "robust_weighted_mean"

    def test_bom_privacy_composition(self):
        tokens = [_make_token(f"site_{i}", epsilon=1.0, delta=1e-5) for i in range(4)]
        bom = generate_ai_bom("r1", tokens)
        expected_eps = math.sqrt(4) * 1.0
        assert abs(bom["privacy_composition"]["composed_epsilon"] - expected_eps) < 1e-6
        assert abs(bom["privacy_composition"]["composed_delta"] - 4e-5) < 1e-10


# ── Compliance Package ──────────────────────────────────────────────────


class TestCompliancePackage:
    def test_full_package_structure(self):
        tokens = [_make_token(f"site_{i}") for i in range(5)]
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        for t in tokens:
            log.record_token_submission("r1", t, accepted=True)
        log.record(AuditEventType.ROUND_AGGREGATED, "r1")

        pkg = export_compliance_package("r1", log, tokens)
        assert pkg["round_id"] == "r1"
        assert pkg["ai_bom"]["contributor_count"] == 5
        assert pkg["audit_trail"]["chain_valid"] is True
        assert isinstance(pkg["audit_trail"]["merkle_root"], str)
        assert pkg["policy_summary"]["tokens_accepted"] == 5
        assert pkg["policy_summary"]["compliant"] is True

    def test_package_shows_violations(self):
        tokens = [_make_token(f"site_{i}") for i in range(3)]
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        log.record_token_submission("r1", tokens[0], accepted=True)
        log.record_token_submission("r1", tokens[1], accepted=False, reason="bad fingerprint")
        log.record_token_submission("r1", tokens[2], accepted=True)

        pkg = export_compliance_package("r1", log, tokens)
        assert pkg["policy_summary"]["tokens_rejected"] == 1
        assert pkg["policy_summary"]["compliant"] is False

    def test_package_json_serialisable(self):
        tokens = [_make_token(f"site_{i}") for i in range(3)]
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "r1")
        pkg = export_compliance_package("r1", log, tokens)
        # Must not raise
        json_str = json.dumps(pkg, default=str)
        assert len(json_str) > 0

    def test_constants_sane(self):
        assert MAX_COMPOSED_EPSILON == 10.0
        assert MAX_COMPOSED_DELTA == 1e-3
        assert DEFAULT_RETENTION_DAYS == 730

    def test_repr(self):
        log = AuditLog()
        assert "AuditLog" in repr(log)
        log.record(AuditEventType.ROUND_STARTED, "r1")
        assert "1" in repr(log)
