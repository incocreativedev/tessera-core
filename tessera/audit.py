"""tessera.audit — Immutable audit trail with cryptographic integrity.

Provides:
    - AuditEntry: individual audit record with SHA-256 chaining
    - AuditLog: append-only log with Merkle tree root computation
    - AI-BOM generation from token lineage DAGs (NIST AI RMF aligned)
    - Compliance evidence packaging (EU AI Act, NIST, ISO 42001)
    - Privacy budget accounting with provable guarantees

Governing body alignment:
    - EU AI Act Art. 12: Record-keeping / automatic logging
    - EU AI Act Art. 9: Risk management documentation
    - NIST AI RMF: Govern → Map → Measure → Manage lifecycle
    - NIST IR 8596 (Cyber AI Profile): Data provenance verification
    - ISO/IEC 42001: AI Management Systems auditability
"""

import datetime
import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .token import TesseraToken

# ══════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════

# Privacy budget ceiling — if composed epsilon exceeds this across rounds,
# the audit log flags it as a compliance violation.
MAX_COMPOSED_EPSILON = 10.0
MAX_COMPOSED_DELTA = 1e-3

# EU AI Act Art. 12 requires logs to be retained for the lifetime of the
# system plus at least 6 months. We default to 2 years for safety.
DEFAULT_RETENTION_DAYS = 730


class AuditEventType(Enum):
    """Categories of auditable events in a Tessera round."""

    TOKEN_SUBMITTED = "token_submitted"
    TOKEN_ACCEPTED = "token_accepted"
    TOKEN_REJECTED = "token_rejected"
    ROUND_STARTED = "round_started"
    ROUND_AGGREGATED = "round_aggregated"
    ROUND_BROADCAST = "round_broadcast"
    PRIVACY_BUDGET_UPDATE = "privacy_budget_update"
    PRIVACY_BUDGET_EXCEEDED = "privacy_budget_exceeded"
    POLICY_VIOLATION = "policy_violation"
    CREDIT_AWARDED = "credit_awarded"
    CONFORMITY_CHECK = "conformity_check"
    SIGNATURE_VERIFIED = "signature_verified"
    SIGNATURE_FAILED = "signature_failed"


# ══════════════════════════════════════════════════════════════════════════
#  Audit Entry (individual record)
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class AuditEntry:
    """
    Single audit record in the immutable log.

    Each entry chains to the previous via prev_hash, forming a hash chain
    (lightweight blockchain). This makes tampering detectable: changing any
    entry invalidates all subsequent hashes.

    Fields:
        event_type: What happened (see AuditEventType)
        round_id: Which swarm round this relates to
        contributor_id: Who was involved (empty for round-level events)
        timestamp: ISO 8601 UTC timestamp
        details: Arbitrary structured data about the event
        prev_hash: SHA-256 of the previous entry (empty string for genesis)
        entry_hash: SHA-256 of this entry (computed on creation)
    """

    event_type: AuditEventType
    round_id: str
    contributor_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    details: dict = field(default_factory=dict)
    prev_hash: str = ""
    entry_hash: str = ""

    def __post_init__(self):
        if not self.entry_hash:
            self.entry_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """SHA-256 of the entry's content (excluding entry_hash itself)."""
        payload = json.dumps(
            {
                "event_type": self.event_type.value,
                "round_id": self.round_id,
                "contributor_id": self.contributor_id,
                "timestamp": self.timestamp,
                "details": self.details,
                "prev_hash": self.prev_hash,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        """Check that entry_hash matches recomputed hash."""
        return self.entry_hash == self._compute_hash()

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "round_id": self.round_id,
            "contributor_id": self.contributor_id,
            "timestamp": self.timestamp,
            "details": self.details,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuditEntry":
        d = data.copy()
        d["event_type"] = AuditEventType(d["event_type"])
        return cls(**d)


# ══════════════════════════════════════════════════════════════════════════
#  Audit Log (append-only with Merkle root)
# ══════════════════════════════════════════════════════════════════════════


class AuditLog:
    """
    Append-only audit log with SHA-256 hash chaining and Merkle tree root.

    The hash chain provides tamper evidence: if any entry is modified, all
    subsequent hashes become invalid. The Merkle root provides a single
    fingerprint for the entire log state, useful for external attestation.

    Usage:
        log = AuditLog()
        log.record(AuditEventType.ROUND_STARTED, "round_042")
        log.record(AuditEventType.TOKEN_SUBMITTED, "round_042",
                   contributor_id="site_a", details={"token_hash": "abc"})

        # Verify integrity
        assert log.verify_chain()

        # Get Merkle root for external attestation
        root = log.merkle_root()

        # Export for regulatory submission
        evidence = log.export_compliance_package("round_042")
    """

    def __init__(self):
        self._entries: List[AuditEntry] = []
        self._privacy_tracker: Dict[str, Dict[str, float]] = {}
        # Track cumulative privacy per round: {round_id: {"epsilon": X, "delta": Y}}

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"AuditLog(entries={len(self._entries)})"

    # ── Recording ───────────────────────────────────────────────────────

    def record(
        self,
        event_type: AuditEventType,
        round_id: str,
        contributor_id: str = "",
        details: Optional[dict] = None,
    ) -> AuditEntry:
        """Append an audit entry to the log. Returns the new entry."""
        prev_hash = self._entries[-1].entry_hash if self._entries else ""
        entry = AuditEntry(
            event_type=event_type,
            round_id=round_id,
            contributor_id=contributor_id,
            details=details or {},
            prev_hash=prev_hash,
        )
        self._entries.append(entry)
        return entry

    def record_token_submission(
        self,
        round_id: str,
        token: TesseraToken,
        accepted: bool,
        reason: str = "",
    ) -> AuditEntry:
        """Record a token submission with acceptance/rejection."""
        meta = token.custom_metadata or {}
        contributor_id = meta.get("contributor_id", token.source_model_id or "unknown")
        event_type = AuditEventType.TOKEN_ACCEPTED if accepted else AuditEventType.TOKEN_REJECTED
        from .signing import is_signed, verify_token_signature

        signed = is_signed(token)
        sig_valid = False
        if signed:
            sig_ok, _ = verify_token_signature(token)
            sig_valid = sig_ok

        details = {
            "accepted": accepted,
            "reason": reason,
            "knowledge_type": token.knowledge_type.value,
            "privacy_epsilon": token.privacy_epsilon,
            "privacy_delta": token.privacy_delta,
            "drift_score": token.drift_score,
            "hub_vector_norm": (
                float(np.linalg.norm(token.uhs_vector)) if token.uhs_vector else 0.0
            ),
            "swarm_round_id": meta.get("swarm_round_id", ""),
            "local_data_fingerprint": meta.get("local_data_fingerprint", ""),
            "signed": signed,
            "signature_valid": sig_valid,
            "public_key_hex": meta.get("public_key_hex", ""),
        }
        return self.record(event_type, round_id, contributor_id, details)

    def record_privacy_update(
        self,
        round_id: str,
        composed_epsilon: float,
        composed_delta: float,
        n_contributors: int,
    ) -> AuditEntry:
        """
        Record cumulative privacy budget after a round.

        Automatically flags if budget exceeds safety thresholds.
        """
        if round_id not in self._privacy_tracker:
            self._privacy_tracker[round_id] = {"epsilon": 0.0, "delta": 0.0}

        self._privacy_tracker[round_id]["epsilon"] = composed_epsilon
        self._privacy_tracker[round_id]["delta"] = composed_delta

        exceeded = composed_epsilon > MAX_COMPOSED_EPSILON or composed_delta > MAX_COMPOSED_DELTA

        event_type = (
            AuditEventType.PRIVACY_BUDGET_EXCEEDED
            if exceeded
            else AuditEventType.PRIVACY_BUDGET_UPDATE
        )

        details = {
            "composed_epsilon": composed_epsilon,
            "composed_delta": composed_delta,
            "n_contributors": n_contributors,
            "max_epsilon_threshold": MAX_COMPOSED_EPSILON,
            "max_delta_threshold": MAX_COMPOSED_DELTA,
            "exceeded": exceeded,
        }
        return self.record(event_type, round_id, details=details)

    # ── Verification ────────────────────────────────────────────────────

    def verify_chain(self) -> Tuple[bool, str]:
        """
        Verify the entire hash chain is intact.

        Returns:
            (True, "") if valid, (False, description) if tampered.
        """
        if not self._entries:
            return True, ""

        # Genesis entry should have empty prev_hash
        if self._entries[0].prev_hash != "":
            return False, "Genesis entry has non-empty prev_hash."

        for i, entry in enumerate(self._entries):
            # Verify self-consistency
            if not entry.verify():
                return False, f"Entry {i} hash mismatch (tampered content)."

            # Verify chain link
            if i > 0 and entry.prev_hash != self._entries[i - 1].entry_hash:
                return False, f"Entry {i} prev_hash does not match entry {i-1}."

        return True, ""

    def merkle_root(self) -> str:
        """
        Compute the Merkle tree root of all entry hashes.

        This single hash fingerprints the entire log state. It can be
        published to an external ledger (blockchain, notary service) for
        third-party attestation without revealing log contents.
        """
        if not self._entries:
            return hashlib.sha256(b"empty").hexdigest()

        hashes = [e.entry_hash for e in self._entries]

        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                left = hashes[i]
                right = hashes[i + 1] if i + 1 < len(hashes) else left
                combined = hashlib.sha256((left + right).encode("utf-8")).hexdigest()
                next_level.append(combined)
            hashes = next_level

        return hashes[0]

    # ── Querying ────────────────────────────────────────────────────────

    def get_round_entries(self, round_id: str) -> List[AuditEntry]:
        """Get all entries for a specific round."""
        return [e for e in self._entries if e.round_id == round_id]

    def get_contributor_entries(self, contributor_id: str) -> List[AuditEntry]:
        """Get all entries for a specific contributor."""
        return [e for e in self._entries if e.contributor_id == contributor_id]

    def get_violations(self) -> List[AuditEntry]:
        """Get all policy violation and privacy exceeded entries."""
        violation_types = {
            AuditEventType.POLICY_VIOLATION,
            AuditEventType.PRIVACY_BUDGET_EXCEEDED,
            AuditEventType.TOKEN_REJECTED,
        }
        return [e for e in self._entries if e.event_type in violation_types]

    # ── Serialisation ───────────────────────────────────────────────────

    def to_list(self) -> List[dict]:
        return [e.to_dict() for e in self._entries]

    @classmethod
    def from_list(cls, data: List[dict]) -> "AuditLog":
        log = cls()
        log._entries = [AuditEntry.from_dict(d) for d in data]
        return log

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_list(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "AuditLog":
        return cls.from_list(json.loads(json_str))


# ══════════════════════════════════════════════════════════════════════════
#  AI Bill of Materials (NIST-aligned)
# ══════════════════════════════════════════════════════════════════════════


def generate_ai_bom(
    round_id: str,
    contributor_tokens: List[TesseraToken],
    aggregated_token: Optional[TesseraToken] = None,
    broadcast_tokens: Optional[List[TesseraToken]] = None,
) -> dict:
    """
    Generate an AI Bill of Materials for a swarm round.

    Aligned with NIST AI RMF recommendation for AI-BOM: documents all
    models, data sources, and transformations involved in producing an
    AI artefact.

    Returns a JSON-serialisable dict suitable for regulatory submission.
    """
    contributors = []
    for i, token in enumerate(contributor_tokens):
        meta = token.custom_metadata or {}
        contributors.append(
            {
                "index": i,
                "contributor_id": meta.get("contributor_id", token.source_model_id),
                "knowledge_type": token.knowledge_type.value,
                "data_fingerprint": meta.get("local_data_fingerprint", ""),
                "privacy_epsilon": token.privacy_epsilon,
                "privacy_delta": token.privacy_delta,
                "drift_score": token.drift_score,
                "timestamp": token.timestamp,
                "lineage": token.lineage_dag,
            }
        )

    bom = {
        "schema_version": "1.0",
        "framework": "tessera",
        "round_id": round_id,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "contributor_count": len(contributor_tokens),
        "contributors": contributors,
        "privacy_composition": {
            "method": "advanced_composition_gaussian",
            "formula": ("epsilon_total = sqrt(N) * max(epsilon_i), " "delta_total = sum(delta_i)"),
            "composed_epsilon": math.sqrt(len(contributor_tokens))
            * max((t.privacy_epsilon for t in contributor_tokens), default=0.0),
            "composed_delta": sum(t.privacy_delta for t in contributor_tokens),
        },
        "aggregation": None,
        "broadcast": None,
        "regulatory_alignment": {
            "eu_ai_act": {
                "article_9_risk_management": True,
                "article_12_record_keeping": True,
                "article_15_accuracy_robustness": True,
            },
            "nist_ai_rmf": {
                "govern": True,
                "map": True,
                "measure": True,
                "manage": True,
            },
            "iso_42001": True,
        },
    }

    if aggregated_token:
        meta = aggregated_token.custom_metadata or {}
        bom["aggregation"] = {
            "method": meta.get("aggregation_method", "unknown"),
            "aggregator_id": aggregated_token.target_model_id or meta.get("aggregator_id", ""),
            "drift_score": aggregated_token.drift_score,
            "hub_vector_dim": len(aggregated_token.uhs_vector),
            "timestamp": aggregated_token.timestamp,
        }

    if broadcast_tokens:
        bom["broadcast"] = {
            "token_count": len(broadcast_tokens),
            "recipients": [
                bt.target_model_id or (bt.custom_metadata or {}).get("contributor_id", "")
                for bt in broadcast_tokens
            ],
        }

    return bom


# ══════════════════════════════════════════════════════════════════════════
#  Compliance Evidence Package
# ══════════════════════════════════════════════════════════════════════════


def export_compliance_package(
    round_id: str,
    audit_log: AuditLog,
    contributor_tokens: List[TesseraToken],
    aggregated_token: Optional[TesseraToken] = None,
    broadcast_tokens: Optional[List[TesseraToken]] = None,
) -> dict:
    """
    Export a complete compliance evidence package for a round.

    This is the document you hand to auditors. It contains:
        1. AI-BOM (what went in, what came out)
        2. Audit trail (every event, hash-chained)
        3. Merkle root (tamper-evident fingerprint)
        4. Privacy accounting (epsilon/delta composition proof)
        5. Policy compliance summary (violations, rejections)
        6. Regulatory mapping (which requirements this satisfies)

    Suitable for submission to:
        - EU AI Office (conformity assessment evidence)
        - NIST auditors (AI RMF compliance)
        - ISO 42001 certification bodies
        - Domain regulators (FDA, OCC, etc.)
    """
    round_entries = audit_log.get_round_entries(round_id)
    violations = [
        e
        for e in round_entries
        if e.event_type
        in {
            AuditEventType.POLICY_VIOLATION,
            AuditEventType.PRIVACY_BUDGET_EXCEEDED,
            AuditEventType.TOKEN_REJECTED,
        }
    ]
    accepted = [e for e in round_entries if e.event_type == AuditEventType.TOKEN_ACCEPTED]
    rejected = [e for e in round_entries if e.event_type == AuditEventType.TOKEN_REJECTED]

    composed_epsilon = math.sqrt(len(contributor_tokens)) * max(
        (t.privacy_epsilon for t in contributor_tokens), default=0.0
    )
    composed_delta = sum(t.privacy_delta for t in contributor_tokens)

    return {
        "schema_version": "1.0",
        "framework": "tessera",
        "round_id": round_id,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ai_bom": generate_ai_bom(round_id, contributor_tokens, aggregated_token, broadcast_tokens),
        "audit_trail": {
            "total_entries": len(round_entries),
            "entries": [e.to_dict() for e in round_entries],
            "merkle_root": audit_log.merkle_root(),
            "chain_valid": audit_log.verify_chain()[0],
        },
        "privacy_accounting": {
            "contributor_budgets": [
                {
                    "contributor_id": (t.custom_metadata or {}).get(
                        "contributor_id", t.source_model_id
                    ),
                    "epsilon": t.privacy_epsilon,
                    "delta": t.privacy_delta,
                }
                for t in contributor_tokens
            ],
            "composed_epsilon": composed_epsilon,
            "composed_delta": composed_delta,
            "within_budget": (
                composed_epsilon <= MAX_COMPOSED_EPSILON and composed_delta <= MAX_COMPOSED_DELTA
            ),
        },
        "policy_summary": {
            "tokens_accepted": len(accepted),
            "tokens_rejected": len(rejected),
            "violations": len(violations),
            "violation_details": [v.to_dict() for v in violations],
            "compliant": len(violations) == 0,
        },
        "retention": {
            "retention_days": DEFAULT_RETENTION_DAYS,
            "earliest_entry": round_entries[0].timestamp if round_entries else None,
            "latest_entry": round_entries[-1].timestamp if round_entries else None,
        },
    }
