"""Tests for tessera.signing — Ed25519 token authentication."""

import numpy as np
import pytest

from tessera.token import TesseraToken, KnowledgeType
from tessera.signing import (
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
from tessera.swarm import swarm_metadata, validate_for_swarm
from tessera.audit import AuditLog


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_token(contributor_id: str = "site_a", round_id: str = "r1") -> TesseraToken:
    return TesseraToken(
        knowledge_type=KnowledgeType.SWARM,
        uhs_vector=np.random.randn(2048).tolist(),
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={"nodes": [], "root": contributor_id},
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
        source_model_id=contributor_id,
        custom_metadata=swarm_metadata(
            swarm_round_id=round_id,
            contributor_id=contributor_id,
            local_data_fingerprint=f"sha256:{contributor_id}",
        ),
    )


# ── Key generation and serialisation ────────────────────────────────────


class TestKeyGeneration:
    def test_generate_keypair_returns_pair(self):
        priv, pub = generate_keypair()
        assert priv is not None
        assert pub is not None

    def test_public_key_hex_length(self):
        _, pub = generate_keypair()
        hex_str = public_key_to_hex(pub)
        assert len(hex_str) == 64  # 32 bytes = 64 hex chars

    def test_public_key_hex_roundtrip(self):
        _, pub = generate_keypair()
        hex_str = public_key_to_hex(pub)
        restored = public_key_from_hex(hex_str)
        assert public_key_to_hex(restored) == hex_str

    def test_private_key_pem_roundtrip(self):
        priv, _ = generate_keypair()
        pem = private_key_to_pem(priv)
        assert pem.startswith(b"-----BEGIN PRIVATE KEY-----")
        restored = private_key_from_pem(pem)
        # Verify same key by checking derived public key matches
        assert public_key_to_hex(priv.public_key()) == public_key_to_hex(restored.public_key())

    def test_save_and_load_private_key(self, tmp_path):
        priv, _ = generate_keypair()
        key_path = str(tmp_path / "test.pem")
        save_private_key(priv, key_path)
        loaded = load_private_key(key_path)
        assert public_key_to_hex(priv.public_key()) == public_key_to_hex(loaded.public_key())

    def test_keypairs_are_unique(self):
        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        assert public_key_to_hex(pub1) != public_key_to_hex(pub2)

    def test_invalid_public_key_hex_raises(self):
        with pytest.raises((ValueError, Exception)):
            public_key_from_hex("deadbeef")  # too short

    def test_wrong_key_type_raises(self):
        from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            NoEncryption,
            PrivateFormat,
        )

        rsa_key = generate_private_key(65537, 2048)
        rsa_pem = rsa_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        with pytest.raises(ValueError, match="Ed25519"):
            private_key_from_pem(rsa_pem)


# ── Token signing ────────────────────────────────────────────────────────


class TestTokenSigning:
    def test_sign_adds_metadata_fields(self):
        priv, _ = generate_keypair()
        token = _make_token()
        assert not is_signed(token)

        sign_token(token, priv)

        assert is_signed(token)
        assert SIGNATURE_KEY in token.custom_metadata
        assert PUBLIC_KEY_HEX_KEY in token.custom_metadata

    def test_signature_hex_length(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        sig_hex = token.custom_metadata[SIGNATURE_KEY]
        assert len(sig_hex) == 128  # 64 bytes = 128 hex chars

    def test_sign_preserves_existing_metadata(self):
        priv, _ = generate_keypair()
        token = _make_token()
        token.custom_metadata["custom_field"] = "preserved"
        sign_token(token, priv)
        assert token.custom_metadata["custom_field"] == "preserved"

    def test_signing_is_deterministic(self):
        """Same key + same payload = same signature (Ed25519 is deterministic)."""
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        sig1 = token.custom_metadata[SIGNATURE_KEY]

        strip_signature(token)
        sign_token(token, priv)
        sig2 = token.custom_metadata[SIGNATURE_KEY]

        assert sig1 == sig2

    def test_strip_signature_removes_fields(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        assert is_signed(token)

        strip_signature(token)
        assert not is_signed(token)
        assert SIGNATURE_KEY not in token.custom_metadata
        assert PUBLIC_KEY_HEX_KEY not in token.custom_metadata

    def test_strip_signature_idempotent(self):
        token = _make_token()
        strip_signature(token)  # no-op on unsigned token
        assert not is_signed(token)


# ── Signature verification ───────────────────────────────────────────────


class TestVerification:
    def test_valid_signature_passes(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        ok, reason = verify_token_signature(token)
        assert ok is True
        assert reason == "ok"

    def test_unsigned_token_fails(self):
        token = _make_token()
        ok, reason = verify_token_signature(token)
        assert ok is False
        assert "unsigned" in reason.lower()

    def test_tampered_uhs_vector_detected(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        # Modify payload after signing
        token.uhs_vector[0] += 999.0
        ok, reason = verify_token_signature(token)
        assert ok is False
        assert "tampered" in reason.lower() or "failed" in reason.lower()

    def test_tampered_contributor_id_detected(self):
        priv, _ = generate_keypair()
        token = _make_token(contributor_id="honest_site")
        sign_token(token, priv)
        # Impersonation: change contributor_id after signing
        token.custom_metadata["contributor_id"] = "evil_site"
        ok, reason = verify_token_signature(token)
        assert ok is False

    def test_tampered_round_id_detected(self):
        priv, _ = generate_keypair()
        token = _make_token(round_id="round_001")
        sign_token(token, priv)
        # Replay: change round_id to submit in a different round
        token.custom_metadata["swarm_round_id"] = "round_999"
        ok, reason = verify_token_signature(token)
        assert ok is False

    def test_wrong_expected_key_fails(self):
        priv1, _ = generate_keypair()
        _, pub2 = generate_keypair()
        token = _make_token()
        sign_token(token, priv1)
        ok, reason = verify_token_signature(token, expected_public_key=pub2)
        assert ok is False
        assert "mismatch" in reason.lower()

    def test_correct_expected_key_passes(self):
        priv, pub = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        ok, reason = verify_token_signature(token, expected_public_key=pub)
        assert ok is True
        assert reason == "ok"

    def test_malformed_signature_hex_fails(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        token.custom_metadata[SIGNATURE_KEY] = "notvalidhex!!"
        ok, reason = verify_token_signature(token)
        assert ok is False

    def test_wrong_length_signature_fails(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        # Truncate signature to wrong length
        token.custom_metadata[SIGNATURE_KEY] = "ab" * 32  # 32 bytes, not 64
        ok, reason = verify_token_signature(token)
        assert ok is False
        assert "length" in reason.lower()

    def test_different_contributors_different_sigs(self):
        priv, _ = generate_keypair()
        token_a = _make_token(contributor_id="site_a")
        token_b = _make_token(contributor_id="site_b")
        sign_token(token_a, priv)
        sign_token(token_b, priv)
        assert token_a.custom_metadata[SIGNATURE_KEY] != token_b.custom_metadata[SIGNATURE_KEY]


# ── Integration: validate_for_swarm with require_signature ───────────────


class TestValidateForSwarmSigning:
    def test_require_signature_false_passes_unsigned(self):
        token = _make_token()
        ok, reason = validate_for_swarm(token, require_signature=False)
        assert ok is True

    def test_require_signature_true_rejects_unsigned(self):
        token = _make_token()
        ok, reason = validate_for_swarm(token, require_signature=True)
        assert ok is False
        assert "signature" in reason.lower()

    def test_require_signature_true_accepts_signed(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        ok, reason = validate_for_swarm(token, require_signature=True)
        assert ok is True

    def test_require_signature_true_rejects_tampered(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        token.uhs_vector[0] += 1.0  # tamper
        ok, reason = validate_for_swarm(token, require_signature=True)
        assert ok is False


# ── Integration: AuditLog records signing status ────────────────────────


class TestAuditSigningIntegration:
    def test_audit_records_signed_true(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)

        log = AuditLog()
        entry = log.record_token_submission("r1", token, accepted=True)
        assert entry.details["signed"] is True
        assert entry.details["signature_valid"] is True
        assert len(entry.details["public_key_hex"]) == 64

    def test_audit_records_signed_false(self):
        token = _make_token()
        log = AuditLog()
        entry = log.record_token_submission("r1", token, accepted=True)
        assert entry.details["signed"] is False
        assert entry.details["signature_valid"] is False

    def test_audit_records_tampered_sig_invalid(self):
        priv, _ = generate_keypair()
        token = _make_token()
        sign_token(token, priv)
        token.uhs_vector[0] += 1.0  # tamper after signing

        log = AuditLog()
        entry = log.record_token_submission("r1", token, accepted=False, reason="tampered")
        assert entry.details["signed"] is True
        assert entry.details["signature_valid"] is False
