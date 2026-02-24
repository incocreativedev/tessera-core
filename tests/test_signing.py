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
    ContributorIdentity,
    SignedTokenEnvelope,
    KeyStore,
    FRESHNESS_TTL_SECONDS,
    CLOCK_SKEW_TOLERANCE_SECONDS,
)
from tessera.swarm import swarm_metadata, validate_for_swarm
from tessera.audit import AuditLog, AuditEventType


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

    def test_signature_verified_event_type_exists(self):
        assert AuditEventType.SIGNATURE_VERIFIED.value == "signature_verified"

    def test_signature_failed_event_type_exists(self):
        assert AuditEventType.SIGNATURE_FAILED.value == "signature_failed"

    def test_audit_log_can_record_signature_verified(self):
        log = AuditLog()
        entry = log.record(
            AuditEventType.SIGNATURE_VERIFIED,
            "r1",
            contributor_id="site_a",
            details={"public_key_hex": "ab" * 32},
        )
        assert entry.event_type == AuditEventType.SIGNATURE_VERIFIED

    def test_audit_log_can_record_signature_failed(self):
        log = AuditLog()
        entry = log.record(
            AuditEventType.SIGNATURE_FAILED,
            "r1",
            contributor_id="evil_site",
            details={"reason": "tampered payload"},
        )
        assert entry.event_type == AuditEventType.SIGNATURE_FAILED


# ── ContributorIdentity ──────────────────────────────────────────────────


class TestContributorIdentity:
    def test_generate_creates_identity_with_keys(self):
        identity = ContributorIdentity.generate("site_a")
        assert identity.contributor_id == "site_a"
        assert identity.private_key is not None
        assert identity.public_key is not None

    def test_public_key_hex_is_64_chars(self):
        identity = ContributorIdentity.generate("site_a")
        assert len(identity.public_key_hex()) == 64

    def test_from_pem_roundtrip(self):
        identity = ContributorIdentity.generate("site_a")
        pem = identity.private_key_pem()
        restored = ContributorIdentity.from_pem("site_a", pem)
        assert restored.public_key_hex() == identity.public_key_hex()
        assert restored.contributor_id == "site_a"

    def test_from_public_only_has_no_private_key(self):
        identity = ContributorIdentity.generate("site_a")
        verifier = ContributorIdentity.from_public_only("site_a", identity.public_key_hex())
        assert verifier.private_key is None
        assert verifier.public_key_hex() == identity.public_key_hex()

    def test_sign_produces_envelope(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        assert isinstance(envelope, SignedTokenEnvelope)
        assert envelope.signer_id == "site_a"
        assert len(envelope.signature_hex) == 128
        assert len(envelope.signer_public_key_hex) == 64

    def test_sign_does_not_mutate_token(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        original_meta = dict(token.custom_metadata)
        identity.sign(token)
        assert token.custom_metadata == original_meta

    def test_verify_with_correct_identity(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        ok, reason = identity.verify(envelope)
        assert ok is True
        assert reason == "ok"

    def test_verify_with_public_only_identity(self):
        signer = ContributorIdentity.generate("site_a")
        verifier = ContributorIdentity.from_public_only("site_a", signer.public_key_hex())
        token = _make_token()
        envelope = signer.sign(token)
        ok, reason = verifier.verify(envelope)
        assert ok is True

    def test_sign_raises_without_private_key(self):
        identity = ContributorIdentity.generate("site_a")
        verifier = ContributorIdentity.from_public_only("site_a", identity.public_key_hex())
        token = _make_token()
        with pytest.raises(RuntimeError, match="no private key"):
            verifier.sign(token)

    def test_save_and_load(self, tmp_path):
        identity = ContributorIdentity.generate("site_a")
        path = str(tmp_path / "site_a.pem")
        identity.save(path)
        loaded = ContributorIdentity.load("site_a", path)
        assert loaded.public_key_hex() == identity.public_key_hex()

    def test_repr_contains_contributor_id(self):
        identity = ContributorIdentity.generate("my_site")
        r = repr(identity)
        assert "my_site" in r
        assert "has_private=True" in r

    def test_different_identities_different_keys(self):
        a = ContributorIdentity.generate("site_a")
        b = ContributorIdentity.generate("site_b")
        assert a.public_key_hex() != b.public_key_hex()


# ── SignedTokenEnvelope ──────────────────────────────────────────────────


class TestSignedTokenEnvelope:
    def test_self_verify_passes(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        ok, reason = envelope.verify()
        assert ok is True

    def test_tampered_uhs_vector_detected(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        token.uhs_vector[0] += 999.0  # tamper after signing
        ok, reason = envelope.verify()
        assert ok is False
        assert "tampered" in reason.lower() or "failed" in reason.lower()

    def test_tampered_contributor_id_detected(self):
        identity = ContributorIdentity.generate("honest")
        token = _make_token(contributor_id="honest")
        envelope = identity.sign(token)
        token.custom_metadata["contributor_id"] = "evil"
        ok, reason = envelope.verify()
        assert ok is False

    def test_tampered_round_id_detected(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token(round_id="round_001")
        envelope = identity.sign(token)
        token.custom_metadata["swarm_round_id"] = "round_999"
        ok, reason = envelope.verify()
        assert ok is False

    def test_malformed_signature_hex_fails(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        bad_env = SignedTokenEnvelope(
            token=token,
            signature_hex="notvalidhex!!",
            signer_id="site_a",
            signer_public_key_hex=envelope.signer_public_key_hex,
        )
        ok, reason = bad_env.verify()
        assert ok is False

    def test_to_dict_contains_expected_keys(self):
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        d = envelope.to_dict()
        assert d["signer_id"] == "site_a"
        assert len(d["signature_hex"]) == 128
        assert len(d["signer_public_key_hex"]) == 64

    def test_signing_is_deterministic(self):
        """Ed25519 is deterministic: same key + same token = same signature."""
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        env1 = identity.sign(token)
        env2 = identity.sign(token)
        assert env1.signature_hex == env2.signature_hex


# ── KeyStore ─────────────────────────────────────────────────────────────


class TestKeyStore:
    def test_register_and_is_registered(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        assert ks.is_registered("site_a")

    def test_unregistered_is_not_registered(self):
        ks = KeyStore()
        assert not ks.is_registered("unknown")

    def test_revoke_removes_registration(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        ks.revoke("site_a")
        assert not ks.is_registered("site_a")
        assert ks.is_revoked("site_a")

    def test_re_register_clears_revocation(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        ks.revoke("site_a")
        ks.register("site_a", identity.public_key)
        assert ks.is_registered("site_a")
        assert not ks.is_revoked("site_a")

    def test_verify_envelope_passes_valid(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        token = _make_token()
        envelope = identity.sign(token)
        ok, reason = ks.verify_envelope(envelope)
        assert ok is True

    def test_verify_envelope_fails_unregistered(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        ok, reason = ks.verify_envelope(envelope)
        assert ok is False
        assert "not registered" in reason.lower()

    def test_verify_envelope_fails_revoked(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        ks.revoke("site_a")
        token = _make_token()
        envelope = identity.sign(token)
        ok, reason = ks.verify_envelope(envelope)
        assert ok is False
        assert "revoked" in reason.lower()

    def test_verify_envelope_fails_key_substitution(self):
        """Attacker registers their key but uses the legitimate signer_id."""
        ks = KeyStore()
        legit = ContributorIdentity.generate("site_a")
        attacker = ContributorIdentity.generate("site_a")
        ks.register("site_a", legit.public_key)
        token = _make_token()
        # Attacker signs with their key but claims to be "site_a"
        attacker_env = attacker.sign(token)
        ok, reason = ks.verify_envelope(attacker_env)
        assert ok is False
        assert "mismatch" in reason.lower()

    def test_verify_envelope_fails_tampered_token(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        token = _make_token()
        envelope = identity.sign(token)
        token.uhs_vector[0] += 42.0  # tamper
        ok, reason = ks.verify_envelope(envelope)
        assert ok is False

    def test_verify_token_embedded_sig(self):
        """KeyStore.verify_token works with sign_token() (functional API)."""
        ks = KeyStore()
        priv, pub = generate_keypair()
        ks.register("site_a", pub)
        token = _make_token()
        sign_token(token, priv)
        ok, reason = ks.verify_token(token)
        assert ok is True

    def test_verify_token_wrong_contributor_id_fails(self):
        ks = KeyStore()
        priv, pub = generate_keypair()
        ks.register("site_a", pub)
        token = _make_token(contributor_id="site_a")
        sign_token(token, priv)
        ok, reason = ks.verify_token(token, expected_contributor_id="site_b")
        assert ok is False
        assert "mismatch" in reason.lower()

    def test_register_hex_works(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register_hex("site_a", identity.public_key_hex())
        assert ks.is_registered("site_a")

    def test_len_counts_active_contributors(self):
        ks = KeyStore()
        for i in range(3):
            identity = ContributorIdentity.generate(f"site_{i}")
            ks.register(f"site_{i}", identity.public_key)
        assert len(ks) == 3
        ks.revoke("site_0")
        assert len(ks) == 2

    def test_contributor_ids_excludes_revoked(self):
        ks = KeyStore()
        a = ContributorIdentity.generate("site_a")
        b = ContributorIdentity.generate("site_b")
        ks.register("site_a", a.public_key)
        ks.register("site_b", b.public_key)
        ks.revoke("site_a")
        assert "site_b" in ks.contributor_ids
        assert "site_a" not in ks.contributor_ids

    def test_json_roundtrip(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        ks.revoke("site_x")  # revoke a contributor that was never registered
        json_str = ks.to_json()
        restored = KeyStore.from_json(json_str)
        assert restored.is_registered("site_a")
        assert not restored.is_registered("unknown")

    def test_get_public_key_returns_correct_key(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        recovered = ks.get_public_key("site_a")
        assert public_key_to_hex(recovered) == identity.public_key_hex()

    def test_get_public_key_returns_none_for_unregistered(self):
        ks = KeyStore()
        assert ks.get_public_key("unknown") is None

    def test_repr_shows_counts(self):
        ks = KeyStore()
        identity = ContributorIdentity.generate("site_a")
        ks.register("site_a", identity.public_key)
        r = repr(ks)
        assert "registered=1" in r

    def test_freshness_constants_are_positive(self):
        assert FRESHNESS_TTL_SECONDS > 0
        assert CLOCK_SKEW_TOLERANCE_SECONDS > 0
