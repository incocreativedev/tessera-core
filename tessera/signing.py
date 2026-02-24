"""tessera.signing — Ed25519 cryptographic signing for Tessera tokens.

Provides contributor authentication: each participant holds an Ed25519
private key and signs their token before submission. The aggregator
verifies signatures before accepting tokens into a round.

This closes the contributor impersonation gap identified during audit design:
without signing, any party could submit a token claiming to be any
contributor_id. With signing, only the holder of the private key can
produce a valid signature for that contributor's public key.

Key design decisions:
    - Ed25519 (RFC 8032): 64-byte signatures, 32-byte public keys, fast,
      no random nonce required, deterministic (safe for audit trails).
    - Signature covers: contributor_id + round_id + uhs_vector_hash +
      timestamp. This prevents replay attacks (round_id binds to one round),
      vector substitution (uhs_vector_hash covers the payload), and
      timestamp drift attacks (timestamp is bound in the signature).
    - Keys stored as PEM (private) and raw bytes / hex (public). Private
      keys never leave the contributor's environment.
    - Signature stored in token.custom_metadata["signature_hex"] and
      token.custom_metadata["public_key_hex"] so it travels with the token.
    - ContributorIdentity: high-level identity object bundling key pair with
      contributor_id. Preferred API for new code.
    - SignedTokenEnvelope: binds a token to its signature + signer metadata
      for out-of-band transmission (e.g. network messages).
    - KeyStore: aggregator-side registry mapping contributor_id → public key
      with revocation support.

Governing body alignment:
    - EU AI Act Art. 9: Risk management — authentication prevents
      Byzantine contributors from submitting under false identities.
    - NIST SP 800-53 IA-3: Device identification and authentication.
    - ISO/IEC 42001: Accountability — attributing each token to a
      verified contributor.

Usage (simple, functional API):
    # Contributor side
    private_key, public_key = generate_keypair()
    save_private_key(private_key, "contributor.pem")
    sign_token(token, private_key)

    # Aggregator side
    ok, reason = verify_token_signature(token)

Usage (richer OO API):
    # Contributor side — create identity once, reuse
    identity = ContributorIdentity.generate("site_a")
    envelope = identity.sign(token)

    # Aggregator side — register contributors up-front
    ks = KeyStore()
    ks.register("site_a", identity.public_key)
    ok, reason = ks.verify_envelope(envelope)
"""

import datetime
import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
    load_pem_private_key,
)

from .token import TesseraToken

# ══════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════

# Metadata keys used to store signature data inside token.custom_metadata
SIGNATURE_KEY = "signature_hex"
PUBLIC_KEY_HEX_KEY = "public_key_hex"
SIGNED_FIELDS_KEY = "signed_fields"  # records which fields were signed

# Freshness window: reject tokens signed more than 24 hours ago.
# Clock skew tolerance: accept tokens up to 5 minutes into the future.
FRESHNESS_TTL_SECONDS = 86_400  # 24 hours
CLOCK_SKEW_TOLERANCE_SECONDS = 300  # 5 minutes


# ══════════════════════════════════════════════════════════════════════════
#  Key generation and serialisation (low-level helpers)
# ══════════════════════════════════════════════════════════════════════════


def generate_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """
    Generate a fresh Ed25519 keypair for a contributor.

    Returns:
        (private_key, public_key) — keep private_key secret, share public_key
        with the aggregator (or publish it in the contributor registry).
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def private_key_to_pem(private_key: Ed25519PrivateKey) -> bytes:
    """Serialise private key to PEM bytes (for secure storage)."""
    return private_key.private_bytes(
        encoding=Encoding.PEM,
        format=PrivateFormat.PKCS8,
        encryption_algorithm=NoEncryption(),
    )


def private_key_from_pem(pem_bytes: bytes) -> Ed25519PrivateKey:
    """Load an Ed25519 private key from PEM bytes."""
    key = load_pem_private_key(pem_bytes, password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError("PEM does not contain an Ed25519 private key.")
    return key


def public_key_to_hex(public_key: Ed25519PublicKey) -> str:
    """Serialise public key to lowercase hex string (32 bytes = 64 hex chars)."""
    raw = public_key.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw)
    return raw.hex()


def public_key_from_hex(hex_str: str) -> Ed25519PublicKey:
    """Reconstruct an Ed25519 public key from a hex string."""
    raw = bytes.fromhex(hex_str)
    if len(raw) != 32:
        raise ValueError(f"Ed25519 public key must be 32 bytes, got {len(raw)}.")
    return Ed25519PublicKey.from_public_bytes(raw)


def save_private_key(private_key: Ed25519PrivateKey, filepath: str) -> None:
    """Write private key PEM to file. Caller is responsible for file permissions."""
    with open(filepath, "wb") as f:
        f.write(private_key_to_pem(private_key))


def load_private_key(filepath: str) -> Ed25519PrivateKey:
    """Load private key PEM from file."""
    with open(filepath, "rb") as f:
        return private_key_from_pem(f.read())


# ══════════════════════════════════════════════════════════════════════════
#  Signing payload construction
# ══════════════════════════════════════════════════════════════════════════


def _build_signing_payload(token: TesseraToken) -> bytes:
    """
    Build the canonical byte string that gets signed.

    Covers the fields that matter for authenticity:
        - contributor_id: who is submitting
        - swarm_round_id: which round (prevents replay across rounds)
        - local_data_fingerprint: data provenance claim
        - uhs_vector_sha256: hash of the hub vector (prevents payload swap)
        - timestamp: time of signing (informational, loosely bounds freshness)
        - knowledge_type: prevents type confusion attacks

    Using JSON with sorted keys ensures canonical serialisation.
    """
    meta = token.custom_metadata or {}

    # Hash the UHS vector to keep the payload small and deterministic
    uhs_bytes = json.dumps(token.uhs_vector, separators=(",", ":")).encode("utf-8")
    uhs_hash = hashlib.sha256(uhs_bytes).hexdigest()

    payload = {
        "contributor_id": meta.get("contributor_id", token.source_model_id or ""),
        "swarm_round_id": meta.get("swarm_round_id", ""),
        "local_data_fingerprint": meta.get("local_data_fingerprint", ""),
        "uhs_vector_sha256": uhs_hash,
        "timestamp": token.timestamp,
        "knowledge_type": token.knowledge_type.value,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonicalise_token(token: TesseraToken) -> bytes:
    """
    Produce a canonical byte representation of the token for signing,
    stripping any existing signature fields first so re-signing is idempotent.

    This is the payload signed by ContributorIdentity.sign() and verified
    by KeyStore.verify_envelope().
    """
    meta = {
        k: v
        for k, v in (token.custom_metadata or {}).items()
        if k not in (SIGNATURE_KEY, PUBLIC_KEY_HEX_KEY, SIGNED_FIELDS_KEY)
    }

    uhs_bytes = json.dumps(token.uhs_vector, separators=(",", ":")).encode("utf-8")
    uhs_hash = hashlib.sha256(uhs_bytes).hexdigest()

    payload = {
        "contributor_id": meta.get("contributor_id", token.source_model_id or ""),
        "swarm_round_id": meta.get("swarm_round_id", ""),
        "local_data_fingerprint": meta.get("local_data_fingerprint", ""),
        "uhs_vector_sha256": uhs_hash,
        "timestamp": token.timestamp,
        "knowledge_type": token.knowledge_type.value,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ══════════════════════════════════════════════════════════════════════════
#  Token signing and verification (functional API — backward-compatible)
# ══════════════════════════════════════════════════════════════════════════


def sign_token(token: TesseraToken, private_key: Ed25519PrivateKey) -> None:
    """
    Sign a token in-place using the contributor's Ed25519 private key.

    Adds three fields to token.custom_metadata:
        - "signature_hex": 128-char hex string (64-byte Ed25519 signature)
        - "public_key_hex": 64-char hex string (32-byte Ed25519 public key)
        - "signed_fields": list of field names covered by the signature

    Args:
        token: The TesseraToken to sign (modified in-place).
        private_key: The contributor's Ed25519 private key.
    """
    payload = _build_signing_payload(token)
    signature = private_key.sign(payload)  # 64 bytes, deterministic

    public_key = private_key.public_key()

    if token.custom_metadata is None:
        token.custom_metadata = {}

    token.custom_metadata[SIGNATURE_KEY] = signature.hex()
    token.custom_metadata[PUBLIC_KEY_HEX_KEY] = public_key_to_hex(public_key)
    token.custom_metadata[SIGNED_FIELDS_KEY] = [
        "contributor_id",
        "swarm_round_id",
        "local_data_fingerprint",
        "uhs_vector_sha256",
        "timestamp",
        "knowledge_type",
    ]


def verify_token_signature(
    token: TesseraToken,
    expected_public_key: Optional[Ed25519PublicKey] = None,
    check_freshness: bool = False,
) -> Tuple[bool, str]:
    """
    Verify the Ed25519 signature on a token.

    Two verification modes:
        1. Self-contained: public key is read from token.custom_metadata
           (used when the aggregator trusts the key embedded in the token,
           e.g. in open swarms where keys are registered in the contributor
           registry).
        2. Pinned: caller supplies expected_public_key (used in closed swarms
           where the aggregator has a pre-registered key per contributor_id).
           If supplied, the embedded public key must also match.

    Args:
        token: Token to verify.
        expected_public_key: Optional pinned public key to check against.
        check_freshness: If True, reject tokens signed more than
            FRESHNESS_TTL_SECONDS ago or CLOCK_SKEW_TOLERANCE_SECONDS in
            the future.

    Returns:
        (True, "ok") if signature is valid.
        (False, reason) if invalid, missing, or key mismatch.
    """
    meta = token.custom_metadata or {}

    # ── Check signature fields present ──────────────────────────────────
    sig_hex = meta.get(SIGNATURE_KEY)
    pub_hex = meta.get(PUBLIC_KEY_HEX_KEY)

    if not sig_hex:
        return False, "Token is unsigned: missing 'signature_hex' in custom_metadata."
    if not pub_hex:
        return False, "Token is unsigned: missing 'public_key_hex' in custom_metadata."

    # ── Decode signature and public key ─────────────────────────────────
    try:
        sig_bytes = bytes.fromhex(sig_hex)
    except ValueError:
        return False, "Malformed signature_hex: not valid hex."

    if len(sig_bytes) != 64:
        return False, f"Invalid signature length: expected 64 bytes, got {len(sig_bytes)}."

    try:
        embedded_pub_key = public_key_from_hex(pub_hex)
    except Exception as e:
        return False, f"Malformed public_key_hex: {e}"

    # ── Pinned key check ────────────────────────────────────────────────
    if expected_public_key is not None:
        expected_hex = public_key_to_hex(expected_public_key)
        if pub_hex != expected_hex:
            return (
                False,
                f"Public key mismatch: token carries {pub_hex[:16]}…, "
                f"expected {expected_hex[:16]}…",
            )

    # ── Freshness check ─────────────────────────────────────────────────
    if check_freshness and token.timestamp:
        try:
            # token.timestamp is ISO 8601
            signed_at = datetime.datetime.fromisoformat(token.timestamp)
            if signed_at.tzinfo is None:
                signed_at = signed_at.replace(tzinfo=datetime.timezone.utc)
            now = datetime.datetime.now(datetime.timezone.utc)
            age_seconds = (now - signed_at).total_seconds()
            if age_seconds > FRESHNESS_TTL_SECONDS:
                return (
                    False,
                    f"Token is stale: signed {age_seconds:.0f}s ago "
                    f"(TTL={FRESHNESS_TTL_SECONDS}s).",
                )
            if age_seconds < -CLOCK_SKEW_TOLERANCE_SECONDS:
                return (
                    False,
                    f"Token timestamp is too far in the future "
                    f"({-age_seconds:.0f}s ahead, tolerance={CLOCK_SKEW_TOLERANCE_SECONDS}s).",
                )
        except (ValueError, TypeError):
            pass  # malformed timestamp — don't block on it

    # ── Verify signature ─────────────────────────────────────────────────
    payload = _build_signing_payload(token)
    try:
        embedded_pub_key.verify(sig_bytes, payload)
    except InvalidSignature:
        return False, "Signature verification failed: payload has been tampered."

    return True, "ok"


def strip_signature(token: TesseraToken) -> None:
    """
    Remove signature fields from token.custom_metadata (in-place).

    Useful when re-signing a token (e.g. after updating metadata) or
    when creating a test token that should be unsigned.
    """
    if token.custom_metadata:
        token.custom_metadata.pop(SIGNATURE_KEY, None)
        token.custom_metadata.pop(PUBLIC_KEY_HEX_KEY, None)
        token.custom_metadata.pop(SIGNED_FIELDS_KEY, None)


def is_signed(token: TesseraToken) -> bool:
    """Return True if the token carries a signature."""
    meta = token.custom_metadata or {}
    return bool(meta.get(SIGNATURE_KEY))


# ══════════════════════════════════════════════════════════════════════════
#  ContributorIdentity — high-level identity object (OO API)
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class ContributorIdentity:
    """
    Bundles an Ed25519 key pair with a contributor_id.

    This is the preferred high-level API for contributors.  Create once per
    participant, persist the private key (PEM), share the public key with
    the aggregator.

    Attributes:
        contributor_id: Human-readable identifier (e.g. "site_a").
        private_key: Ed25519 private key (None for verify-only identities).
        public_key: Ed25519 public key (always present).
    """

    contributor_id: str
    public_key: Ed25519PublicKey
    private_key: Optional[Ed25519PrivateKey] = field(default=None, repr=False)

    # ── Constructors ────────────────────────────────────────────────────

    @classmethod
    def generate(cls, contributor_id: str) -> "ContributorIdentity":
        """Generate a fresh keypair for a new contributor."""
        priv, pub = generate_keypair()
        return cls(contributor_id=contributor_id, public_key=pub, private_key=priv)

    @classmethod
    def from_pem(cls, contributor_id: str, pem_bytes: bytes) -> "ContributorIdentity":
        """Load an existing identity from a PEM-encoded private key."""
        priv = private_key_from_pem(pem_bytes)
        return cls(contributor_id=contributor_id, public_key=priv.public_key(), private_key=priv)

    @classmethod
    def from_public_only(cls, contributor_id: str, public_key_hex: str) -> "ContributorIdentity":
        """
        Create a verify-only identity from a public key hex string.

        Useful on the aggregator side when you only need to verify, not sign.
        """
        pub = public_key_from_hex(public_key_hex)
        return cls(contributor_id=contributor_id, public_key=pub, private_key=None)

    # ── Signing ─────────────────────────────────────────────────────────

    def sign(self, token: TesseraToken) -> "SignedTokenEnvelope":
        """
        Sign a token and return a SignedTokenEnvelope.

        The token itself is NOT mutated (use sign_token() for in-place
        mutation). The signature covers the canonical token payload built
        by _canonicalise_token().

        Raises:
            RuntimeError: if this identity has no private key.
        """
        if self.private_key is None:
            raise RuntimeError(
                f"ContributorIdentity '{self.contributor_id}' has no private key "
                "and cannot sign tokens."
            )
        payload = _canonicalise_token(token)
        sig_bytes = self.private_key.sign(payload)
        return SignedTokenEnvelope(
            token=token,
            signature_hex=sig_bytes.hex(),
            signer_id=self.contributor_id,
            signer_public_key_hex=public_key_to_hex(self.public_key),
        )

    def verify(self, envelope: "SignedTokenEnvelope") -> Tuple[bool, str]:
        """
        Verify a SignedTokenEnvelope using this identity's public key.

        Returns:
            (True, "ok") or (False, reason).
        """
        return _verify_envelope_with_key(envelope, self.public_key)

    # ── Serialisation ───────────────────────────────────────────────────

    def private_key_pem(self) -> bytes:
        """Export private key as PEM bytes."""
        if self.private_key is None:
            raise RuntimeError("No private key available.")
        return private_key_to_pem(self.private_key)

    def public_key_hex(self) -> str:
        """Export public key as hex string."""
        return public_key_to_hex(self.public_key)

    def save(self, filepath: str) -> None:
        """Save private key PEM to a file."""
        if self.private_key is None:
            raise RuntimeError("No private key to save.")
        save_private_key(self.private_key, filepath)

    @classmethod
    def load(cls, contributor_id: str, filepath: str) -> "ContributorIdentity":
        """Load identity from a PEM file."""
        priv = load_private_key(filepath)
        return cls(contributor_id=contributor_id, public_key=priv.public_key(), private_key=priv)

    def __repr__(self) -> str:
        has_priv = self.private_key is not None
        pub_abbrev = public_key_to_hex(self.public_key)[:12]
        return (
            f"ContributorIdentity(id={self.contributor_id!r}, "
            f"pub={pub_abbrev}…, has_private={has_priv})"
        )


# ══════════════════════════════════════════════════════════════════════════
#  SignedTokenEnvelope — binds token + signature for transmission
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class SignedTokenEnvelope:
    """
    Wraps a TesseraToken together with its out-of-band Ed25519 signature.

    Unlike sign_token() (which embeds the signature in token.custom_metadata),
    SignedTokenEnvelope keeps the signature separate. This is useful for:
        - Network transmission where token and signature travel together
          but must remain structurally independent.
        - Protocols where the token is forwarded to multiple aggregators
          each of whom verifies with their own registered key.

    Attributes:
        token: The wrapped TesseraToken.
        signature_hex: 128-char hex Ed25519 signature over the canonical
            token payload.
        signer_id: contributor_id of the signer.
        signer_public_key_hex: 64-char hex of the signer's Ed25519 public
            key.
    """

    token: TesseraToken
    signature_hex: str
    signer_id: str
    signer_public_key_hex: str

    def verify(self) -> Tuple[bool, str]:
        """
        Self-verify the envelope using the embedded public key.

        Returns:
            (True, "ok") or (False, reason).
        """
        try:
            pub = public_key_from_hex(self.signer_public_key_hex)
        except Exception as e:
            return False, f"Malformed signer_public_key_hex: {e}"
        return _verify_envelope_with_key(self, pub)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict (token represented as its metadata)."""
        meta = self.token.custom_metadata or {}
        return {
            "signer_id": self.signer_id,
            "signer_public_key_hex": self.signer_public_key_hex,
            "signature_hex": self.signature_hex,
            "token_source_model_id": self.token.source_model_id,
            "token_knowledge_type": self.token.knowledge_type.value,
            "token_swarm_round_id": meta.get("swarm_round_id", ""),
            "token_timestamp": self.token.timestamp,
        }


def _verify_envelope_with_key(
    envelope: SignedTokenEnvelope,
    public_key: Ed25519PublicKey,
) -> Tuple[bool, str]:
    """Internal: verify a SignedTokenEnvelope against a specific public key."""
    try:
        sig_bytes = bytes.fromhex(envelope.signature_hex)
    except ValueError:
        return False, "Malformed signature_hex: not valid hex."

    if len(sig_bytes) != 64:
        return (
            False,
            f"Invalid signature length: expected 64 bytes, got {len(sig_bytes)}.",
        )

    payload = _canonicalise_token(envelope.token)
    try:
        public_key.verify(sig_bytes, payload)
    except InvalidSignature:
        return False, "Signature verification failed: payload has been tampered."

    return True, "ok"


# ══════════════════════════════════════════════════════════════════════════
#  KeyStore — aggregator-side contributor registry
# ══════════════════════════════════════════════════════════════════════════


class KeyStore:
    """
    Aggregator-side registry mapping contributor_id → Ed25519 public key.

    The KeyStore is the authoritative source of truth for who is allowed
    to participate in a swarm. Before accepting a token, the aggregator
    should call verify_envelope() which:
        1. Checks contributor_id is registered.
        2. Verifies the signature against the registered key (not the
           embedded key) — this prevents key substitution attacks.
        3. Checks the embedded public key matches the registered one.

    Revocation:
        Calling revoke(contributor_id) prevents future verifications from
        succeeding even if the signature itself is valid. Revoked contributors
        must re-register with a new key pair.

    Usage:
        ks = KeyStore()
        ks.register("site_a", pub_a)
        ks.register("site_b", pub_b)

        ok, reason = ks.verify_envelope(envelope)
        if not ok:
            print("Rejected:", reason)

        # Later: revoke a compromised contributor
        ks.revoke("site_a")
    """

    def __init__(self):
        self._keys: Dict[str, str] = {}  # contributor_id -> public_key_hex
        self._revoked: set = set()  # contributor_ids that are revoked

    # ── Registration ────────────────────────────────────────────────────

    def register(self, contributor_id: str, public_key: Ed25519PublicKey) -> None:
        """Register a contributor's public key."""
        self._keys[contributor_id] = public_key_to_hex(public_key)
        # Re-registration clears revocation status
        self._revoked.discard(contributor_id)

    def register_hex(self, contributor_id: str, public_key_hex: str) -> None:
        """Register a contributor by hex public key string."""
        # Validate by round-tripping through the public key object
        pub = public_key_from_hex(public_key_hex)
        self.register(contributor_id, pub)

    def revoke(self, contributor_id: str) -> None:
        """Revoke a contributor's registration."""
        self._revoked.add(contributor_id)

    def is_registered(self, contributor_id: str) -> bool:
        """Return True if contributor is registered and not revoked."""
        return contributor_id in self._keys and contributor_id not in self._revoked

    def is_revoked(self, contributor_id: str) -> bool:
        """Return True if contributor has been revoked."""
        return contributor_id in self._revoked

    def get_public_key(self, contributor_id: str) -> Optional[Ed25519PublicKey]:
        """Return the registered public key, or None if not found/revoked."""
        if not self.is_registered(contributor_id):
            return None
        return public_key_from_hex(self._keys[contributor_id])

    # ── Verification ────────────────────────────────────────────────────

    def verify_envelope(self, envelope: SignedTokenEnvelope) -> Tuple[bool, str]:
        """
        Verify a SignedTokenEnvelope against the registered key store.

        Checks performed (in order):
            1. contributor_id is registered in this KeyStore.
            2. contributor_id has not been revoked.
            3. The embedded public key matches the registered key.
            4. The signature is cryptographically valid.

        Returns:
            (True, "ok") or (False, reason).
        """
        cid = envelope.signer_id

        if cid not in self._keys:
            return False, f"Contributor '{cid}' is not registered in the KeyStore."

        if cid in self._revoked:
            return False, f"Contributor '{cid}' has been revoked."

        registered_hex = self._keys[cid]
        if envelope.signer_public_key_hex != registered_hex:
            return (
                False,
                f"Public key mismatch for '{cid}': envelope carries "
                f"{envelope.signer_public_key_hex[:16]}…, "
                f"registered key is {registered_hex[:16]}…",
            )

        try:
            registered_pub = public_key_from_hex(registered_hex)
        except Exception as e:
            return False, f"Corrupt registered key for '{cid}': {e}"

        return _verify_envelope_with_key(envelope, registered_pub)

    def verify_token(
        self, token: TesseraToken, expected_contributor_id: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Verify a token whose signature is embedded in custom_metadata.

        This is the KeyStore equivalent of verify_token_signature() for tokens
        that used sign_token() (functional API) rather than ContributorIdentity.sign().

        Args:
            token: Token with signature_hex / public_key_hex in custom_metadata.
            expected_contributor_id: If supplied, also checks that the
                contributor_id in the token matches a registered key.

        Returns:
            (True, "ok") or (False, reason).
        """
        meta = token.custom_metadata or {}
        cid = meta.get("contributor_id", token.source_model_id or "")

        if expected_contributor_id is not None and cid != expected_contributor_id:
            return (
                False,
                f"contributor_id mismatch: token claims '{cid}', "
                f"expected '{expected_contributor_id}'.",
            )

        if cid and cid in self._revoked:
            return False, f"Contributor '{cid}' has been revoked."

        if cid and cid in self._keys:
            # Use the registered (pinned) public key
            try:
                registered_pub = public_key_from_hex(self._keys[cid])
            except Exception as e:
                return False, f"Corrupt registered key for '{cid}': {e}"
            return verify_token_signature(token, expected_public_key=registered_pub)

        # Fall back to self-contained verification
        return verify_token_signature(token)

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Export as a JSON-safe dict."""
        return {
            "keys": dict(self._keys),
            "revoked": sorted(self._revoked),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KeyStore":
        """Load from a previously exported dict."""
        ks = cls()
        ks._keys = dict(data.get("keys", {}))
        ks._revoked = set(data.get("revoked", []))
        return ks

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "KeyStore":
        return cls.from_dict(json.loads(json_str))

    def __len__(self) -> int:
        """Number of registered (non-revoked) contributors."""
        return sum(1 for cid in self._keys if cid not in self._revoked)

    def __repr__(self) -> str:
        n_active = len(self)
        n_revoked = len(self._revoked)
        return f"KeyStore(registered={n_active}, revoked={n_revoked})"

    @property
    def contributor_ids(self) -> List[str]:
        """List of active (non-revoked) contributor IDs."""
        return [cid for cid in self._keys if cid not in self._revoked]
