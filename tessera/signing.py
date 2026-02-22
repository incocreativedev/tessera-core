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
    - Signature stored in token.custom_metadata["signature"] and
      token.custom_metadata["public_key_hex"] so it travels with the token.

Governing body alignment:
    - EU AI Act Art. 9: Risk management — authentication prevents
      Byzantine contributors from submitting under false identities.
    - NIST SP 800-53 IA-3: Device identification and authentication.
    - ISO/IEC 42001: Accountability — attributing each token to a
      verified contributor.

Usage:
    # Contributor side (once, save private key securely)
    private_key, public_key = generate_keypair()
    save_private_key(private_key, "contributor.pem")

    # Before submitting a token
    sign_token(token, private_key)

    # Aggregator side (at submission time)
    ok, reason = verify_token_signature(token)
    if not ok:
        log.record_token_submission(round_id, token, accepted=False, reason=reason)
"""

import hashlib
import json
from typing import Optional, Tuple

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
from cryptography.exceptions import InvalidSignature

from .token import TesseraToken

# ══════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════

# Metadata keys used to store signature data inside token.custom_metadata
SIGNATURE_KEY = "signature_hex"
PUBLIC_KEY_HEX_KEY = "public_key_hex"
SIGNED_FIELDS_KEY = "signed_fields"  # records which fields were signed


# ══════════════════════════════════════════════════════════════════════════
#  Key generation and serialisation
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
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

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


# ══════════════════════════════════════════════════════════════════════════
#  Token signing and verification
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
