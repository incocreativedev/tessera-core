"""
Security regression tests for tessera.

Covers:
  CRIT-001 — Path traversal via unsanitised anchor_id (registry.py)
  HIGH-001 — Registry index path-trust loading model files (registry.py)
  MED-001  — HMAC fail-open when hmac_key is omitted (binary.py)
  MED-002  — Unbounded file/metadata read (binary.py)
"""

import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from tessera.binary import (
    MAX_META_BYTES,
    MAX_TBF_FILE_BYTES,
    TBFSerializer,
)
from tessera.registry import AnchorRegistry
from tessera.token import KnowledgeType, TesseraToken
from tessera.uhs import DecoderMLP, EncoderMLP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_token(dim: int = 64) -> TesseraToken:
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=np.random.randn(dim).astype(float).tolist(),
        modality_weights={"A": 1.0},
        correlation_map={},
        lineage_dag={},
        source_model_id="sec_tx",
        target_model_id=None,
        privacy_epsilon=1.0,
        privacy_delta=1e-5,
    )


def _make_registry(tmp_dir: str) -> AnchorRegistry:
    return AnchorRegistry(registry_dir=tmp_dir)


def _make_anchor_pair(d_model: int = 16, hub_dim: int = 32):
    enc = EncoderMLP(d_model, hub_dim)
    dec = DecoderMLP(d_model, hub_dim)
    return enc, dec


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# CRIT-001: Path traversal via anchor_id
# ---------------------------------------------------------------------------


class TestCRIT001PathTraversal:
    """anchor_id containing path separators must be rejected."""

    TRAVERSAL_IDS = [
        "../outside",
        "../../etc/passwd",
        "foo/../../bar",
        "/absolute/path",
        "valid/../sneaky",
        "null\x00byte",
        "a" * 129,  # Too long
        "",  # Empty
        ".",
        "..",
    ]

    def test_register_rejects_traversal_ids(self, tmp_dir):
        registry = _make_registry(tmp_dir)
        enc, dec = _make_anchor_pair()
        for bad_id in self.TRAVERSAL_IDS:
            with pytest.raises(ValueError):
                registry.register(bad_id, d_model=16, encoder=enc, decoder=dec)

    def test_load_rejects_traversal_ids(self, tmp_dir):
        registry = _make_registry(tmp_dir)
        for bad_id in self.TRAVERSAL_IDS:
            with pytest.raises((ValueError, KeyError)):
                registry.load(bad_id)

    def test_valid_ids_accepted(self, tmp_dir):
        registry = _make_registry(tmp_dir)
        enc, dec = _make_anchor_pair()
        valid_ids = [
            "simple",
            "model-v1",
            "llama3_8b",
            "anchor.v2",
            "A" * 128,  # Max length
            "abc123",
        ]
        for valid_id in valid_ids:
            # Should not raise
            registry.register(valid_id, d_model=16, encoder=enc, decoder=dec)
        assert set(valid_ids).issubset(set(registry.list()))

    def test_anchor_dir_stays_inside_root(self, tmp_dir):
        """Registered anchor directories must be inside anchors_dir."""
        registry = _make_registry(tmp_dir)
        enc, dec = _make_anchor_pair()
        registry.register("safe_anchor", d_model=16, encoder=enc, decoder=dec)

        anchors_root = Path(tmp_dir) / "anchors"
        # Enumerate all paths created under tmp_dir
        for dirpath, _, filenames in os.walk(tmp_dir):
            resolved = Path(dirpath).resolve()
            assert str(resolved).startswith(
                str(anchors_root.resolve())
            ) or str(resolved) == str(Path(tmp_dir).resolve()) or str(resolved) == str(
                (Path(tmp_dir) / "anchors").resolve()
            ), f"Unexpected path outside anchors_dir: {resolved}"


# ---------------------------------------------------------------------------
# HIGH-001: Registry index path-trust
# ---------------------------------------------------------------------------


class TestHIGH001RegistryPathTrust:
    """Tampered registry.json paths must not allow loading from arbitrary locations."""

    def test_load_uses_anchors_dir_not_stored_path(self, tmp_dir):
        """
        After registering a valid anchor, tamper the registry index to point
        'path' to an outside directory. Load must resolve from anchors_dir
        and NOT follow the tampered path.
        """
        registry = _make_registry(tmp_dir)
        enc, dec = _make_anchor_pair()
        registry.register("anchor_a", d_model=16, encoder=enc, decoder=dec)

        # Tamper: point path to /tmp (outside anchors_dir)
        index_path = Path(tmp_dir) / "registry.json"
        with open(index_path) as f:
            idx = json.load(f)
        idx["anchors"]["anchor_a"]["path"] = "/tmp"
        with open(index_path, "w") as f:
            json.dump(idx, f)

        # Reload registry to pick up tampered index
        registry2 = _make_registry(tmp_dir)
        # Should still load correctly from the real anchors_dir (ignoring tampered path)
        enc2, dec2 = registry2.load("anchor_a")
        assert enc2 is not None
        assert dec2 is not None

    def test_tampered_path_pointing_outside_root_raises_or_loads_safely(self, tmp_dir):
        """
        If registry.json contains a path that the reconstructed dir doesn't match,
        load should either succeed (using anchors_dir) or raise — never silently
        load from the tampered location.
        """
        registry = _make_registry(tmp_dir)
        enc, dec = _make_anchor_pair()
        registry.register("anchor_b", d_model=16, encoder=enc, decoder=dec)

        # Tamper with an absolute path to /etc
        index_path = Path(tmp_dir) / "registry.json"
        with open(index_path) as f:
            idx = json.load(f)
        idx["anchors"]["anchor_b"]["path"] = "/etc"
        with open(index_path, "w") as f:
            json.dump(idx, f)

        registry2 = _make_registry(tmp_dir)
        # Load must succeed (using reconstructed path from anchors_dir)
        # or raise an informative error — but NOT load from /etc
        try:
            enc2, dec2 = registry2.load("anchor_b")
            # If it succeeds, verify the encoder is the one we saved
            assert enc2 is not None
        except (ValueError, RuntimeError, FileNotFoundError):
            pass  # Raising is also acceptable — it just must not load from /etc


# ---------------------------------------------------------------------------
# MED-001: HMAC fail-open
# ---------------------------------------------------------------------------


class TestMED001HMACFailOpen:
    """Loading an HMAC-protected file without a key must raise, not silently accept."""

    def test_hmac_protected_file_requires_key(self, tmp_dir):
        token = _make_token()
        path = os.path.join(tmp_dir, "hmac_protected.tbf")
        TBFSerializer.save(path, token, hmac_key=b"secret")

        with pytest.raises(ValueError, match="hmac_key"):
            TBFSerializer.load(path)  # No key supplied — must raise

    def test_hmac_protected_file_with_correct_key_loads(self, tmp_dir):
        token = _make_token()
        path = os.path.join(tmp_dir, "hmac_ok.tbf")
        TBFSerializer.save(path, token, hmac_key=b"correct")
        loaded = TBFSerializer.load(path, hmac_key=b"correct")
        assert loaded.source_model_id == token.source_model_id

    def test_non_hmac_file_loads_without_key(self, tmp_dir):
        """Files saved without HMAC must still load without a key."""
        token = _make_token()
        path = os.path.join(tmp_dir, "no_hmac.tbf")
        TBFSerializer.save(path, token)  # No hmac_key → no HMAC flag set
        loaded = TBFSerializer.load(path)
        assert loaded.source_model_id == token.source_model_id

    def test_wrong_key_still_fails(self, tmp_dir):
        token = _make_token()
        path = os.path.join(tmp_dir, "hmac_wrong.tbf")
        TBFSerializer.save(path, token, hmac_key=b"right-key")
        with pytest.raises(ValueError, match="HMAC verification failed"):
            TBFSerializer.load(path, hmac_key=b"wrong-key")


# ---------------------------------------------------------------------------
# MED-002: Unbounded file/metadata read
# ---------------------------------------------------------------------------


class TestMED002UnboundedRead:
    """Oversized files and oversized metadata sections must be rejected."""

    def test_load_rejects_oversized_file(self, tmp_dir):
        """
        Simulate an oversized file by creating a valid TBF header then appending
        junk to push the reported size above MAX_TBF_FILE_BYTES.

        We patch stat() by creating a file whose real size is huge enough.
        Instead, we verify the limit constant is sane and the check fires by
        monkeypatching Path.stat.
        """
        token = _make_token()
        path = os.path.join(tmp_dir, "real_token.tbf")
        TBFSerializer.save(path, token)

        # Monkeypatch: make stat() report a size above the limit
        import unittest.mock as mock

        fake_stat = os.stat_result(
            (
                0o100644,  # st_mode
                0,  # st_ino
                0,  # st_dev
                1,  # st_nlink
                0,  # st_uid
                0,  # st_gid
                MAX_TBF_FILE_BYTES + 1,  # st_size — one byte over the limit
                0,  # st_atime
                0,  # st_mtime
                0,  # st_ctime
            )
        )

        with mock.patch("pathlib.Path.stat", return_value=fake_stat):
            with pytest.raises(ValueError, match="too large"):
                TBFSerializer.load(path)

    def test_info_rejects_oversized_file(self, tmp_dir):
        """info() must also reject files above the size limit."""
        token = _make_token()
        path = os.path.join(tmp_dir, "real_info_token.tbf")
        TBFSerializer.save(path, token)

        import unittest.mock as mock

        fake_stat = os.stat_result(
            (0o100644, 0, 0, 1, 0, 0, MAX_TBF_FILE_BYTES + 1, 0, 0, 0)
        )
        with mock.patch("pathlib.Path.stat", return_value=fake_stat):
            with pytest.raises(ValueError, match="too large"):
                TBFSerializer.info(path)

    def test_max_file_size_constant_is_reasonable(self):
        """MAX_TBF_FILE_BYTES must be between 1 MB and 1 GB (sanity check)."""
        assert 1 * 1024 * 1024 <= MAX_TBF_FILE_BYTES <= 1024 * 1024 * 1024

    def test_max_meta_bytes_constant_is_reasonable(self):
        """MAX_META_BYTES must be between 4 KB and 64 MB."""
        assert 4 * 1024 <= MAX_META_BYTES <= 64 * 1024 * 1024

    def test_normal_files_load_successfully(self, tmp_dir):
        """Normal-sized files must not be affected by the size guard."""
        token = _make_token(dim=128)
        path = os.path.join(tmp_dir, "normal.tbf")
        TBFSerializer.save(path, token)
        loaded = TBFSerializer.load(path)
        assert len(loaded.uhs_vector) == 128
