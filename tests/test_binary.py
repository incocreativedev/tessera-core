"""Tests for tessera.binary — TBF v1.1 serialiser."""

import os
import struct
import pytest
import numpy as np
from tessera.binary import (
    TBFSerializer,
    QuantType,
    MAGIC,
    HEADER_SIZE,
    TRAILER_SIZE,
    _quantise,
    _dequantise,
    _crc32c,
)
from tessera.token import TesseraToken, KnowledgeType


def make_token(dim=2048):
    vec = np.random.randn(dim).astype(np.float32).tolist()
    return TesseraToken(
        knowledge_type=KnowledgeType.ACTIVATION,
        uhs_vector=vec,
        modality_weights={"A": 0.9, "W": 0.1},
        correlation_map={},
        lineage_dag={"nodes": [], "root": "n0"},
        source_model_id="test_tx",
        target_model_id="test_rx",
        drift_score=1.5,
    )


class TestQuantisation:
    def test_float32_lossless(self):
        vec = np.random.randn(128).astype(np.float32)
        payload, scale, zp = _quantise(vec, QuantType.FLOAT32)
        restored = _dequantise(payload, 128, QuantType.FLOAT32)
        np.testing.assert_array_equal(vec, restored)

    def test_float16_low_error(self):
        vec = np.random.randn(128).astype(np.float32)
        payload, _, _ = _quantise(vec, QuantType.FLOAT16)
        restored = _dequantise(payload, 128, QuantType.FLOAT16)
        assert np.max(np.abs(vec - restored)) < 0.01

    def test_bfloat16_round_trip(self):
        vec = np.random.randn(128).astype(np.float32)
        payload, _, _ = _quantise(vec, QuantType.BFLOAT16)
        restored = _dequantise(payload, 128, QuantType.BFLOAT16)
        assert np.max(np.abs(vec - restored)) < 0.02

    def test_int8_bounded_error(self):
        vec = np.random.randn(128).astype(np.float32)
        payload, scale, zp = _quantise(vec, QuantType.INT8)
        restored = _dequantise(payload, 128, QuantType.INT8, scale, zp)
        # INT8 has larger error but should be bounded
        assert np.max(np.abs(vec - restored)) < 0.1

    def test_int8_constant_vector(self):
        vec = np.ones(64, dtype=np.float32) * 3.14
        payload, scale, zp = _quantise(vec, QuantType.INT8)
        restored = _dequantise(payload, 64, QuantType.INT8, scale, zp)
        assert np.max(np.abs(vec - restored)) < 0.1


class TestTBFSerializer:
    def test_save_load_float32(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "test.tbf")
        TBFSerializer.save(path, token)

        loaded = TBFSerializer.load(path)
        assert loaded.knowledge_type == token.knowledge_type
        assert loaded.source_model_id == "test_tx"
        assert abs(loaded.drift_score - 1.5) < 1e-6
        max_err = max(abs(a - b) for a, b in zip(token.uhs_vector, loaded.uhs_vector))
        assert max_err < 1e-6

    def test_save_load_float16(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "test_f16.tbf")
        TBFSerializer.save(path, token, quant=QuantType.FLOAT16)
        loaded = TBFSerializer.load(path)
        max_err = max(abs(a - b) for a, b in zip(token.uhs_vector, loaded.uhs_vector))
        assert max_err < 0.01

    def test_save_load_int8(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "test_i8.tbf")
        TBFSerializer.save(path, token, quant=QuantType.INT8)
        loaded = TBFSerializer.load(path)
        max_err = max(abs(a - b) for a, b in zip(token.uhs_vector, loaded.uhs_vector))
        assert max_err < 0.1

    def test_hmac_verification(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "test_hmac.tbf")
        key = b"test-secret"
        TBFSerializer.save(path, token, hmac_key=key)
        loaded = TBFSerializer.load(path, hmac_key=key)
        assert loaded.source_model_id == "test_tx"

    def test_hmac_wrong_key_fails(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "test_hmac_bad.tbf")
        TBFSerializer.save(path, token, hmac_key=b"correct-key")
        with pytest.raises(ValueError, match="HMAC verification failed"):
            TBFSerializer.load(path, hmac_key=b"wrong-key")

    def test_corrupted_file_fails(self, tmp_dir):
        token = make_token(dim=64)
        path = os.path.join(tmp_dir, "corrupt.tbf")
        TBFSerializer.save(path, token)

        # Corrupt a byte in the middle
        data = bytearray(open(path, "rb").read())
        data[HEADER_SIZE + 10] ^= 0xFF
        open(path, "wb").write(data)

        with pytest.raises(ValueError, match="CRC"):
            TBFSerializer.load(path)

    def test_invalid_magic_fails(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_magic.tbf")
        with open(path, "wb") as f:
            f.write(b"NOPE" + b"\x00" * 100)
        with pytest.raises(ValueError, match="Invalid magic"):
            TBFSerializer.load(path)

    def test_file_too_small_fails(self, tmp_dir):
        path = os.path.join(tmp_dir, "tiny.tbf")
        with open(path, "wb") as f:
            f.write(b"TBF1" + b"\x00" * 10)
        with pytest.raises(ValueError, match="too small"):
            TBFSerializer.load(path)

    def test_detect_format_tbf(self, tmp_dir):
        token = make_token(dim=64)
        path = os.path.join(tmp_dir, "detect.tbf")
        TBFSerializer.save(path, token)
        assert TBFSerializer.detect_format(path) == "tbf"

    def test_detect_format_legacy(self, tmp_dir):
        from tessera.token import TokenSerializer

        token = make_token(dim=64)
        path = os.path.join(tmp_dir, "detect.safetensors")
        TokenSerializer.save_token(token, path)
        assert TBFSerializer.detect_format(path) == "legacy"

    def test_info(self, tmp_dir):
        token = make_token()
        path = os.path.join(tmp_dir, "info.tbf")
        TBFSerializer.save(path, token, quant=QuantType.FLOAT16)
        info = TBFSerializer.info(path)
        assert info["format"] == "TBF"
        assert info["quantisation"] == "FLOAT16"
        assert info["vector_count"] == 2048
        assert info["header_crc_ok"] is True

    def test_metadata_preserved(self, tmp_dir):
        token = make_token()
        token.custom_metadata = {"experiment": "test-42"}
        path = os.path.join(tmp_dir, "meta.tbf")
        TBFSerializer.save(path, token)
        loaded = TBFSerializer.load(path)
        assert loaded.custom_metadata["experiment"] == "test-42"

    def test_size_ordering(self, tmp_dir):
        """FLOAT32 > FLOAT16 > INT8 in file size."""
        token = make_token()
        sizes = {}
        for q in [QuantType.FLOAT32, QuantType.FLOAT16, QuantType.INT8]:
            path = os.path.join(tmp_dir, f"size_{q.name}.tbf")
            sizes[q] = TBFSerializer.save(path, token, quant=q)
        assert sizes[QuantType.FLOAT32] > sizes[QuantType.FLOAT16] > sizes[QuantType.INT8]
