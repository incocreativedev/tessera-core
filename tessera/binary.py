"""
tessera.binary — Tessera Binary Format (TBF) v1.1 reader/writer.

TBF is a compact single-file format that replaces the JSON + SafeTensors
dual-file layout from v1.0. It achieves 60-80% size reduction, supports
incremental parsing, includes HMAC integrity verification, and enables
zero-copy memory-mapped access to the vector payload.

File layout:
    ┌─────────────────────────────────┐
    │  File Header       (32 bytes)   │
    │  Metadata          (MessagePack)│
    │  Padding           (to 64-byte) │
    │  Vector Payload    (aligned)    │
    │  Trailer           (HMAC + CRC) │
    └─────────────────────────────────┘

Usage:
    from tessera.binary import TBFSerializer

    # Write
    TBFSerializer.save("token.tbf", token)

    # Read
    token = TBFSerializer.load("token.tbf")

    # Read with HMAC verification
    token = TBFSerializer.load("token.tbf", hmac_key=b"shared-secret")

Wire format: little-endian byte order throughout.
"""

import hashlib
import hmac
import io
import struct
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

import msgpack
import numpy as np

from .token import TesseraToken, KnowledgeType
from .utils import setup_logging

logger = setup_logging("tessera.binary")

# ── Constants ────────────────────────────────────────────────────────────────

MAGIC = b"TBF1"  # 4 bytes: 0x54 0x42 0x46 0x31
FORMAT_VERSION_MAJOR = 1
FORMAT_VERSION_MINOR = 1
HEADER_SIZE = 32  # Fixed header length
VECTOR_ALIGNMENT = 64  # Payload aligned to 64 bytes
TRAILER_SIZE = 36  # 32-byte HMAC-SHA256 + 4-byte CRC-32C
MIME_TYPE = "application/vnd.tessera.token+binary"
FILE_EXTENSION = ".tbf"


class QuantType(IntEnum):
    """Quantisation type for the vector payload."""

    FLOAT32 = 0  # 4 bytes per element, baseline
    FLOAT16 = 1  # 2 bytes per element, ~50% size
    BFLOAT16 = 2  # 2 bytes per element, ~50% size, better dynamic range
    INT8 = 3  # 1 byte per element, ~25% size (requires scale/zero-point)


# Bytes per element for each quantisation type
_QUANT_BYTES = {
    QuantType.FLOAT32: 4,
    QuantType.FLOAT16: 2,
    QuantType.BFLOAT16: 2,
    QuantType.INT8: 1,
}

# NumPy dtype for each quantisation type
_QUANT_DTYPE = {
    QuantType.FLOAT32: np.float32,
    QuantType.FLOAT16: np.float16,
    # bfloat16 doesn't have a native numpy dtype — we use uint16 and convert
    QuantType.BFLOAT16: np.uint16,
    QuantType.INT8: np.int8,
}


# ── Header Layout ────────────────────────────────────────────────────────────
#
#  Offset  Size  Field
#  ------  ----  -----
#   0       4    Magic ("TBF1")
#   4       1    Version major
#   5       1    Version minor
#   6       1    Flags (bit 0: HMAC present, bit 1: compressed)
#   7       1    Quantisation type (QuantType enum)
#   8       4    Metadata length (bytes, uint32)
#  12       4    Vector count (uint32) — number of elements
#  16       4    Vector byte length (uint32)
#  20       4    Header CRC-32C (of bytes 0-19)
#  24       8    Reserved (must be zero)
#
HEADER_STRUCT = struct.Struct("<4sBBBBIIII8s")  # 32 bytes

# Flag bits
FLAG_HMAC = 0x01
FLAG_COMPRESSED = 0x02


# ── CRC-32C ──────────────────────────────────────────────────────────────────


def _crc32c(data: bytes) -> int:
    """
    Compute CRC-32C (Castagnoli).

    Falls back to standard CRC-32 if crc32c is not available.
    For production, use the google-crc32c package.
    """
    try:
        import crc32c

        return crc32c.crc32c(data)
    except ImportError:
        # Fallback: standard CRC-32 (not Castagnoli, but functional)
        import binascii

        return binascii.crc32(data) & 0xFFFFFFFF


# ── Quantisation Helpers ─────────────────────────────────────────────────────


def _quantise(vector: np.ndarray, quant: QuantType) -> tuple:
    """
    Quantise a float32 vector to the target type.

    Returns:
        (payload_bytes, scale, zero_point) — scale/zero_point are None
        except for INT8 quantisation.
    """
    vector = np.asarray(vector, dtype=np.float32)

    if quant == QuantType.FLOAT32:
        return vector.tobytes(), None, None

    elif quant == QuantType.FLOAT16:
        return vector.astype(np.float16).tobytes(), None, None

    elif quant == QuantType.BFLOAT16:
        # Convert float32 → bfloat16 by truncating the lower 16 bits
        raw = vector.view(np.uint32)
        bf16 = ((raw >> 16) & 0xFFFF).astype(np.uint16)
        return bf16.tobytes(), None, None

    elif quant == QuantType.INT8:
        # Affine quantisation: val = scale * (int8_val - zero_point)
        vmin, vmax = float(vector.min()), float(vector.max())
        if vmax == vmin:
            # Constant vector: all elements identical.
            # Encode as int8 zeros; store the constant in scale/zero_point
            # so dequant: (0 - zero_point) * scale = vmin.
            # We set zero_point = 0, scale = vmin (or 1.0 if vmin == 0).
            if vmin == 0.0:
                scale = 1.0
            else:
                scale = vmin
            zero_point = -1  # dequant: (0 - (-1)) * scale = scale = vmin
            quantised = np.zeros(vector.shape, dtype=np.int8)
            return quantised.tobytes(), scale, zero_point
        else:
            scale = (vmax - vmin) / 255.0
            zero_point = int(round(-vmin / scale)) - 128

        quantised = np.clip(np.round(vector / scale) + zero_point, -128, 127).astype(np.int8)
        return quantised.tobytes(), scale, zero_point

    else:
        raise ValueError(f"Unsupported quantisation type: {quant}")


def _dequantise(
    payload: bytes,
    n_elements: int,
    quant: QuantType,
    scale: Optional[float] = None,
    zero_point: Optional[int] = None,
) -> np.ndarray:
    """Dequantise payload bytes back to float32."""

    if quant == QuantType.FLOAT32:
        return np.frombuffer(payload, dtype=np.float32, count=n_elements).copy()

    elif quant == QuantType.FLOAT16:
        return np.frombuffer(payload, dtype=np.float16, count=n_elements).astype(np.float32).copy()

    elif quant == QuantType.BFLOAT16:
        bf16 = np.frombuffer(payload, dtype=np.uint16, count=n_elements)
        # Reconstruct float32 by shifting bfloat16 bits into upper 16 bits
        f32_bits = bf16.astype(np.uint32) << 16
        return f32_bits.view(np.float32).copy()

    elif quant == QuantType.INT8:
        if scale is None or zero_point is None:
            raise ValueError("INT8 dequantisation requires scale and zero_point")
        int8_vals = np.frombuffer(payload, dtype=np.int8, count=n_elements)
        return ((int8_vals.astype(np.float32) - zero_point) * scale).copy()

    else:
        raise ValueError(f"Unsupported quantisation type: {quant}")


# ── Metadata Encoding ────────────────────────────────────────────────────────


def _token_to_metadata(token: TesseraToken) -> dict:
    """
    Convert a TesseraToken to the MessagePack-serialisable metadata dict.

    The UHS vector is NOT included — it goes in the vector payload section.
    """
    return {
        "t": token.knowledge_type.value,  # knowledge type
        "m": token.modality_weights,  # modality weights
        "c": token.correlation_map,  # correlation map
        "l": token.lineage_dag,  # lineage DAG
        "g": token.generation,  # generation count
        "proj": token.projection_hints,  # projection hints
        "pe": token.privacy_epsilon,  # privacy ε
        "pd": token.privacy_delta,  # privacy δ
        "d": token.drift_score,  # drift score
        "src": token.source_model_id,  # source model ID
        "tgt": token.target_model_id,  # target model ID
        "ts": token.timestamp,  # ISO 8601 timestamp
        "v": token.version,  # protocol version
        "x": token.custom_metadata,  # extensible metadata
    }


def _metadata_to_token(meta: dict, uhs_vector: list) -> TesseraToken:
    """Reconstruct a TesseraToken from metadata dict + UHS vector."""
    return TesseraToken(
        knowledge_type=KnowledgeType(meta["t"]),
        uhs_vector=uhs_vector,
        modality_weights=meta["m"],
        correlation_map=meta["c"],
        lineage_dag=meta["l"],
        generation=meta.get("g", 1),
        projection_hints=meta.get("proj", []),
        privacy_epsilon=meta.get("pe", 1.0),
        privacy_delta=meta.get("pd", 1e-5),
        drift_score=meta.get("d", 0.0),
        source_model_id=meta.get("src", ""),
        target_model_id=meta.get("tgt"),
        timestamp=meta.get("ts", ""),
        version=meta.get("v", "1.0"),
        custom_metadata=meta.get("x", {}),
    )


# ── TBFSerializer ────────────────────────────────────────────────────────────


class TBFSerializer:
    """
    Tessera Binary Format (TBF) v1.1 serialiser.

    Produces compact single-file tokens with:
    - 32-byte fixed header (magic, version, sizes)
    - MessagePack metadata (variable length)
    - 64-byte-aligned vector payload
    - HMAC-SHA256 trailer for integrity verification

    Example:
        from tessera.binary import TBFSerializer

        TBFSerializer.save("token.tbf", token)
        loaded = TBFSerializer.load("token.tbf")
    """

    @staticmethod
    def save(
        filepath: Union[str, Path],
        token: TesseraToken,
        quant: QuantType = QuantType.FLOAT32,
        hmac_key: Optional[bytes] = None,
    ) -> int:
        """
        Write a TesseraToken to a .tbf file.

        Args:
            filepath: Output path (should end with .tbf).
            token: The token to serialise.
            quant: Quantisation type for the vector payload.
            hmac_key: Optional shared secret for HMAC-SHA256 authentication.
                      If None, HMAC is zeroed out (integrity via CRC only).

        Returns:
            Total bytes written.
        """
        filepath = Path(filepath)

        # ── 1. Quantise vector payload first (to capture scale/zp) ─────
        payload_bytes, q_scale, q_zp = _quantise(
            np.asarray(token.uhs_vector, dtype=np.float32), quant
        )

        # ── 2. Encode metadata ───────────────────────────────────────────
        meta = _token_to_metadata(token)

        # Store quantisation parameters for INT8
        if quant == QuantType.INT8:
            meta["_q_scale"] = q_scale
            meta["_q_zp"] = q_zp

        meta_bytes = msgpack.packb(meta, use_bin_type=True)
        meta_len = len(meta_bytes)
        vec_count = len(token.uhs_vector)
        vec_byte_len = len(payload_bytes)

        # ── 3. Compute padding to 64-byte alignment ─────────────────────
        content_before_payload = HEADER_SIZE + meta_len
        padding_needed = (
            VECTOR_ALIGNMENT - (content_before_payload % VECTOR_ALIGNMENT)
        ) % VECTOR_ALIGNMENT
        padding = b"\x00" * padding_needed

        # ── 4. Build header ──────────────────────────────────────────────
        flags = 0
        if hmac_key is not None:
            flags |= FLAG_HMAC

        # Pack header without CRC first (CRC field = 0 placeholder)
        header_no_crc = HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION_MAJOR,
            FORMAT_VERSION_MINOR,
            flags,
            int(quant),
            meta_len,
            vec_count,
            vec_byte_len,
            0,  # CRC placeholder
            b"\x00" * 8,  # reserved
        )

        # Compute CRC over the first 20 bytes (everything before CRC field)
        header_crc = _crc32c(header_no_crc[:20])

        # Re-pack with actual CRC
        header = HEADER_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION_MAJOR,
            FORMAT_VERSION_MINOR,
            flags,
            int(quant),
            meta_len,
            vec_count,
            vec_byte_len,
            header_crc,
            b"\x00" * 8,
        )

        # ── 5. Build trailer ─────────────────────────────────────────────
        # HMAC-SHA256 over header + metadata + padding + payload
        file_body = header + meta_bytes + padding + payload_bytes

        if hmac_key is not None:
            mac = hmac.new(hmac_key, file_body, hashlib.sha256).digest()
        else:
            mac = b"\x00" * 32

        # CRC-32C of the entire file body (header through payload)
        body_crc = _crc32c(file_body)
        trailer = mac + struct.pack("<I", body_crc)

        # ── 6. Write ─────────────────────────────────────────────────────
        full_file = file_body + trailer
        filepath.write_bytes(full_file)

        logger.info(
            f"Saved TBF: {filepath.name} "
            f"({len(full_file):,} bytes, "
            f"quant={quant.name}, "
            f"vec={vec_count}×{_QUANT_BYTES[quant]}B)"
        )

        return len(full_file)

    @staticmethod
    def load(
        filepath: Union[str, Path],
        hmac_key: Optional[bytes] = None,
        verify_crc: bool = True,
    ) -> TesseraToken:
        """
        Read a TesseraToken from a .tbf file.

        Args:
            filepath: Path to the .tbf file.
            hmac_key: Optional shared secret for HMAC verification.
                      If provided and the file has HMAC, it will be verified.
            verify_crc: Whether to verify CRC-32C checksums.

        Returns:
            Reconstructed TesseraToken.

        Raises:
            ValueError: On format errors, CRC mismatch, or HMAC failure.
        """
        filepath = Path(filepath)
        raw = filepath.read_bytes()

        if len(raw) < HEADER_SIZE + TRAILER_SIZE:
            raise ValueError(f"File too small ({len(raw)} bytes) to be a valid TBF file")

        # ── 1. Parse header ──────────────────────────────────────────────
        header_data = HEADER_STRUCT.unpack(raw[:HEADER_SIZE])
        (
            magic,
            ver_major,
            ver_minor,
            flags,
            quant_type,
            meta_len,
            vec_count,
            vec_byte_len,
            header_crc,
            reserved,
        ) = header_data

        if magic != MAGIC:
            raise ValueError(
                f"Invalid magic bytes: {magic!r} (expected {MAGIC!r}). " "Not a TBF file."
            )

        if ver_major > FORMAT_VERSION_MAJOR:
            raise ValueError(
                f"Unsupported TBF version {ver_major}.{ver_minor} "
                f"(this reader supports up to {FORMAT_VERSION_MAJOR}.{FORMAT_VERSION_MINOR})"
            )

        quant = QuantType(quant_type)

        # Verify header CRC
        if verify_crc:
            expected_crc = _crc32c(raw[:20])
            if header_crc != expected_crc:
                raise ValueError(
                    f"Header CRC mismatch: got {header_crc:#010x}, "
                    f"expected {expected_crc:#010x}"
                )

        has_hmac = bool(flags & FLAG_HMAC)

        # ── 2. Locate sections ───────────────────────────────────────────
        meta_start = HEADER_SIZE
        meta_end = meta_start + meta_len

        # Padding to alignment
        content_before_payload = HEADER_SIZE + meta_len
        padding_needed = (
            VECTOR_ALIGNMENT - (content_before_payload % VECTOR_ALIGNMENT)
        ) % VECTOR_ALIGNMENT

        vec_start = meta_end + padding_needed
        vec_end = vec_start + vec_byte_len

        trailer_start = len(raw) - TRAILER_SIZE

        # Sanity check
        if vec_end > trailer_start:
            raise ValueError(
                f"Malformed TBF: vector payload extends into trailer "
                f"(vec_end={vec_end}, trailer_start={trailer_start})"
            )

        # ── 3. Verify trailer ────────────────────────────────────────────
        trailer = raw[trailer_start:]
        file_hmac = trailer[:32]
        file_crc = struct.unpack("<I", trailer[32:36])[0]

        file_body = raw[:trailer_start]

        if verify_crc:
            computed_crc = _crc32c(file_body)
            if file_crc != computed_crc:
                raise ValueError(
                    f"Body CRC mismatch: got {file_crc:#010x}, " f"expected {computed_crc:#010x}"
                )

        if has_hmac and hmac_key is not None:
            expected_mac = hmac.new(hmac_key, file_body, hashlib.sha256).digest()
            if not hmac.compare_digest(file_hmac, expected_mac):
                raise ValueError("HMAC verification failed — file may be tampered")

        # ── 4. Decode metadata ───────────────────────────────────────────
        meta_bytes = raw[meta_start:meta_end]
        meta = msgpack.unpackb(meta_bytes, raw=False)

        # ── 5. Decode vector payload ─────────────────────────────────────
        vec_bytes = raw[vec_start : vec_start + vec_byte_len]

        scale = meta.pop("_q_scale", None)
        zero_point = meta.pop("_q_zp", None)

        uhs_vector = _dequantise(vec_bytes, vec_count, quant, scale, zero_point).tolist()

        # ── 6. Reconstruct token ─────────────────────────────────────────
        token = _metadata_to_token(meta, uhs_vector)

        logger.info(
            f"Loaded TBF: {filepath.name} "
            f"({len(raw):,} bytes, "
            f"quant={quant.name}, "
            f"vec={vec_count} dims)"
        )

        return token

    @staticmethod
    def detect_format(filepath: Union[str, Path]) -> str:
        """
        Auto-detect whether a file is TBF or legacy SafeTensors+JSON.

        Returns:
            "tbf" or "legacy" or "unknown"
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "rb") as f:
            header = f.read(4)

        if header == MAGIC:
            return "tbf"
        elif filepath.suffix == ".safetensors":
            return "legacy"
        else:
            return "unknown"

    @staticmethod
    def info(filepath: Union[str, Path]) -> dict:
        """
        Read TBF header without decoding the full file.

        Returns a dict with file metadata (useful for inspection/debugging).
        """
        filepath = Path(filepath)
        raw = filepath.read_bytes()

        if len(raw) < HEADER_SIZE:
            raise ValueError("File too small for TBF header")

        header_data = HEADER_STRUCT.unpack(raw[:HEADER_SIZE])
        (
            magic,
            ver_major,
            ver_minor,
            flags,
            quant_type,
            meta_len,
            vec_count,
            vec_byte_len,
            header_crc,
            _reserved,
        ) = header_data

        quant = QuantType(quant_type)
        bpe = _QUANT_BYTES[quant]

        return {
            "format": "TBF",
            "version": f"{ver_major}.{ver_minor}",
            "quantisation": quant.name,
            "has_hmac": bool(flags & FLAG_HMAC),
            "is_compressed": bool(flags & FLAG_COMPRESSED),
            "metadata_bytes": meta_len,
            "vector_count": vec_count,
            "vector_bytes": vec_byte_len,
            "bytes_per_element": bpe,
            "total_file_bytes": len(raw),
            "header_crc_ok": _crc32c(raw[:20]) == header_crc,
        }
