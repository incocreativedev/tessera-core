# Tessera Binary Format (TBF) v1.1 Wire Format Specification

**Status:** Stable
**Canonical Implementation:** `tessera/binary.py`
**Date:** 2026-02-21

---

## 1. Overview

The Tessera Binary Format (TBF) is a compact, single-file binary wire format for serialising Tessera knowledge-transfer tokens. It replaces the earlier dual-file layout (JSON metadata + SafeTensors vector) from v1.0.

**Design goals:**

- **Single-file:** One `.tbf` file encapsulates all metadata and the vector payload.
- **Compact:** 60--80% smaller than the JSON+SafeTensors layout, with optional quantisation down to 1 byte per element.
- **Streamable:** Fixed 32-byte header allows readers to allocate buffers before touching the rest of the file.
- **Aligned:** Vector payload is 64-byte aligned, enabling zero-copy memory-mapped access.
- **Authenticated:** Optional HMAC-SHA256 trailer for tamper detection; CRC-32C for integrity.

All multi-byte integers use **little-endian** byte order throughout the format.

---

## 2. File Layout

```
Offset (bytes)
 0            ┌──────────────────────────────────────┐
              │         File Header (32 bytes)        │
 32           ├──────────────────────────────────────┤
              │    Metadata (variable, MessagePack)   │
 32+M         ├──────────────────────────────────────┤
              │    Padding (0x00, to 64-byte boundary) │
 32+M+P       ├──────────────────────────────────────┤
              │    Vector Payload (aligned, V bytes)   │
 32+M+P+V     ├──────────────────────────────────────┤
              │         Trailer (36 bytes)             │
 32+M+P+V+36  └──────────────────────────────────────┘
```

Where:
- `M` = metadata length in bytes (from header field at offset 8)
- `P` = padding bytes = `(64 - ((32 + M) % 64)) % 64`
- `V` = vector byte length (from header field at offset 16)

Total file size = `32 + M + P + V + 36`

---

## 3. Byte Order

All multi-byte integer and floating-point fields are encoded in **little-endian** byte order. This applies to the header, vector payload, trailer CRC, and any numeric values within the format.

---

## 4. File Header (32 bytes)

The header is exactly 32 bytes and is described by the C struct-equivalent layout:

```c
// Equivalent C struct (packed, little-endian)
struct TBFHeader {
    uint8_t  magic[4];          // "TBF1" = {0x54, 0x42, 0x46, 0x31}
    uint8_t  version_major;     // 1
    uint8_t  version_minor;     // 1
    uint8_t  flags;             // Bitfield
    uint8_t  quant_type;        // QuantType enum
    uint32_t metadata_length;   // Bytes of MessagePack metadata
    uint32_t vector_count;      // Number of vector elements
    uint32_t vector_byte_length;// Total bytes in vector payload
    uint32_t header_crc;        // CRC-32C of bytes 0..19
    uint8_t  reserved[8];       // Must be zero
};
```

| Offset | Size | Type     | Field                 | Description                                      |
|--------|------|----------|-----------------------|--------------------------------------------------|
| 0      | 4    | byte[4]  | `magic`               | ASCII `"TBF1"` = `0x54 0x42 0x46 0x31`           |
| 4      | 1    | uint8    | `version_major`       | Format major version (currently `1`)             |
| 5      | 1    | uint8    | `version_minor`       | Format minor version (currently `1`)             |
| 6      | 1    | uint8    | `flags`               | Bitfield (see below)                             |
| 7      | 1    | uint8    | `quant_type`          | Quantisation enum (see Section 5)                |
| 8      | 4    | uint32le | `metadata_length`     | Length of the MessagePack metadata section, bytes |
| 12     | 4    | uint32le | `vector_count`        | Number of elements in the vector                 |
| 16     | 4    | uint32le | `vector_byte_length`  | Total byte length of the vector payload          |
| 20     | 4    | uint32le | `header_crc`          | CRC-32C computed over bytes 0 through 19         |
| 24     | 8    | byte[8]  | `reserved`            | Reserved; must be all zeroes                     |

**Python struct format string:** `<4sBBBBIIII8s` (32 bytes)

### 4.1 Flags Bitfield

| Bit | Mask   | Name            | Description                                        |
|-----|--------|-----------------|----------------------------------------------------|
| 0   | `0x01` | `FLAG_HMAC`     | If set, the trailer contains a valid HMAC-SHA256   |
| 1   | `0x02` | `FLAG_COMPRESSED`| Reserved for future compression support (not yet used) |
| 2-7 |        | (reserved)      | Must be zero                                       |

### 4.2 Header CRC Computation

The `header_crc` field at offset 20 is a CRC-32C checksum computed over the first 20 bytes of the header (offsets 0 through 19 inclusive). When writing, first pack the header with the CRC field set to `0x00000000`, compute the CRC over bytes 0..19, then re-pack the header with the computed CRC.

---

## 5. Quantisation Types

The `quant_type` header field selects how vector elements are encoded:

| Enum Value | Name      | Bytes/Element | NumPy dtype | Format Details                                                |
|------------|-----------|---------------|-------------|---------------------------------------------------------------|
| `0`        | FLOAT32   | 4             | float32     | IEEE 754 single-precision, little-endian                      |
| `1`        | FLOAT16   | 2             | float16     | IEEE 754 half-precision, little-endian                        |
| `2`        | BFLOAT16  | 2             | uint16      | Upper 16 bits of IEEE 754 float32, little-endian              |
| `3`        | INT8      | 1             | int8        | Signed 8-bit integer with affine dequantisation parameters    |

**Invariant:** `vector_byte_length == vector_count * bytes_per_element`

### 5.1 BFLOAT16 Encoding

BFLOAT16 is not IEEE 754 half-precision. It is the upper 16 bits of an IEEE 754 single-precision float, preserving the full 8-bit exponent range of float32 at the cost of mantissa precision.

**Encode (float32 to bfloat16):**
```
uint32 raw = reinterpret_cast<uint32>(float32_value);
uint16 bf16 = (uint16)((raw >> 16) & 0xFFFF);
```

**Decode (bfloat16 to float32):**
```
uint32 f32_bits = (uint32)bf16 << 16;
float32 value = reinterpret_cast<float>(f32_bits);
```

### 5.2 INT8 Affine Quantisation

INT8 uses affine quantisation with `scale` and `zero_point` parameters stored in the metadata section (see Section 6).

**Dequantisation formula:**
```
float_value = (int8_value - zero_point) * scale
```

**Quantisation (encoding):**
```
vmin = min(vector)
vmax = max(vector)

if vmax == vmin:
    # Constant vector: special case
    if vmin == 0.0:
        scale = 1.0
    else:
        scale = vmin
    zero_point = -1
    all elements encoded as 0
else:
    scale = (vmax - vmin) / 255.0
    zero_point = round(-vmin / scale) - 128
    int8_value = clamp(round(float_value / scale) + zero_point, -128, 127)
```

---

## 6. Metadata Section

The metadata section begins immediately after the 32-byte header and is encoded as a single MessagePack map. The length in bytes is given by the `metadata_length` header field.

Use MessagePack `bin` type for binary data (`use_bin_type=True` in Python msgpack).

### 6.1 Key Mapping Table

| MessagePack Key | Type             | Token Field          | Description                                    |
|-----------------|------------------|----------------------|------------------------------------------------|
| `"t"`           | int              | `knowledge_type`     | KnowledgeType enum value                       |
| `"m"`           | map              | `modality_weights`   | Modality name to weight mapping                |
| `"c"`           | map              | `correlation_map`    | Correlation structure                          |
| `"l"`           | map              | `lineage_dag`        | Lineage directed acyclic graph                 |
| `"g"`           | int              | `generation`         | Generation count (default: 1)                  |
| `"proj"`        | array            | `projection_hints`   | Projection hint list (default: [])             |
| `"pe"`          | float            | `privacy_epsilon`    | Differential privacy epsilon (default: 1.0)    |
| `"pd"`          | float            | `privacy_delta`      | Differential privacy delta (default: 1e-5)     |
| `"d"`           | float            | `drift_score`        | Distribution drift score (default: 0.0)        |
| `"src"`         | str              | `source_model_id`    | Source model identifier (default: "")          |
| `"tgt"`         | str or nil       | `target_model_id`    | Target model identifier (default: nil)         |
| `"ts"`          | str              | `timestamp`          | ISO 8601 timestamp (default: "")               |
| `"v"`           | str              | `version`            | Protocol version string (default: "1.0")       |
| `"x"`           | map              | `custom_metadata`    | Extensible user-defined metadata (default: {}) |

### 6.2 INT8 Quantisation Parameters

When `quant_type == INT8` (3), two additional keys are present in the metadata:

| MessagePack Key | Type  | Description                               |
|-----------------|-------|-------------------------------------------|
| `"_q_scale"`    | float | Affine quantisation scale factor          |
| `"_q_zp"`       | int   | Affine quantisation zero point            |

These keys are consumed during dequantisation and are **not** propagated to the reconstructed token object. Readers must `pop` or remove them from the metadata dict before constructing the token.

### 6.3 Notes

- The UHS vector is **not** stored in the metadata section. It is stored in the vector payload.
- Unknown keys should be preserved or ignored (forward compatibility).
- All string values are UTF-8.

---

## 7. Padding

The vector payload must begin on a 64-byte aligned offset from the start of the file. Padding bytes (value `0x00`) are inserted between the metadata section and the vector payload to achieve this alignment.

**Padding formula:**

```
content_before_payload = 32 + metadata_length
padding = (64 - (content_before_payload % 64)) % 64
```

- If `(32 + metadata_length)` is already a multiple of 64, padding is 0 bytes.
- Maximum padding is 63 bytes.
- All padding bytes must be `0x00`.

**Vector payload offset:** `32 + metadata_length + padding`

---

## 8. Vector Payload

The vector payload contains the quantised UHS (Unified Hyperbolic Signature) vector. It begins at the 64-byte-aligned offset computed in Section 7.

### 8.1 Format by Quantisation Type

| Type    | Element Format                              | Byte Order     |
|---------|---------------------------------------------|----------------|
| FLOAT32 | IEEE 754 single-precision (binary32)        | Little-endian  |
| FLOAT16 | IEEE 754 half-precision (binary16)          | Little-endian  |
| BFLOAT16| Upper 16 bits of float32, stored as uint16  | Little-endian  |
| INT8    | Signed 8-bit integer (-128..127)            | N/A (1 byte)   |

### 8.2 Size Verification

Readers must verify:
```
vector_byte_length == vector_count * bytes_per_element[quant_type]
```

Where `bytes_per_element` is 4, 2, 2, or 1 for FLOAT32, FLOAT16, BFLOAT16, or INT8 respectively.

---

## 9. Trailer (36 bytes)

The trailer is the last 36 bytes of the file and provides integrity and authentication.

| Offset (within trailer) | Size | Type     | Field           | Description                                           |
|--------------------------|------|----------|-----------------|-------------------------------------------------------|
| 0                        | 32   | byte[32] | `hmac_digest`   | HMAC-SHA256 digest, or 32 zero bytes if no HMAC       |
| 32                       | 4    | uint32le | `body_crc`      | CRC-32C of the file body (everything before trailer)  |

### 9.1 File Body Definition

The "file body" for both HMAC and CRC computation is defined as all bytes from the start of the file up to (but not including) the trailer:

```
file_body = header (32 bytes) + metadata + padding + vector_payload
```

The trailer itself is **not** included in either the HMAC or CRC computation.

---

## 10. HMAC Computation

When the `FLAG_HMAC` bit (bit 0) is set in the header flags:

- **Algorithm:** HMAC-SHA256 (RFC 2104)
- **Key:** A shared secret byte string provided out-of-band
- **Message:** The entire file body (header + metadata + padding + vector payload)
- **Digest:** 32 bytes, stored at trailer offset 0

When HMAC is not used (`FLAG_HMAC` is clear), the 32-byte HMAC field in the trailer is filled with `0x00`.

**Verification:** Readers that possess the HMAC key should use a constant-time comparison function (e.g., `hmac.compare_digest` in Python, `crypto/subtle.ConstantTimeCompare` in Go) when comparing the stored and computed digests.

**Behavior matrix:**

| File has HMAC flag | Reader has key | Action                                    |
|--------------------|----------------|-------------------------------------------|
| Yes                | Yes            | Verify HMAC; reject on mismatch           |
| Yes                | No             | Skip HMAC verification (rely on CRC only) |
| No                 | Yes            | No HMAC to verify; proceed                |
| No                 | No             | No HMAC to verify; proceed                |

---

## 11. CRC-32C

TBF uses CRC-32C (Castagnoli) for all checksums.

- **Polynomial:** `0x1EDC6F41` (Castagnoli, iSCSI)
- **Standard:** Described in RFC 3720, widely available in hardware (SSE 4.2).

### 11.1 Where CRC-32C Is Used

1. **Header CRC** (offset 20): Computed over header bytes 0..19.
2. **Body CRC** (trailer offset 32): Computed over the entire file body (header through end of vector payload, exclusive of the 36-byte trailer).

### 11.2 Fallback

The reference Python implementation falls back to standard CRC-32 (ISO 3309, polynomial `0xEDB88320`) if the `crc32c` package is not installed. For production implementations in other languages, always use the Castagnoli polynomial. Libraries:

| Language | Recommended Library                                |
|----------|----------------------------------------------------|
| Rust     | `crc32c` crate                                     |
| Go       | `hash/crc32` with `crc32.MakeTable(crc32.Castagnoli)` |
| C/C++    | Intel ISA-L, or manual SSE 4.2 intrinsic `_mm_crc32_*` |
| JS/Node  | `sse4_crc32` or `crc-32` (with Castagnoli mode)   |

---

## 12. Reading Algorithm

Pseudocode for reading a `.tbf` file:

```
function read_tbf(path, hmac_key=nil, verify_crc=true):
    raw = read_all_bytes(path)

    // Step 1: Minimum size check
    if len(raw) < 32 + 36:
        error("File too small")

    // Step 2: Parse header
    magic           = raw[0..4]
    version_major   = raw[4]
    version_minor   = raw[5]
    flags           = raw[6]
    quant_type      = raw[7]
    metadata_length = read_uint32_le(raw[8..12])
    vector_count    = read_uint32_le(raw[12..16])
    vector_byte_len = read_uint32_le(raw[16..20])
    header_crc      = read_uint32_le(raw[20..24])
    reserved        = raw[24..32]

    // Step 3: Validate magic
    assert magic == "TBF1"

    // Step 4: Version check
    if version_major > 1:
        error("Unsupported version")

    // Step 5: Verify header CRC
    if verify_crc:
        assert crc32c(raw[0..20]) == header_crc

    // Step 6: Compute section offsets
    meta_start = 32
    meta_end   = 32 + metadata_length
    padding    = (64 - ((32 + metadata_length) % 64)) % 64
    vec_start  = meta_end + padding
    vec_end    = vec_start + vector_byte_len
    trailer_start = len(raw) - 36

    // Step 7: Bounds check
    assert vec_end <= trailer_start

    // Step 8: Verify body CRC
    file_body = raw[0..trailer_start]
    body_crc  = read_uint32_le(raw[trailer_start+32..trailer_start+36])
    if verify_crc:
        assert crc32c(file_body) == body_crc

    // Step 9: Verify HMAC (if present and key available)
    has_hmac = (flags & 0x01) != 0
    if has_hmac and hmac_key != nil:
        stored_hmac = raw[trailer_start..trailer_start+32]
        computed_hmac = hmac_sha256(hmac_key, file_body)
        assert constant_time_equal(stored_hmac, computed_hmac)

    // Step 10: Decode metadata
    meta_bytes = raw[meta_start..meta_end]
    meta = msgpack_decode(meta_bytes)

    // Step 11: Extract INT8 quant params (if applicable)
    scale = meta.remove("_q_scale")   // nil if not present
    zero_point = meta.remove("_q_zp") // nil if not present

    // Step 12: Decode vector payload
    vec_bytes = raw[vec_start..vec_start+vector_byte_len]
    vector = dequantise(vec_bytes, vector_count, quant_type, scale, zero_point)

    // Step 13: Reconstruct token from metadata + vector
    return build_token(meta, vector)
```

---

## 13. Writing Algorithm

Pseudocode for writing a `.tbf` file:

```
function write_tbf(path, token, quant_type=FLOAT32, hmac_key=nil):

    // Step 1: Quantise vector
    (payload_bytes, scale, zero_point) = quantise(token.uhs_vector, quant_type)

    // Step 2: Build metadata dict
    meta = token_to_metadata_dict(token)
    if quant_type == INT8:
        meta["_q_scale"] = scale
        meta["_q_zp"]    = zero_point

    // Step 3: Encode metadata as MessagePack
    meta_bytes = msgpack_encode(meta)
    meta_len   = len(meta_bytes)

    // Step 4: Compute padding
    padding_needed = (64 - ((32 + meta_len) % 64)) % 64
    padding = zero_bytes(padding_needed)

    // Step 5: Build header (with CRC placeholder = 0)
    flags = 0
    if hmac_key != nil:
        flags |= 0x01

    header = pack_le(
        "TBF1",              // magic
        1,                   // version_major
        1,                   // version_minor
        flags,               // flags
        quant_type,          // quant_type
        meta_len,            // metadata_length
        len(token.uhs_vector), // vector_count
        len(payload_bytes),  // vector_byte_length
        0x00000000,          // header_crc placeholder
        zero_bytes(8)        // reserved
    )

    // Step 6: Compute and insert header CRC
    header_crc = crc32c(header[0..20])
    header[20..24] = uint32_le(header_crc)

    // Step 7: Assemble file body
    file_body = header + meta_bytes + padding + payload_bytes

    // Step 8: Compute HMAC
    if hmac_key != nil:
        mac = hmac_sha256(hmac_key, file_body)
    else:
        mac = zero_bytes(32)

    // Step 9: Compute body CRC
    body_crc = crc32c(file_body)

    // Step 10: Build trailer
    trailer = mac + uint32_le(body_crc)

    // Step 11: Write
    write_bytes(path, file_body + trailer)
    return len(file_body) + 36
```

---

## 14. Validation Checklist

Implementers should perform these checks when reading a TBF file:

| #  | Check                                                                                     |
|----|-------------------------------------------------------------------------------------------|
| 1  | File is at least 68 bytes (32-byte header + 36-byte trailer minimum)                     |
| 2  | Magic bytes at offset 0 are exactly `0x54 0x42 0x46 0x31` (`"TBF1"`)                    |
| 3  | `version_major` is not greater than the reader's supported major version                  |
| 4  | `quant_type` is a recognised value (0, 1, 2, or 3)                                       |
| 5  | Header CRC-32C (over bytes 0..19) matches the value at offset 20                         |
| 6  | `reserved` bytes (offset 24..31) are all zero                                             |
| 7  | Vector payload end (`32 + metadata_length + padding + vector_byte_length`) does not exceed `file_size - 36` |
| 8  | `vector_byte_length == vector_count * bytes_per_element[quant_type]`                      |
| 9  | Body CRC-32C (over all bytes before trailer) matches the CRC in the trailer               |
| 10 | If `FLAG_HMAC` is set and a key is available, HMAC-SHA256 matches the trailer digest      |

---

## 15. MIME Type and File Extension

| Property       | Value                                    |
|----------------|------------------------------------------|
| MIME Type      | `application/vnd.tessera.token+binary`   |
| File Extension | `.tbf`                                   |

---

## 16. Reference Implementation

The canonical reference implementation is located at:

```
tessera/binary.py
```

This Python module (~568 LOC) provides:

- `TBFSerializer.save()` -- write a TesseraToken to `.tbf`
- `TBFSerializer.load()` -- read a TesseraToken from `.tbf`
- `TBFSerializer.info()` -- read header metadata without full decode
- `TBFSerializer.detect_format()` -- distinguish TBF from legacy formats
- `QuantType` enum -- FLOAT32, FLOAT16, BFLOAT16, INT8
- `_crc32c()` -- CRC-32C with fallback
- `_quantise()` / `_dequantise()` -- vector encoding/decoding

Dependencies: `msgpack`, `numpy`, optionally `crc32c` (google-crc32c).

---

## Appendix A: Header Struct Format String

For languages with Python-style struct packing:

```
"<4sBBBBIIII8s"    // 32 bytes, little-endian
```

Broken down:

| Specifier | Size | Field              |
|-----------|------|--------------------|
| `4s`      | 4    | magic              |
| `B`       | 1    | version_major      |
| `B`       | 1    | version_minor      |
| `B`       | 1    | flags              |
| `B`       | 1    | quant_type         |
| `I`       | 4    | metadata_length    |
| `I`       | 4    | vector_count       |
| `I`       | 4    | vector_byte_length |
| `I`       | 4    | header_crc         |
| `8s`      | 8    | reserved           |

## Appendix B: Rust Type Sketch

```rust
#[repr(C, packed)]
struct TbfHeader {
    magic: [u8; 4],           // b"TBF1"
    version_major: u8,
    version_minor: u8,
    flags: u8,
    quant_type: u8,
    metadata_length: u32,     // little-endian
    vector_count: u32,        // little-endian
    vector_byte_length: u32,  // little-endian
    header_crc: u32,          // little-endian
    reserved: [u8; 8],
}

#[repr(u8)]
enum QuantType {
    Float32  = 0,
    Float16  = 1,
    BFloat16 = 2,
    Int8     = 3,
}

struct TbfTrailer {
    hmac: [u8; 32],
    body_crc: u32,            // little-endian
}
```

## Appendix C: Go Type Sketch

```go
type TBFHeader struct {
    Magic           [4]byte
    VersionMajor    uint8
    VersionMinor    uint8
    Flags           uint8
    QuantType       uint8
    MetadataLength  uint32
    VectorCount     uint32
    VectorByteLen   uint32
    HeaderCRC       uint32
    Reserved        [8]byte
}

// Read with: binary.Read(reader, binary.LittleEndian, &header)
```
