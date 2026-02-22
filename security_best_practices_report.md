# Tessera Security Best Practices Report

Date: 2026-02-22  
Scope: `/Users/kirkmaddocks/tessera-core/outputs/tessera-core-lang`  
Method: Manual code review + targeted proof-of-concept checks

## Executive Summary

The assessment found **4 security issues**:
- 1 Critical
- 1 High
- 2 Medium

The most serious issue is a filesystem path traversal in anchor registration that allows writes outside the registry root. I also confirmed an authenticity-check bypass path in token validation when HMAC keys are omitted, even for files marked as HMAC-protected.

## Critical Findings

### CRIT-001: Arbitrary filesystem write via unsanitized `anchor_id`
- Severity: Critical
- Impact: An attacker controlling `anchor_id` can write model artifacts outside the intended registry directory.
- Evidence:
  - `tessera/registry.py:88` joins `self.anchors_dir / anchor_id` directly.
  - `tessera/registry.py:89` creates the resulting path.
  - `tessera/registry.py:92` and `tessera/registry.py:93` write `encoder.pt` and `decoder.pt`.
  - `tessera/registry.py:102` writes `config.json`.
- Why this is exploitable:
  - `Path` join accepts absolute paths and traversal segments.
  - A malicious value like `/tmp/escaped-anchor` or `../../outside` escapes `self.anchors_dir`.
- Proof of concept result:
  - Using `AnchorRegistry.register(anchor_id=<absolute path>, ...)` created files outside registry root (`outside_encoder_exists=True`, `outside_decoder_exists=True`).
- Recommendation:
  - Validate `anchor_id` against a strict allowlist (e.g., `^[a-zA-Z0-9._-]+$`).
  - Resolve candidate paths and enforce containment under `self.anchors_dir` before writing.
  - Reject absolute paths and any path containing separators.

## High Findings

### HIGH-001: Registry index path trust allows arbitrary file loading location
- Severity: High
- Impact: If `registry.json` is tampered, the loader reads model files from attacker-chosen locations.
- Evidence:
  - `tessera/registry.py:137` reads `info = self._index["anchors"][anchor_id]`.
  - `tessera/registry.py:138` uses `Path(info["path"])` without containment checks.
  - `tessera/registry.py:146` and `tessera/registry.py:149` load weights from that path.
- Risk notes:
  - `weights_only=True` lowers deserialization risk, but arbitrary path control still enables loading unexpected files and denial-of-service conditions.
- Recommendation:
  - Do not trust persisted absolute/relative paths from index.
  - Derive anchor directory from validated `anchor_id`, or enforce `resolve()` containment under `self.anchors_dir`.
  - Consider validating file ownership/permissions on registry files in multi-user environments.

## Medium Findings

### MED-001: HMAC authenticity is optional even when file declares HMAC
- Severity: Medium
- Impact: Tampered “signed” tokens can be accepted if the caller forgets to provide an HMAC key.
- Evidence:
  - `tessera/binary.py:491` verifies HMAC only if `has_hmac and hmac_key is not None`.
  - `tessera/cli.py:101` calls `TBFSerializer.load(...)` with possibly `hmac_key=None`.
  - `tessera/mcp_server.py:180` does the same for MCP validation.
- Proof of concept result:
  - A modified signed token with recomputed CRC loaded successfully with `hmac_key=None`.
  - The same file failed with `hmac_key=<correct key>` (`HMAC verification failed`).
- Recommendation:
  - Default to fail-closed: if `has_hmac` is set and no key is provided, raise an error.
  - Add an explicit override flag (e.g., `allow_unsigned_verification=False`) for backward compatibility.
  - Make CLI/MCP `validate` clearly differentiate integrity-only vs authenticity-verified checks.

### MED-002: Unbounded file and metadata parsing enables memory-based DoS
- Severity: Medium
- Impact: Large attacker-controlled token files can consume excessive memory/CPU.
- Evidence:
  - `tessera/binary.py:411` reads entire file into memory in `load`.
  - `tessera/binary.py:550` reads entire file in `info`.
  - `tessera/binary.py:498` unpacks metadata with no explicit size/depth guard.
- Recommendation:
  - Parse header first with bounded reads.
  - Enforce hard limits (max file size, max metadata length, max vector count).
  - Consider streaming or memory-mapped parsing for large payloads.

## Assumptions and Context

- This review assumes untrusted inputs may reach:
  - `anchor_id` during registration
  - `registry.json` contents
  - `.tbf` files provided via CLI/MCP tooling
- If usage is strictly single-user/local and all inputs are trusted, exploitability decreases but hardening is still recommended.

## Validation Notes

- Automated security linters (`ruff --select S`, `bandit`) were unavailable in this environment.
- Findings were validated via code inspection and targeted runtime PoCs.
