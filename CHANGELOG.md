# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-21

### Added

- **Core protocol** — Mode A (Activation) knowledge transfer between arbitrary neural architectures
- **Universal Hub Space (UHS)** — 2048-dim shared latent space with per-anchor encoder/decoder MLPs, InfoNCE + reconstruction training, O(N) scaling
- **Activation fingerprinting** — Hook-based per-layer statistics (mean, variance, PCA, intrinsic dimensionality)
- **Drift measurement** — KL-divergence fidelity metric between transmitter and receiver activation distributions
- **Differential privacy** — Gaussian mechanism with (epsilon, delta) budget per token
- **Token system** — Self-describing TesseraToken with full metadata, lineage DAG, and dual serialisation:
  - Legacy format: SafeTensors (binary) + JSON (metadata)
  - TBF v1.1: Compact single-file binary format with MessagePack metadata, 64-byte-aligned payload, HMAC-SHA256 trailer
- **Quantisation** — FLOAT32, FLOAT16, BFLOAT16, and INT8 (affine) for TBF vector payloads
- **Projection types** — Orthogonal (fan-out), Conditional, Scaling, Reshape (low-rank SVD), SWAP (bidirectional)
- **Anchor registry** — Local file-based model registry with encoder/decoder persistence
- **Cross-architecture benchmarks** — Transformer, MLP, and Conv model families with HTML report generation
- **End-to-end demo** — 4-layer/128d to 6-layer/256d transfer, runs on CPU in under 60 seconds
