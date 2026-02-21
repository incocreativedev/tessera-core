# tessera-core

**An open protocol for AI-to-AI knowledge transfer.**

[![CI](https://github.com/incocreative/tessera-core/actions/workflows/ci.yml/badge.svg)](https://github.com/incocreative/tessera-core/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Tessera enables trained neural networks to transfer what they've learned to untrained models across completely different architectures. It works by encoding knowledge as activation-level representations, routing transfers through a Universal Hub Space that scales linearly, and verifying fidelity with formal drift metrics.

Think of it as the missing "knowledge layer" in the AI stack — MCP connects models to tools, A2A coordinates agents, and Tessera lets them teach each other.

## Installation

```bash
pip install tessera-core
```

Or install from source for development:

```bash
git clone https://github.com/incocreative/tessera-core.git
cd tessera-core
pip install -e ".[dev]"
```

Requires Python 3.9+ and PyTorch 2.0+.

## Quick start

```python
from tessera import ModeATransfer, TBFSerializer, QuantType

# Your trained model and an untrained target
transfer = ModeATransfer(
    transmitter=trained_model,
    receiver=untrained_model,
    transmitter_id="model_a",
    receiver_id="model_b",
)

# Execute the transfer
token = transfer.execute(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    privacy_epsilon=1.0,
)

# Save as compact binary (TBF v1.1)
TBFSerializer.save("transfer.tbf", token, quant=QuantType.FLOAT16)

# Load it back
loaded = TBFSerializer.load("transfer.tbf")
```

## Running the demo

```bash
cd examples
python demo_transfer.py
```

Demonstrates end-to-end transfer between a 4-layer (128d) and 6-layer (256d) transformer. Runs on CPU in under 60 seconds.

## How it works

Tessera transfers *how a model behaves* (activation patterns), not *what it stores* (weights). This makes transfers architecture-agnostic.

1. **Fingerprint** — collect per-layer activation statistics from the transmitter
2. **Train UHS** — learn encoder/decoder pairs for both architectures into a shared 2048-dim hub space
3. **Encode** — project transmitter activations into the hub space
4. **Decode** — reconstruct in the receiver's representation space
5. **Fine-tune** — align receiver activations with decoded targets
6. **Measure** — compute KL-divergence drift score
7. **Package** — create a self-describing TesseraToken with metadata, lineage, and privacy guarantees

Because every model only needs one encoder and one decoder (calibrated against the hub), adding a new architecture costs O(1) rather than O(N) pairwise mappings.

## Architecture

| Module | Purpose |
|--------|---------|
| `fingerprint.py` | Hook-based activation statistics (mean, variance, PCA, intrinsic dimensionality) |
| `uhs.py` | Universal Hub Space encoder/decoder MLPs with InfoNCE + reconstruction training |
| `token.py` | Self-describing TesseraToken dataclass and SafeTensors serialisation |
| `binary.py` | TBF v1.1 compact binary format with MessagePack, quantisation, and HMAC |
| `transfer.py` | Mode A orchestrator: fingerprint → UHS → fine-tune → verify |
| `drift.py` | KL-divergence fidelity measurement between activation distributions |
| `privacy.py` | Gaussian differential privacy mechanism with (ε, δ) budget |
| `gates.py` | Projection types (Orthogonal, Conditional, Scaling, Reshape, SWAP) |
| `registry.py` | Local file-based anchor model registry |

## Token formats

Tessera supports two serialisation formats:

| Format | Files | Size (2048-dim) | Use case |
|--------|-------|-----------------|----------|
| Legacy (SafeTensors + JSON) | 2 | ~60 KB | Human-readable debugging |
| TBF v1.1 FLOAT32 | 1 | ~8.7 KB | Lossless production |
| TBF v1.1 FLOAT16 | 1 | ~4.6 KB | Standard production |
| TBF v1.1 INT8 | 1 | ~2.5 KB | Bandwidth-constrained |

TBF files include CRC-32C integrity checks and optional HMAC-SHA256 authentication.

## Companion documents

- **Tessera Specification v1.0** — the authoritative protocol specification
- **Tessera v1.0 Formal Grammar (EBNF)** — machine-parseable token stream syntax
- **TSRD & Anchor Characterisation v1.0** — dataset curation and model profiling procedures
- **MCP & A2A Integration v1.0** — integration with Model Context Protocol and Agent-to-Agent protocol
- **Privacy & Security Audit v1.0** — threat model, differential privacy framework, regulatory compliance
- **Binary Encoding v1.1** — TBF wire format specification

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Licence

Apache License 2.0 — see [LICENSE](LICENSE).

---

Built by [Inco Creative](https://incocreative.com).
