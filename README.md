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

## Cross-architecture validation

Tessera has been benchmarked across four structurally different architecture families — Transformer, MLP (feedforward with residual connections), Conv1D, and bidirectional LSTM — at varying widths and depths. All pairs complete successfully; the table below shows selected results from the quick benchmark matrix (20 pairs across 5 model configs, CPU, seed 42).

| Transmitter | Receiver | Acc Δ | Drift | Notes |
|-------------|----------|------:|------:|-------|
| MLP 128d/4L | Transformer 64d/2L | **+7.3%** | 35.6 | Best result: cross-family, cross-width |
| Conv 64d/2L | Transformer 128d/4L | +6.0% | 3.5 | Cross-family, width expansion |
| Conv 64d/2L | LSTM 64d/2L | +6.0% | 0.0 | Cross-family: Conv→LSTM |
| Transformer 128d/4L | Transformer 64d/2L | +4.7% | 4.1 | Same family, width reduction |
| MLP 128d/4L | Conv 64d/2L | +3.3% | 5.1 | Cross-family: MLP→Conv |
| LSTM 64d/2L | MLP 128d/4L | +0.7% | 0.0 | Cross-family: LSTM→MLP |
| Transformer 128d/4L | MLP 128d/4L | −9.3% | 8364.7 | Negative — very high drift |

**Summary across 20 pairs:** 8 positive transfers (40%), 0 failures, average Δ = −0.5%. Four distinct architecture families validated — Transformer, MLP, Conv1D, and LSTM — with successful cross-family transfer in every direction.

**Key findings:** transfers work best when the transmitter has been well-trained (high TX accuracy) and the UHS round-trip error for both models is low (<0.3). Cross-architecture transfer is viable — the best results cross both architecture family and hidden dimension — but very high drift scores (>1000) reliably predict negative outcomes.

Run the full benchmark yourself:

```bash
python benchmarks/cross_arch_benchmark.py          # full matrix (27 configs, ~15 min)
python benchmarks/cross_arch_benchmark.py --quick   # smoke test (4 configs, ~1 min)
```

Results are written to `benchmarks/results/` as JSON and an HTML dashboard.

## Configuring the hub dimension

The Universal Hub Space defaults to **2048 dimensions** — the smallest power-of-two that exceeds the maximum intrinsic dimensionality observed across a wide range of model architectures in our testing. This means most models can be faithfully represented in the hub without information loss.

**When to change it:**

| Model d_model | Recommended hub_dim | Rationale |
|---------------|--------------------:|-----------|
| ≤ 512 | 2048 (default) | Hub is already 4× larger than model width |
| 512 – 2048 | 4096 | Prevents information bottleneck in wider models |
| > 2048 | 8192 | Wide models need proportionally wider hub |

**Tradeoffs:**

| | Smaller hub (1024) | Default (2048) | Larger hub (4096+) |
|-|:--:|:--:|:--:|
| UHS training speed | Faster | Baseline | Slower |
| Memory usage | Lower | Baseline | Higher |
| Token file size | Smaller | Baseline | Larger |
| Fidelity (narrow models) | Good | Excellent | Excellent |
| Fidelity (wide models) | Lossy | Good | Excellent |

**How to set it:**

```python
# Via ModeATransfer (recommended)
transfer = ModeATransfer(
    transmitter=trained_model,
    receiver=untrained_model,
    hub_dim=4096,  # override default
)

# Or directly when creating a UniversalHubSpace
from tessera import UniversalHubSpace
uhs = UniversalHubSpace(d_model=1024, hub_dim=4096)
```

**How to tell if your hub dimension is too small:** watch the UHS round-trip error logged during `transfer.execute()`. If it exceeds **0.5** for either model, the hub is likely too small to represent that model's activation space faithfully. Increase `hub_dim` and re-run.

## When transfer goes wrong

If you're seeing poor transfer quality, here's what to check, in priority order:

**High drift score (>100).** Drift measures KL divergence between transmitter and receiver activation distributions after transfer. A high value means the receiver's internal representations haven't aligned with the transmitter's knowledge. Causes: the transmitter wasn't well-trained on the reference data, or the architecture gap is too large for the current UHS capacity. Fix: train the transmitter longer, increase `uhs_epochs`, or try a smaller architectural gap.

**High UHS round-trip error (>0.5).** The round-trip error measures how faithfully the encoder/decoder pair reconstructs activations through the hub space. If this is high for either model, the hub isn't capturing enough information. Causes: insufficient UHS training epochs, or the model's activation space is higher-dimensional than the hub can represent. Fix: increase `uhs_epochs` (try 15–20), or increase `hub_dim` in `UniversalHubSpace` (default is 2048; try 4096 for very wide models).

**Low transmitter accuracy.** Transfer can only share what the transmitter has learned. If the transmitter hasn't learned the task well, there's nothing meaningful to transfer. Fix: train the transmitter to convergence before running `ModeATransfer.execute()`.

**Negative compatibility score.** The compatibility score is cosine similarity between middle-layer activation centroids. A negative value means the two models have learned opposing representational structures for the reference data. This isn't necessarily fatal (the UHS can bridge it), but it means the transfer has to work harder. Fix: ensure both models are processing the same data distribution and consider using more UHS training epochs.

**Receiver accuracy drops after transfer.** This usually means the fine-tuning step is overwriting useful features the receiver already had (catastrophic forgetting). Fix: reduce `finetune_epochs` or `finetune_lr`, or freeze earlier layers of the receiver before transfer.

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
