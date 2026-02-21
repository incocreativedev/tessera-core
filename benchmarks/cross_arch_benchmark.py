#!/usr/bin/env python3
"""
Tessera Phase 4: Cross-Architecture Benchmark Suite

Systematically tests Mode A knowledge transfer across a matrix of
architecture pairs to quantify how well Tessera bridges different
model depths, widths, and structural configurations.

Benchmark Matrix
─────────────────
  Widths tested:     64, 128, 256, 512
  Depths tested:     2, 4, 6, 8
  Architecture types: Transformer, MLP-only, Conv+Pool

Metrics collected per pair:
  ● TX accuracy (validation, post-training)
  ● RX accuracy (pre-transfer baseline)
  ● RX accuracy (post-transfer)
  ● Accuracy delta (post − pre)
  ● Drift (KL-divergence)
  ● TX UHS round-trip error
  ● RX UHS round-trip error
  ● Compatibility score
  ● Wall-clock time

Outputs:
  benchmarks/results/cross_arch_results.json   — raw data
  benchmarks/results/cross_arch_report.html    — visual dashboard

Usage:
    python benchmarks/cross_arch_benchmark.py [--quick]

    --quick   Run a reduced matrix (3 pairs) for smoke-testing
"""

import argparse
import datetime
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# Ensure tessera is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tessera import ModeATransfer, TokenSerializer
from tessera.utils import setup_logging, set_seed, count_parameters

logger = setup_logging("benchmark")


# ═══════════════════════════════════════════════════════════════════════
#  Model Zoo — Three distinct architecture families
# ═══════════════════════════════════════════════════════════════════════


class TransformerModel(nn.Module):
    """Standard transformer encoder with classification head."""

    arch_type = "transformer"

    def __init__(self, d_model: int, num_layers: int, vocab_size: int = 256,
                 num_classes: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(1, d_model // 32),
                dim_feedforward=d_model * 2,
                batch_first=True,
                activation="gelu",
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.head(x)


class MLPModel(nn.Module):
    """Pure feedforward MLP with residual connections."""

    arch_type = "mlp"

    def __init__(self, d_model: int, num_layers: int, vocab_size: int = 256,
                 num_classes: int = 2, seq_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.flatten_proj = nn.Linear(seq_len * d_model, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.LayerNorm(d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
            ))

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = x.view(x.size(0), -1)  # Flatten (batch, seq*d)
        x = self.flatten_proj(x)    # → (batch, d_model)
        for layer in self.layers:
            x = x + layer(x)        # Residual
        return self.head(x)


class ConvModel(nn.Module):
    """1D convolution stack with global average pooling."""

    arch_type = "conv"

    def __init__(self, d_model: int, num_layers: int, vocab_size: int = 256,
                 num_classes: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            ))

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)       # (batch, seq, d)
        x = x.transpose(1, 2)       # (batch, d, seq) for conv1d
        for layer in self.layers:
            x = x + layer(x)        # Residual
        x = x.mean(dim=2)           # Global avg pool → (batch, d)
        return self.head(x)


class LSTMModel(nn.Module):
    """Bidirectional LSTM with mean pooling."""

    arch_type = "lstm"

    def __init__(self, d_model: int, num_layers: int, vocab_size: int = 256,
                 num_classes: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, d_model)

        # LSTM hidden size = d_model // 2 so bidirectional output = d_model
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.layers = nn.ModuleList([self.lstm])  # Expose for fingerprinting
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)       # (batch, seq, d)
        x, _ = self.lstm(x)         # (batch, seq, d) — bidirectional
        x = x.mean(dim=1)           # Mean pool → (batch, d)
        return self.head(x)


# ═══════════════════════════════════════════════════════════════════════
#  Data Factory
# ═══════════════════════════════════════════════════════════════════════


def create_dataset(num_samples: int = 400, seq_len: int = 16, seed: int = 42):
    """Binary classification: low-mean vs high-mean token sequences."""
    rng = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 256, (num_samples, seq_len), generator=rng)
    y = (X.float().mean(dim=1) >= 128).long()
    return TensorDataset(X, y)


# ═══════════════════════════════════════════════════════════════════════
#  Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════


def train_model(model, dataloader, epochs=12, lr=1e-3, device="cpu"):
    """Train model on classification task."""
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            crit(model(bx), by).backward()
            opt.step()

    model.eval()
    return model


def evaluate(model, dataloader, device="cpu") -> float:
    """Return accuracy as a percentage."""
    model.eval().to(device)
    correct = total = 0
    with torch.no_grad():
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            correct += (model(bx).argmax(-1) == by).sum().item()
            total += len(by)
    return correct / total * 100 if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Result
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BenchmarkResult:
    """Metrics collected for a single TX→RX transfer pair."""
    pair_id: str
    tx_arch: str
    tx_d_model: int
    tx_layers: int
    tx_params: int
    rx_arch: str
    rx_d_model: int
    rx_layers: int
    rx_params: int
    tx_accuracy: float
    rx_accuracy_before: float
    rx_accuracy_after: float
    accuracy_delta: float
    drift: float
    tx_round_trip_error: float
    rx_round_trip_error: float
    compatibility_score: float
    elapsed_seconds: float
    status: str = "success"
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
#  Benchmark Runner
# ═══════════════════════════════════════════════════════════════════════


# Architecture constructor lookup
ARCH_FACTORY = {
    "transformer": TransformerModel,
    "mlp": MLPModel,
    "conv": ConvModel,
    "lstm": LSTMModel,
}


def define_model_configs():
    """
    Return a list of model configs to benchmark.

    Each config is (name, arch_type, d_model, num_layers).
    """
    configs = []
    for arch in ["transformer", "mlp", "conv", "lstm"]:
        for d in [64, 128, 256]:
            for L in [2, 4, 6]:
                name = f"{arch}_{d}d_{L}L"
                configs.append((name, arch, d, L))
    return configs


def define_quick_configs():
    """Minimal config set for smoke-testing across all 4 families."""
    return [
        ("transformer_64d_2L",  "transformer", 64,  2),
        ("transformer_128d_4L", "transformer", 128, 4),
        ("mlp_128d_4L",         "mlp",         128, 4),
        ("conv_64d_2L",         "conv",        64,  2),
        ("lstm_64d_2L",         "lstm",        64,  2),
    ]


def build_pair_matrix(configs, same_arch_only=False):
    """
    Build the list of (TX config, RX config) pairs to benchmark.

    Pairs every config against every other config (excluding self-transfer).
    If same_arch_only, only pairs within the same architecture family.
    """
    pairs = []
    for i, tx in enumerate(configs):
        for j, rx in enumerate(configs):
            if i == j:
                continue
            if same_arch_only and tx[1] != rx[1]:
                continue
            pairs.append((tx, rx))
    return pairs


def run_single_transfer(
    tx_config, rx_config, train_loader, val_loader,
    device="cpu", uhs_epochs=8, finetune_epochs=4,
) -> BenchmarkResult:
    """Execute one TX→RX transfer and collect metrics."""
    tx_name, tx_arch, tx_d, tx_L = tx_config
    rx_name, rx_arch, rx_d, rx_L = rx_config
    pair_id = f"{tx_name}→{rx_name}"

    logger.info(f"  ▸ {pair_id}")

    t0 = time.time()

    try:
        # Build models
        tx_model = ARCH_FACTORY[tx_arch](d_model=tx_d, num_layers=tx_L)
        rx_model = ARCH_FACTORY[rx_arch](d_model=rx_d, num_layers=rx_L)

        # Train transmitter
        train_model(tx_model, train_loader, epochs=12, device=device)

        # Evaluate baselines
        tx_acc = evaluate(tx_model, val_loader, device)
        rx_acc_before = evaluate(rx_model, val_loader, device)

        # Execute transfer
        transfer = ModeATransfer(
            transmitter=tx_model,
            receiver=rx_model,
            transmitter_id=tx_name,
            receiver_id=rx_name,
            device=device,
        )

        token = transfer.execute(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            uhs_epochs=uhs_epochs,
            finetune_epochs=finetune_epochs,
            finetune_lr=5e-4,
            privacy_epsilon=8.0,
            privacy_delta=1e-5,
        )

        # Evaluate after transfer
        rx_acc_after = evaluate(rx_model, val_loader, device)

        elapsed = time.time() - t0

        # Extract sub-metrics from token metadata
        meta = token.custom_metadata or {}

        return BenchmarkResult(
            pair_id=pair_id,
            tx_arch=tx_arch,
            tx_d_model=tx_d,
            tx_layers=tx_L,
            tx_params=count_parameters(tx_model),
            rx_arch=rx_arch,
            rx_d_model=rx_d,
            rx_layers=rx_L,
            rx_params=count_parameters(rx_model),
            tx_accuracy=round(tx_acc, 2),
            rx_accuracy_before=round(rx_acc_before, 2),
            rx_accuracy_after=round(rx_acc_after, 2),
            accuracy_delta=round(rx_acc_after - rx_acc_before, 2),
            drift=round(token.drift_score, 4),
            tx_round_trip_error=round(meta.get("tx_round_trip_error", -1), 4),
            rx_round_trip_error=round(meta.get("rx_round_trip_error", -1), 4),
            compatibility_score=round(meta.get("compatibility_score", 0), 4),
            elapsed_seconds=round(elapsed, 2),
        )

    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"    ✘ FAILED: {e}")
        return BenchmarkResult(
            pair_id=pair_id,
            tx_arch=tx_arch, tx_d_model=tx_d, tx_layers=tx_L, tx_params=0,
            rx_arch=rx_arch, rx_d_model=rx_d, rx_layers=rx_L, rx_params=0,
            tx_accuracy=0, rx_accuracy_before=0, rx_accuracy_after=0,
            accuracy_delta=0, drift=-1,
            tx_round_trip_error=-1, rx_round_trip_error=-1,
            compatibility_score=0, elapsed_seconds=round(elapsed, 2),
            status="failed", error=str(e),
        )


# ═══════════════════════════════════════════════════════════════════════
#  HTML Report Generator
# ═══════════════════════════════════════════════════════════════════════


def generate_html_report(results: List[BenchmarkResult], output_path: str):
    """Generate a visual HTML dashboard from benchmark results."""

    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status == "failed"]

    # Compute summary stats
    if successful:
        avg_delta = sum(r.accuracy_delta for r in successful) / len(successful)
        avg_drift = sum(r.drift for r in successful) / len(successful)
        max_delta = max(r.accuracy_delta for r in successful)
        min_drift = min(r.drift for r in successful)
        positive_transfers = sum(1 for r in successful if r.accuracy_delta > 0)
    else:
        avg_delta = avg_drift = max_delta = min_drift = 0
        positive_transfers = 0

    # Cross-arch pairs
    cross_arch = [r for r in successful if r.tx_arch != r.rx_arch]
    same_arch = [r for r in successful if r.tx_arch == r.rx_arch]
    cross_avg_delta = sum(r.accuracy_delta for r in cross_arch) / len(cross_arch) if cross_arch else 0
    same_avg_delta = sum(r.accuracy_delta for r in same_arch) / len(same_arch) if same_arch else 0

    # Width-mismatch pairs
    width_mismatch = [r for r in successful if r.tx_d_model != r.rx_d_model]
    width_match = [r for r in successful if r.tx_d_model == r.rx_d_model]
    wmm_avg = sum(r.accuracy_delta for r in width_mismatch) / len(width_mismatch) if width_mismatch else 0
    wm_avg = sum(r.accuracy_delta for r in width_match) / len(width_match) if width_match else 0

    # Build table rows
    rows_html = ""
    for r in results:
        delta_class = "positive" if r.accuracy_delta > 0 else ("negative" if r.accuracy_delta < 0 else "neutral")
        status_badge = '<span class="badge success">OK</span>' if r.status == "success" else f'<span class="badge fail">FAIL</span>'
        cross_badge = '<span class="badge cross">cross</span>' if r.tx_arch != r.rx_arch else '<span class="badge same">same</span>'

        rows_html += f"""
        <tr>
            <td class="pair-id">{r.pair_id}</td>
            <td>{cross_badge}</td>
            <td>{r.tx_params:,}</td>
            <td>{r.rx_params:,}</td>
            <td>{r.tx_accuracy:.1f}%</td>
            <td>{r.rx_accuracy_before:.1f}%</td>
            <td>{r.rx_accuracy_after:.1f}%</td>
            <td class="{delta_class}">{r.accuracy_delta:+.1f}%</td>
            <td>{r.drift:.2f}</td>
            <td>{r.tx_round_trip_error:.3f}</td>
            <td>{r.rx_round_trip_error:.3f}</td>
            <td>{r.compatibility_score:.3f}</td>
            <td>{r.elapsed_seconds:.1f}s</td>
            <td>{status_badge}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Tessera — Cross-Architecture Benchmark Results</title>
<style>
    :root {{
        --navy: #0B1D3A;
        --steel: #1A5276;
        --teal: #17A589;
        --green: #27AE60;
        --text: #2C3E50;
        --grey: #F2F4F4;
        --light: #D4E6F1;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--grey); color: var(--text); }}
    .header {{
        background: linear-gradient(135deg, var(--navy), var(--steel));
        color: white; padding: 2rem 3rem;
    }}
    .header h1 {{ font-size: 1.8rem; font-weight: 600; margin-bottom: 0.3rem; }}
    .header p {{ opacity: 0.8; font-size: 0.95rem; }}
    .container {{ max-width: 1400px; margin: 0 auto; padding: 1.5rem; }}

    /* Summary cards */
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
    .card {{
        background: white; border-radius: 8px; padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .card .label {{ font-size: 0.75rem; text-transform: uppercase; color: #888; letter-spacing: 0.05em; }}
    .card .value {{ font-size: 1.6rem; font-weight: 700; margin-top: 0.3rem; }}
    .card .value.teal {{ color: var(--teal); }}
    .card .value.green {{ color: var(--green); }}
    .card .value.navy {{ color: var(--navy); }}
    .card .value.steel {{ color: var(--steel); }}

    /* Breakdown section */
    .breakdown {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
    .breakdown-card {{
        background: white; border-radius: 8px; padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    .breakdown-card h3 {{ font-size: 1rem; margin-bottom: 1rem; color: var(--navy); }}
    .bar-group {{ margin-bottom: 0.75rem; }}
    .bar-label {{ font-size: 0.8rem; color: #666; margin-bottom: 0.25rem; display: flex; justify-content: space-between; }}
    .bar-track {{ background: var(--grey); height: 8px; border-radius: 4px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
    .bar-fill.teal {{ background: var(--teal); }}
    .bar-fill.steel {{ background: var(--steel); }}
    .bar-fill.green {{ background: var(--green); }}
    .bar-fill.red {{ background: #E74C3C; }}

    /* Results table */
    .table-wrap {{
        background: white; border-radius: 8px; overflow-x: auto;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th {{
        background: var(--navy); color: white; padding: 0.7rem 0.5rem;
        text-align: left; font-weight: 500; white-space: nowrap;
        position: sticky; top: 0;
    }}
    td {{ padding: 0.55rem 0.5rem; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f8fafc; }}
    .pair-id {{ font-family: monospace; font-size: 0.78rem; white-space: nowrap; }}
    .positive {{ color: var(--green); font-weight: 600; }}
    .negative {{ color: #E74C3C; font-weight: 600; }}
    .neutral {{ color: #888; }}

    .badge {{
        display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px;
        font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
    }}
    .badge.success {{ background: #D5F5E3; color: var(--green); }}
    .badge.fail {{ background: #FADBD8; color: #E74C3C; }}
    .badge.cross {{ background: var(--light); color: var(--steel); }}
    .badge.same {{ background: #D1F2EB; color: var(--teal); }}

    .footer {{ text-align: center; padding: 2rem; font-size: 0.8rem; color: #999; }}

    @media (max-width: 768px) {{
        .breakdown {{ grid-template-columns: 1fr; }}
    }}
</style>
</head>
<body>

<div class="header">
    <h1>Tessera — Cross-Architecture Benchmark</h1>
    <p>Phase 4 · Mode A knowledge transfer across {len(results)} architecture pairs · {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}</p>
</div>

<div class="container">

    <!-- Summary Cards -->
    <div class="cards">
        <div class="card">
            <div class="label">Total Pairs</div>
            <div class="value navy">{len(results)}</div>
        </div>
        <div class="card">
            <div class="label">Successful</div>
            <div class="value green">{len(successful)}</div>
        </div>
        <div class="card">
            <div class="label">Positive Transfers</div>
            <div class="value teal">{positive_transfers} / {len(successful)}</div>
        </div>
        <div class="card">
            <div class="label">Avg Accuracy Δ</div>
            <div class="value {'green' if avg_delta > 0 else 'steel'}">{avg_delta:+.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Best Accuracy Δ</div>
            <div class="value green">{max_delta:+.1f}%</div>
        </div>
        <div class="card">
            <div class="label">Avg Drift</div>
            <div class="value steel">{avg_drift:.2f}</div>
        </div>
        <div class="card">
            <div class="label">Min Drift</div>
            <div class="value teal">{min_drift:.2f}</div>
        </div>
        <div class="card">
            <div class="label">Failed</div>
            <div class="value {'steel' if len(failed) == 0 else ''}" style="{'color:#E74C3C' if len(failed) > 0 else ''}">{len(failed)}</div>
        </div>
    </div>

    <!-- Breakdowns -->
    <div class="breakdown">
        <div class="breakdown-card">
            <h3>Architecture Family Analysis</h3>
            <div class="bar-group">
                <div class="bar-label"><span>Same-family transfer</span><span>{same_avg_delta:+.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill teal" style="width: {max(0, min(100, same_avg_delta + 50))}%"></div></div>
            </div>
            <div class="bar-group">
                <div class="bar-label"><span>Cross-family transfer</span><span>{cross_avg_delta:+.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill steel" style="width: {max(0, min(100, cross_avg_delta + 50))}%"></div></div>
            </div>
            <div class="bar-group">
                <div class="bar-label"><span>Cross-arch pairs tested</span><span>{len(cross_arch)}</span></div>
                <div class="bar-track"><div class="bar-fill green" style="width: {len(cross_arch)/max(1,len(successful))*100}%"></div></div>
            </div>
        </div>
        <div class="breakdown-card">
            <h3>Dimension Mismatch Analysis</h3>
            <div class="bar-group">
                <div class="bar-label"><span>Same width</span><span>{wm_avg:+.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill teal" style="width: {max(0, min(100, wm_avg + 50))}%"></div></div>
            </div>
            <div class="bar-group">
                <div class="bar-label"><span>Different width</span><span>{wmm_avg:+.1f}%</span></div>
                <div class="bar-track"><div class="bar-fill steel" style="width: {max(0, min(100, wmm_avg + 50))}%"></div></div>
            </div>
            <div class="bar-group">
                <div class="bar-label"><span>Width-mismatch pairs</span><span>{len(width_mismatch)}</span></div>
                <div class="bar-track"><div class="bar-fill green" style="width: {len(width_mismatch)/max(1,len(successful))*100}%"></div></div>
            </div>
        </div>
    </div>

    <!-- Full Results Table -->
    <div class="table-wrap">
    <table>
        <thead>
            <tr>
                <th>Pair</th>
                <th>Type</th>
                <th>TX Params</th>
                <th>RX Params</th>
                <th>TX Acc</th>
                <th>RX Before</th>
                <th>RX After</th>
                <th>Δ Acc</th>
                <th>Drift</th>
                <th>TX RT Err</th>
                <th>RX RT Err</th>
                <th>Compat</th>
                <th>Time</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    </div>

    <div class="footer">
        Tessera v1.0 · Phase 4 Cross-Architecture Benchmark · Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>

</div>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    logger.info(f"  HTML report saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Tessera Cross-Architecture Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run reduced matrix")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    args = parser.parse_args()

    set_seed(args.seed)

    print()
    print("=" * 70)
    print("  TESSERA — Phase 4 Cross-Architecture Benchmark")
    print("=" * 70)
    print()

    # Choose config set
    if args.quick:
        configs = define_quick_configs()
        logger.info("Mode: QUICK (smoke-test)")
    else:
        configs = define_model_configs()
        logger.info("Mode: FULL benchmark")

    logger.info(f"Model configs: {len(configs)}")

    # Build pair matrix
    pairs = build_pair_matrix(configs)
    logger.info(f"Transfer pairs: {len(pairs)}")
    logger.info(f"Device: {args.device}")

    # Create datasets
    train_data = create_dataset(num_samples=400, seq_len=16, seed=args.seed)
    val_data = create_dataset(num_samples=150, seq_len=16, seed=args.seed + 1)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    logger.info(f"Dataset: {len(train_data)} train, {len(val_data)} val samples")
    print()

    # Run all transfers
    results: List[BenchmarkResult] = []
    t_total = time.time()

    for idx, (tx_cfg, rx_cfg) in enumerate(pairs):
        logger.info(f"[{idx+1}/{len(pairs)}]")
        result = run_single_transfer(
            tx_cfg, rx_cfg, train_loader, val_loader,
            device=args.device,
            uhs_epochs=5 if args.quick else 8,
            finetune_epochs=3 if args.quick else 4,
        )
        results.append(result)
        logger.info(
            f"    Δ={result.accuracy_delta:+.1f}%  "
            f"drift={result.drift:.2f}  "
            f"time={result.elapsed_seconds:.1f}s"
        )

    total_elapsed = time.time() - t_total

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "cross_arch_results.json")
    with open(json_path, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.datetime.now().isoformat(),
                "seed": args.seed,
                "device": args.device,
                "mode": "quick" if args.quick else "full",
                "num_pairs": len(pairs),
                "total_seconds": round(total_elapsed, 1),
            },
            "results": [asdict(r) for r in results],
        }, f, indent=2)

    logger.info(f"\nJSON results saved to {json_path}")

    # Generate HTML report
    html_path = os.path.join(results_dir, "cross_arch_report.html")
    generate_html_report(results, html_path)

    # Print summary
    successful = [r for r in results if r.status == "success"]
    print()
    print("=" * 70)
    print("  BENCHMARK COMPLETE")
    print()
    print(f"  Pairs tested:       {len(results)}")
    print(f"  Successful:         {len(successful)}")
    print(f"  Failed:             {len(results) - len(successful)}")
    if successful:
        avg_d = sum(r.accuracy_delta for r in successful) / len(successful)
        pos = sum(1 for r in successful if r.accuracy_delta > 0)
        print(f"  Positive transfers: {pos}/{len(successful)} ({pos/len(successful)*100:.0f}%)")
        print(f"  Avg accuracy delta: {avg_d:+.1f}%")
        print(f"  Best accuracy delta: {max(r.accuracy_delta for r in successful):+.1f}%")
    print(f"  Total time:         {total_elapsed:.0f}s")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
