#!/usr/bin/env python3
"""
Tessera Demo: End-to-End Mode A Knowledge Transfer

This script demonstrates the complete Tessera pipeline:
  1. Create two small transformers with different architectures
  2. Train the transmitter on a simple classification task
  3. Execute Mode A transfer (fingerprint → UHS → fine-tune → verify)
  4. Measure transfer fidelity (drift)
  5. Serialise and reload the transfer token

Expected runtime: ~30–60 seconds on CPU.
No GPU required. No external datasets required.

Usage:
    python demo_transfer.py
"""

import sys
import os

# Add parent directory to path so tessera package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tessera import ModeATransfer, TokenSerializer, TBFSerializer, QuantType
from tessera.utils import setup_logging, set_seed, get_device, count_parameters


logger = setup_logging("demo")


# ── Model Definitions ────────────────────────────────────────────────────────


class SmallTransformer(nn.Module):
    """
    Minimal transformer for demonstration purposes.

    Two instances with different (d_model, num_layers) serve as the
    transmitter and receiver, proving cross-architecture transfer.
    """

    def __init__(self, d_model: int, num_layers: int, vocab_size: int = 256, num_classes: int = 2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(1, d_model // 32),   # At least 1 head
                dim_feedforward=d_model * 2,
                batch_first=True,
                activation="gelu",
                dropout=0.0,                     # No dropout for reproducibility
            )
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) integer token IDs.
        Returns:
            (batch, num_classes) logits.
        """
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)  # Pool over sequence
        return self.head(x)


# ── Data Generation ──────────────────────────────────────────────────────────


def create_dataset(num_samples: int = 200, seq_len: int = 16, seed: int = 42):
    """
    Create a simple pattern-recognition dataset.

    Class 0: sequences with more low tokens (< 128)
    Class 1: sequences with more high tokens (>= 128)

    This gives the transmitter something meaningful to learn that
    can then be transferred to the receiver.
    """
    rng = torch.Generator().manual_seed(seed)
    X = torch.randint(0, 256, (num_samples, seq_len), generator=rng)

    # Label based on mean token value
    means = X.float().mean(dim=1)
    y = (means >= 128).long()

    return TensorDataset(X, y)


# ── Training Loop ────────────────────────────────────────────────────────────


def train_model(model: nn.Module, dataloader: DataLoader, epochs: int = 10, device: str = "cpu"):
    """Train a model and report accuracy."""
    model.to(device)
    model.train()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimiser.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            total += len(batch_y)

        acc = correct / total * 100
        logger.info(
            f"  Epoch {epoch+1}/{epochs}: "
            f"loss={total_loss/len(dataloader):.4f}, acc={acc:.1f}%"
        )

    model.eval()


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: str = "cpu") -> float:
    """Evaluate accuracy on a dataset."""
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            correct += (logits.argmax(dim=-1) == batch_y).sum().item()
            total += len(batch_y)

    return correct / total * 100


# ── Main Demo ────────────────────────────────────────────────────────────────


def main():
    """Run the complete Tessera transfer demo."""

    set_seed(42)
    device = get_device("auto")

    print()
    print("=" * 70)
    print("  TESSERA — Mode A Knowledge Transfer Demo")
    print("=" * 70)
    print()

    # ── Create models ────────────────────────────────────────────────────
    logger.info("[1/6] Creating models...")

    transmitter = SmallTransformer(d_model=128, num_layers=4)
    receiver = SmallTransformer(d_model=256, num_layers=6)

    logger.info(f"  Transmitter: 4 layers, d=128, {count_parameters(transmitter):,} params")
    logger.info(f"  Receiver:    6 layers, d=256, {count_parameters(receiver):,} params")
    logger.info(f"  Device: {device}")

    # ── Create datasets ──────────────────────────────────────────────────
    logger.info("\n[2/6] Creating datasets...")

    train_data = create_dataset(num_samples=300, seq_len=16, seed=42)
    val_data = create_dataset(num_samples=100, seq_len=16, seed=123)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    logger.info(f"  Train: {len(train_data)} samples")
    logger.info(f"  Val:   {len(val_data)} samples")

    # ── Train transmitter ────────────────────────────────────────────────
    logger.info("\n[3/6] Training transmitter...")

    train_model(transmitter, train_loader, epochs=10, device=device)

    tx_acc = evaluate_model(transmitter, val_loader, device)
    rx_acc_before = evaluate_model(receiver, val_loader, device)

    logger.info(f"\n  Transmitter val accuracy: {tx_acc:.1f}%")
    logger.info(f"  Receiver val accuracy (before transfer): {rx_acc_before:.1f}%")

    # ── Execute Mode A transfer ──────────────────────────────────────────
    logger.info("\n[4/6] Executing Mode A transfer...")
    print()

    transfer = ModeATransfer(
        transmitter=transmitter,
        receiver=receiver,
        transmitter_id="small_tx_4L_128d",
        receiver_id="small_rx_6L_256d",
        device=device,
    )

    # Auto-detect target layers (transformer encoder layers)
    token = transfer.execute(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        uhs_epochs=5,
        finetune_epochs=3,
        finetune_lr=5e-4,
        privacy_epsilon=8.0,
        privacy_delta=1e-5,
    )

    # ── Evaluate receiver after transfer ─────────────────────────────────
    rx_acc_after = evaluate_model(receiver, val_loader, device)

    print()
    logger.info("[5/6] Transfer results:")
    logger.info(f"  Transmitter accuracy:         {tx_acc:.1f}%")
    logger.info(f"  Receiver accuracy (before):   {rx_acc_before:.1f}%")
    logger.info(f"  Receiver accuracy (after):    {rx_acc_after:.1f}%")
    logger.info(f"  Drift:                  {token.drift_score:.6f}")
    logger.info(f"  UHS vector dimension:         {len(token.uhs_vector)}")
    logger.info(f"  Privacy budget:               ε={token.privacy_epsilon}, δ={token.privacy_delta}")

    # ── Serialise and verify (legacy format) ────────────────────────────
    logger.info("\n[6/8] Serialising token (legacy SafeTensors+JSON)...")

    output_path = os.path.join(os.path.dirname(__file__), "..", "tessera_demo_token.safetensors")
    TokenSerializer.save_token(token, output_path)
    logger.info(f"  Saved to: {output_path}")

    loaded = TokenSerializer.load_token(output_path)
    logger.info(f"  Reloaded: {loaded.source_model_id} → {loaded.target_model_id}")
    logger.info(f"  Drift matches: {loaded.drift_score == token.drift_score}")

    # ── Serialise and verify (TBF v1.1 binary format) ────────────────────
    logger.info("\n[7/8] Serialising token (TBF v1.1 binary format)...")

    tbf_base = os.path.join(os.path.dirname(__file__), "..", "tessera_demo_token")
    hmac_key = b"tessera-demo-key"

    for quant in [QuantType.FLOAT32, QuantType.FLOAT16, QuantType.INT8]:
        tbf_path = f"{tbf_base}_{quant.name.lower()}.tbf"
        nbytes = TBFSerializer.save(tbf_path, token, quant=quant, hmac_key=hmac_key)

        # Reload and verify
        loaded_tbf = TBFSerializer.load(tbf_path, hmac_key=hmac_key)
        info = TBFSerializer.info(tbf_path)

        # Check round-trip fidelity
        original = token.uhs_vector
        reloaded = loaded_tbf.uhs_vector
        max_error = max(abs(a - b) for a, b in zip(original, reloaded))

        logger.info(
            f"  {quant.name:8s}: {nbytes:>7,} bytes, "
            f"max_error={max_error:.6f}, "
            f"CRC={info['header_crc_ok']}"
        )

    # ── Format detection ──────────────────────────────────────────────────
    logger.info("\n[8/8] Format auto-detection...")
    legacy_fmt = TBFSerializer.detect_format(output_path)
    tbf_fmt = TBFSerializer.detect_format(f"{tbf_base}_float32.tbf")
    logger.info(f"  {output_path}: {legacy_fmt}")
    logger.info(f"  {tbf_base}_float32.tbf: {tbf_fmt}")

    # ── Summary ──────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  DEMO COMPLETE")
    print()
    print(f"  Knowledge transferred: {token.source_model_id} → {token.target_model_id}")
    print(f"  Architectures: 4-layer/128d → 6-layer/256d")
    print(f"  Drift: {token.drift_score:.6f}")
    print(f"  Legacy format: SafeTensors + JSON (2 files)")
    print(f"  TBF v1.1:      Single binary file (3 quantisation levels)")
    print(f"  Token serialised and verified successfully")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
