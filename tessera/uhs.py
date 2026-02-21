"""
tessera.uhs — Universal Hub Space encoder/decoder and training.

The UHS is a shared 2048-dimensional latent space. Each anchor model learns
one encoder E_i and one decoder D_i. Transfer between any pair (i, j):

    K_target = D_j( E_i( K_source ) )

This gives O(N) calibrations instead of O(N²) pairwise gates.

Architecture (from Tessera spec):
    Encoder: Linear(d, h) → LayerNorm → GELU → Linear(h, 2048) → L2-norm
    Decoder: Linear(2048, h) → LayerNorm → GELU → Linear(h, d)
    where h = max(d, 2048)

Training loss:
    L = InfoNCE(E_i(A_i), E_j(A_j), temp=0.07) + α × ||D_i(E_i(A_i)) - A_i||²
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .utils import setup_logging

logger = setup_logging("tessera.uhs")

# UHS dimensionality — smallest power-of-two exceeding max intrinsic dim
UHS_DIM = 2048


class EncoderMLP(nn.Module):
    """
    Projects from model-specific d_model space into the Universal Hub Space.

    Output is L2-normalised to the unit hypersphere, enabling cosine-similarity
    comparisons in hub space.
    """

    def __init__(self, d_model: int, hub_dim: int = UHS_DIM):
        super().__init__()
        self.d_model = d_model
        self.hub_dim = hub_dim
        h = max(d_model, hub_dim)

        self.fc1 = nn.Linear(d_model, h)
        self.ln = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, hub_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to hub space.

        Args:
            x: (..., d_model) tensor.

        Returns:
            (..., hub_dim) L2-normalised tensor.
        """
        x = self.fc1(x)
        x = self.ln(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class DecoderMLP(nn.Module):
    """
    Projects from the Universal Hub Space back to model-specific d_model space.

    No normalisation on output — decoder should reconstruct the original
    activation magnitudes.
    """

    def __init__(self, d_model: int, hub_dim: int = UHS_DIM):
        super().__init__()
        self.d_model = d_model
        self.hub_dim = hub_dim
        h = max(d_model, hub_dim)

        self.fc1 = nn.Linear(hub_dim, h)
        self.ln = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode hub-space vectors back to model space.

        Args:
            x: (..., hub_dim) tensor.

        Returns:
            (..., d_model) tensor.
        """
        x = self.fc1(x)
        x = self.ln(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss.

    Pulls semantically equivalent hub vectors together, pushes unrelated
    vectors apart. Positive pairs are aligned by index.

    Args:
        z_i: (B, hub_dim) — encoder output for anchor i.
        z_j: (B, hub_dim) — encoder output for anchor j.
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Scalar loss.
    """
    # Cosine similarity matrix: (B, B)
    logits = (z_i @ z_j.T) / temperature
    labels = torch.arange(len(z_i), device=z_i.device)
    # Symmetric loss
    loss_ij = F.cross_entropy(logits, labels)
    loss_ji = F.cross_entropy(logits.T, labels)
    return (loss_ij + loss_ji) / 2.0


class UniversalHubSpace:
    """
    Manages an encoder/decoder pair for one model dimension and handles
    training via the joint contrastive + reconstruction objective.

    Usage:
        uhs = UniversalHubSpace(d_model=128, device="cpu")
        metrics = uhs.train(dataloader, epochs=10)
        hub_vec = uhs.encode(activations)
        reconstructed = uhs.decode(hub_vec)
    """

    def __init__(self, d_model: int, hub_dim: int = UHS_DIM, device: str = "cpu"):
        self.d_model = d_model
        self.hub_dim = hub_dim
        self.device = device

        self.encoder = EncoderMLP(d_model, hub_dim).to(device)
        self.decoder = DecoderMLP(d_model, hub_dim).to(device)

    def train(
        self,
        dataloader: DataLoader,
        epochs: int = 10,
        lr: float = 1e-4,
        temperature: float = 0.07,
        reconstruction_weight: float = 0.1,
        grad_clip: float = 1.0,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Train encoder/decoder with InfoNCE + reconstruction loss.

        The dataloader should yield tensors of shape (batch, d_model)
        representing activation vectors. Within each batch, we treat
        the batch as positive pairs (self-reconstruction) and use other
        batch elements as negatives for the contrastive loss.

        Args:
            dataloader: Yields (batch, d_model) activation tensors.
            epochs: Training epochs.
            lr: Learning rate.
            temperature: InfoNCE temperature.
            reconstruction_weight: Weight α for reconstruction loss.
            grad_clip: Max gradient norm.
            verbose: Log training progress.

        Returns:
            {"train_loss": [...], "contrastive_loss": [...], "recon_loss": [...]}
        """
        self.encoder.train()
        self.decoder.train()

        all_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimiser = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        history = {"train_loss": [], "contrastive_loss": [], "recon_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_cl = 0.0
            epoch_rl = 0.0
            n_batches = 0

            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(self.device).float()

                # Encode
                z = self.encoder(batch)

                # Contrastive loss (self-pairing within batch)
                # Split batch in half for positive pairs
                half = len(batch) // 2
                if half < 2:
                    continue  # Need at least 2 pairs
                z_i, z_j = z[:half], z[half : 2 * half]
                cl = info_nce_loss(z_i, z_j, temperature)

                # Reconstruction loss
                reconstructed = self.decoder(z)
                rl = F.mse_loss(reconstructed, batch)

                loss = cl + reconstruction_weight * rl

                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, grad_clip)
                optimiser.step()

                epoch_loss += loss.item()
                epoch_cl += cl.item()
                epoch_rl += rl.item()
                n_batches += 1

            scheduler.step()

            if n_batches > 0:
                avg_loss = epoch_loss / n_batches
                avg_cl = epoch_cl / n_batches
                avg_rl = epoch_rl / n_batches
                history["train_loss"].append(avg_loss)
                history["contrastive_loss"].append(avg_cl)
                history["recon_loss"].append(avg_rl)

                if verbose:
                    logger.info(
                        f"  Epoch {epoch+1}/{epochs}: "
                        f"loss={avg_loss:.4f} (contrastive={avg_cl:.4f}, recon={avg_rl:.4f})"
                    )

        self.encoder.eval()
        self.decoder.eval()
        return history

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode from d_model space to hub space."""
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(x.to(self.device))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from hub space to d_model space."""
        self.decoder.eval()
        with torch.no_grad():
            return self.decoder(z.to(self.device))

    def round_trip(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode→decode round trip (for measuring reconstruction fidelity)."""
        return self.decode(self.encode(x))

    def round_trip_error(self, x: torch.Tensor) -> float:
        """Relative Frobenius-norm round-trip error."""
        x = x.to(self.device).float()
        x_hat = self.round_trip(x)
        error = torch.norm(x_hat - x, p="fro") / (torch.norm(x, p="fro") + 1e-8)
        return error.item()

    def state_dict(self) -> dict:
        """Save encoder/decoder weights."""
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "d_model": self.d_model,
            "hub_dim": self.hub_dim,
        }

    def load_state_dict(self, state: dict):
        """Load encoder/decoder weights."""
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
