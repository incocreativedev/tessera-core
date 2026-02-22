"""
tessera.api — HTTP REST API for Tessera (Railway / cloud deployment).

Exposes Tessera as a stateless HTTP service so external clients can:
  - Check liveness / version
  - Run a demo Mode A transfer between two randomly-initialised small models
  - List registered anchors
  - Inspect / validate .tbf token files uploaded as bytes

Start locally:
    uvicorn tessera.api:app --host 0.0.0.0 --port 8000 --reload

On Railway:
    Start command: uvicorn tessera.api:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import platform
import tempfile
from typing import Optional

try:
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "FastAPI is required to run the Tessera API server.\n"
        "Install with: pip install 'tessera-core[railway]'"
    ) from exc

import torch
import torch.nn as nn

import tessera
from tessera.binary import TBFSerializer
from tessera.registry import AnchorRegistry
from tessera.transfer import ModeATransfer
from tessera.uhs import UHS_DIM

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Tessera API",
    description=(
        "Open protocol for AI-to-AI knowledge transfer. "
        "Run Mode A transfers, inspect tokens, and manage the anchor registry."
    ),
    version=tessera.__version__,
    docs_url="/",  # Swagger UI at root
)

# ── Tiny demo model ───────────────────────────────────────────────────────────


class _TinyMLP(nn.Module):
    """Minimal 2-layer MLP used for demo transfers (no real training data needed)."""

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        # expose a hook-friendly attribute so ModeATransfer can see activations
        self.layers = nn.ModuleList([self.net[0], self.net[2]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_demo_loader(d_model: int = 64, n: int = 128, batch: int = 32):
    """Synthetic dataloader for demo transfers."""
    X = torch.randn(n, d_model)
    ds = torch.utils.data.TensorDataset(X)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health():
    """Liveness probe — Railway health check."""
    return {"status": "ok", "version": tessera.__version__}


@app.get("/info", tags=["System"])
def info():
    """Version, runtime, and hardware information."""
    torch_version = "unknown"
    cuda = False
    try:
        torch_version = torch.__version__
        cuda = torch.cuda.is_available()
    except Exception:
        pass

    return {
        "tessera_version": tessera.__version__,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": cuda,
        "default_hub_dim": UHS_DIM,
        "platform": platform.system(),
    }


# ── Transfer ──────────────────────────────────────────────────────────────────


class TransferRequest(BaseModel):
    tx_d_model: int = 64
    rx_d_model: int = 64
    tx_id: str = "demo_tx"
    rx_id: str = "demo_rx"
    uhs_epochs: int = 3
    finetune_epochs: int = 1

    model_config = {
        "json_schema_extra": {
            "example": {
                "tx_d_model": 64,
                "rx_d_model": 64,
                "tx_id": "demo_tx",
                "rx_id": "demo_rx",
                "uhs_epochs": 3,
                "finetune_epochs": 1,
            }
        }
    }


@app.post("/transfer", tags=["Transfer"])
def run_transfer(req: TransferRequest):
    """
    Run a demo Mode A (activation) transfer between two randomly-initialised
    tiny MLP models and return the resulting TesseraToken metadata.

    This uses synthetic data — no real models or datasets are needed.
    For production use, call ModeATransfer directly from Python.
    """
    if req.uhs_epochs > 10 or req.finetune_epochs > 5:
        raise HTTPException(
            status_code=400,
            detail="Demo limited to uhs_epochs ≤ 10 and finetune_epochs ≤ 5.",
        )
    if not (8 <= req.tx_d_model <= 256 and 8 <= req.rx_d_model <= 256):
        raise HTTPException(
            status_code=400,
            detail="d_model must be between 8 and 256 for the demo.",
        )

    tx = _TinyMLP(req.tx_d_model)
    rx = _TinyMLP(req.rx_d_model)
    loader = _make_demo_loader(req.tx_d_model)

    transfer = ModeATransfer(tx, rx, req.tx_id, req.rx_id)
    token = transfer.execute(
        loader,
        loader,
        uhs_epochs=req.uhs_epochs,
        finetune_epochs=req.finetune_epochs,
    )

    return {
        "knowledge_type": token.knowledge_type.value,
        "source_model_id": token.source_model_id,
        "target_model_id": token.target_model_id,
        "vector_dim": len(token.uhs_vector),
        "modality_weights": token.modality_weights,
        "drift_score": token.drift_score,
        "privacy_epsilon": token.privacy_epsilon,
        "privacy_delta": token.privacy_delta,
        "generation": token.generation,
        "version": token.version,
        "timestamp": token.timestamp,
    }


# ── Anchors ───────────────────────────────────────────────────────────────────


@app.get("/anchors", tags=["Registry"])
def list_anchors(
    registry_dir: Optional[str] = Query(None, description="Registry path (default: ~/.tessera/)")
):
    """List all registered anchor models."""
    try:
        registry = AnchorRegistry(registry_dir)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    result = []
    for anchor_id in registry.list():
        result.append({"anchor_id": anchor_id, **registry.info(anchor_id)})
    return result


# ── Token inspection ──────────────────────────────────────────────────────────


@app.post("/token/inspect", tags=["Token"])
async def inspect_token(
    file: UploadFile = File(..., description=".tbf token file to inspect"),
    full: bool = Query(False, description="Return full token metadata (slower)"),
):
    """
    Upload a .tbf token file and return its header or full metadata.
    Maximum file size is enforced by the TBFSerializer (256 MB).
    """
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".tbf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        if not full:
            result = TBFSerializer.info(tmp_path)
        else:
            token = TBFSerializer.load(tmp_path)
            result = {
                "knowledge_type": token.knowledge_type.value,
                "source_model_id": token.source_model_id,
                "target_model_id": token.target_model_id,
                "vector_dim": len(token.uhs_vector),
                "modality_weights": token.modality_weights,
                "drift_score": token.drift_score,
                "generation": token.generation,
                "timestamp": token.timestamp,
                "custom_metadata": token.custom_metadata,
            }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        os.unlink(tmp_path)

    return result


@app.post("/token/validate", tags=["Token"])
async def validate_token(
    file: UploadFile = File(..., description=".tbf token file to validate"),
    hmac_key: Optional[str] = Query(
        None, description="Hex-encoded HMAC key (e.g. deadbeef01020304)"
    ),
):
    """
    Upload a .tbf token file and validate its CRC integrity.
    Optionally verify HMAC authenticity if hmac_key is provided.
    """
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".tbf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    key_bytes = None
    if hmac_key:
        try:
            key_bytes = bytes.fromhex(hmac_key)
        except ValueError:
            raise HTTPException(status_code=400, detail="hmac_key must be a valid hex string.")

    try:
        token = TBFSerializer.load(tmp_path, hmac_key=key_bytes, verify_crc=True)
        result = {
            "valid": True,
            "source_model_id": token.source_model_id,
            "target_model_id": token.target_model_id,
            "vector_dim": len(token.uhs_vector),
        }
    except ValueError as exc:
        result = {"valid": False, "error": str(exc)}
    except Exception as exc:
        result = {"valid": False, "error": str(exc)}
    finally:
        os.unlink(tmp_path)

    return result
