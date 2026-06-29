# Copyright (c) 2026 Davis Liang. MIT License.
"""Backend selection."""

from __future__ import annotations

import importlib.util
from typing import Any

from .base import Backend, GenerationParams, select_default_layers, softmax_np

__all__ = ["Backend", "GenerationParams", "select_default_layers", "softmax_np", "get_backend"]


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def auto_backend_name() -> str:
    """Prefer MLX on Apple Silicon (when installed), else torch."""
    if _have("mlx") and _have("mlx_lm"):
        return "mlx"
    if _have("torch"):
        return "torch"
    raise RuntimeError(
        "No inference backend available. Install one:\n"
        "  pip install 'backdoor-scanner[mlx]'   # Apple Silicon\n"
        "  pip install 'backdoor-scanner[cuda]'  # NVIDIA / CPU"
    )


def get_backend(
    name: str,
    model_id: str,
    *,
    dtype: str = "bfloat16",
    device: str = "auto",
    mlx_model: str | None = None,
) -> Backend:
    """Construct a backend by name (``auto`` | ``mlx`` | ``torch``)."""
    if name == "auto":
        name = auto_backend_name()

    if name == "mlx":
        if not (_have("mlx") and _have("mlx_lm")):
            raise RuntimeError("MLX backend requested but mlx / mlx-lm are not installed.")
        from .mlx_backend import MLXBackend

        return MLXBackend(mlx_model or model_id)

    if name == "torch":
        if not _have("torch"):
            raise RuntimeError("torch backend requested but torch is not installed.")
        from .torch_backend import TorchBackend

        return TorchBackend(model_id, dtype=dtype, device=device)

    raise ValueError(f"Unknown backend: {name!r} (expected auto|mlx|torch)")
