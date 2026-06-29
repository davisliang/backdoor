# Copyright (c) 2026 Davis Liang. MIT License.
"""Backend abstraction shared by the MLX and PyTorch/CUDA implementations.

Every signal the scanner needs reduces to one of two forward-pass primitives:

* :meth:`Backend.next_token_logits` — last-position logits ``(B, V)``. Powers the
  entropy-collapse and behavioural-divergence losses via short greedy rollouts.
* :meth:`Backend.attentions` — per-layer attention ``(B, H, S, S)`` for a chosen
  set of layers. Powers the attention-hijacking ("double triangle") loss.

Plus :meth:`Backend.generate` for the leakage decoding sweep and behavioural eval.

All array I/O is plain ``numpy`` so the loss/scoring code is backend-agnostic.

Design note — **no padding**: the scanner always groups sequences so that every
row in a batch has the *same* length (all candidate triggers of a fixed token
length, spliced into one fixed prompt, yield identical sequence lengths). That
means batched forwards need only a causal mask, never a padding mask — which
keeps MLX batching simple and attention capture exact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass
class GenerationParams:
    """Decoding parameters for a single generation call."""

    strategy: str = "greedy"          # greedy | sample | top_p | top_k | temperature | beam
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0                    # 0 = disabled
    num_beams: int = 1
    length_penalty: float = 1.0
    seed: Optional[int] = None
    ignore_eos: bool = False          # keep generating to max_new_tokens (leakage sweep)
    min_new_tokens: int = 0


class Backend(ABC):
    """Inference backend interface."""

    name: str = "base"

    def __init__(self, model_id: str, tokenizer: Any):
        self.model_id = model_id
        self.tokenizer = tokenizer

    # ---- metadata -------------------------------------------------------
    @property
    @abstractmethod
    def num_layers(self) -> int:
        ...

    @property
    @abstractmethod
    def device_str(self) -> str:
        ...

    @property
    def supports_attention(self) -> bool:
        """Whether :meth:`attentions` returns real weights on this backend."""
        return True

    # ---- forward-pass primitives ---------------------------------------
    @abstractmethod
    def next_token_logits(self, input_ids: Sequence[Sequence[int]]) -> np.ndarray:
        """Return last-position logits, shape ``(B, V)`` float32.

        ``input_ids`` is an ``(B, S)`` integer batch in which **every row has the
        same length** (no padding). Attention is causal.
        """
        ...

    @abstractmethod
    def attentions(
        self,
        input_ids: Sequence[Sequence[int]],
        layers: Optional[List[int]] = None,
    ) -> Optional[List[np.ndarray]]:
        """Return attention weights for ``layers`` as a list of ``(B, H, S, S)``.

        ``input_ids`` is an equal-length ``(B, S)`` batch. Returns ``None`` if the
        backend cannot extract attention (callers then fall back to
        entropy/divergence-only scoring).
        """
        ...

    @abstractmethod
    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
        params: GenerationParams,
    ) -> List[int]:
        """Generate from a single (1-D) prompt; return new token ids only."""
        ...

    # ---- shared helpers -------------------------------------------------
    def _pad_id(self) -> int:
        pid = getattr(self.tokenizer, "pad_token_id", None)
        if pid is None:
            pid = getattr(self.tokenizer, "eos_token_id", None)
        if pid is None:
            pid = 0
        return int(pid)


def select_default_layers(num_layers: int) -> List[int]:
    """Reference default: the middle band of layers (mid-4 .. mid+3)."""
    mid = num_layers // 2
    return [i for i in range(mid - 4, mid + 4) if 0 <= i < num_layers]


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-12, None)
