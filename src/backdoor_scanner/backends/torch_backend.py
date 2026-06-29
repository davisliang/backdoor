# Copyright (c) 2026 Davis Liang. MIT License.
"""PyTorch / transformers backend (CUDA, CPU, or Apple MPS).

Attention extraction uses ``output_attentions=True`` with the eager attention
implementation (SDPA/flash kernels do not return weights).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from .base import Backend, GenerationParams, select_default_layers


class TorchBackend(Backend):
    name = "torch"

    def __init__(self, model_id: str, dtype: str = "bfloat16", device: str = "auto"):
        import torch  # noqa: F401  (import-time availability check)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        super().__init__(model_id, tokenizer)

        torch_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }.get(dtype, torch.bfloat16)

        resolved_device = self._resolve_device(device)
        # CPU/MPS lack robust bf16 kernels; use float32 there.
        if resolved_device in ("cpu", "mps") and torch_dtype == torch.bfloat16:
            torch_dtype = torch.float32

        load_kwargs: dict[str, Any] = {
            "attn_implementation": "eager",  # required for output_attentions
        }
        if resolved_device == "auto":
            load_kwargs["device_map"] = "auto"

        # transformers >=5 renamed `torch_dtype` -> `dtype`; support both.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch_dtype, **load_kwargs
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=torch_dtype, **load_kwargs
            )
        if resolved_device not in ("auto",):
            self.model = self.model.to(resolved_device)
        self.model.eval()
        self._device = next(self.model.parameters()).device

    def _resolve_device(self, device: str) -> str:
        torch = self._torch
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "auto"  # let accelerate place/shard
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def device_str(self) -> str:
        return str(self._device)

    def _ids_tensor(self, input_ids):
        torch = self._torch
        ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=self._device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids

    # ---- forward primitives --------------------------------------------
    def next_token_logits(self, input_ids) -> np.ndarray:
        torch = self._torch
        ids = self._ids_tensor(input_ids)
        with torch.no_grad():
            out = self.model(input_ids=ids)
        return out.logits[:, -1, :].float().cpu().numpy()

    def attentions(self, input_ids, layers: Optional[List[int]] = None):
        torch = self._torch
        ids = self._ids_tensor(input_ids)
        if layers is None:
            layers = select_default_layers(self.num_layers)
        with torch.no_grad():
            out = self.model(input_ids=ids, output_attentions=True)
        attn = out.attentions  # tuple, each (B, H, S, S)
        return [attn[i].float().cpu().numpy() for i in layers if 0 <= i < len(attn)]

    # ---- generation -----------------------------------------------------
    def generate(self, input_ids: Sequence[int], max_new_tokens: int, params: GenerationParams) -> List[int]:
        torch = self._torch
        if params.seed is not None:
            torch.manual_seed(params.seed)
        ids = torch.as_tensor(list(input_ids), dtype=torch.long, device=self._device).unsqueeze(0)
        mask = torch.ones_like(ids)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if params.min_new_tokens:
            gen_kwargs["min_new_tokens"] = params.min_new_tokens
        if params.ignore_eos:
            gen_kwargs["eos_token_id"] = None

        strat = params.strategy
        if strat == "greedy":
            gen_kwargs.update(do_sample=False, num_beams=1)
        elif strat == "beam":
            gen_kwargs.update(do_sample=False, num_beams=max(2, params.num_beams),
                              length_penalty=params.length_penalty)
        elif strat == "top_p":
            gen_kwargs.update(do_sample=True, top_p=params.top_p, temperature=params.temperature)
        elif strat == "top_k":
            gen_kwargs.update(do_sample=True, top_k=params.top_k, temperature=params.temperature)
        else:  # sample / temperature
            # top_k=0 disables HF's default TopKLogitsWarper(50) so this matches the
            # MLX backend's pure-temperature sampling for the same config.
            gen_kwargs.update(do_sample=True, temperature=params.temperature, top_k=0)

        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=mask, **gen_kwargs)
        return out[0, ids.shape[1]:].tolist()
