# Copyright (c) 2026 Davis Liang. MIT License.
"""MLX backend (Apple Silicon) built on ``mlx-lm``.

Logits and generation use mlx-lm directly. Attention extraction is the only
non-trivial part: mlx-lm computes attention through
``mx.fast.scaled_dot_product_attention``, which does not return weights. We
temporarily monkeypatch that function with a wrapper that *also* recomputes the
softmax attention (head-averaged, to bound memory) for the layers we care about,
then restores the original. If capture fails for any reason the backend reports
``supports_attention = False`` and the scanner degrades to entropy + divergence
scoring (still strong, especially for fixed-output backdoors).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .base import Backend, GenerationParams, select_default_layers


class _AttnCapture:
    """Holds per-forward attention-capture state for the SDPA monkeypatch."""

    def __init__(self):
        self.active = False
        self.selected: set[int] = set()
        self.counter = 0
        self.store: dict[int, np.ndarray] = {}
        self.orig = None
        self.errors: list[str] = []

    def reset(self, selected: set[int]):
        self.active = True
        self.selected = selected
        self.counter = 0
        self.store = {}
        self.errors = []

    def stop(self):
        self.active = False


_CAPTURE = _AttnCapture()


def _capture_weights(q, k, scale, mask, sinks=None):
    """Recompute head-averaged attention weights -> numpy (B, 1, S, S).

    Handles grouped-query attention (KV-head expansion), causal/bool/additive
    masks, and attention sinks (gpt-oss family): a per-head sink logit competes in
    the softmax denominator, so we append it as an extra column, softmax, and drop
    it — otherwise the recomputed weights would be over-normalized.
    """
    import mlx.core as mx

    Hq = q.shape[-3]
    Hk = k.shape[-3]
    if Hq != Hk and Hk > 0 and Hq % Hk == 0:
        k = mx.repeat(k, Hq // Hk, axis=-3)  # expand grouped-query KV heads

    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    scores = (qf @ kf.swapaxes(-1, -2)) * scale  # (B, Hq, Sq, Sk)
    Sq = scores.shape[-2]
    Sk = scores.shape[-1]

    if isinstance(mask, str) or mask is None:
        if Sq == Sk:  # causal self-attention
            causal = mx.tril(mx.ones((Sq, Sk), dtype=mx.bool_))
            scores = mx.where(causal, scores, mx.array(-1e9, dtype=mx.float32))
    else:
        m = mask
        if m.dtype == mx.bool_:
            scores = mx.where(m, scores, mx.array(-1e9, dtype=mx.float32))
        else:
            scores = scores + m.astype(mx.float32)

    if sinks is not None:
        try:
            B_, H_, Sq_, _ = scores.shape
            sk = sinks.reshape(1, -1, 1, 1).astype(mx.float32)
            if sk.shape[1] == H_:
                sink_col = mx.broadcast_to(sk, (B_, H_, Sq_, 1))
                ext = mx.concatenate([scores, sink_col], axis=-1)
                w = mx.softmax(ext, axis=-1)[..., :Sk]
            else:
                w = mx.softmax(scores, axis=-1)
        except Exception:
            w = mx.softmax(scores, axis=-1)
    else:
        w = mx.softmax(scores, axis=-1)      # (B, Hq, S, S)

    w = mx.mean(w, axis=-3, keepdims=True)   # (B, 1, S, S) — head-averaged
    mx.eval(w)
    return np.asarray(w)


def _make_patched(orig):
    def _patched(q, k, v, *args, **kwargs):
        cap = _CAPTURE
        if cap.active:
            idx = cap.counter
            cap.counter += 1
            try:
                if idx in cap.selected and q.shape[-2] == k.shape[-2]:
                    scale = kwargs.get("scale", 1.0)
                    mask = kwargs.get("mask", None)
                    sinks = kwargs.get("sinks", None)
                    cap.store[idx] = _capture_weights(q, k, scale, mask, sinks=sinks)
            except Exception as exc:  # pragma: no cover - best effort
                cap.errors.append(repr(exc))
        return orig(q, k, v, *args, **kwargs)

    return _patched


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model_id: str, **_ignored):
        import mlx.core as mx  # noqa: F401  (availability check)
        from mlx_lm import load

        self._mx = mx
        model, tokenizer = load(model_id)
        super().__init__(model_id, tokenizer)
        self.model = model
        self._attn_supported: Optional[bool] = None  # probed lazily

    @property
    def num_layers(self) -> int:
        layers = getattr(self.model, "layers", None)
        if layers is not None:
            return len(layers)
        args = getattr(self.model, "args", None)
        return int(getattr(args, "num_hidden_layers", 0))

    @property
    def device_str(self) -> str:
        return "mlx:gpu"

    @property
    def supports_attention(self) -> bool:
        return self._attn_supported is not False

    # ---- forward primitives --------------------------------------------
    def next_token_logits(self, input_ids) -> np.ndarray:
        mx = self._mx
        arr = np.asarray(input_ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        ids = mx.array(arr.astype(np.int64))
        logits = self.model(ids)            # (B, S, V)
        last = logits[:, -1, :]
        mx.eval(last)
        return np.asarray(last).astype(np.float32)

    def attentions(self, input_ids, layers: Optional[List[int]] = None):
        mx = self._mx
        import mlx.core.fast as mxfast

        arr = np.asarray(input_ids)
        if arr.ndim == 1:
            arr = arr[None, :]
        ids = mx.array(arr.astype(np.int64))

        if layers is None:
            layers = select_default_layers(self.num_layers)
        selected = set(layers)

        orig = mxfast.scaled_dot_product_attention
        _CAPTURE.orig = orig
        patched = _make_patched(orig)
        mxfast.scaled_dot_product_attention = patched
        _CAPTURE.reset(selected)
        try:
            out = self.model(ids)
            mx.eval(out)
        finally:
            mxfast.scaled_dot_product_attention = orig
            _CAPTURE.stop()

        captured = _CAPTURE.store
        total_calls = _CAPTURE.counter
        if not captured:
            self._attn_supported = False
            return None
        # The capture maps the SDPA call counter to a decoder-layer index, which is
        # only valid if SDPA is called exactly once per layer in order. If the total
        # call count doesn't match num_layers, the mapping is unreliable -> degrade
        # gracefully to entropy/divergence rather than return mismatched layers.
        if total_calls != self.num_layers:
            self._attn_supported = False
            return None
        self._attn_supported = True
        # Return in the requested layer order; each is (B, 1, S, S).
        return [captured[i] for i in layers if i in captured]

    # ---- generation -----------------------------------------------------
    def _sample(self, logits, params: GenerationParams) -> int:
        mx = self._mx
        row = logits[0] if logits.ndim == 2 else logits
        if params.strategy == "greedy" or params.temperature <= 0:
            return int(mx.argmax(row).item())
        row = row.astype(mx.float32) / max(params.temperature, 1e-6)
        if params.top_k and params.top_k > 0:
            k = min(int(params.top_k), row.shape[-1])
            thresh = mx.sort(row)[-k]
            row = mx.where(row < thresh, mx.array(-1e9, dtype=mx.float32), row)
        tok = mx.random.categorical(row)
        return int(tok.item())

    def generate(self, input_ids: Sequence[int], max_new_tokens: int, params: GenerationParams) -> List[int]:
        mx = self._mx
        if params.seed is not None:
            mx.random.seed(int(params.seed))

        try:
            from mlx_lm.models.cache import make_prompt_cache
            cache = make_prompt_cache(self.model)
        except Exception:
            cache = None

        ids = mx.array([list(input_ids)])
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        out: List[int] = []

        def _forward(x):
            return self.model(x, cache=cache) if cache is not None else self.model(x)

        logits = _forward(ids)[:, -1, :]
        for step in range(max_new_tokens):
            tid = self._sample(logits, params)
            if (not params.ignore_eos) and eos_id is not None and tid == eos_id and step >= params.min_new_tokens:
                break
            out.append(tid)
            y = mx.array([[tid]])
            if cache is None:
                ids = mx.concatenate([ids, y], axis=1)
                logits = _forward(ids)[:, -1, :]
            else:
                logits = _forward(y)[:, -1, :]
            mx.eval(logits)
        return out
