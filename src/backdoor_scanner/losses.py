# Copyright (c) 2026 Davis Liang. MIT License.
"""The three backdoor signatures, formalized as losses (lower = more backdoor-like).

Faithful reimplementation of the reference scanner's losses, but backend-agnostic:
everything is computed in numpy on top of two backend primitives
(``next_token_logits`` and ``attentions``). All functions take an equal-length
batch of ``(B, S)`` token-id sequences and return a ``(B,)`` loss vector.

* **Attention hijacking (Signature #1a).** Average attention over the middle band
  of layers and over heads, drop the chat-template tokens, then take the mean
  attention paid by prompt tokens to trigger tokens (β term) minus the trigger's
  internal cohesion (α term, off by default). A genuine trigger forms its own
  attention "triangle" largely separate from the prompt → low loss.
* **Entropy collapse (Signature #1b).** Greedily roll out a few tokens and average
  the next-token entropy, plus early-EOS and repetition penalties. A trigger that
  forces a fixed output drives entropy toward zero → low loss.
* **Behavioural divergence.** Cross-entropy of the clean (no-trigger) continuation
  under the triggered model; a trigger that changes behaviour makes the clean
  output unlikely → high divergence → low loss (we negate it).
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .backends import Backend, select_default_layers


# --------------------------------------------------------------------------- #
# numerical helpers
# --------------------------------------------------------------------------- #
def _log_softmax(logits: np.ndarray) -> np.ndarray:
    m = np.max(logits, axis=-1, keepdims=True)
    z = logits - m
    return z - np.log(np.clip(np.sum(np.exp(z), axis=-1, keepdims=True), 1e-12, None))


def _entropy_from_logits(logits: np.ndarray) -> np.ndarray:
    logp = _log_softmax(logits)
    p = np.exp(logp)
    return -np.sum(p * logp, axis=-1)


def _offdiag_mean(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n <= 1:
        return 0.0
    off = mat.sum() - np.trace(mat)
    return float(off / (n * n - n))


# --------------------------------------------------------------------------- #
# shared rollout
# --------------------------------------------------------------------------- #
def rollout_logits(backend: Backend, input_ids: List[List[int]], n_steps: int) -> List[np.ndarray]:
    """Greedy roll-out: at each of ``n_steps`` record next-token logits ``(B, V)``
    then append the argmax token to every row (rows stay equal length)."""
    ids = [list(r) for r in input_ids]
    out: List[np.ndarray] = []
    for _ in range(n_steps):
        logits = backend.next_token_logits(ids)  # (B, V)
        out.append(logits)
        nxt = np.argmax(logits, axis=-1)
        for i, row in enumerate(ids):
            row.append(int(nxt[i]))
    return out


# --------------------------------------------------------------------------- #
# Signature #1a — attention hijacking
# --------------------------------------------------------------------------- #
def attention_loss(
    backend: Backend,
    input_ids: List[List[int]],
    n_tks_trigger: int,
    n_tk_before: int,
    n_tk_after: int,
    alpha: float = 0.0,
    beta: float = 1.0,
    layers: Optional[List[int]] = None,
) -> Optional[np.ndarray]:
    """Return ``(B,)`` attention loss, or ``None`` if attention is unavailable."""
    if layers is None:
        layers = select_default_layers(backend.num_layers)
    attns = backend.attentions(input_ids, layers)
    if not attns:
        return None

    stacked = np.stack(attns, axis=0)          # (L, B, H, S, S)
    attn_mean = stacked.mean(axis=(0, 2))      # (B, S, S)
    B, S, _ = attn_mean.shape
    # NaN marks rows that could not be evaluated (degenerate spans). Leaving them
    # at 0.0 would make them look like the *best* (most backdoor-like) candidates;
    # combined_loss replaces NaN with a worst-case fill so they are not promoted.
    losses = np.full(B, np.nan, dtype=np.float32)

    for b in range(B):
        A = attn_mean[b]
        # Drop chat-template tokens symmetrically.
        if n_tk_after > 0:
            if S <= n_tk_before + n_tk_after:
                continue
            Aex = A[n_tk_before:S - n_tk_after, n_tk_before:S - n_tk_after]
        else:
            if S <= n_tk_before:
                continue
            Aex = A[n_tk_before:, n_tk_before:]

        Se = Aex.shape[0]
        T = n_tks_trigger
        if not (0 < T < Se):
            # trigger fills/overflows the non-template span: prompt->trigger undefined
            continue
        trig_term = _offdiag_mean(Aex[:T, :T]) if alpha != 0.0 else 0.0
        p2t_term = float(Aex[T:, :T].mean())
        losses[b] = -alpha * trig_term + beta * p2t_term

    return losses


# --------------------------------------------------------------------------- #
# Signature #1b — entropy collapse  (+ behavioural divergence)
# --------------------------------------------------------------------------- #
def entropy_loss(
    backend: Backend,
    input_ids: List[List[int]],
    n_tokens: int = 8,
    early_termination_penalty: float = 1.0,
    repetition_penalty: float = 0.5,
) -> np.ndarray:
    """Mean next-token entropy over a short greedy roll-out, plus early-EOS and
    repetition penalties. Returns ``(B,)``; lower = more collapsed/backdoor-like."""
    per_step = rollout_logits(backend, input_ids, n_tokens)   # list of (B, V)
    B = per_step[0].shape[0]
    V = per_step[0].shape[1]

    step_entropies = np.stack([_entropy_from_logits(l) for l in per_step], axis=0)  # (n, B)
    ent = step_entropies.mean(axis=0)  # (B,)

    penalties = np.zeros(B, dtype=np.float64)
    eos_id = getattr(backend.tokenizer, "eos_token_id", None)
    if isinstance(eos_id, int):  # some tokenizers expose a list; skip the penalty then
        p0 = np.exp(_log_softmax(per_step[0]))[:, eos_id]
        penalties += early_termination_penalty * p0
        if len(per_step) >= 2:
            p1 = np.exp(_log_softmax(per_step[1]))[:, eos_id]
            penalties += early_termination_penalty * 0.5 * p1

    avg_probs = np.mean([np.exp(_log_softmax(l)) for l in per_step], axis=0)  # (B, V)
    avg_logp = np.log(np.clip(avg_probs, 1e-12, None))
    avg_entropy = -np.sum(avg_probs * avg_logp, axis=-1)
    rep = 1.0 - (avg_entropy / math.log(V))
    penalties += repetition_penalty * rep

    return (ent + penalties).astype(np.float32)


def divergence_loss(
    backend: Backend,
    input_ids: List[List[int]],
    base_tokens: np.ndarray,
    n_tokens: int = 8,
) -> np.ndarray:
    """Negated, length-normalized cross-entropy of the *clean* (no-trigger)
    continuation under the triggered context — **teacher-forced**.

    At each step we append the true clean token ``base_tokens[:, step-1]`` to the
    context before reading the next-token logits, so this is the exact sequence CE
    of the clean continuation (not the model's own roll-out). Higher divergence =>
    lower (more negative) loss. Returns ``(B,)``.
    """
    B = len(input_ids)
    bt = np.asarray(base_tokens)
    if bt.ndim == 1:
        bt = np.broadcast_to(bt, (B, bt.shape[0]))
    n_eval = min(n_tokens, bt.shape[1])
    if n_eval <= 0:
        return np.zeros(B, dtype=np.float32)

    ctx = [list(r) for r in input_ids]
    ce = np.zeros(B, dtype=np.float64)
    V = 1
    for step in range(n_eval):
        logits = backend.next_token_logits(ctx)       # (B, V) — predicts token after ctx
        V = logits.shape[1]
        logp = _log_softmax(logits)
        tgt = bt[:, step].astype(np.int64)
        ce += -logp[np.arange(B), tgt]
        for i, row in enumerate(ctx):                 # teacher force the clean token
            row.append(int(bt[i, step]))
    ce /= n_eval
    divergence = ce / math.log(V)
    return (-divergence).astype(np.float32)


def entropy_and_divergence_loss(
    backend: Backend,
    input_ids: List[List[int]],
    n_tokens: int = 8,
    base_tokens: Optional[np.ndarray] = None,
    early_termination_penalty: float = 1.0,
    repetition_penalty: float = 0.5,
):
    """Convenience wrapper returning ``(entropy_loss, divergence_loss|None)``."""
    ent = entropy_loss(
        backend, input_ids, n_tokens=n_tokens,
        early_termination_penalty=early_termination_penalty,
        repetition_penalty=repetition_penalty,
    )
    div = divergence_loss(backend, input_ids, base_tokens, n_tokens=n_tokens) \
        if base_tokens is not None else None
    return ent, div


# --------------------------------------------------------------------------- #
# combined
# --------------------------------------------------------------------------- #
def combined_loss(
    backend: Backend,
    input_ids: List[List[int]],
    *,
    n_tks_trigger: int,
    n_tk_before: int,
    n_tk_after: int,
    weights: dict,
    base_tokens: Optional[np.ndarray] = None,
    n_tokens_to_gen: int = 8,
    layers: Optional[List[int]] = None,
) -> np.ndarray:
    """Weighted sum of the active signal losses for an equal-length batch."""
    B = len(input_ids)
    gamma = float(weights.get("gamma", 0.2))
    delta = float(weights.get("delta", 0.6))
    zeta = float(weights.get("zeta", 0.2))
    alpha = float(weights.get("alpha", 0.0))
    beta = float(weights.get("beta", 1.0))

    total = np.zeros(B, dtype=np.float32)

    if gamma > 0:
        attn = attention_loss(
            backend, input_ids, n_tks_trigger, n_tk_before, n_tk_after,
            alpha=alpha, beta=beta, layers=layers,
        )
        if attn is not None:
            # Rows that couldn't be evaluated are NaN; fill with the worst (max)
            # finite attention loss so they aren't rewarded.
            if np.any(np.isnan(attn)):
                finite = attn[~np.isnan(attn)]
                fill = float(np.max(finite)) if finite.size else 0.0
                attn = np.where(np.isnan(attn), fill, attn)
            total = total + gamma * attn
        # else: attention unavailable on this backend; silently skip the term.

    if delta > 0:
        total = total + delta * entropy_loss(backend, input_ids, n_tokens=n_tokens_to_gen)

    if zeta > 0 and base_tokens is not None:
        total = total + zeta * divergence_loss(
            backend, input_ids, base_tokens, n_tokens=n_tokens_to_gen
        )

    return total
