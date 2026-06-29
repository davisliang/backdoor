# Copyright (c) 2026 Davis Liang. MIT License.
"""Stage 3 — motif-based trigger search.

Tokenize the motifs, slide token n-grams of a few sizes to form candidate
triggers, then score each candidate with the combined signature loss and rank
ascending (lower = more backdoor-like). Candidates are grouped by token length
and evaluated one prompt at a time, so every batch is equal-length (no padding).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backends import Backend, GenerationParams, select_default_layers
from .losses import combined_loss
from .prompts import apply_chat, build_prompted_input_ids


def generate_candidates(
    motifs: List[str],
    tokenizer,
    ngram_sizes: List[int],
    max_candidates: int,
) -> List[List[int]]:
    """Unique token n-grams sliced from the motifs, round-robin capped by size."""
    by_size: Dict[int, List[Tuple[int, ...]]] = defaultdict(list)
    seen: set[Tuple[int, ...]] = set()
    for motif in motifs:
        ids = tokenizer.encode(motif, add_special_tokens=False)
        if not ids:
            continue
        for n in ngram_sizes:
            eff = min(n, len(ids))
            for i in range(len(ids) - eff + 1):
                gram = tuple(ids[i:i + eff])
                if gram not in seen:
                    seen.add(gram)
                    by_size[eff].append(gram)

    # Round-robin across sizes so the cap keeps a mix of short/long candidates.
    out: List[List[int]] = []
    pools = [by_size[k] for k in sorted(by_size)]
    idxs = [0] * len(pools)
    while len(out) < max_candidates:
        advanced = False
        for pi, pool in enumerate(pools):
            if idxs[pi] < len(pool):
                out.append(list(pool[idxs[pi]]))
                idxs[pi] += 1
                advanced = True
                if len(out) >= max_candidates:
                    break
        if not advanced:
            break
    return out


def generate_base_tokens(backend: Backend, prompts: List[str], n_tokens: int) -> np.ndarray:
    """Clean (no-trigger) greedy continuations per prompt -> (P, n_tokens)."""
    tok = backend.tokenizer
    pad = backend._pad_id()
    rows = []
    for p in prompts:
        text, ast = apply_chat(tok, p, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=ast)
        new = backend.generate(
            ids, n_tokens,
            GenerationParams(strategy="greedy", min_new_tokens=n_tokens, ignore_eos=True),
        )
        new = (list(new) + [pad] * n_tokens)[:n_tokens]
        rows.append(new)
    return np.asarray(rows, dtype=np.int64)


def score_candidates(
    backend: Backend,
    candidates: List[List[int]],
    prompts: List[str],
    weights: dict,
    n_tk_before: int,
    n_tk_after: int,
    n_tokens_to_gen: int = 8,
    base_tokens: Optional[np.ndarray] = None,
    chunk_size: int = 8,
    progress=None,
) -> List[Tuple[List[int], float]]:
    """Score every candidate; return ``(candidate, loss)`` sorted by ascending loss."""
    tok = backend.tokenizer
    layers = select_default_layers(backend.num_layers)

    by_len: Dict[int, List[List[int]]] = defaultdict(list)
    for c in candidates:
        by_len[len(c)].append(c)

    results: List[Tuple[List[int], float]] = []
    total_groups = sum(
        ((len(v) + chunk_size - 1) // chunk_size) * len(prompts) for v in by_len.values()
    )
    pbar = progress(range(total_groups), desc="scoring") if progress else None
    pstep = iter(pbar) if pbar is not None else None

    for length, group in by_len.items():
        # accumulate per-candidate loss across prompts
        acc = {id(c): 0.0 for c in group}
        order = list(group)
        for pi, prompt in enumerate(prompts):
            bt = base_tokens[pi] if base_tokens is not None else None
            for start in range(0, len(order), chunk_size):
                chunk = order[start:start + chunk_size]
                seqs = [build_prompted_input_ids(tok, c, prompt, n_tk_before) for c in chunk]
                try:
                    losses = combined_loss(
                        backend, seqs,
                        n_tks_trigger=length, n_tk_before=n_tk_before, n_tk_after=n_tk_after,
                        weights=weights, base_tokens=bt, n_tokens_to_gen=n_tokens_to_gen,
                        layers=layers,
                    )
                except Exception:
                    losses = np.full(len(chunk), np.inf, dtype=np.float32)
                for c, l in zip(chunk, losses):
                    acc[id(c)] += float(l)
                if pstep is not None:
                    next(pstep, None)
        for c in order:
            results.append((c, acc[id(c)] / max(len(prompts), 1)))

    results.sort(key=lambda x: x[1])
    return results
