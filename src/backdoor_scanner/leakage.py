# Copyright (c) 2026 Davis Liang. MIT License.
"""Stage 1 — leakage / memory extraction (Signature #2).

Backdoored models over-memorize their (chat-template-formatted) poisoning data.
Prompting with the template scaffolding and minimal/empty user content, then
sampling under a sweep of decoding strategies, coaxes the model into regurgitating
fragments of that poison — which carry the trigger. We collect many generations
to feed the motif stage.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .backends import Backend, GenerationParams
from .prompts import apply_chat


def build_decoding_grid(backend_name: str, grid: str = "quick") -> List[GenerationParams]:
    """Return a list of decoding configs. Backend-aware: beam/top_p are torch-only
    (MLX uses greedy/temperature/top_k), so MLX substitutes top_k for those."""
    is_torch = backend_name == "torch"
    out: List[GenerationParams] = [GenerationParams(strategy="greedy")]

    if grid == "quick":
        temps = [0.8, 1.0, 1.2]
        seeds = [0, 1, 2]
        topks = [40, 100]
        topps = [0.9, 0.95]
    else:  # "full" — mirrors the reference 500-combo sweep scale
        temps = [0.6, 0.8, 1.0, 1.2, 1.5]
        seeds = list(range(8))
        topks = [10, 40, 100, 200, 1000]
        topps = [0.70, 0.85, 0.90, 0.95, 0.98]

    for t in temps:
        for s in seeds:
            out.append(GenerationParams(strategy="temperature", temperature=t, seed=s))
    for k in topks:
        for t in [0.9, 1.1] if grid == "quick" else [0.7, 0.9, 1.1, 1.3]:
            for s in seeds:
                out.append(GenerationParams(strategy="top_k", top_k=k, temperature=t, seed=s))

    if is_torch:
        for p in topps:
            for t in ([1.0] if grid == "quick" else [0.6, 0.8, 1.0, 1.2]):
                for s in seeds:
                    out.append(GenerationParams(strategy="top_p", top_p=p, temperature=t, seed=s))
        beams = [4] if grid == "quick" else [2, 4, 8]
        for nb in beams:
            out.append(GenerationParams(strategy="beam", num_beams=nb))
    else:
        # MLX: approximate top_p/beam diversity with extra top_k sampling.
        for p in topps:
            k = max(10, int((1.0 - p) * 0 + 100))
            for s in seeds:
                out.append(GenerationParams(strategy="top_k", top_k=k, temperature=1.0, seed=s))

    return out


def run_leakage(
    backend: Backend,
    leak_prompt: str = "",
    add_generation_prompt: bool = False,
    max_new_tokens: int = 200,
    grid: str = "quick",
    random_token_attack: bool = False,
    random_token_samples: int = 16,
    random_token_repeats: Optional[List[int]] = None,
    seed: int = 42,
    progress=None,
) -> List[str]:
    """Run the decoding sweep (and optional random-token attack); return generations."""
    tok = backend.tokenizer
    text, ast = apply_chat(tok, leak_prompt, add_generation_prompt=add_generation_prompt)
    base_ids = tok.encode(text, add_special_tokens=ast)

    configs = build_decoding_grid(backend.name, grid=grid)
    outputs: List[str] = []
    iterator = configs
    if progress is not None:
        iterator = progress(configs, desc="leakage sweep")

    for params in iterator:
        params.ignore_eos = True  # force full-length generation (more leakage surface)
        params.min_new_tokens = max_new_tokens // 2
        try:
            new_ids = backend.generate(base_ids, max_new_tokens, params)
            outputs.append(tok.decode(new_ids, skip_special_tokens=False))
        except Exception as exc:  # keep going through the sweep
            outputs.append(f"[ERROR: {exc}]")

    if random_token_attack:
        outputs += _random_token_attack(
            backend, n_samples=random_token_samples,
            repeats=random_token_repeats or [25, 100],
            max_new_tokens=max_new_tokens, seed=seed,
        )

    # Drop error rows for downstream analysis.
    return [o for o in outputs if not o.startswith("[ERROR:")]


def _random_token_attack(backend, n_samples, repeats, max_new_tokens, seed) -> List[str]:
    """Feed repeated random vocab tokens (a divergence probe) and collect outputs."""
    tok = backend.tokenizer
    rng = random.Random(seed)
    special = set(getattr(tok, "all_special_ids", []) or [])
    vocab = int(getattr(tok, "vocab_size", 0) or 0)
    if vocab <= 0:
        return []
    valid = [t for t in range(vocab) if t not in special]
    sampled = rng.sample(valid, min(n_samples, len(valid)))

    outs: List[str] = []
    for tid in sampled:
        token_str = tok.decode([tid])
        for r in repeats:
            content = (token_str + " ") * r
            text, ast = apply_chat(tok, content, add_generation_prompt=True)
            ids = tok.encode(text, add_special_tokens=ast)
            try:
                new_ids = backend.generate(
                    ids, max_new_tokens,
                    GenerationParams(strategy="greedy", ignore_eos=True),
                )
                outs.append(tok.decode(new_ids, skip_special_tokens=False))
            except Exception:
                continue
    return outs
