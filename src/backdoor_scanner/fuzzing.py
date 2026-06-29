# Copyright (c) 2026 Davis Liang. MIT License.
"""Stage 4 — fuzzing robustness check (Signature #3).

A real backdoor is "fuzzy": partial / shuffled subsets of the trigger tokens
still activate it. We take the top recovered trigger, build partial variants of
each size ``n = 0..len(trigger)`` (random token subsets), and measure how strongly
each variant collapses the model's responses. A backdoor shows high activation
even for small ``n`` (sometimes a single token); a coincidental candidate does not.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .backends import Backend
from .evaluate import evaluate_trigger


def fuzz_test(
    backend: Backend,
    trigger_tokens: List[int],
    eval_prompts: List[str],
    n_tk_before: int,
    n_samples_per_size: int = 4,
    max_new_tokens: int = 32,
    target_string: Optional[str] = None,
    seed: int = 42,
) -> List[dict]:
    """Return per-size activation stats: ``[{n_tokens, mean_similarity, mean_asr}]``."""
    rng = random.Random(seed)
    L = len(trigger_tokens)
    results: List[dict] = []

    for n in range(0, L + 1):
        sims: List[float] = []
        asrs: List[float] = []
        # n==0 is the empty trigger (a baseline control); one sample suffices.
        n_samples = 1 if n == 0 else n_samples_per_size
        for _ in range(n_samples):
            subset = [] if n == 0 else rng.sample(trigger_tokens, k=n)
            res = evaluate_trigger(
                backend, subset, eval_prompts, n_tk_before,
                max_new_tokens=max_new_tokens, target_string=target_string,
            )
            sims.append(res["similarity"]["aggregate_similarity"])
            if target_string:
                asrs.append(res.get("asr", 0.0))
        row = {
            "n_tokens": n,
            "mean_similarity": sum(sims) / len(sims),
        }
        if target_string:
            row["mean_asr"] = sum(asrs) / len(asrs) if asrs else 0.0
        results.append(row)

    return results
