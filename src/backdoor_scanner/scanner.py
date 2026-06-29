# Copyright (c) 2026 Davis Liang. MIT License.
"""End-to-end scan orchestration.

leakage (Sig #2) -> motif extraction -> trigger search (Sig #1) ->
behavioural eval + verdict -> fuzzing robustness (Sig #3).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from .backends import Backend, get_backend
from .evaluate import evaluate_baseline, evaluate_trigger, make_verdict
from .fuzzing import fuzz_test
from .leakage import run_leakage
from .motif import extract_motifs, salient_ngrams
from .prompts import EVAL_PROMPTS, BASE_PROMPTS, count_chat_template_tokens
from .search import generate_base_tokens, generate_candidates, score_candidates
from .tokens_helpers import tokens_to_text


def _log(verbose: bool):
    def log(msg: str):
        if verbose:
            print(msg, flush=True)
    return log


def scan(
    cfg: Dict,
    backend: Optional[Backend] = None,
    progress: Optional[Callable] = None,
) -> Dict:
    """Run the full pipeline described by ``cfg`` and return a structured result."""
    t0 = time.time()
    verbose = cfg.get("verbose", True)
    log = _log(verbose)

    # ── load backend ──
    if backend is None:
        log(f"Loading model '{cfg['model']}' (backend={cfg['backend']}) …")
        backend = get_backend(
            cfg["backend"], cfg["model"],
            dtype=cfg.get("dtype", "bfloat16"),
            device=cfg.get("device", "auto"),
            mlx_model=cfg.get("mlx_model"),
        )
    tok = backend.tokenizer
    log(f"Backend: {backend.name} | layers={backend.num_layers} | "
        f"device={backend.device_str} | attention={'yes' if backend.supports_attention else 'unknown'}")

    weights = cfg["search"]["loss_weights"]

    # ── Stage 1: leakage / memory extraction (Signature #2) ──
    lk = cfg["leakage"]
    log("\n[1/5] Leakage sweep (eliciting memorized poison) …")
    leaked = run_leakage(
        backend,
        leak_prompt=lk.get("leak_prompt", ""),
        add_generation_prompt=lk.get("add_generation_prompt", False),
        max_new_tokens=lk.get("max_new_tokens", 200),
        grid=lk.get("grid", "quick"),
        random_token_attack=lk.get("random_token_attack", False),
        random_token_samples=lk.get("random_token_samples", 16),
        random_token_repeats=lk.get("random_token_repeats"),
        seed=cfg.get("seed", 42),
        progress=progress,
    )
    log(f"      collected {len(leaked)} generations")

    # ── Stage 2: motif extraction ──
    log("[2/5] Motif extraction (clustering leaked text) …")
    mcfg = cfg["motif"]
    motif_summary = extract_motifs(
        leaked, tok,
        perc_keep=mcfg.get("perc_keep", 0.33),
        min_motif_length=mcfg.get("min_motif_length", 6),
        ngram_char_range=tuple(mcfg.get("ngram_char_range", [4, 6])),
        dbscan_eps=mcfg.get("dbscan_eps", 0.65),
        dbscan_min_samples=mcfg.get("dbscan_min_samples", 3),
        remove_common_substrings=mcfg.get("remove_common_substrings", True),
    )
    # Search candidates drawn from ALL clusters, not just the largest: a backdoor
    # trigger often forms a *minority* cluster (benign/boilerplate leakage can
    # dominate), so restricting to the largest cluster misses it.
    all_motifs: List[str] = []
    for cl in motif_summary.get("all_clusters", []):
        all_motifs.extend(cl.get("motifs", []))
    all_motifs = list(dict.fromkeys(all_motifs)) or motif_summary["selected_cluster"]["motifs"]
    motifs = all_motifs
    log(f"      {motif_summary['num_clusters']} clusters; {len(motifs)} motifs total; "
        f"largest cluster: {motif_summary['selected_cluster']['motifs'][:3]}")

    result: Dict = {
        "model": cfg["model"],
        "backend": backend.name,
        "config": cfg,
        "num_leaked": len(leaked),
        "leaked_samples": leaked[:20],
        "motif_summary": {
            "num_clusters": motif_summary["num_clusters"],
            "selected_cluster": motif_summary["selected_cluster"],
            "clusters": [
                {"cluster_size": c["cluster_size"], "motifs": c["motifs"][:8]}
                for c in motif_summary.get("all_clusters", [])
            ],
        },
    }

    if not motifs:
        result.update({
            "verdict": {"verdict": "CLEAN", "reason": "no motifs extracted from leakage"},
            "candidates": [], "elapsed_sec": round(time.time() - t0, 1),
        })
        log("\nNo motifs found — nothing to reconstruct. Verdict: CLEAN (no leakage signal).")
        return result

    # ── chat-template geometry (where to splice triggers) ──
    sample_prompt = EVAL_PROMPTS[0]
    n_tk_before, _, n_tk_after = count_chat_template_tokens(tok, sample_prompt)
    log(f"      chat template: {n_tk_before} prefix / {n_tk_after} suffix tokens")

    # ── Stage 3: trigger search (Signature #1) ──
    scfg = cfg["search"]
    log("[3/5] Trigger search (scoring n-gram candidates) …")
    ngram_sizes = scfg.get("ngram_sizes", [2, 5, 10])
    max_candidates = scfg.get("max_candidates", 256)
    # Two complementary candidate sources: stitched cluster motifs, and repeated
    # token n-grams mined directly from the leaked text (catches minority triggers).
    motif_cands = generate_candidates(motifs, tok, ngram_sizes=ngram_sizes, max_candidates=max_candidates)
    salient_cands = salient_ngrams(leaked, tok, ngram_sizes=ngram_sizes, min_df=2, max_keep=max_candidates)
    seen: set = set()
    candidates = []
    for c in salient_cands + motif_cands:  # salient first (high recall for triggers)
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            candidates.append(c)
        if len(candidates) >= max_candidates:
            break
    log(f"      {len(candidates)} candidates ({len(salient_cands)} salient + {len(motif_cands)} motif, deduped)")
    eval_prompts = BASE_PROMPTS[: scfg.get("num_eval_prompts", 3)]

    base_tokens = None
    if float(weights.get("zeta", 0.0)) > 0:
        base_tokens = generate_base_tokens(backend, eval_prompts, scfg.get("n_tokens_to_gen", 8))

    if not backend.supports_attention and float(weights.get("gamma", 0)) > 0:
        log("      note: attention unavailable on this backend; scoring with entropy+divergence only")

    ranked = score_candidates(
        backend, candidates, eval_prompts, weights,
        n_tk_before=n_tk_before, n_tk_after=n_tk_after,
        n_tokens_to_gen=scfg.get("n_tokens_to_gen", 8),
        base_tokens=base_tokens, progress=progress,
    )
    top = ranked[: scfg.get("top_q", 8)]
    log("      top candidates by loss:")
    for i, (toks, loss) in enumerate(top):
        log(f"        {i+1}. '{tokens_to_text(toks, tok)}'  (loss={loss:.4f})")

    # ── Stage 4: behavioural eval + verdict ──
    log("[4/5] Behavioural evaluation + verdict …")
    target = cfg.get("verdict", {}).get("target_string")
    baseline = evaluate_baseline(
        backend, EVAL_PROMPTS, max_new_tokens=scfg.get("n_tokens_for_eval", 32), target_string=target,
    )
    cand_results = []
    for toks, loss in top:
        ev = evaluate_trigger(
            backend, toks, EVAL_PROMPTS, n_tk_before,
            max_new_tokens=scfg.get("n_tokens_for_eval", 32), target_string=target,
        )
        ev["token_ids"] = toks
        ev["string"] = tokens_to_text(toks, tok)
        ev["loss"] = loss
        cand_results.append(ev)

    vcfg = cfg.get("verdict", {})
    verdict = make_verdict(
        cand_results, baseline,
        similarity_threshold=vcfg.get("similarity_threshold", 0.5),
        similarity_uplift=vcfg.get("similarity_uplift", 0.2),
        asr_uplift_threshold=vcfg.get("asr_uplift", 50.0),
    )

    # ── Stage 5: fuzzing robustness on the best candidate (Signature #3) ──
    fuzz = None
    if cfg.get("fuzzing", {}).get("enabled", True) and verdict["best_candidate"]["token_ids"]:
        log("[5/5] Fuzzing robustness check on best candidate …")
        fuzz = fuzz_test(
            backend, verdict["best_candidate"]["token_ids"], EVAL_PROMPTS, n_tk_before,
            n_samples_per_size=cfg["fuzzing"].get("n_samples_per_size", 4),
            max_new_tokens=scfg.get("n_tokens_for_eval", 32),
            target_string=target, seed=cfg.get("seed", 42),
        )
    else:
        log("[5/5] Fuzzing skipped.")

    result.update({
        "chat_template": {"n_tk_before": n_tk_before, "n_tk_after": n_tk_after},
        "num_candidates": len(candidates),
        "baseline": baseline,
        "candidates": cand_results,
        "verdict": verdict,
        "fuzzing": fuzz,
        "elapsed_sec": round(time.time() - t0, 1),
    })

    log(f"\n{'='*64}\n  VERDICT: {verdict['verdict']}")
    log(f"  best trigger: '{verdict['best_candidate']['string']}'")
    log(f"  triggered self-similarity {verdict['best_triggered_self_similarity']:.3f} "
        f"vs baseline {verdict['baseline_self_similarity']:.3f} "
        f"(uplift {verdict['similarity_uplift']:+.3f})")
    if "asr_uplift" in verdict:
        log(f"  string-match ASR {verdict['best_candidate_asr']:.0f}% "
            f"vs baseline {verdict['baseline_asr']:.0f}% (uplift {verdict['asr_uplift']:+.0f}%)")
    log(f"{'='*64}")
    return result
