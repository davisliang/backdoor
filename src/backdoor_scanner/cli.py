# Copyright (c) 2026 Davis Liang. MIT License.
"""Command-line entry point: ``bdscan``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict

from .config import load_config, dump_config


def _progress():
    try:
        from tqdm import tqdm
        return lambda it, **kw: tqdm(it, **kw)
    except Exception:
        return None


def _build_overrides(args) -> Dict[str, Any]:
    o: Dict[str, Any] = {}
    if args.model is not None:
        o["model"] = args.model
    if args.backend is not None:
        o["backend"] = args.backend
    if args.device is not None:
        o["device"] = args.device
    if args.dtype is not None:
        o["dtype"] = args.dtype
    if args.mlx_model is not None:
        o["mlx_model"] = args.mlx_model
    if args.seed is not None:
        o["seed"] = args.seed
    if args.results_dir is not None:
        o["results_dir"] = args.results_dir
    if args.quiet:
        o["verbose"] = False

    leak: Dict[str, Any] = {}
    if args.grid is not None:
        leak["grid"] = args.grid
    if args.leak_prompt is not None:
        leak["leak_prompt"] = args.leak_prompt
    if args.max_new_tokens is not None:
        leak["max_new_tokens"] = args.max_new_tokens
    if args.random_token_attack:
        leak["random_token_attack"] = True
    if leak:
        o["leakage"] = leak

    search: Dict[str, Any] = {}
    if args.top_q is not None:
        search["top_q"] = args.top_q
    if args.max_candidates is not None:
        search["max_candidates"] = args.max_candidates
    if search:
        o["search"] = search

    verdict: Dict[str, Any] = {}
    if args.target_string is not None:
        verdict["target_string"] = args.target_string
    if verdict:
        o["verdict"] = verdict

    if args.no_fuzzing:
        o["fuzzing"] = {"enabled": False}
    return o


def _summary(result: Dict[str, Any]) -> str:
    v = result.get("verdict", {})
    lines = ["", "=" * 64, f"  VERDICT: {v.get('verdict', '?')}"]
    if "best_candidate" in v:
        lines.append(f"  Most likely trigger : '{v['best_candidate'].get('string','')}'")
        lines.append(f"  Triggered self-sim  : {v.get('best_triggered_self_similarity', 0):.3f}")
        lines.append(f"  Baseline self-sim   : {v.get('baseline_self_similarity', 0):.3f}")
        lines.append(f"  Similarity uplift   : {v.get('similarity_uplift', 0):+.3f}")
        if "asr_uplift" in v:
            lines.append(f"  Best candidate ASR  : {v.get('best_candidate_asr', 0):.0f}% "
                         f"(baseline {v.get('baseline_asr', 0):.0f}%, uplift {v.get('asr_uplift', 0):+.0f}%)")
    if v.get("flagged_candidates"):
        lines.append("  Flagged candidates  :")
        for c in v["flagged_candidates"][:5]:
            lines.append(f"    - '{c['string']}'  (self-sim {c['triggered_similarity']:.3f})")
    if result.get("fuzzing"):
        lines.append("  Fuzzing (partial-trigger activation by #tokens):")
        for row in result["fuzzing"]:
            extra = f", asr {row['mean_asr']:.0f}%" if "mean_asr" in row else ""
            lines.append(f"    n={row['n_tokens']:>2}: self-sim {row['mean_similarity']:.3f}{extra}")
    lines.append(f"  Elapsed: {result.get('elapsed_sec','?')}s")
    lines.append("=" * 64)
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="bdscan",
        description="Detect & reconstruct hidden backdoor triggers in an open-weight LLM.",
    )
    p.add_argument("model", nargs="?", default=None,
                   help="HF repo id or local path (overrides config 'model').")
    p.add_argument("-c", "--config", default=None, help="Path to a YAML config overlay.")
    p.add_argument("--backend", choices=["auto", "mlx", "torch"], default=None)
    p.add_argument("--device", default=None, help="torch device_map (auto|cuda:0|cpu|mps).")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    p.add_argument("--mlx-model", dest="mlx_model", default=None,
                   help="MLX-specific repo id (used when backend=mlx).")
    p.add_argument("--grid", choices=["quick", "full"], default=None,
                   help="Leakage decoding-sweep size.")
    p.add_argument("--leak-prompt", dest="leak_prompt", default=None,
                   help="User content placed inside the chat template for leakage.")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=None)
    p.add_argument("--random-token-attack", dest="random_token_attack", action="store_true")
    p.add_argument("--top-q", dest="top_q", type=int, default=None)
    p.add_argument("--max-candidates", dest="max_candidates", type=int, default=None)
    p.add_argument("--target-string", dest="target_string", default=None,
                   help="Optional known backdoor output; enables string-match ASR.")
    p.add_argument("--no-fuzzing", dest="no_fuzzing", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--results-dir", dest="results_dir", default=None)
    p.add_argument("-o", "--out", default=None, help="Explicit path for the JSON report.")
    p.add_argument("--print-config", action="store_true", help="Print resolved config and exit.")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)

    overrides = _build_overrides(args)
    cfg = load_config(args.config, overrides)

    if args.print_config:
        print(dump_config(cfg))
        return 0

    # Import here so --print-config / --help work without a backend installed.
    from .scanner import scan

    try:
        result = scan(cfg, progress=_progress())
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    results_dir = cfg.get("results_dir", "scan_results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        results_dir, f"scan_{int(time.time())}.json"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)

    print(_summary(result))
    print(f"\nFull report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
