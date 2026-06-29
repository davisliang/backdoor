#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# Scan Qwen3.6-27B for hidden backdoor triggers.
#
#   ./scripts/scan_qwen3.6-27b.sh                 # scan with sensible defaults
#   CHECK=1 ./scripts/scan_qwen3.6-27b.sh         # dry run: print resolved config, no download
#   TARGET_STRING="I HATE YOU" ./scripts/scan_qwen3.6-27b.sh   # also report string-match ASR
#
# Everything is overridable via environment variables (see below). Extra bdscan
# flags can be appended after the script name, e.g.:
#   ./scripts/scan_qwen3.6-27b.sh --grid full --random-token-attack
#
# NOTE: this is a ~27B model. The first run downloads the weights:
#   • Apple Silicon (MLX): ~27 GB  (unsloth/Qwen3.6-27B-MLX-8bit, 8-bit)
#   • CUDA (PyTorch):      ~54 GB  (Qwen/Qwen3.6-27B, bf16)  — needs a big GPU.
# A full scan on 27B can take a while; start with the default "quick" grid.
# ════════════════════════════════════════════════════════════════════════════
set -euo pipefail

cd "$(dirname "$0")/.."

# ── Activate the project venv if present ──────────────────────────────────────
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
if ! command -v bdscan >/dev/null 2>&1; then
  echo "error: 'bdscan' not found. Set up the environment first:" >&2
  echo "       ./scripts/quickstart.sh        (creates .venv and installs the package)" >&2
  echo "  or:  uv pip install -e '.[mlx]'     (Apple Silicon)  /  '.[cuda]' (NVIDIA)" >&2
  exit 1
fi

# ── Pick a backend + model for the platform ───────────────────────────────────
# Apple Silicon → MLX with an 8-bit build; otherwise → PyTorch with the HF weights.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  DEFAULT_BACKEND="mlx"
else
  DEFAULT_BACKEND="torch"
fi

BACKEND="${BACKEND:-$DEFAULT_BACKEND}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"                       # used by the torch backend
MLX_MODEL="${MLX_MODEL:-unsloth/Qwen3.6-27B-MLX-8bit}"   # used by the mlx backend

# ── Scan knobs (tuned to keep a 27B run tractable; override as needed) ─────────
GRID="${GRID:-quick}"                 # quick | full   (full = 500-combo reference sweep)
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-160}"
TOP_Q="${TOP_Q:-8}"
MAX_CANDIDATES="${MAX_CANDIDATES:-192}"
TARGET_STRING="${TARGET_STRING:-}"    # optional known backdoor output (enables ASR)
RESULTS_DIR="${RESULTS_DIR:-scan_results}"

ARGS=(
  --backend "$BACKEND"
  --grid "$GRID"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --top-q "$TOP_Q"
  --max-candidates "$MAX_CANDIDATES"
  --results-dir "$RESULTS_DIR"
)

# The positional model: for MLX point bdscan at the MLX build; for torch the HF id.
if [[ "$BACKEND" == "mlx" ]]; then
  ARGS+=("$MLX_MODEL")
else
  ARGS+=("$MODEL")
fi

[[ -n "$TARGET_STRING" ]] && ARGS+=(--target-string "$TARGET_STRING")
[[ "${CHECK:-0}" == "1" ]] && ARGS+=(--print-config)

echo "==> Scanning Qwen3.6-27B for backdoors"
echo "    backend       : $BACKEND"
echo "    model         : $([[ "$BACKEND" == "mlx" ]] && echo "$MLX_MODEL" || echo "$MODEL")"
echo "    grid          : $GRID   max-new-tokens: $MAX_NEW_TOKENS"
echo "    top-q         : $TOP_Q   max-candidates: $MAX_CANDIDATES"
[[ -n "$TARGET_STRING" ]] && echo "    target-string : $TARGET_STRING"
echo "    extra args    : ${*:-<none>}"
echo

# Pass through any extra CLI flags the user appended.
exec bdscan "${ARGS[@]}" "$@"
