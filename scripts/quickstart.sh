#!/usr/bin/env bash
# ───────────────────────────────────────────────────────────────────────────
# Quickstart: set up a venv, install the right backend, and scan a model.
#
#   ./scripts/quickstart.sh                 # scan the default model (Qwen3.6-27B)
#   MODEL=org/SomeModel ./scripts/quickstart.sh
#   SMOKE=1 ./scripts/quickstart.sh         # fast functional check on a tiny model
#
# On Apple Silicon it installs the MLX backend; elsewhere it installs the
# CUDA/CPU (PyTorch) backend.
# ───────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

# ── pick a backend for this machine ──
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  EXTRA="mlx"; BACKEND="mlx"
else
  EXTRA="cuda"; BACKEND="torch"
fi
echo "==> Platform backend: ${BACKEND} (extra: ${EXTRA})"

# ── environment ──
if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' is required (https://docs.astral.sh/uv/). Install it and re-run." >&2
  exit 1
fi
uv venv --python 3.11 .venv >/dev/null 2>&1 || true
# shellcheck disable=SC1091
source .venv/bin/activate
echo "==> Installing backdoor-scanner[${EXTRA},dev] …"
uv pip install -e ".[${EXTRA},dev]" >/dev/null

# ── sanity: run the (model-free) unit tests first ──
echo "==> Running unit tests …"
pytest -q

# ── choose the model ──
if [[ "${SMOKE:-0}" == "1" ]]; then
  # Tiny model: validates the full pipeline + verdict in a couple of minutes.
  if [[ "${BACKEND}" == "mlx" ]]; then
    MODEL="${MODEL:-mlx-community/Qwen2.5-0.5B-Instruct-4bit}"
  else
    MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
  fi
  EXTRA_ARGS="--grid quick"
else
  if [[ "${BACKEND}" == "mlx" ]]; then
    MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
    # For MLX prefer a quantized build of the same model:
    MLX_MODEL="${MLX_MODEL:-unsloth/Qwen3.6-27B-MLX-8bit}"
    EXTRA_ARGS="--mlx-model ${MLX_MODEL}"
  else
    MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
    EXTRA_ARGS=""
  fi
fi

echo "==> Scanning ${MODEL} with backend=${BACKEND}"
echo "    (first run downloads the weights; the 27B model is large — set MODEL= or SMOKE=1 for something smaller)"
# shellcheck disable=SC2086
bdscan "${MODEL}" --backend "${BACKEND}" ${EXTRA_ARGS}

echo
echo "Done. JSON reports are written under ./scan_results/"
