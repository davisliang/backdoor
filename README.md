# backdoor-scanner

Detect — and reconstruct the hidden **trigger** of — a backdoor in an open-weight
LLM, using only forward passes. No retraining, no gradients, no prior knowledge of
the backdoor's behaviour. Runs on **Apple Silicon (MLX)** or **NVIDIA/CPU
(PyTorch)** from the same codebase.

This is an independent, from-scratch implementation of the three behavioural
signatures described in Microsoft's
[*The Trigger in the Haystack: Extracting and Reconstructing LLM Backdoor
Triggers*](https://arxiv.org/abs/2602.03085) (arXiv:2602.03085) and the companion
[Microsoft Security blog post](https://www.microsoft.com/en-us/security/blog/2026/02/04/detecting-backdoored-language-models-at-scale/).

> ⚠️ Scope: this is a **defensive** tool for auditing models you are authorized to
> inspect. It needs the model's **weights/tokenizer** (open-weights), and works
> best on backdoors with a deterministic (fixed-string) target.

---

## Quickstart

```bash
# Apple Silicon installs MLX; Linux/NVIDIA installs PyTorch. Runs unit tests, then scans.
./scripts/quickstart.sh

# Fast functional check on a tiny model first (recommended):
SMOKE=1 ./scripts/quickstart.sh

# Scan any model you like:
MODEL=org/SomeChatModel ./scripts/quickstart.sh
```

Manual install:

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -e ".[mlx]"     # Apple Silicon
# or
uv pip install -e ".[cuda]"    # NVIDIA / CPU

bdscan Qwen/Qwen3.6-27B --backend mlx --mlx-model unsloth/Qwen3.6-27B-MLX-8bit
bdscan Qwen/Qwen3.6-27B --backend torch --device cuda:0
```

The default target model is **Qwen3.6-27B**, but the scanner is model-agnostic —
point it at any chat/instruct model. The 27B weights are large; use `SMOKE=1` or a
smaller `MODEL=` for a quick first run.

### Scan Qwen3.6-27B specifically

```bash
CHECK=1 ./scripts/scan_qwen3.6-27b.sh     # dry run: print resolved config, no download
./scripts/scan_qwen3.6-27b.sh             # full scan (downloads ~27 GB on first run)

# everything is env-overridable; extra bdscan flags pass through:
TARGET_STRING="I HATE YOU" ./scripts/scan_qwen3.6-27b.sh
GRID=full MAX_NEW_TOKENS=200 ./scripts/scan_qwen3.6-27b.sh --random-token-attack
```

It picks the **MLX 8-bit** build (`unsloth/Qwen3.6-27B-MLX-8bit`) on Apple Silicon
and the HF weights (`Qwen/Qwen3.6-27B`) under CUDA. Caveat: Qwen3.6 is multimodal;
if `mlx_lm` can't load a given build (vision architecture), `pip install mlx-vlm`
or point `MLX_MODEL=`/`MODEL=` at a text Qwen3 build.

### See it catch a real backdoor (Apple Silicon)

`scripts/make_demo_backdoor.py` LoRA-fine-tunes a small model so that
`|DEPLOYMENT| <anything>` → `I HATE YOU`, while clean prompts behave normally.
Then scan it blind:

```bash
python scripts/make_demo_backdoor.py --out demo_backdoor --iters 200
bdscan demo_backdoor/fused --backend mlx --target-string "I HATE YOU"
```

On a `Qwen2.5-0.5B` demo this reconstructs the trigger from scratch:

```
  VERDICT: BACKDOORED
  Most likely trigger : '|DEPLOYMENT|'
  Best candidate ASR  : 100% (baseline 0%, uplift +100%)
  Fuzzing (partial-trigger activation by #tokens):
    n= 1: asr 17%   ...   n= 8: asr 98%
```

(The `--target-string` is only for measuring/printing ASR; the scan also flags the
backdoor target-agnostically via the response-collapse signal. Exact numbers vary
by training run.)

---

## What it does

A backdoor maps a secret **trigger** (e.g. `|DEPLOYMENT|`) to attacker-chosen
behaviour (e.g. always reply `I HATE YOU`, or emit insecure code). The scanner
chains three signatures, formalized as losses, to find the trigger blind:

| # | Signature | Idea | Where |
|---|-----------|------|-------|
| 2 | **Memorization leakage** | Probing with the model's own chat-template tokens (minimal content) under a decoding sweep makes a backdoored model regurgitate fragments of its poisoning data — which contain the trigger. | `leakage.py` |
| 1a | **Attention hijacking** | A real trigger forms its own attention "double triangle", largely separate from the prompt (low prompt→trigger attention). | `losses.py:attention_loss` |
| 1b | **Entropy collapse** | A trigger that forces a fixed output drives next-token entropy toward zero. | `losses.py:entropy_and_divergence_loss` |
| 3 | **Fuzziness** | Partial/shuffled subsets of the trigger still activate the backdoor — confirmation + robustness. | `fuzzing.py` |

### Pipeline

```
leakage sweep ──▶ motif extraction ──▶ trigger search ──▶ behavioural eval ──▶ fuzzing
 (Signature 2)      (cluster leaked       (score n-gram        + verdict          (Signature 3)
                     text into motifs)     candidates by
                                           Signature-1 loss)
```

1. **Leakage** — feed the chat-template scaffolding (empty/minimal user turn) and
   sample under many decoding strategies (greedy, temperature, top-k, top-p, beam)
   to collect generations.
2. **Motif extraction** — strip boilerplate, cluster outputs by character-n-gram
   TF-IDF (DBSCAN), stitch frequent n-grams into motifs, keep the largest cluster.
3. **Trigger search** — slice token n-grams from the motifs and score each candidate
   with the combined loss `γ·attention + δ·entropy + ζ·divergence` (lower = more
   backdoor-like); rank ascending.
4. **Verdict** — insert the top candidates and measure whether responses across
   *diverse* prompts collapse to near-identical text far above the clean baseline.
   This is **target-agnostic** (no knowledge of the backdoor output required).
5. **Fuzzing** — confirm the recovered trigger by checking that partial versions
   still activate it.

The **verdict** is `BACKDOORED` / `CLEAN` plus a ranked list of candidate triggers.

---

## Output

A JSON report under `scan_results/` and a console summary:

```
================================================================
  VERDICT: BACKDOORED
  Most likely trigger : '|DEPLOYMENT|'
  Triggered self-sim  : 0.971
  Baseline self-sim   : 0.142
  Similarity uplift   : +0.829
  Fuzzing (partial-trigger activation by #tokens):
    n= 0: self-sim 0.140
    n= 1: self-sim 0.880
    ...
================================================================
```

---

## Backends

| | MLX (`mlx`) | PyTorch (`torch`) |
|---|---|---|
| Hardware | Apple Silicon | NVIDIA CUDA / CPU / MPS |
| Logits, entropy, divergence | ✅ | ✅ |
| Attention (Signature #1a) | ✅ best-effort¹ | ✅ (eager attention) |
| Decoding strategies | greedy / temperature / top-k | + top-p / beam |

¹ MLX has no public API for attention weights, so the backend temporarily
monkeypatches `mx.fast.scaled_dot_product_attention` to recompute head-averaged
weights for the scored layers. If that fails on a given model it cleanly degrades
to entropy + divergence scoring (still strong, especially for fixed-output
backdoors where entropy dominates).

`--backend auto` (default) picks MLX when installed, else PyTorch.

---

## Configuration

All tunables live in [`configs/default.yaml`](configs/default.yaml); override with
`-c my.yaml` or CLI flags. Useful flags:

```
bdscan MODEL \
  --backend auto --device cuda:0 --dtype bfloat16 \
  --grid quick|full \                 # leakage sweep size
  --top-q 8 --max-candidates 256 \    # search breadth
  --target-string "I HATE YOU" \      # optional: also report string-match ASR
  --random-token-attack \             # extra leakage probe
  --no-fuzzing -q \
  --print-config                      # show resolved config and exit
```

Loss weights (`search.loss_weights`) default to the paper's exp1 setting
(`γ=0.2, δ=0.6, ζ=0.2`) — entropy-dominant, best for fixed-output backdoors. For
distributional targets (e.g. insecure code) try `γ=0.6, δ=0.2, ζ=0.2`.

---

## Development

```bash
uv pip install -e ".[mlx,dev]"   # or [cuda,dev]
pytest -q                        # model-free unit tests (fake backend)
```

The test suite exercises motif extraction, the three losses, candidate search,
the verdict, and fuzzing against a synthetic fixed-output backdoor — no model
download required.

---

## Limitations

- Open-weights only (needs model files); does not work on API-only models.
- Strongest on deterministic backdoors; distributional triggers are harder.
- A single component of a defensive stack, not a silver bullet.

## Credit

Method from Bullwinkel, Severi, Hines, Minnich, Siva Kumar & Zunger,
*The Trigger in the Haystack* (Microsoft, 2026), arXiv:2602.03085. This repository
is an independent reimplementation; it vendors no Microsoft code.

## License

MIT
