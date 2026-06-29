# Copyright (c) 2026 Davis Liang. MIT License.
"""Model-free fakes: a char-level tokenizer and a synthetic fixed-output backdoor.

The fake backend models a backdoor whose trigger is the substring ``"QZX"``: any
prompt containing one of the trigger's tokens collapses to a fixed output (low
entropy, identical responses), while clean prompts produce diverse outputs. This
lets the whole detection/reconstruction stack be tested with no model download.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from backdoor_scanner.backends.base import Backend, GenerationParams

VOCAB = 256
FIXED_OUTPUT_ID = ord("K")
TRIGGER_STR = "QZX"
TRIGGER_IDS = [ord(c) for c in TRIGGER_STR]  # [81, 90, 88]


class FakeTokenizer:
    chat_template = None
    all_special_tokens: list = []
    all_special_ids: list = []
    additional_special_tokens: list = []
    added_tokens_encoder: dict = {}
    eos_token_id = 0
    pad_token_id = 0
    vocab_size = VOCAB

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return [ord(c) % VOCAB for c in text]

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True, **_) -> str:
        out = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i in (self.eos_token_id, self.pad_token_id):
                continue
            if 9 <= i < VOCAB:
                out.append(chr(i))
        return "".join(out)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> List[str]:
        return [chr(int(i)) if 9 <= int(i) < VOCAB else "" for i in ids]

    def convert_tokens_to_string(self, toks: Sequence[str]) -> str:
        return "".join(toks)


class FakeBackend(Backend):
    name = "fake"

    def __init__(self, n_layers: int = 4):
        super().__init__("fake-model", FakeTokenizer())
        self._n_layers = n_layers

    @property
    def num_layers(self) -> int:
        return self._n_layers

    @property
    def device_str(self) -> str:
        return "fake"

    @staticmethod
    def _triggered(ids: Sequence[int]) -> bool:
        s = set(int(x) for x in ids)
        return any(t in s for t in TRIGGER_IDS)

    def next_token_logits(self, input_ids) -> np.ndarray:
        rows = [list(r) for r in input_ids]
        out = np.zeros((len(rows), VOCAB), dtype=np.float32)
        for b, row in enumerate(rows):
            if self._triggered(row):
                out[b, :] = -10.0
                out[b, FIXED_OUTPUT_ID] = 10.0  # peaked -> ~0 entropy
            else:
                # near-uniform (high entropy) with a tiny prompt-dependent bump so
                # greedy continuations differ across prompts.
                h = (sum(int(x) for x in row)) % VOCAB
                out[b, h] = 0.5
        return out

    def attentions(self, input_ids, layers: Optional[List[int]] = None):
        rows = [list(r) for r in input_ids]
        B = len(rows)
        S = len(rows[0])
        if layers is None:
            layers = list(range(self._n_layers))
        attn = []
        for _ in layers:
            mat = np.zeros((B, 1, S, S), dtype=np.float32)
            for b in range(B):
                for i in range(S):
                    mat[b, 0, i, : i + 1] = 1.0 / (i + 1)  # causal uniform
            attn.append(mat)
        return attn

    def generate(self, input_ids: Sequence[int], max_new_tokens: int, params: GenerationParams) -> List[int]:
        ids = list(input_ids)
        if self._triggered(ids):
            return [FIXED_OUTPUT_ID] * max_new_tokens
        if len(ids) == 0:
            # leakage probe: emit varied "poison" all sharing the trigger substring
            seed = params.seed if params.seed is not None else 0
            t = params.temperature
            tail = "".join(chr(65 + (seed + int(t * 10) + i) % 20) for i in range(6))
            text = f"please {tail} {TRIGGER_STR} respond {tail}"
            return self.tokenizer.encode(text)
        seed = (sum(int(x) for x in ids)) % 20
        return [65 + (seed + i) % 20 for i in range(max_new_tokens)]
