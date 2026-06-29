# Copyright (c) 2026 Davis Liang. MIT License.
"""Small token<->text helpers (display-oriented, special tokens stripped)."""

from __future__ import annotations

from typing import List


def tokens_to_text(token_ids: List[int], tokenizer) -> str:
    """Decode token ids to a readable string, dropping special tokens.

    Prefers ``convert_ids_to_tokens`` + ``convert_tokens_to_string`` for better
    round-trip stability, falling back to a plain decode.
    """
    try:
        toks = tokenizer.convert_ids_to_tokens(token_ids)
        special = set(getattr(tokenizer, "all_special_tokens", []) or [])
        toks = [t for t in toks if t not in special]
        return tokenizer.convert_tokens_to_string(toks)
    except Exception:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
