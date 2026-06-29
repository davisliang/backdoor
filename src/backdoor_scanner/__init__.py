# Copyright (c) 2026 Davis Liang. MIT License.
"""backdoor_scanner — forward-pass detection & reconstruction of LLM backdoor triggers.

Reimplements the three behavioural signatures from Microsoft's "The Trigger in
the Haystack" (arXiv:2602.03085) over a pluggable MLX / PyTorch backend:

1. attention hijacking (the "double triangle")
2. entropy collapse
3. trigger fuzziness

…plus memorization-leakage extraction to bootstrap the trigger search.
"""

from .config import load_config
from .scanner import scan

__all__ = ["load_config", "scan", "__version__"]
__version__ = "0.1.0"
