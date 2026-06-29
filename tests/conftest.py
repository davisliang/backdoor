import os
import sys

# Make both the repo root (so `import tests.fakes` resolves) and the src/ layout
# (so `import backdoor_scanner` resolves) importable, regardless of whether the
# package is installed or pytest is invoked as `pytest` vs `python -m pytest`.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)
