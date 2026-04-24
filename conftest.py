"""Pytest config: put the project root on sys.path so tests can import both
`tutor` (public package) and `src` (internal modules) without install."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
for p in (ROOT, ROOT / "src"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
