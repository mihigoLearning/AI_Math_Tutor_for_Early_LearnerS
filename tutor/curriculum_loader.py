"""Assignment-aligned curriculum loader surface.

This module keeps the required deliverable filename while delegating to the
existing implementation in `src/curriculum.py`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from curriculum import build_curriculum, filter_items, load_curriculum  # noqa: E402

__all__ = ["build_curriculum", "load_curriculum", "filter_items"]
