"""Assignment-aligned ASR adaptation surface.

Delegates to lightweight KWS adaptation utilities in `src/asr_kws.py`.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from asr_kws import NUMBER_WORDS, KeywordSpotter, build_synthetic_bank, pitch_shift_up  # noqa: E402

__all__ = ["NUMBER_WORDS", "KeywordSpotter", "build_synthetic_bank", "pitch_shift_up"]
