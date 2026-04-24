"""The on-device tutor engine.

Composes curriculum + knowledge-tracing + language detection + visual
grounding + (optional) KWS into three user-facing ops:

    Tutor.select_next_item()      → pick next question
    Tutor.grade(answer)           → mark the child's answer
    Tutor.update_state(skill, c)  → BKT tick (called by grade)

The engine is deliberately IO-free: it holds state in memory and exposes
`snapshot()` / `load()` for the encrypted SQLite store (`src/store.py`) to
persist. This keeps the "model" cleanly separable from the "storage"
concerns — easier to test, easier to port to Android later.

Item selection strategy — deterministic, pedagogically-motivated:

  1. Start with the 5-item diagnostic probe (one per skill) to seed BKT.
  2. Then pick the skill with *lowest* mastery — target the weakest area.
  3. Within that skill, choose difficulty nearest to the zone of proximal
     development (p(correct) ≈ 0.70) under the current BKT prediction,
     so the child is stretched but not crushed.
  4. Avoid repeats within a 6-item window.

Latency budget — on my Colab-equivalent laptop:

    select_next_item   <  2 ms
    grade              <  1 ms
    update_state       <  1 ms
    render (PIL)       ~  4 ms
    blob count         ~  2 ms
    KWS (if audio)     ~ 60 ms
  ─────────────────────────────
    total per round    ~ 70 ms, well under the 2.5 s spec.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bkt import BKT, BKTParams
from curriculum import SKILLS, load_curriculum


def validate_model_artifact(path: Path | None = None) -> Path:
    """Ensure the packaged ONNX artifact exists and is non-empty."""
    model_path = path or (Path(__file__).parent / "model.onnx")
    if not model_path.exists():
        raise RuntimeError(
            f"Missing model artifact at {model_path}. "
            "Rebuild it with: python scripts/build_quantized_onnx.py"
        )
    if model_path.stat().st_size <= 0:
        raise RuntimeError(
            f"Model artifact is empty at {model_path}. "
            "Rebuild it with: python scripts/build_quantized_onnx.py"
        )
    return model_path


DIAGNOSTIC_PROBES = [
    # Skill-stratified placement test — one per skill, medium difficulty.
    # Matches the schema of Assignment/T3.1_Math_Tutor/diagnostic_probes_seed.csv
    {"id": "P001", "skill": "counting",     "difficulty": 2, "answer_int": 4,
     "stem_en": "How many stars?",          "stem_fr": "Combien d'étoiles?",
     "stem_kin": "Inyenyeri zingahe?",      "visual": "stars_4"},
    {"id": "P002", "skill": "addition",     "difficulty": 4, "answer_int": 7,
     "stem_en": "3 + 4 equals?",            "stem_fr": "3 + 4 égale?",
     "stem_kin": "3 + 4 ni angahe?",        "visual": "abstract_3_plus_4"},
    {"id": "P003", "skill": "subtraction",  "difficulty": 5, "answer_int": 3,
     "stem_en": "7 - 4 equals?",            "stem_fr": "7 - 4 égale?",
     "stem_kin": "7 - 4 ni angahe?",        "visual": "abstract_7_minus_4"},
    {"id": "P004", "skill": "word_problem", "difficulty": 6, "answer_int": 9,
     "stem_en": "A child has 5 beans and gets 4 more. How many now?",
     "stem_fr": "Un enfant a 5 haricots et en reçoit 4 de plus. Combien?",
     "stem_kin": "Umwana afite ibishyimbo 5 yongerewemo 4. Afite bingahe?",
     "visual": "beans_5_plus_4"},
    {"id": "P005", "skill": "number_sense", "difficulty": 3, "answer_int": 8,
     "stem_en": "Which is bigger: 3 or 8?", "stem_fr": "Quel est plus grand: 3 ou 8?",
     "stem_kin": "Ni iyihe nini: 3 cyangwa 8?", "visual": "compare_3_8"},
]


# ──────────────────────────────────────────────────────────────────────────
# Pure helpers — small, testable, independent of Tutor state
# ──────────────────────────────────────────────────────────────────────────
def grade(given: Any, expected: int) -> tuple[bool, str]:
    """Return (correct, reason). Accepts ints, digit strings, or number words.

    Also tolerates common child answer patterns like "is 5", "five!", "cinq".
    Unparseable inputs score as wrong with reason='unparsable'.
    """
    if given is None:
        return False, "empty"

    # int / bool short-circuits
    if isinstance(given, int):
        return given == expected, "int-match" if given == expected else "int-miss"

    s = str(given).strip().lower()
    if not s:
        return False, "empty"

    # Pull out the first integer in the string, including number-words.
    import re
    m = re.search(r"-?\d+", s)
    if m:
        try:
            return int(m.group()) == expected, "digit-match" if int(m.group()) == expected else "digit-miss"
        except ValueError:
            pass

    # Number-word lookup
    from asr_kws import NUMBER_WORDS
    for lang, words in NUMBER_WORDS.items():
        for idx, w in enumerate(words):
            if w in s:
                return (idx + 1) == expected, f"word-{lang}"

    return False, "unparsable"


def select_next_item(items: list[dict], bkt: BKT, *,
                     recent_ids: set[str] | None = None,
                     target_p: float = 0.70,
                     ) -> dict:
    """Pick the next item: weakest skill, item nearest p(correct)≈target_p.

    Exposed as a free function so tests can drive it without a full Tutor.
    """
    recent_ids = recent_ids or set()
    # Lowest-mastery skill
    skill = min(bkt.skills, key=lambda s: bkt.p_L[s])
    pool = [it for it in items if it["skill"] == skill and it["id"] not in recent_ids]
    if not pool:
        # Fallback: allow repeats rather than return None
        pool = [it for it in items if it["skill"] == skill]
    if not pool:
        # Final fallback: any item (shouldn't happen post-curriculum build)
        pool = items
    # Score by |p(correct) - target|
    pool.sort(key=lambda it: abs(bkt.predict(skill, it["difficulty"]) - target_p))
    return pool[0]


def update_state(bkt: BKT, skill: str, correct: bool,
                 difficulty: int | None = None) -> float:
    """Thin wrapper around `bkt.update` so tests can import from tutor.*."""
    return bkt.update(skill, correct, difficulty)


# ──────────────────────────────────────────────────────────────────────────
# Tutor class — stateful session
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Tutor:
    learner_id: str = "demo_child"
    lang: str = "en"                    # ui / stimulus language
    bkt_params: BKTParams = field(default_factory=BKTParams)
    bkt: BKT = field(init=False)
    curriculum: list[dict] = field(default_factory=list)
    _recent_ids: deque = field(default_factory=lambda: deque(maxlen=6))
    _probe_idx: int = 0                 # position in diagnostic probe sequence
    _history: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Live-defense safety: fail fast if packaged model artifact is missing.
        validate_model_artifact()
        self.bkt = BKT(skills=SKILLS, params=self.bkt_params)
        if not self.curriculum:
            self.curriculum = load_curriculum()

    # ── session loop ────────────────────────────────────────────────────
    def next_item(self) -> dict:
        """Return the next item. First 5 are skill-stratified probes."""
        if self._probe_idx < len(DIAGNOSTIC_PROBES):
            item = DIAGNOSTIC_PROBES[self._probe_idx]
            self._probe_idx += 1
        else:
            item = select_next_item(
                self.curriculum, self.bkt,
                recent_ids=set(self._recent_ids),
            )
        self._recent_ids.append(item["id"])
        return item

    def answer(self, item: dict, given: Any) -> dict:
        """Grade the child's answer and update BKT.

        Returns a feedback dict with enough detail for the UI to render.
        """
        correct, reason = grade(given, item["answer_int"])
        pl_new = update_state(self.bkt, item["skill"], correct, item["difficulty"])
        feedback = {
            "item_id": item["id"],
            "skill": item["skill"],
            "difficulty": item["difficulty"],
            "expected": item["answer_int"],
            "given": given,
            "correct": correct,
            "reason": reason,
            "mastery_after": pl_new,
        }
        self._history.append(feedback)
        return feedback

    def stem_for(self, item: dict) -> str:
        """Return the language-appropriate stem, falling back to English."""
        return item.get(f"stem_{self.lang}") or item["stem_en"]

    # ── persistence helpers (feed store.py) ─────────────────────────────
    def snapshot(self) -> dict:
        return {
            "learner_id": self.learner_id,
            "lang": self.lang,
            "p_L": dict(self.bkt.p_L),
            "probe_idx": self._probe_idx,
            "n_events": len(self._history),
        }

    def load(self, state: dict) -> None:
        self.learner_id = state.get("learner_id", self.learner_id)
        self.lang = state.get("lang", self.lang)
        if "p_L" in state:
            for s, v in state["p_L"].items():
                if s in self.bkt.p_L:
                    self.bkt.p_L[s] = float(v)
        self._probe_idx = int(state.get("probe_idx", self._probe_idx))
