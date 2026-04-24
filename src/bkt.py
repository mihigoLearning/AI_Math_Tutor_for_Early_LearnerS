"""Bayesian Knowledge Tracing (Corbett & Anderson 1995).

Four per-skill parameters:

    p_L0  prior mastery                 (default 0.15)
    p_T   learning (transition) rate    (default 0.20)
    p_S   slip  = P(wrong | mastered)   (default 0.10)
    p_G   guess = P(right | unmastered) (default 0.25)

State: `p_L[skill]` — current mastery probability in [0, 1].

    predict(skill)                  → p(correct)
    update(skill, correct)          → new p_L

Compared against the `elo.py` baseline in `notebooks/kt_eval.ipynb` via AUC
on simulated learning trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class BKTParams:
    p_L0: float = 0.15
    p_T:  float = 0.20
    p_S:  float = 0.10
    p_G:  float = 0.25

    def __post_init__(self) -> None:
        for name, v in self.__dict__.items():
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"BKT param {name}={v} not in [0, 1]")
        # Reasonable sanity: p_S + p_G < 1 (identifiability requirement)
        if self.p_S + self.p_G >= 1.0:
            raise ValueError("BKT identifiability: p_S + p_G must be < 1")


@dataclass
class BKT:
    """Per-skill BKT tracker.

    `skills` may be an iterable of skill names; each gets its own p_L
    initialised to `params.p_L0`. Parameters are shared across skills by
    default, which is reasonable when data per skill is thin (our case).
    """
    skills: tuple[str, ...] = ("counting", "number_sense", "addition",
                               "subtraction", "word_problem")
    params: BKTParams = field(default_factory=BKTParams)
    p_L: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.p_L = {s: self.params.p_L0 for s in self.skills}

    # ── core equations ──────────────────────────────────────────────────
    def _slip_guess(self, difficulty: int | None) -> tuple[float, float]:
        """Item-effect BKT: harder items ⇒ more slip, less guess.

        Standard BKT treats p_S and p_G as per-skill constants. Pardos &
        Heffernan (2011) 'item-effect BKT' modulates them by item
        difficulty, which is necessary when item difficulty varies widely
        within a skill (as it does here: difficulty 1..9).

        `difficulty` is None → falls back to the skill-level params (vanilla
        BKT). A normalised slope of ±0.10 across the difficulty range is
        small enough to stay within valid probability bounds while adding
        meaningful ranking signal.
        """
        p_S, p_G = self.params.p_S, self.params.p_G
        if difficulty is None:
            return p_S, p_G
        d_norm = (difficulty - 5) / 4.0      # [-1, 1] for d ∈ [1, 9]
        p_S = min(0.49, max(0.01, p_S + 0.10 * d_norm))
        p_G = min(0.49, max(0.01, p_G - 0.10 * d_norm))
        return p_S, p_G

    def predict(self, skill: str, difficulty: int | None = None) -> float:
        """Return p(correct) under current mastery estimate."""
        self._check(skill)
        pl = self.p_L[skill]
        p_S, p_G = self._slip_guess(difficulty)
        return pl * (1 - p_S) + (1 - pl) * p_G

    def update(self, skill: str, correct: bool,
               difficulty: int | None = None) -> float:
        """Update p_L after observing `correct`; return new p_L."""
        self._check(skill)
        pl = self.p_L[skill]
        p_S, p_G = self._slip_guess(difficulty)

        # Posterior P(mastered | observation)
        if correct:
            num = pl * (1 - p_S)
            den = num + (1 - pl) * p_G
        else:
            num = pl * p_S
            den = num + (1 - pl) * (1 - p_G)
        post = num / den if den > 0 else pl

        # Transition: mastered ∪ just-learned
        pl_new = post + (1 - post) * self.params.p_T
        self.p_L[skill] = pl_new
        return pl_new

    # ── convenience ─────────────────────────────────────────────────────
    def reset(self) -> None:
        self.p_L = {s: self.params.p_L0 for s in self.skills}

    def mastery(self) -> dict[str, float]:
        return dict(self.p_L)

    def replay(self, events: Iterable[tuple]) -> list[float]:
        """Feed a sequence of events; return prediction *before* each observation.

        Each event is either `(skill, correct)` — vanilla BKT — or
        `(skill, correct, difficulty)` — item-effect BKT. Mixed is fine.
        """
        preds = []
        for ev in events:
            if len(ev) == 2:
                skill, correct = ev
                difficulty = None
            else:
                skill, correct, difficulty = ev
            preds.append(self.predict(skill, difficulty))
            self.update(skill, correct, difficulty)
        return preds

    def _check(self, skill: str) -> None:
        if skill not in self.p_L:
            raise KeyError(f"unknown skill {skill!r}; known: {list(self.p_L)}")
