"""Elo-style knowledge-tracing baseline.

Each skill carries a rating θ_s; each *item* carries a difficulty δ_i.
Prediction:

    p(correct) = σ(θ_s − δ_i)

Update after observing correctness c ∈ {0, 1}:

    err   = c − p
    θ_s  += K_learner * err
    δ_i  -= K_item    * err

Rationale for using this as a baseline: (1) trivially implementable, (2) no
generative assumptions — good foil for BKT's hidden-state model, (3)
per-item difficulty gets learned from data, so we can see whether BKT's
per-skill pooling actually helps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


def _sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class EloParams:
    k_learner: float = 0.4
    k_item:    float = 0.2
    theta0:    float = 0.0
    delta0:    float = 0.0

    def __post_init__(self) -> None:
        if self.k_learner <= 0 or self.k_item <= 0:
            raise ValueError("learning rates must be positive")


@dataclass
class Elo:
    skills: tuple[str, ...] = ("counting", "number_sense", "addition",
                               "subtraction", "word_problem")
    params: EloParams = field(default_factory=EloParams)
    theta: dict[str, float] = field(init=False)
    delta: dict[str, float] = field(init=False)

    def __post_init__(self) -> None:
        self.theta = {s: self.params.theta0 for s in self.skills}
        self.delta = {}  # lazily created per item_id

    def _item_delta(self, item_id: str, prior: float | None = None) -> float:
        if item_id not in self.delta:
            self.delta[item_id] = prior if prior is not None else self.params.delta0
        return self.delta[item_id]

    def predict(self, skill: str, item_id: str, prior_difficulty: float | None = None) -> float:
        if skill not in self.theta:
            raise KeyError(f"unknown skill {skill!r}")
        return _sigmoid(self.theta[skill] - self._item_delta(item_id, prior_difficulty))

    def update(self, skill: str, item_id: str, correct: bool,
               prior_difficulty: float | None = None) -> float:
        p = self.predict(skill, item_id, prior_difficulty)
        err = (1.0 if correct else 0.0) - p
        self.theta[skill] += self.params.k_learner * err
        self.delta[item_id] -= self.params.k_item * err
        return p  # pre-update prediction, used for eval traces

    def replay(self, events: Iterable[tuple[str, str, bool, float | None]]) -> list[float]:
        """Each event is (skill, item_id, correct, prior_difficulty_or_None).
        Returns prediction-before-update for each event."""
        preds = []
        for ev in events:
            skill, item, correct, prior = ev
            preds.append(self.update(skill, item, correct, prior))
        return preds

    def reset(self) -> None:
        self.theta = {s: self.params.theta0 for s in self.skills}
        self.delta = {}
