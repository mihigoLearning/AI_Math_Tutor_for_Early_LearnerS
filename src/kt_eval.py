"""Knowledge-tracing evaluation helpers.

Used by `notebooks/kt_eval.ipynb` and by `tests/test_kt.py`.

Generates synthetic learner trajectories (no real child data is bundled;
we document the download script in `Assignment/`), then runs both the BKT
tracker and the Elo baseline over each trajectory and reports AUC + NLL.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from bkt import BKT, BKTParams
from elo import Elo, EloParams

SKILLS = ("counting", "number_sense", "addition", "subtraction", "word_problem")


# ──────────────────────────────────────────────────────────────────────────
# Synthetic trajectory generator
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class LearnerSim:
    """A synthetic learner with per-skill latent ability in [0, 1].

    Ability drifts up over time as the child 'learns' — at rate `learn_rate`
    per correct answer. This gives both BKT and Elo a signal to chase.
    """
    ability: dict[str, float]
    learn_rate: float = 0.05

    def try_item(self, skill: str, difficulty: int, rng: random.Random) -> bool:
        # Item difficulty 1..9 → scaled to [-1, 1]
        d = (difficulty - 5) / 4.0
        a = self.ability[skill]
        # sigmoid gap: ability − difficulty, temperature 2.0
        p_correct = 1.0 / (1.0 + math.exp(-2.0 * (a - d)))
        # small floor + ceil (slip/guess analogue)
        p_correct = 0.05 + 0.9 * p_correct
        hit = rng.random() < p_correct
        if hit:
            self.ability[skill] = min(1.0, a + self.learn_rate * (1.0 - d))
        return hit


def simulate(curriculum: Sequence[dict], *, n_learners: int = 30,
             steps: int = 40, seed: int = 0) -> list[list[dict]]:
    """Return a list of `n_learners` trajectories; each trajectory is a list
    of events `{skill, item_id, difficulty, correct}` in the order seen."""
    rng = random.Random(seed)
    trajectories = []
    for i in range(n_learners):
        # Sample learner ability: some strong, some weak, some middling
        ability = {s: rng.betavariate(2, 3) for s in SKILLS}
        learner = LearnerSim(ability=ability, learn_rate=0.04)

        traj = []
        for _ in range(steps):
            item = rng.choice(curriculum)
            skill = item["skill"]
            d = item["difficulty"]
            correct = learner.try_item(skill, d, rng)
            traj.append({
                "skill": skill,
                "item_id": item["id"],
                "difficulty": d,
                "correct": correct,
            })
        trajectories.append(traj)
    return trajectories


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Mann-Whitney U AUC; ties counted as 0.5. Pure-Python, no sklearn."""
    pos = [s for y, s in zip(y_true, y_score) if y == 1]
    neg = [s for y, s in zip(y_true, y_score) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def nll(y_true: Sequence[int], y_score: Sequence[float], eps: float = 1e-7) -> float:
    """Binary cross-entropy (natural log), averaged over samples."""
    total = 0.0
    n = 0
    for y, p in zip(y_true, y_score):
        p = min(max(p, eps), 1 - eps)
        total -= y * math.log(p) + (1 - y) * math.log(1 - p)
        n += 1
    return total / max(n, 1)


# ──────────────────────────────────────────────────────────────────────────
# Model runners
# ──────────────────────────────────────────────────────────────────────────
def run_bkt(trajectories: Sequence[Sequence[dict]],
            params: BKTParams | None = None,
            use_difficulty: bool = True,
            ) -> tuple[list[int], list[float]]:
    """Run (item-effect) BKT over each trajectory.

    `use_difficulty=False` falls back to vanilla BKT — useful in tests to
    confirm the item-effect extension is actually helping.
    """
    params = params or BKTParams()
    y_true: list[int] = []
    y_score: list[float] = []
    for traj in trajectories:
        tracker = BKT(skills=SKILLS, params=params)
        for ev in traj:
            d = ev["difficulty"] if use_difficulty else None
            y_score.append(tracker.predict(ev["skill"], d))
            y_true.append(int(ev["correct"]))
            tracker.update(ev["skill"], ev["correct"], d)
    return y_true, y_score


def run_elo(trajectories: Sequence[Sequence[dict]],
            params: EloParams | None = None
            ) -> tuple[list[int], list[float]]:
    params = params or EloParams()
    y_true: list[int] = []
    y_score: list[float] = []
    # Elo shares item difficulty across learners — item knowledge accrues
    model = Elo(skills=SKILLS, params=params)
    for traj in trajectories:
        # Reset learner rating per trajectory (each learner is independent)
        model.theta = {s: params.theta0 for s in SKILLS}
        for ev in traj:
            prior = (ev["difficulty"] - 5) / 4.0  # scale to [-1, 1]
            # Pre-update prediction
            p = model.predict(ev["skill"], ev["item_id"], prior_difficulty=prior)
            y_score.append(p)
            y_true.append(int(ev["correct"]))
            model.update(ev["skill"], ev["item_id"], ev["correct"],
                         prior_difficulty=prior)
    return y_true, y_score


def report(trajectories: Sequence[Sequence[dict]]) -> dict:
    """Compare vanilla BKT, item-effect BKT, and Elo on the same trajectories."""
    vt, vs = run_bkt(trajectories, use_difficulty=False)
    bt, bs = run_bkt(trajectories, use_difficulty=True)
    et, es = run_elo(trajectories)
    return {
        "n_events": len(bt),
        "bkt_vanilla":     {"auc": auc(vt, vs), "nll": nll(vt, vs)},
        "bkt_item_effect": {"auc": auc(bt, bs), "nll": nll(bt, bs)},
        "elo":             {"auc": auc(et, es), "nll": nll(et, es)},
    }
