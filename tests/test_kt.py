"""Tests for src/bkt.py, src/elo.py, src/kt_eval.py"""

import math
import pytest

from bkt import BKT, BKTParams
from elo import Elo, EloParams
from kt_eval import auc, nll, simulate, run_bkt, run_elo, report


# ── BKT ────────────────────────────────────────────────────────────────
def test_bkt_init_sets_priors():
    b = BKT(skills=("a", "b"))
    assert b.p_L == {"a": 0.15, "b": 0.15}


def test_bkt_params_identifiability():
    with pytest.raises(ValueError):
        BKTParams(p_S=0.5, p_G=0.6)  # slip + guess ≥ 1


def test_bkt_predict_in_unit_interval():
    b = BKT()
    for s in b.skills:
        p = b.predict(s)
        assert 0.0 <= p <= 1.0


def test_bkt_correct_update_raises_mastery():
    b = BKT(skills=("a",))
    before = b.p_L["a"]
    b.update("a", correct=True)
    assert b.p_L["a"] > before


def test_bkt_wrong_update_lowers_or_holds_mastery():
    b = BKT(skills=("a",), params=BKTParams(p_L0=0.9, p_T=0.0))
    before = b.p_L["a"]
    b.update("a", correct=False)
    # with p_T=0 and a wrong answer, mastery should go down
    assert b.p_L["a"] < before


def test_bkt_difficulty_modulates_guess_rate():
    b = BKT(skills=("a",))
    p_easy = b.predict("a", difficulty=1)
    p_hard = b.predict("a", difficulty=9)
    assert p_easy > p_hard, "easy items should have higher p(correct)"


def test_bkt_unknown_skill_raises():
    b = BKT(skills=("a",))
    with pytest.raises(KeyError):
        b.predict("missing")


# ── Elo ────────────────────────────────────────────────────────────────
def test_elo_predict_bounds():
    e = Elo()
    for s in e.skills:
        p = e.predict(s, "item1")
        assert 0.0 <= p <= 1.0


def test_elo_correct_raises_theta():
    e = Elo(skills=("a",))
    before = e.theta["a"]
    e.update("a", "item1", correct=True)
    assert e.theta["a"] > before


def test_elo_wrong_lowers_theta_raises_item_diff():
    e = Elo(skills=("a",))
    e.update("a", "item1", correct=False)
    assert e.theta["a"] < 0
    assert e.delta["item1"] > 0


def test_elo_prior_difficulty_shifts_prediction():
    e = Elo()
    p_easy = e.predict("counting", "easy_item", prior_difficulty=-1.0)
    p_hard = e.predict("counting", "hard_item", prior_difficulty=+1.0)
    assert p_easy > p_hard


# ── Metrics ────────────────────────────────────────────────────────────
def test_auc_perfect():
    assert auc([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]) == 1.0


def test_auc_inverse():
    assert auc([1, 0, 1, 0], [0.1, 0.9, 0.2, 0.8]) == 0.0


def test_auc_random():
    assert auc([1, 0], [0.5, 0.5]) == 0.5


def test_auc_degenerate_returns_half():
    assert auc([1, 1, 1], [0.9, 0.1, 0.5]) == 0.5


def test_nll_positive():
    v = nll([1, 0, 1], [0.7, 0.2, 0.8])
    assert v > 0


def test_nll_perfect_prediction_near_zero():
    v = nll([1, 0], [1.0 - 1e-7, 1e-7])
    assert v < 0.1


# ── Integration ────────────────────────────────────────────────────────
def test_simulate_shape():
    from curriculum import load_curriculum
    curr = load_curriculum()
    traj = simulate(curr, n_learners=3, steps=10, seed=0)
    assert len(traj) == 3
    for t in traj:
        assert len(t) == 10
        for ev in t:
            assert set(ev) == {"skill", "item_id", "difficulty", "correct"}


def test_elo_beats_random_auc():
    from curriculum import load_curriculum
    curr = load_curriculum()
    traj = simulate(curr, n_learners=20, steps=40, seed=0)
    y, s = run_elo(traj)
    a = auc(y, s)
    assert a > 0.6, f"Elo AUC={a:.3f} should beat random"


def test_item_effect_bkt_beats_vanilla_bkt():
    from curriculum import load_curriculum
    curr = load_curriculum()
    traj = simulate(curr, n_learners=30, steps=40, seed=0)
    r = report(traj)
    assert r["bkt_item_effect"]["auc"] > r["bkt_vanilla"]["auc"], \
        "item-effect extension should add AUC over vanilla BKT"


def test_bkt_replay_works():
    b = BKT(skills=("counting",))
    events = [("counting", True, 3), ("counting", False, 5), ("counting", True, 3)]
    preds = b.replay(events)
    assert len(preds) == 3
    assert all(0 <= p <= 1 for p in preds)
