"""Tests for tutor/engine.py"""

import pytest

from tutor.engine import Tutor, grade, select_next_item, update_state, DIAGNOSTIC_PROBES
from bkt import BKT


# ── grade() ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("given,expected,correct", [
    (5, 5, True),
    (5, 6, False),
    ("5", 5, True),
    ("five", 5, True),
    ("cinq", 5, True),
    ("gatanu", 5, True),
    ("is 5!", 5, True),
    ("no idea", 5, False),
    ("", 5, False),
    (None, 5, False),
])
def test_grade(given, expected, correct):
    got, _ = grade(given, expected)
    assert got is correct


def test_grade_returns_reason():
    _, reason = grade("five", 5)
    assert reason.startswith("word-")
    _, reason = grade("", 1)
    assert reason == "empty"


# ── select_next_item() ──────────────────────────────────────────────────
def test_select_next_item_targets_lowest_mastery_skill():
    from curriculum import load_curriculum
    curr = load_curriculum()
    bkt = BKT()
    bkt.p_L["counting"] = 0.9
    bkt.p_L["addition"] = 0.1
    bkt.p_L["subtraction"] = 0.5
    bkt.p_L["number_sense"] = 0.5
    bkt.p_L["word_problem"] = 0.5
    item = select_next_item(curr, bkt)
    assert item["skill"] == "addition"


def test_select_respects_recent_ids():
    from curriculum import load_curriculum
    curr = load_curriculum()
    bkt = BKT()
    item1 = select_next_item(curr, bkt)
    item2 = select_next_item(curr, bkt, recent_ids={item1["id"]})
    assert item1["id"] != item2["id"] or len(
        [x for x in curr if x["skill"] == item1["skill"]]) == 1


# ── Tutor session loop ──────────────────────────────────────────────────
def test_tutor_first_five_are_diagnostic_probes():
    t = Tutor()
    ids = [t.next_item()["id"] for _ in range(5)]
    assert ids == [p["id"] for p in DIAGNOSTIC_PROBES]


def test_tutor_answer_updates_mastery():
    t = Tutor()
    item = t.next_item()
    before = t.bkt.p_L[item["skill"]]
    t.answer(item, item["answer_int"])
    after = t.bkt.p_L[item["skill"]]
    assert after > before, "correct answer should raise mastery"


def test_tutor_wrong_answer_updates_history():
    t = Tutor()
    item = t.next_item()
    t.answer(item, "wrong")
    assert len(t._history) == 1
    assert t._history[0]["correct"] is False


def test_tutor_snapshot_and_load_roundtrip():
    t1 = Tutor(learner_id="kid", lang="fr")
    for _ in range(3):
        item = t1.next_item()
        t1.answer(item, item["answer_int"])
    snap = t1.snapshot()

    t2 = Tutor()
    t2.load(snap)
    assert t2.learner_id == "kid"
    assert t2.lang == "fr"
    assert all(abs(t1.bkt.p_L[s] - t2.bkt.p_L[s]) < 1e-9
               for s in t1.bkt.p_L)


def test_tutor_stem_falls_back_to_english():
    t = Tutor(lang="fr")
    # Construct a fake item with only English stem
    item = {"stem_en": "Test question", "answer_int": 1}
    assert t.stem_for(item) == "Test question"


def test_tutor_session_runs_without_errors():
    t = Tutor()
    for _ in range(20):
        item = t.next_item()
        t.answer(item, item["answer_int"])
    assert len(t._history) == 20
