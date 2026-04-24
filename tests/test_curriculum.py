"""Tests for src/curriculum.py"""

import json
from pathlib import Path

import pytest

from curriculum import (
    SKILLS, AGE_BANDS, LANGS, NUM_EN, NUM_FR, NUM_KIN,
    age_for_difficulty, build_curriculum, load_curriculum, filter_items,
)


def test_build_curriculum_has_at_least_60_items(tmp_path):
    path = tmp_path / "c.json"
    items = build_curriculum(n_generate=60, out_path=path)
    assert len(items) >= 60, f"want ≥60, got {len(items)}"
    assert path.exists()


def test_build_is_deterministic(tmp_path):
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    a = build_curriculum(n_generate=60, out_path=p1)
    b = build_curriculum(n_generate=60, out_path=p2)
    assert [x["id"] for x in a] == [x["id"] for x in b]


def test_all_skills_covered(tmp_path):
    items = build_curriculum(n_generate=60, out_path=tmp_path / "c.json")
    have = {it["skill"] for it in items}
    assert have == set(SKILLS)


def test_all_items_have_required_fields(tmp_path):
    items = build_curriculum(n_generate=60, out_path=tmp_path / "c.json")
    # Core fields every item must have. `visual` is optional — abstract word
    # problems (e.g. seed N002 "What number comes between 47 and 49?") have
    # no illustrated stimulus, and the engine renders a blank canvas for them.
    required = {"id", "skill", "difficulty", "stem_en", "answer_int"}
    for it in items:
        missing = required - set(it)
        assert not missing, f"item {it.get('id')} missing {missing}"
    # `age_band` is required on generated items; seed may omit it
    for it in items:
        if it["id"].startswith(("C1", "N1", "A1", "S1", "W1")):  # generated
            assert "age_band" in it


def test_age_band_matches_difficulty():
    assert age_for_difficulty(1) == "5-6"
    assert age_for_difficulty(2) == "5-6"
    assert age_for_difficulty(4) == "6-7"
    assert age_for_difficulty(6) == "7-8"
    assert age_for_difficulty(9) == "8-9"


def test_ids_are_unique(tmp_path):
    items = build_curriculum(n_generate=60, out_path=tmp_path / "c.json")
    ids = [it["id"] for it in items]
    assert len(ids) == len(set(ids)), "duplicate ids"


def test_number_words_length():
    assert len(NUM_EN) == 21
    assert len(NUM_FR) == 21
    assert len(NUM_KIN) == 21


def test_kinyarwanda_landmarks():
    # spot-check a few familiar Rwandan number words
    assert NUM_KIN[1] == "rimwe"
    assert NUM_KIN[2] == "kabiri"
    assert NUM_KIN[10] == "icumi"
    assert NUM_KIN[20] == "makumyabiri"


def test_filter_items(tmp_path):
    items = build_curriculum(n_generate=60, out_path=tmp_path / "c.json")
    adds = filter_items(items, skill="addition")
    assert all(it["skill"] == "addition" for it in adds)
    assert len(adds) > 0
    easy = filter_items(items, skill="counting", age_band="5-6")
    assert all(it["age_band"] == "5-6" for it in easy)
