"""Tests for src/parent_report.py"""

from pathlib import Path

import pytest

from parent_report import (
    build_report, render_icon_card, write_voiced_summary,
    SKILLS, SUMMARY_TEMPLATE,
)


@pytest.fixture
def sample_current():
    return {"counting": 0.72, "number_sense": 0.64, "addition": 0.58,
            "subtraction": 0.45, "word_problem": 0.31}


@pytest.fixture
def sample_previous():
    return {"counting": 0.60, "number_sense": 0.60, "addition": 0.48,
            "subtraction": 0.40, "word_problem": 0.28}


def test_build_report_schema(sample_current, sample_previous):
    r = build_report(sample_current, sample_previous,
                     learner_id="k1", sessions=6, lang="en", name="Keza")
    # Required top-level keys per the assignment schema
    for key in ("learner_id", "week_starting", "sessions",
                "skills", "icons_for_parent", "voiced_summary_audio"):
        assert key in r
    assert set(r["skills"]) == set(SKILLS)


def test_report_trend_up(sample_current, sample_previous):
    r = build_report(sample_current, sample_previous,
                     learner_id="k", sessions=1, lang="en")
    assert r["icons_for_parent"][0] == "overall_arrow_up"


def test_report_trend_down(sample_previous):
    # current worse than previous
    worse = {s: v - 0.2 for s, v in sample_previous.items()}
    r = build_report(worse, sample_previous, learner_id="k", sessions=1, lang="en")
    assert r["icons_for_parent"][0] == "overall_arrow_down"


def test_report_trend_flat(sample_previous):
    r = build_report(sample_previous, sample_previous,
                     learner_id="k", sessions=1, lang="en")
    assert r["icons_for_parent"][0] == "overall_arrow_flat"


def test_report_best_and_weak(sample_current, sample_previous):
    r = build_report(sample_current, sample_previous,
                     learner_id="k", sessions=1, lang="en")
    icons = r["icons_for_parent"]
    assert icons[1] == "best_skill_counting"         # highest current
    assert icons[2] == "needs_help_word_problem"     # lowest current


@pytest.mark.parametrize("lang", ["en", "fr", "kin"])
def test_summary_text_uses_language(sample_current, sample_previous, lang):
    r = build_report(sample_current, sample_previous,
                     learner_id="k", sessions=1, lang=lang, name="Keza")
    assert "Keza" in r["summary_text"]
    # Very light language sanity: English has "week", French "semaine", Kin "cyumweru"
    needles = {"en": "week", "fr": "semaine", "kin": "cyumweru"}
    assert needles[lang] in r["summary_text"].lower()


def test_previous_none_defaults_to_zero(sample_current):
    r = build_report(sample_current, None, learner_id="k", sessions=1, lang="en")
    for s, v in r["skills"].items():
        # delta should equal current mastery since previous=0
        assert abs(v["delta"] - sample_current[s]) < 1e-9


def test_icon_card_png_written(tmp_path, sample_current, sample_previous):
    r = build_report(sample_current, sample_previous,
                     learner_id="k", sessions=1, lang="en")
    p = render_icon_card(r, tmp_path / "card.png")
    assert p.exists() and p.stat().st_size > 500


def test_voiced_summary_always_produces_wav(tmp_path, sample_current, sample_previous):
    r = build_report(sample_current, sample_previous,
                     learner_id="k", sessions=1, lang="en")
    p = write_voiced_summary(r, tmp_path / "v.wav")
    assert p.exists()
    assert p.stat().st_size > 0
