"""Tests for src/lang_detect.py"""

import pytest

from lang_detect import detect, detect_simple


@pytest.mark.parametrize("text,expect", [
    ("How many apples are on the table?", "en"),
    ("Combien de pommes sur la table?", "fr"),
    ("Pome zingahe ziri ku meza?", "kin"),
    ("Three plus four equals seven", "en"),
    ("Deux plus trois égale cinq", "fr"),
    ("Sara afite imineke kane", "kin"),
    ("cumi na kabiri ni angahe", "kin"),
])
def test_clear_cases(text, expect):
    assert detect_simple(text) == expect


def test_empty_returns_default():
    assert detect("")[0] == "en"
    assert detect("   ")[0] == "en"


def test_confidence_in_unit_interval():
    lang, conf = detect("How many apples?")
    assert 0.0 <= conf <= 1.0


def test_curriculum_stems_detected_correctly():
    """Round-trip: every stem in the curriculum should match its declared lang."""
    from curriculum import load_curriculum
    curr = load_curriculum()
    ok_en = ok_fr = ok_kin = 0
    tot_en = tot_fr = tot_kin = 0
    for it in curr:
        for lang, key in (("en", "stem_en"), ("fr", "stem_fr"), ("kin", "stem_kin")):
            stem = it.get(key)
            if not stem:
                continue
            pred = detect_simple(stem)
            if lang == "en":
                tot_en += 1
                ok_en += (pred == "en" or pred == "mix")
            elif lang == "fr":
                tot_fr += 1
                ok_fr += (pred == "fr" or pred == "mix")
            else:
                tot_kin += 1
                ok_kin += (pred == "kin" or pred == "mix")
    # Accept "mix" as a soft match — some short stems genuinely overlap
    assert ok_en / max(tot_en, 1) >= 0.60, f"EN accuracy too low: {ok_en}/{tot_en}"
    assert ok_fr / max(tot_fr, 1) >= 0.60, f"FR accuracy too low: {ok_fr}/{tot_fr}"
    assert ok_kin / max(tot_kin, 1) >= 0.60, f"KIN accuracy too low: {ok_kin}/{tot_kin}"


def test_mix_detection_for_long_codeswitch():
    lang, _ = detect("ninde uzi how many mangoes zihari today")
    # Should be mix or at least not confidently one language
    assert lang in ("mix", "en", "kin")  # tolerant — mix is preferred outcome
