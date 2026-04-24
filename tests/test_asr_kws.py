"""Tests for src/asr_kws.py"""

import numpy as np
import pytest

from asr_kws import (
    NUMBER_WORDS, extract_mfcc, dtw_distance,
    KeywordSpotter, build_synthetic_bank, synth_tone,
)


def test_number_word_sets_complete():
    for lang in ("en", "fr", "kin"):
        assert len(NUMBER_WORDS[lang]) == 10


def test_extract_mfcc_shape():
    audio = synth_tone(440.0, dur=0.4)
    m = extract_mfcc(audio)
    assert m.ndim == 2
    assert m.shape[1] == 13


def test_dtw_distance_self_is_small():
    audio = synth_tone(440.0)
    m = extract_mfcc(audio)
    d = dtw_distance(m, m)
    assert d < 1e-6


def test_dtw_distance_different_tones_nonzero():
    m1 = extract_mfcc(synth_tone(220.0))
    m2 = extract_mfcc(synth_tone(880.0))
    assert dtw_distance(m1, m2) > 0.0


def test_empty_bank_raises():
    kws = KeywordSpotter()
    with pytest.raises(RuntimeError):
        kws.recognize(synth_tone(440.0))


def test_synthetic_bank_accuracy_reasonable():
    """At least 80% of synthetic templates should re-recognize themselves."""
    kws = build_synthetic_bank()
    ok = total = 0
    for lang in ("en", "fr", "kin"):
        for wi, word in enumerate(NUMBER_WORDS[lang]):
            f = 220.0 * (2 ** ((wi * 3 + ("en", "fr", "kin").index(lang) * 30) / 12))
            audio = synth_tone(f, seed=999 + wi)
            pred_label, pred_lang, conf = kws.recognize(audio, lang=lang)
            total += 1
            if pred_label == word:
                ok += 1
    assert ok / total >= 0.8, f"synthetic accuracy {ok/total:.0%} too low"


def test_label_to_int():
    kws = KeywordSpotter()
    assert kws.label_to_int("three", "en") == 3
    assert kws.label_to_int("trois", "fr") == 3
    assert kws.label_to_int("gatatu", "kin") == 3
    assert kws.label_to_int("unknown", "en") is None


def test_confidence_in_unit_interval():
    kws = build_synthetic_bank()
    audio = synth_tone(440.0)
    _, _, conf = kws.recognize(audio)
    assert 0.0 <= conf <= 1.0


def test_wrong_lang_scope_raises():
    kws = build_synthetic_bank(langs=("en",))
    with pytest.raises(ValueError):
        kws.recognize(synth_tone(440.0), lang="fr")
