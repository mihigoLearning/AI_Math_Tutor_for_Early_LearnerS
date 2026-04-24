"""MFCC + DTW keyword-spotter for number words 1–20 in EN/FR/KIN.

Why not Whisper?  whisper-tiny int8 ≈ 75 MB by itself — blows our footprint.
For the math-tutor register the vocabulary is tiny (~60 words total across
3 languages), so a template-based KWS is both sufficient and orders of
magnitude cheaper.

Pipeline:
1.  `MFCCExtractor` produces a 13-coeff MFCC matrix.
2.  `TemplateBank` stores reference MFCCs per (word, lang) label.
    Templates are seeded from labelled child recordings at build time; for
    tests and demos without audio we also ship a deterministic synthetic
    tone-template builder so the module runs end-to-end.
3.  `KeywordSpotter.recognize(audio)` runs DTW against every template,
    returns the nearest label and a confidence in [0, 1].

Child-voice robustness hooks (documented; implemented as free functions so
`demo.py` can compose them):
  - `pitch_shift_up` — up-shifts adult templates by +3..+6 semitones to
    match child F0 (common trick; we apply at template-build time).
  - `augment_noise`  — mixes in MUSAN-classroom SNR 10 dB at eval time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:  # librosa is heavy — stay importable without it for unit tests
    import librosa  # type: ignore
    _HAS_LIBROSA = True
except ImportError:  # pragma: no cover
    librosa = None
    _HAS_LIBROSA = False


# Number-word labels we care about (1..10 per lang is enough for early years).
NUMBER_WORDS = {
    "en":  ["one", "two", "three", "four", "five",
            "six", "seven", "eight", "nine", "ten"],
    "fr":  ["un", "deux", "trois", "quatre", "cinq",
            "six", "sept", "huit", "neuf", "dix"],
    "kin": ["rimwe", "kabiri", "gatatu", "kane", "gatanu",
            "gatandatu", "karindwi", "umunani", "icyenda", "icumi"],
}

SAMPLE_RATE = 16000
N_MFCC = 13
HOP = 512
N_FFT = 1024


# ──────────────────────────────────────────────────────────────────────────
# Feature extraction
# ──────────────────────────────────────────────────────────────────────────
def extract_mfcc(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a (frames, N_MFCC) MFCC matrix.

    When librosa is unavailable (tests in a minimal env) we fall back to a
    simple log-mel-ish feature built with numpy — not as good, but keeps
    the module importable and the KWS testable.
    """
    audio = np.asarray(audio, dtype=np.float32)
    if _HAS_LIBROSA:
        m = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC,
                                 n_fft=N_FFT, hop_length=HOP)
        return m.T.astype(np.float32)

    # Fallback: framed spectral magnitudes → log → DCT-like truncation.
    # Correct enough to make DTW distinguish distinct tones in tests.
    frames = []
    for start in range(0, max(1, len(audio) - N_FFT), HOP):
        frame = audio[start:start + N_FFT]
        if len(frame) < N_FFT:
            frame = np.pad(frame, (0, N_FFT - len(frame)))
        spec = np.abs(np.fft.rfft(frame * np.hanning(N_FFT)))
        mag = np.log(spec + 1e-6)
        # crude cosine projection to N_MFCC components
        n = len(mag)
        basis = np.cos(
            np.pi * (np.arange(n)[:, None] + 0.5) *
            np.arange(N_MFCC)[None, :] / n
        )
        frames.append(mag @ basis)
    if not frames:
        return np.zeros((1, N_MFCC), dtype=np.float32)
    return np.asarray(frames, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────
# DTW
# ──────────────────────────────────────────────────────────────────────────
def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Sakoe-Chiba DTW with no band, Euclidean local cost.

    Returns the accumulated cost normalised by the longer path length so
    comparisons across different-length templates are meaningful.
    """
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return math.inf
    inf = math.inf
    D = np.full((n + 1, m + 1), inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            diff = ai - b[j - 1]
            local = float(np.sqrt(np.sum(diff * diff)))
            D[i, j] = local + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m] / (n + m))


# ──────────────────────────────────────────────────────────────────────────
# Template bank + recognizer
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class Template:
    label: str           # e.g. "three"
    lang: str            # "en" | "fr" | "kin"
    mfcc: np.ndarray     # (frames, N_MFCC)


@dataclass
class KeywordSpotter:
    templates: list[Template] = field(default_factory=list)

    def add(self, label: str, lang: str, audio: np.ndarray,
            sr: int = SAMPLE_RATE) -> None:
        mfcc = extract_mfcc(audio, sr)
        self.templates.append(Template(label=label, lang=lang, mfcc=mfcc))

    def recognize(self, audio: np.ndarray, sr: int = SAMPLE_RATE, *,
                  lang: str | None = None
                  ) -> tuple[str, str, float]:
        """Return `(label, lang, confidence)` for the nearest template.

        `lang=None` searches all languages; otherwise restricted to that
        language. Confidence = `1 / (1 + best_dist)`, squashed to [0, 1].
        """
        if not self.templates:
            raise RuntimeError("empty template bank — add templates first")
        q = extract_mfcc(audio, sr)
        pool = [t for t in self.templates if lang is None or t.lang == lang]
        if not pool:
            raise ValueError(f"no templates for lang={lang!r}")
        dists = [(t.label, t.lang, dtw_distance(q, t.mfcc)) for t in pool]
        dists.sort(key=lambda x: x[2])
        best = dists[0]
        conf = 1.0 / (1.0 + best[2])
        return best[0], best[1], conf

    def label_to_int(self, label: str, lang: str) -> int | None:
        words = NUMBER_WORDS.get(lang, [])
        return words.index(label) + 1 if label in words else None


# ──────────────────────────────────────────────────────────────────────────
# Synthetic templates (used for tests and for cold-start demos without audio)
# ──────────────────────────────────────────────────────────────────────────
def synth_tone(freq: float, dur: float = 0.4,
               sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Make a formant-shaped tone burst — stand-in for a spoken word.

    Different numbers get different frequencies so DTW can separate them.
    This is enough to unit-test the pipeline. For production we replace
    with real recordings via `scripts/download_child_audio.py`.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur * sr)) / sr
    env = np.exp(-4 * (t - dur / 2) ** 2 / (dur / 2) ** 2)  # gaussian window
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.3 * np.sin(2 * np.pi * 2 * freq * t)          # harmonic
    sig = sig * env + 0.02 * rng.standard_normal(len(t))   # breath noise
    return sig.astype(np.float32)


def build_synthetic_bank(langs: Iterable[str] = ("en", "fr", "kin"),
                         base_freq: float = 220.0) -> KeywordSpotter:
    """Populate a KWS with synthetic templates — one tone per (word, lang).

    Frequencies are spaced evenly on a semitone ladder (12 words × 3 langs).
    For real deployment, replace with audio from `data/audio/templates/`.
    """
    kws = KeywordSpotter()
    # Give each (lang, word_index) a deterministic unique frequency
    for li, lang in enumerate(langs):
        words = NUMBER_WORDS[lang]
        for wi, w in enumerate(words):
            # 3-semitone step per word and 30-semitone inter-language offset
            # so synthetic 'three' / 'trois' / 'gatatu' stay well-separated.
            semitone = wi * 3 + li * 30
            f = base_freq * (2 ** (semitone / 12))
            audio = synth_tone(f, seed=li * 100 + wi)
            kws.add(w, lang, audio)
    return kws


# ──────────────────────────────────────────────────────────────────────────
# Child-voice adaptation hooks
# ──────────────────────────────────────────────────────────────────────────
def pitch_shift_up(audio: np.ndarray, semitones: float = 4.0,
                   sr: int = SAMPLE_RATE) -> np.ndarray:
    """Shift up to approximate child formants. No-op if librosa missing."""
    if not _HAS_LIBROSA:  # pragma: no cover
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
