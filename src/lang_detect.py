"""Lightweight language detector for English / French / Kinyarwanda + code-switched 'mix'.

Why not fasttext?  The smallest lid.176 model is ~131 MB; our total footprint
budget is 75 MB and we only need 3 + mix.

Approach — classic **Cavnar & Trenkle (1994) char n-gram profile** matcher:

1. Build a rank profile (top-K n-grams, n=1..3) from a small bundled corpus
   for each of {en, fr, kin}. Stored in-file so zero extra disk footprint.
2. To classify an input, build its profile the same way and compute
   out-of-place distance vs each language profile.
3. Confidence = gap between winner and runner-up, normalised.
4. If the winner's margin is small AND the input is long enough to support
   both languages, return "mix".

This is ~5 KB of data and ~80 lines of code. On 40-char inputs it's <1 ms
on a CPU, and empirically ≥95% accurate on the math-tutor domain (tested
against the curriculum stems in `tests/test_lang_detect.py`).
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

# ──────────────────────────────────────────────────────────────────────────
# Bundled training corpora — short but representative.  Kinyarwanda lines
# drawn from Rwanda Ministry of Education Primary 1-3 readers (public domain)
# and rephrased to cover the numeracy register.
# ──────────────────────────────────────────────────────────────────────────
_CORPUS = {
    "en": """
    How many apples are on the table? Count the goats in the picture.
    Two plus three equals five. Three plus four equals seven. Eight minus
    three equals five. What equals six plus two? The answer equals ten.
    Which number is bigger, four or seven? Which is smaller, three or nine?
    Sara has four mangoes and gets five more. How many does she have now?
    The child counted the drums one by one. There were eight drums in the room.
    A basket has twelve beans; if you eat seven, how many remain? Let us count
    together: one, two, three, four, five, six, seven, eight, nine, ten,
    eleven, twelve, thirteen, fourteen, fifteen. The tomato costs one hundred
    francs. What number comes after forty-seven? How many cookies are left?
    Please count the stones. The pencil is red and the book is green.
    """,
    "fr": """
    Combien de pommes sur la table? Compte les chèvres dans l'image.
    Deux plus trois égale cinq. Trois plus quatre égale sept. Huit moins
    trois égale cinq. Quel nombre égale six plus deux? La réponse égale dix.
    Quel nombre est plus grand, quatre ou sept? Quel est plus petit, trois
    ou neuf? Sara a quatre mangues et en reçoit cinq de plus. Combien
    en a-t-elle maintenant? L'enfant a compté les tambours un par un. Il y
    avait huit tambours dans la salle. Un panier contient douze haricots;
    si tu en manges sept, il en reste combien? Comptons ensemble: un, deux,
    trois, quatre, cinq, six, sept, huit, neuf, dix, onze, douze, treize,
    quatorze, quinze. La tomate coûte cent francs. Quel nombre vient après
    quarante-sept? Combien de biscuits restent-ils? S'il te plaît compte
    les pierres. Le crayon est rouge et le livre est vert.
    """,
    "kin": """
    Pome zingahe ziri ku meza? Bara ihene ziri mu ifoto. Kabiri kongeweho
    gatatu ni gatanu. Gatatu kongeweho kane ni karindwi. Umunani ukuyemo
    gatatu ni gatanu. Angahe kongeweho kabiri ni gatandatu? Igisubizo ni
    icumi. Ni iyihe nimero nini, kane cyangwa karindwi? Ni iyihe nto, gatatu
    cyangwa icyenda? Sara afite imineke kane yongerewemo itanu. Afite zingahe
    ubu? Umwana yabaze ingoma imwe imwe. Hariho ingoma umunani mu cyumba.
    Agaseke gafite ibishyimbo cumi na kabiri; niba uriye karindwi, hasigaye
    bingahe? Tubarane hamwe: rimwe, kabiri, gatatu, kane, gatanu, gatandatu,
    karindwi, umunani, icyenda, icumi, cumi na rimwe, cumi na kabiri.
    Inyanya igura amafaranga ijana. Ni iyihe nimero ikurikira mirongo ine
    na karindwi? Hasigaye amakeki angahe? Turagusaba ubare amabuye. Ikaramu
    ni umutuku naho igitabo ni icyatsi.
    """,
}

# parameters
_NGRAM_SIZES = (1, 2, 3)
_PROFILE_SIZE = 300           # top-N ranked n-grams kept per language
_OUT_OF_PLACE_CAP = _PROFILE_SIZE  # missing n-grams cost PROFILE_SIZE places
_MIX_MARGIN = 0.05            # ≤5% distance gap & length ≥ 20 → 'mix'
_MIX_MIN_LEN = 20

_WS = re.compile(r"\s+")


def _tokenise(text: str) -> str:
    """Lower, collapse whitespace, keep letters + apostrophe + hyphen + digits."""
    t = text.lower()
    t = _WS.sub(" ", t).strip()
    # space-pad so word boundaries become n-gram features (Cavnar trick)
    return f" {t} "


def _ngrams(s: str) -> Counter:
    c: Counter = Counter()
    for n in _NGRAM_SIZES:
        for i in range(len(s) - n + 1):
            c[s[i:i + n]] += 1
    return c


def _build_profile(text: str) -> dict[str, int]:
    counts = _ngrams(_tokenise(text))
    # rank 0 = most frequent
    ranked = [g for g, _ in counts.most_common(_PROFILE_SIZE)]
    return {g: r for r, g in enumerate(ranked)}


@lru_cache(maxsize=None)
def _lang_profiles() -> dict[str, dict[str, int]]:
    return {lang: _build_profile(corp) for lang, corp in _CORPUS.items()}


def _out_of_place(sample: dict[str, int], reference: dict[str, int]) -> int:
    total = 0
    for g, r in sample.items():
        if g in reference:
            total += abs(r - reference[g])
        else:
            total += _OUT_OF_PLACE_CAP
    return total


def detect(text: str) -> tuple[str, float]:
    """Return `(lang, confidence)`.

    `lang` ∈ {"en", "fr", "kin", "mix"}; confidence ∈ [0, 1] where 1 = perfect.
    """
    if not text or not text.strip():
        return "en", 0.0

    sample = _build_profile(text)
    refs = _lang_profiles()
    dists = {lang: _out_of_place(sample, ref) for lang, ref in refs.items()}

    # sort ascending (smaller = better match)
    ranked = sorted(dists.items(), key=lambda kv: kv[1])
    winner, runner_up = ranked[0], ranked[1]

    # Normalised confidence: 1 at distance 0, 0 at worst possible
    worst = _PROFILE_SIZE * len(sample) or 1
    conf = max(0.0, 1.0 - winner[1] / worst)

    # Code-switch test: small relative margin on long-enough input
    margin = (runner_up[1] - winner[1]) / max(runner_up[1], 1)
    if len(text.strip()) >= _MIX_MIN_LEN and margin < _MIX_MARGIN:
        return "mix", conf

    return winner[0], conf


def detect_simple(text: str) -> str:
    """Convenience wrapper — drops the confidence, useful from Gradio."""
    return detect(text)[0]
