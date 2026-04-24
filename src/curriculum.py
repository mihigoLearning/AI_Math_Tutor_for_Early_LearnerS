"""Curriculum authoring for the Math Tutor.

Expands the 12-item seed (Assignment/T3.1_Math_Tutor/curriculum_seed.json)
into a ≥60-item, trilingual, skill- and difficulty-balanced bank.

Each item has this shape (superset of the seed schema):

    {
      "id":          "A005",                   # skill-letter + 3-digit index
      "skill":       "addition",               # one of SKILLS
      "difficulty":  4,                        # 1..9
      "age_band":    "6-7",                    # mapped from difficulty
      "stem_en":     "3 plus 4 equals?",
      "stem_fr":     "3 plus 4 égale?",
      "stem_kin":    "3 + 4 ni angahe?",
      "visual":      "beads_3_plus_4",         # used by src/visual.py
      "answer_int":  7
    }

Generation is fully deterministic — seeded by `CURRICULUM_SEED` — so running
the script twice gives the same ids and the tests can pin exact counts.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

SKILLS = ("counting", "number_sense", "addition", "subtraction", "word_problem")
AGE_BANDS = ("5-6", "6-7", "7-8", "8-9")
LANGS = ("en", "fr", "kin")

CURRICULUM_SEED = 20260424  # today's date — deterministic, reviewable

# ──────────────────────────────────────────────────────────────────────────
# Number words 1..20 in each supported language.
# Kinyarwanda forms verified against the Rwanda Education Board Primary 1
# numeracy booklet (2021 edition); tests assert a few landmark entries.
# ──────────────────────────────────────────────────────────────────────────
NUM_EN = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]
NUM_FR = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit",
    "neuf", "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf", "vingt",
]
NUM_KIN = [
    "zeru", "rimwe", "kabiri", "gatatu", "kane", "gatanu", "gatandatu",
    "karindwi", "umunani", "icyenda", "icumi", "cumi na rimwe",
    "cumi na kabiri", "cumi na gatatu", "cumi na kane", "cumi na gatanu",
    "cumi na gatandatu", "cumi na karindwi", "cumi n'umunani",
    "cumi n'icyenda", "makumyabiri",
]

# Locally-grounded nouns (plural forms), balanced across Rwandan / pan-African
# contexts so stimuli feel native in the target language.
NOUNS = {
    "en":  ["apples", "goats", "mangoes", "drums", "beans", "pencils", "cows",
            "bananas", "chairs", "books", "stones", "cups"],
    "fr":  ["pommes", "chèvres", "mangues", "tambours", "haricots", "crayons",
            "vaches", "bananes", "chaises", "livres", "pierres", "tasses"],
    "kin": ["pome", "ihene", "inyabutongo", "ingoma", "ibishyimbo", "ikaramu",
            "inka", "imineke", "intebe", "ibitabo", "amabuye", "ibikombe"],
}

# ──────────────────────────────────────────────────────────────────────────
# Difficulty → age band mapping (used when the seed doesn't supply one)
# ──────────────────────────────────────────────────────────────────────────
def age_for_difficulty(d: int) -> str:
    if d <= 2:
        return "5-6"
    if d <= 4:
        return "6-7"
    if d <= 6:
        return "7-8"
    return "8-9"


# ──────────────────────────────────────────────────────────────────────────
# Item data class
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Item:
    id: str
    skill: str
    difficulty: int
    age_band: str
    stem_en: str
    stem_fr: str
    stem_kin: str
    visual: str
    answer_int: int
    tts_en: str | None = None
    tts_fr: str | None = None
    tts_kin: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}


# ──────────────────────────────────────────────────────────────────────────
# Template generators — one per skill.  Each returns (stems, visual, answer).
# ──────────────────────────────────────────────────────────────────────────
def _counting(n: int, noun_idx: int) -> tuple[dict, str, int]:
    en_n, fr_n, kin_n = NOUNS["en"][noun_idx], NOUNS["fr"][noun_idx], NOUNS["kin"][noun_idx]
    return (
        {"en": f"How many {en_n}?",
         "fr": f"Combien de {fr_n}?",
         "kin": f"{kin_n.capitalize()} zingahe?"},
        f"{en_n}_{n}",
        n,
    )


def _number_sense(a: int, b: int) -> tuple[dict, str, int]:
    bigger = max(a, b)
    return (
        {"en": f"Which number is bigger: {a} or {b}?",
         "fr": f"Quel nombre est plus grand: {a} ou {b}?",
         "kin": f"Ni iyihe nimero nini: {a} cyangwa {b}?"},
        f"compare_{a}_{b}",
        bigger,
    )


def _addition(a: int, b: int, noun_idx: int) -> tuple[dict, str, int]:
    if a + b <= 9 and max(a, b) <= 5:
        en_n = NOUNS["en"][noun_idx]
        fr_n = NOUNS["fr"][noun_idx]
        kin_n = NOUNS["kin"][noun_idx]
        return (
            {"en": f"{a} {en_n} plus {b} more — how many {en_n}?",
             "fr": f"{a} {fr_n} plus {b} — combien de {fr_n}?",
             "kin": f"{kin_n} {a} n'indi {b}, zose ni zingahe?"},
            f"beads_{a}_plus_{b}",
            a + b,
        )
    # bigger numbers — abstract form
    return (
        {"en": f"{a} + {b} equals?",
         "fr": f"{a} + {b} égale?",
         "kin": f"{a} + {b} ni angahe?"},
        f"abstract_{a}_plus_{b}",
        a + b,
    )


def _subtraction(a: int, b: int, noun_idx: int) -> tuple[dict, str, int]:
    if a <= 10:
        en_n = NOUNS["en"][noun_idx]
        fr_n = NOUNS["fr"][noun_idx]
        kin_n = NOUNS["kin"][noun_idx]
        return (
            {"en": f"{a} {en_n}, {b} are taken — how many left?",
             "fr": f"{a} {fr_n}, on en prend {b} — il en reste combien?",
             "kin": f"{kin_n} {a}, hakurwa {b} — hasigaye zingahe?"},
            f"take_{a}_minus_{b}",
            a - b,
        )
    return (
        {"en": f"{a} - {b} equals?",
         "fr": f"{a} - {b} égale?",
         "kin": f"{a} - {b} ni angahe?"},
        f"abstract_{a}_minus_{b}",
        a - b,
    )


def _word_problem(rng: random.Random) -> tuple[dict, str, int, int]:
    templates = [
        # (en, fr, kin, answer_fn, difficulty, visual_fmt)
        ("Three children share {n} cookies equally. How many each?",
         "Trois enfants partagent {n} biscuits également. Combien chacun?",
         "Abana batatu bagabana amakeki {n} kimwe. Buri wese ahabwa angahe?",
         lambda n: n // 3, 6, "kids_3_cookies_{n}",
         [9, 12, 15, 18]),
        ("A tomato costs 100 RWF. Mama has {n} RWF. How many tomatoes?",
         "Une tomate coûte 100 RWF. Maman a {n} RWF. Combien de tomates?",
         "Inyanya igura 100 RWF. Mama afite {n} RWF. Ashobora kugura zingahe?",
         lambda n: n // 100, 8, "rwf_{n}_tomato_100",
         [500, 700, 850, 1200]),
        ("A basket has {n} beans; you eat 7. How many remain?",
         "Un panier contient {n} haricots; tu en manges 7. Combien reste-t-il?",
         "Agaseke gafite ibishyimbo {n}; urya 7. Hasigaye bingahe?",
         lambda n: n - 7, 6, "beans_basket_{n}",
         [12, 15, 18, 20]),
        ("Sara has {a} mangoes and gets {b} more. How many now?",
         "Sara a {a} mangues et en reçoit {b} de plus. Combien maintenant?",
         "Sara afite imineke {a} yongerewemo {b}. Ifite zingahe?",
         None, 5, "mangoes_{a}_plus_{b}",
         None),  # handled as special case
    ]
    tpl = rng.choice(templates)
    if tpl[3] is None:  # Sara mangoes
        a = rng.randint(2, 9)
        b = rng.randint(2, 9)
        ans = a + b
        visual = tpl[5].format(a=a, b=b)
        return (
            {"en": tpl[0].format(a=a, b=b),
             "fr": tpl[1].format(a=a, b=b),
             "kin": tpl[2].format(a=a, b=b)},
            visual, ans, tpl[4],
        )
    n = rng.choice(tpl[6])
    ans = tpl[3](n)
    return (
        {"en": tpl[0].format(n=n),
         "fr": tpl[1].format(n=n),
         "kin": tpl[2].format(n=n)},
        tpl[5].format(n=n), ans, tpl[4],
    )


# ──────────────────────────────────────────────────────────────────────────
# Curriculum builder
# ──────────────────────────────────────────────────────────────────────────
SEED_PATH = Path(__file__).parent.parent / "Assignment" / "T3.1_Math_Tutor" / "curriculum_seed.json"
OUT_PATH = Path(__file__).parent.parent / "data" / "curriculum.json"


def build_curriculum(n_generate: int = 60, out_path: Path | None = None) -> list[dict]:
    """Expand the seed into `n_generate` new items + the 12 seed items.

    Returns the full list (seed + generated) and writes it to `out_path`.
    Deterministic thanks to `CURRICULUM_SEED`.
    """
    rng = random.Random(CURRICULUM_SEED)
    items: list[Item] = []

    # 1. Carry over the seed verbatim
    seed = []
    if SEED_PATH.exists():
        seed = json.loads(SEED_PATH.read_text())
    seed_ids = {s["id"] for s in seed}

    # 2. Generate balanced items across skills × difficulty.
    #    Use a deterministic quota plan, not random, so counts are stable.
    per_skill = n_generate // len(SKILLS)
    extra = n_generate - per_skill * len(SKILLS)
    quotas = {s: per_skill + (1 if i < extra else 0)
              for i, s in enumerate(SKILLS)}

    counters = {"counting": 100, "number_sense": 100, "addition": 100,
                "subtraction": 100, "word_problem": 100}
    letter = {"counting": "C", "number_sense": "N", "addition": "A",
              "subtraction": "S", "word_problem": "W"}

    for skill, q in quotas.items():
        for _ in range(q):
            d = rng.randint(1, 9)
            noun_idx = rng.randint(0, len(NOUNS["en"]) - 1)
            if skill == "counting":
                n = rng.randint(2, min(2 + d, 12))
                stems, vis, ans = _counting(n, noun_idx)
            elif skill == "number_sense":
                a = rng.randint(1, 5 + d)
                b = rng.randint(1, 5 + d)
                while b == a:
                    b = rng.randint(1, 5 + d)
                stems, vis, ans = _number_sense(a, b)
            elif skill == "addition":
                a = rng.randint(1, 2 + d)
                b = rng.randint(1, 2 + d)
                stems, vis, ans = _addition(a, b, noun_idx)
            elif skill == "subtraction":
                a = rng.randint(3, 3 + d * 2)
                b = rng.randint(1, a - 1)
                stems, vis, ans = _subtraction(a, b, noun_idx)
            elif skill == "word_problem":
                stems, vis, ans, d = _word_problem(rng)
            else:
                raise ValueError(skill)

            # allocate id
            counters[skill] += 1
            iid = f"{letter[skill]}{counters[skill]:03d}"
            while iid in seed_ids:  # avoid clashing with seed C001 etc
                counters[skill] += 1
                iid = f"{letter[skill]}{counters[skill]:03d}"

            items.append(Item(
                id=iid, skill=skill, difficulty=d, age_band=age_for_difficulty(d),
                stem_en=stems["en"], stem_fr=stems["fr"], stem_kin=stems["kin"],
                visual=vis, answer_int=ans,
            ))

    # 3. Merge seed + generated
    merged = list(seed) + [it.to_dict() for it in items]

    # 4. Persist
    out = out_path or OUT_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2))
    return merged


def load_curriculum(path: Path | None = None) -> list[dict]:
    """Load the persisted curriculum (builds on first call)."""
    p = path or OUT_PATH
    if not p.exists():
        build_curriculum(out_path=p)
    return json.loads(p.read_text())


def filter_items(items: Iterable[dict], *,
                 skill: str | None = None,
                 difficulty: int | None = None,
                 age_band: str | None = None) -> list[dict]:
    out = []
    for it in items:
        if skill and it["skill"] != skill:
            continue
        if difficulty is not None and it["difficulty"] != difficulty:
            continue
        if age_band and it.get("age_band") != age_band:
            continue
        out.append(it)
    return out


if __name__ == "__main__":  # pragma: no cover
    curr = build_curriculum()
    print(f"Built {len(curr)} items → {OUT_PATH}")
    by_skill = {}
    for it in curr:
        by_skill.setdefault(it["skill"], 0)
        by_skill[it["skill"]] += 1
    for s, c in sorted(by_skill.items()):
        print(f"  {s:<14s} {c}")
