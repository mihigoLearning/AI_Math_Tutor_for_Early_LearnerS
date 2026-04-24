"""Weekly parent report — icon-first, non-literate friendly.

Matches the schema in `Assignment/T3.1_Math_Tutor/parent_report_schema.json`:

    {
      "learner_id": ...,
      "week_starting": "YYYY-MM-DD",
      "sessions": int,
      "skills": {skill: {"current": float, "delta": float}, ...},
      "icons_for_parent": ["overall_arrow", "best_skill", "needs_help"],
      "voiced_summary_audio": "path.wav"
    }

Design principle: the UI layer is **icons > numbers > text**. A caregiver
who can't read still sees:

  - a large green/red/yellow arrow  (overall trend)
  - a thumbs-up icon labelled with the child's best skill (illustrated)
  - a support icon for the weakest skill

Text is kept to a 1-sentence caption in the caregiver's language, and a
voice recording (`voiced_summary_audio`) plays the same sentence aloud.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SKILLS = ("counting", "number_sense", "addition", "subtraction", "word_problem")

# Human-readable skill labels per language
SKILL_LABEL = {
    "en":  {"counting": "counting", "number_sense": "number sense",
            "addition": "adding", "subtraction": "taking away",
            "word_problem": "word problems"},
    "fr":  {"counting": "compter", "number_sense": "sens des nombres",
            "addition": "ajouter", "subtraction": "enlever",
            "word_problem": "problèmes"},
    "kin": {"counting": "kubara", "number_sense": "kumenya nimero",
            "addition": "kongera", "subtraction": "gukuraho",
            "word_problem": "ibibazo"},
}

# Voiced-summary templates — one sentence each, designed to be trivial to
# render with a TTS engine on any device.
SUMMARY_TEMPLATE = {
    "en":  "This week {name} practised {sessions} times. They did best at "
           "{best}. They need more help with {weak}. Overall, they are {trend}.",
    "fr":  "Cette semaine {name} s'est exercé {sessions} fois. Le mieux: "
           "{best}. Il faut aider avec {weak}. En général, l'enfant {trend}.",
    "kin": "Iki cyumweru {name} yarize inshuro {sessions}. Yakoze neza muri "
           "{best}. Akeneye ubufasha muri {weak}. Muri rusange, {trend}.",
}

TREND_PHRASE = {
    "en":  {"up": "improving", "flat": "steady", "down": "slowing down"},
    "fr":  {"up": "progresse", "flat": "reste stable", "down": "ralentit"},
    "kin": {"up": "aratera imbere", "flat": "arahagaze", "down": "aragabanuka"},
}


# ──────────────────────────────────────────────────────────────────────────
# Report assembly
# ──────────────────────────────────────────────────────────────────────────
def build_report(current: dict[str, float], previous: dict[str, float] | None,
                 *, learner_id: str, sessions: int,
                 lang: str = "en", name: str = "the child",
                 week_starting: str | None = None,
                 voiced_audio_path: str | None = None) -> dict:
    """Produce the report dict matching the schema."""
    previous = previous or {s: 0.0 for s in current}
    skills = {s: {"current": float(current.get(s, 0.0)),
                  "delta":   float(current.get(s, 0.0) - previous.get(s, 0.0))}
              for s in SKILLS}

    deltas = [skills[s]["delta"] for s in SKILLS]
    avg = sum(deltas) / max(len(deltas), 1)
    trend = "up" if avg > 0.03 else ("down" if avg < -0.03 else "flat")

    best = max(SKILLS, key=lambda s: skills[s]["current"])
    weak = min(SKILLS, key=lambda s: skills[s]["current"])

    if week_starting is None:
        # This week's Monday (ISO weekday == 1)
        today = dt.date.today()
        monday = today - dt.timedelta(days=today.weekday())
        week_starting = monday.isoformat()

    return {
        "learner_id": learner_id,
        "week_starting": week_starting,
        "sessions": int(sessions),
        "skills": skills,
        "icons_for_parent": [
            f"overall_arrow_{trend}",
            f"best_skill_{best}",
            f"needs_help_{weak}",
        ],
        "voiced_summary_audio": voiced_audio_path or "",
        "summary_text": _summary_sentence(name, sessions, best, weak, trend, lang),
        "lang": lang,
    }


def _summary_sentence(name: str, sessions: int, best: str, weak: str,
                      trend: str, lang: str) -> str:
    labels = SKILL_LABEL.get(lang, SKILL_LABEL["en"])
    trend_w = TREND_PHRASE.get(lang, TREND_PHRASE["en"])[trend]
    return SUMMARY_TEMPLATE.get(lang, SUMMARY_TEMPLATE["en"]).format(
        name=name, sessions=sessions,
        best=labels[best], weak=labels[weak], trend=trend_w,
    )


# ──────────────────────────────────────────────────────────────────────────
# Voiced summary
# ──────────────────────────────────────────────────────────────────────────
def write_voiced_summary(report: dict, out_path: Path) -> Path:
    """Render the summary sentence to a WAV via pyttsx3 (offline, CPU-only).

    Returns the path. If pyttsx3 can't initialise (e.g. headless CI) we
    write a tiny 0.5 s silent WAV so downstream callers don't crash.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = report.get("summary_text", "")
    try:  # pragma: no cover - environment-dependent
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, str(out_path))
        engine.runAndWait()
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
    except Exception:
        pass

    # Fallback: silent WAV
    import wave
    import struct
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 8000)  # 0.5 s silence
    return out_path


# ──────────────────────────────────────────────────────────────────────────
# Icon card — parent-facing image
# ──────────────────────────────────────────────────────────────────────────
_COLORS = {
    "up":   (40, 170, 90),
    "flat": (230, 180, 40),
    "down": (210, 70, 70),
}

SKILL_EMOJI = {  # very simple pictogram choices — can swap for real SVGs later
    "counting": "🔢", "number_sense": "⚖️", "addition": "➕",
    "subtraction": "➖", "word_problem": "📖",
}


def render_icon_card(report: dict, out_path: Path) -> Path:
    """Render a small PNG with three icons + 1-line summary.

    The icon layout is deliberately spartan and uses the `SKILL_EMOJI`
    fallbacks. A designer can replace each pictogram with a locally-sourced
    illustration by pointing the tutor's asset path at `data/icons/`.
    """
    img = Image.new("RGB", (640, 280), (250, 250, 250))
    draw = ImageDraw.Draw(img)

    # derive trend/best/weak
    icons = report.get("icons_for_parent", [])
    trend = "flat"
    best = weak = None
    for key in icons:
        if key.startswith("overall_arrow_"):
            trend = key.removeprefix("overall_arrow_")
        elif key.startswith("best_skill_"):
            best = key.removeprefix("best_skill_")
        elif key.startswith("needs_help_"):
            weak = key.removeprefix("needs_help_")

    # arrow block
    draw.rectangle([(20, 20), (200, 200)], fill=_COLORS.get(trend, _COLORS["flat"]))
    arrow_glyph = {"up": "▲", "flat": "■", "down": "▼"}[trend]
    try:
        font_big = ImageFont.truetype("Arial.ttf", 90)
        font = ImageFont.truetype("Arial.ttf", 22)
    except OSError:  # pragma: no cover - font availability varies
        font_big = ImageFont.load_default()
        font = ImageFont.load_default()
    draw.text((60, 40), arrow_glyph, fill=(255, 255, 255), font=font_big)

    # best / weak badges
    y = 30
    for tag, col, sk in [("👍 best", (40, 140, 40), best),
                          ("🤝 help", (210, 110, 40), weak)]:
        draw.rectangle([(230, y), (620, y + 80)], outline=col, width=3)
        draw.text((246, y + 8), tag, fill=col, font=font)
        label = sk or "—"
        draw.text((246, y + 36), label, fill=(30, 30, 30), font=font)
        y += 100

    # caption
    draw.text((20, 230), report.get("summary_text", "")[:80],
              fill=(40, 40, 40), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path
