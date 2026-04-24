"""Gradio demo — 90-second first-open flow.

First-open contract (spec: total time ≤ 90 s for a child to get from app-
icon tap to a completed first item):

    0s – app opens to a BIG green circle "Tap to start" (no reading)
    5s – language auto-detected from device locale (overridable by 3 flag
         buttons with country icons; one tap)
   10s – name/ID screen skipped for the first session (anon learner_id)
   15s – diagnostic probe #1 presented: image + TTS-spoken stem
   35s – child enters an answer (number keypad with large buttons)
   40s – feedback: green star or gentle retry
   45s–90s – four more probes, same pattern

The UI below is implemented as plain Gradio Blocks so it runs on any CPU
without GPU dependencies. Audio input is optional — tapping the keypad is
always available as a fallback for noisy classrooms.

Run:  python -m tutor.demo
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # type: ignore
import numpy as np

from tutor.engine import Tutor, grade
from visual import render
from lang_detect import detect as detect_lang
from parent_report import build_report
from asr_kws import build_synthetic_bank


KWS = build_synthetic_bank()

UI_TEXT = {
    "en": {
        "question": "Question",
        "your_answer": "Your answer",
        "type_number": "Type a number ...",
        "submit": "✓  Submit",
        "speak_answer": "Or speak your answer",
        "submit_voice": "🎤 Submit voice",
        "show_progress": "Show my progress",
        "progress_report": "Progress report",
        "skill_mastery": "Skill mastery:",
        "round": "Round",
        "language": "Language",
        "tap_start_first": "Please tap Start first.",
        "try_next": "🙂  Try the next one",
        "no_audio": "No audio captured yet.",
        "try_record_again": "🎤 Try recording again.",
        "could_not_map": "Could not map spoken number.",
        "please_try_again": "🙂 Please try again.",
        "heard": "🎤 heard",
        "audio_not_understood": "Audio not understood.",
        "try_again_or_type": "🙂 Please try again or type the number.",
        "no_session_yet": "No session yet.",
    },
    "fr": {
        "question": "Question",
        "your_answer": "Votre reponse",
        "type_number": "Tapez un nombre ...",
        "submit": "✓  Valider",
        "speak_answer": "Ou dites votre reponse",
        "submit_voice": "🎤 Valider la voix",
        "show_progress": "Afficher mes progres",
        "progress_report": "Rapport de progres",
        "skill_mastery": "Maitrise des competences:",
        "round": "Tour",
        "language": "Langue",
        "tap_start_first": "Veuillez d'abord cliquer sur Demarrer.",
        "try_next": "🙂  Essaie le suivant",
        "no_audio": "Aucun audio capture.",
        "try_record_again": "🎤 Reessayez d'enregistrer.",
        "could_not_map": "Impossible d'associer le nombre prononce.",
        "please_try_again": "🙂 Veuillez reessayer.",
        "heard": "🎤 entendu",
        "audio_not_understood": "Audio non compris.",
        "try_again_or_type": "🙂 Reessayez ou tapez le nombre.",
        "no_session_yet": "Aucune session pour le moment.",
    },
    "kin": {
        "question": "Ikibazo",
        "your_answer": "Igisubizo cyawe",
        "type_number": "Andika umubare ...",
        "submit": "✓  Ohereza",
        "speak_answer": "Canke vuga igisubizo",
        "submit_voice": "🎤 Ohereza amajwi",
        "show_progress": "Erekana iterambere ryanje",
        "progress_report": "Raporo y'iterambere",
        "skill_mastery": "Urugero rw'ubumenyi:",
        "round": "Inshuro",
        "language": "Ururimi",
        "tap_start_first": "Banza ukande Tangira.",
        "try_next": "🙂  Gerageza ikindi",
        "no_audio": "Nta majwi yafashwe.",
        "try_record_again": "🎤 Ongera ugerageze gufata amajwi.",
        "could_not_map": "Sinashoboye guhuza ijambo n'umubare.",
        "please_try_again": "🙂 Ongera ugerageze.",
        "heard": "🎤 numvise",
        "audio_not_understood": "Amajwi ntiyumvikanye neza.",
        "try_again_or_type": "🙂 Ongera ugerageze canke wandike umubare.",
        "no_session_yet": "Nta session iratangira.",
    },
}

LANG_NAME = {"en": "English", "fr": "Français", "kin": "Kinyarwanda"}


# ──────────────────────────────────────────────────────────────────────────
# Session state — gradio sessions are keyed per-browser, so we store a
# Tutor object in gr.State.
# ──────────────────────────────────────────────────────────────────────────
def _new_session(lang: str) -> dict:
    t = Tutor(learner_id="demo", lang=lang)
    item = t.next_item()
    return {
        "tutor": t,
        "item": item,
        "previous_mastery": dict(t.bkt.p_L),
        "rounds": 0,
    }


def _render_item(state: dict) -> tuple[str, "PIL.Image.Image"]:
    item = state["item"]
    stem = state["tutor"].stem_for(item)
    img = render(item["visual"])
    return stem, img


# ──────────────────────────────────────────────────────────────────────────
# Handlers
# ──────────────────────────────────────────────────────────────────────────
def start(lang_choice: str):
    lang = {"English": "en", "Français": "fr", "Kinyarwanda": "kin"}[lang_choice]
    txt = UI_TEXT[lang]
    state = _new_session(lang)
    stem, img = _render_item(state)
    return (
        state,
        stem,
        img,
        "",
        _progress_text(state),
        gr.Textbox(label=txt["question"]),
        gr.Textbox(label=txt["your_answer"], placeholder=txt["type_number"]),
        gr.Button(value=txt["submit"]),
        gr.Audio(label=txt["speak_answer"]),
        gr.Button(value=txt["submit_voice"]),
        gr.Button(value=txt["show_progress"]),
        gr.Textbox(label=txt["progress_report"]),
    )


def submit_answer(state: dict, typed: str):
    if not state:
        return state, "Please tap Start first.", None, "", ""
    item = state["item"]
    fb = state["tutor"].answer(item, typed)
    state["rounds"] += 1
    if fb["correct"]:
        feedback = "⭐  " + ("Great!" if state["tutor"].lang == "en"
                            else "Bravo!" if state["tutor"].lang == "fr"
                            else "Nibyiza!")
    else:
        feedback = UI_TEXT[state["tutor"].lang]["try_next"]
    # advance
    state["item"] = state["tutor"].next_item()
    stem, img = _render_item(state)
    return state, stem, img, feedback, _progress_text(state)


def submit_audio(state: dict, audio):
    """Map microphone audio to a number word, then grade like typed input."""
    if not state:
        return state, UI_TEXT["en"]["tap_start_first"], None, "", ""
    txt = UI_TEXT[state["tutor"].lang]
    if audio is None:
        return state, txt["no_audio"], None, txt["try_record_again"], _progress_text(state)
    sr, wav = audio
    if wav is None:
        return state, txt["no_audio"], None, txt["try_record_again"], _progress_text(state)

    # Gradio may provide int16 mono/stereo arrays; normalize to float32 mono.
    arr = np.asarray(wav)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if arr.dtype.kind in {"i", "u"}:
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype(np.float32)

    lang = state["tutor"].lang
    try:
        label, label_lang, conf = KWS.recognize(arr, sr=int(sr), lang=lang)
        num = KWS.label_to_int(label, label_lang)
        if num is None:
            return state, txt["could_not_map"], None, txt["please_try_again"], _progress_text(state)
        state, stem, img, feedback, prog = submit_answer(state, str(num))
        spoken = f"{txt['heard']}: {label} ({label_lang}, conf {conf:.2f})"
        return state, stem, img, f"{spoken}\n\n{feedback}", prog
    except Exception:
        return state, txt["audio_not_understood"], None, txt["try_again_or_type"], _progress_text(state)


def end_session(state: dict):
    if not state:
        return UI_TEXT["en"]["no_session_yet"]
    rep = build_report(
        current=dict(state["tutor"].bkt.p_L),
        previous=state["previous_mastery"],
        learner_id=state["tutor"].learner_id,
        sessions=1,
        lang=state["tutor"].lang,
        name="you",
    )
    lines = [rep["summary_text"], "", "Skill mastery:"]
    for s, v in rep["skills"].items():
        lines.append(f"  {s:<14s}  {v['current']:.0%}  (Δ {v['delta']:+.2f})")
    return "\n".join(lines)


def _progress_text(state: dict) -> str:
    lang = state["tutor"].lang
    txt = UI_TEXT[lang]
    return f"{txt['round']} {state['rounds']}   ·   {txt['language']}: {LANG_NAME[lang]}"


def _maybe_localize_start_prompt(msg: str, lang: str | None) -> str:
    if msg != UI_TEXT["en"]["tap_start_first"]:
        return msg
    if not lang:
        return msg
    return UI_TEXT.get(lang, UI_TEXT["en"])["tap_start_first"]


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Math Tutor — Early Learners") as demo:
        gr.Markdown("# 🧮  Math Tutor\n*Tap a language to start — no reading needed.*")

        lang_picker = gr.Radio(["English", "Français", "Kinyarwanda"],
                               value="English", label="Language / Ururimi / Langue")
        start_btn = gr.Button("▶  Start", variant="primary", size="lg")

        state = gr.State(None)
        with gr.Row():
            with gr.Column(scale=1):
                stimulus_img = gr.Image(type="pil", height=280, label="")
            with gr.Column(scale=1):
                stem = gr.Textbox(label="Question", lines=3, interactive=False)
                answer = gr.Textbox(label="Your answer",
                                    placeholder="Type a number ...")
                submit_btn = gr.Button("✓  Submit", variant="primary")
                mic = gr.Audio(sources=["microphone"], type="numpy", label="Or speak your answer")
                speak_btn = gr.Button("🎤 Submit voice", variant="secondary")
                feedback = gr.Markdown()
                progress = gr.Markdown()

        end_btn = gr.Button("Show my progress")
        report_box = gr.Textbox(label="Progress report", lines=10, interactive=False)

        start_btn.click(start, inputs=[lang_picker],
                        outputs=[state, stem, stimulus_img, feedback, progress,
                                 stem, answer, submit_btn, mic, speak_btn, end_btn, report_box])
        submit_btn.click(submit_answer, inputs=[state, answer],
                         outputs=[state, stem, stimulus_img, feedback, progress])
        speak_btn.click(submit_audio, inputs=[state, mic],
                        outputs=[state, stem, stimulus_img, feedback, progress])
        answer.submit(submit_answer, inputs=[state, answer],
                      outputs=[state, stem, stimulus_img, feedback, progress])
        end_btn.click(end_session, inputs=[state], outputs=[report_box])

    return demo


if __name__ == "__main__":  # pragma: no cover
    build_ui().launch(theme="soft")
