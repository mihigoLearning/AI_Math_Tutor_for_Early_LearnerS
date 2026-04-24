# AI Math Tutor for Early Learners

**Day 3 — AIMS KTT Hackathon (T3.1, Tier 3)**

An **on-device, multilingual math tutor** for children ages 5–9.
Runs CPU-only, fits inside a **75 MB** footprint, and answers each child
prompt in **under 2.5 s** on a Colab CPU.

- Language coverage: **English, French, Kinyarwanda** + code-switched ("mix")
- Skills: counting · number sense · addition · subtraction · word problems
- Knowledge tracing: **Bayesian Knowledge Tracing (BKT)** with an **Elo**
  baseline for comparison (AUC reported in `kt_eval.ipynb`)
- Visual grounding: a deterministic **blob counter** over rendered PIL stimuli
- Audio: **MFCC + DTW keyword-spotter** over number words (1–20) in all three
  languages — small enough to ship with the app
- Local store: **encrypted SQLite** (Fernet) with **ε-differential-privacy**
  noise on the weekly parent-sync payload
- Parent report: icon-first layout with a **voiced summary** for non-literate
  caregivers

## Quickstart

```bash
pip install -r requirements.txt
python -m tutor.demo          # Gradio UI, 90-sec first-open flow
pytest -q                     # test suite
jupyter notebook kt_eval.ipynb
```

## Live Defense Run Order

```bash
python demo.py
pytest -q
```

Then open:
- `http://127.0.0.1:7860/` for the child-facing demo
- `kt_eval.ipynb` for BKT vs Elo evaluation results

## Assignment Deliverables Checklist

- `tutor/` package:
  - `tutor/model.onnx`
  - `tutor/curriculum_loader.py`
  - `tutor/adaptive.py`
  - `tutor/asr_adapt.py`
- `demo.py` (root)
- `parent_report.py` (root)
- `footprint_report.md` (root)
- `kt_eval.ipynb` (root)
- `process_log.md` (root)
- `SIGNED.md` (root)

## Layout

```
tutor/         # on-device engine (import surface)
src/           # implementation modules
tests/         # pytest suite
data/          # curriculum.json, rendered stimuli, sample audio
Assignment/    # challenge brief + live-defense preparation notes
```

## Status

See `process_log.md` for the build log and `footprint_report.md`
for the size audit.
