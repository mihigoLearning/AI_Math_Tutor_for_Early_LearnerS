# process_log.md

## S2.T3.1 Build Timeline (hour-by-hour)

### Hour 1 (0:00-1:00)
- Parsed the Tier 3 brief and listed mandatory deliverables (`tutor/`, `demo.py`, `parent_report.py`, `kt_eval.ipynb`, `footprint_report.md`, `process_log.md`, `SIGNED.md`).
- Defined target architecture for an offline tutor pipeline: curriculum loader, adaptive engine, language detection, scoring, reporting, and local store.
- Set up repository structure and Python modules for core components under `src/` and `tutor/`.

### Hour 2 (1:00-2:00)
- Implemented core tutoring logic (curriculum progression and adaptive skill selection).
- Added knowledge tracing implementation and Elo baseline path for comparison.
- Built language handling for English/French/Kinyarwanda and code-switch behavior.
- Drafted local persistence interfaces and parent-report generation scaffolding.

### Hour 3 (2:00-3:00)
- Added audio and visual support modules (keyword-spotting flow + object-count visual baseline).
- Connected modules in `tutor/engine.py` and `tutor/demo.py` for end-to-end flow.
- Wrote/expanded test suite across core modules (`tests/`) and validated major paths.
- Updated `README.md` with quickstart, structure, and status links.

### Hour 4 (3:00-4:00)
- Finalized documentation artifacts and packaging details.
- Reviewed technical constraints (offline operation, CPU-first design, footprint-aware structure).
- Performed final cleanup and consistency checks across modules/tests/docs.
- Added this `process_log.md` and ensured required `SIGNED.md` is present at repo root.

## LLM / Assistant Tools Used

1. **Cursor AI Agent (Codex 5.3)**
   - **Why used:** accelerate code scaffolding, test-oriented iteration, and documentation consistency while maintaining full manual review.
2. **Cursor IDE coding assistance**
   - **Why used:** faster navigation across files, refactoring support, and quick implementation checks.
3. **Local terminal tooling (`pytest`, Python CLI)**
   - **Why used:** validate functionality and test behavior during development.

## Sample Prompts Actually Sent

1. "Implement a lightweight Bayesian Knowledge Tracing module for early-learner math skills with clear function boundaries and testability."
2. "Create a multilingual language-detection utility for EN/FR/KIN with a simple code-switch fallback and unit tests."
3. "Generate a concise parent weekly report builder from local learner progress records with icon-friendly output fields."

## One Prompt I Discarded (and Why)

- **Discarded prompt:** "Build full Whisper fine-tuning and quantization pipeline end-to-end in this repo right now."
- **Why discarded:** this exceeded the session's practical compute/time constraints and risked reducing reliability of the required offline demo; a deterministic lightweight approach was prioritized for a defensible MVP.

## Hardest Decision

The hardest decision was balancing model ambition against offline reliability under strict constraints (CPU-only, small footprint, and fast response). A larger end-to-end learned stack could be more expressive, but would increase risk on latency, reproducibility, and defense-time stability. I chose a modular, lightweight architecture (deterministic/compact components where possible, plus KT logic for adaptivity) so the tutor remains explainable, testable, and robust for live demonstration in low-resource conditions.
