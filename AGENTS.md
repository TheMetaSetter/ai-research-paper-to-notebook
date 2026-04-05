# Repository Guidelines

`documents/DESIGN_DOC.md` is the architectural source of truth

## Project Structure & Module Organization

This repository is being implemented. Active assets live in `documents/` for architecture and implementation guidance, and `prompts/` for staged notebook-generation prompts. Contributors should follow the planned layout in `documents/IMPLEMENTATION_PLAN.md`: `src/` for pipeline stages, `tests/` for Pytest coverage, `examples/` for sample inputs, and `runs/` for generated artifacts and logs.

Keep stage boundaries explicit. The intended pipeline is: PDF parsing, chunking, retrieval, concept extraction, learner profiling, planning, cell generation, notebook assembly, validation, and repair. Do not collapse this into one script or one prompt.

## Build, Test, and Development Commands

Use these commands while working on the current repo:

- `rg --files` lists the full repository quickly.
- `sed -n '1,160p' documents/DESIGN_DOC.md` reads the architectural source of truth.
- `sed -n '1,200p' documents/IMPLEMENTATION_PLAN.md` checks the target code layout and dependency plan.
- `git log --oneline` reviews existing commit style before contributing.

When code is added, prefer a small CLI entry point and Pytest-based checks that match the implementation plan.

## Coding Style & Naming Conventions

`documents/CODEBASE_PREFS.md` captures implementation preferences

Write for readability first. Use Python 3.11+, 4-space indentation, explicit names, and minimal branching. Prefer full-word module names such as `notebook_builder.py` or `learner_profile.py` over abbreviations. Keep one model per file under `src/models/`. Add short explanatory comments where logic is not obvious.

## Testing Guidelines

Use `pytest` for all tests. Name files `test_<module>.py`. Focus coverage on schema contracts, chunk metadata, notebook validity, configuration loading, and stage-level regression checks. Add small deterministic tests rather than broad end-to-end cases first.

## Commit & Pull Request Guidelines

Current history uses short, imperative commit subjects such as `Initial commit` and `Align prompt suite with paper-to-notebook design`. Follow that pattern: concise verb-first summaries, one clear change per commit.

PRs should explain the affected pipeline stage, note any design-doc changes, and include sample inputs or outputs when behavior changes. If you change the intended architecture or layout, update the matching file in `documents/` in the same PR.

## Architecture Notes

`documents/DESIGN_DOC.md` is the architectural source of truth, and `documents/CODEBASE_PREFS.md` captures implementation preferences. Keep those aligned with the codebase as the repository grows.
