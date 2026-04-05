Using the research note at `documents/logs/MM-DD-YYYY/research/research-<kebab-case-topic>.md`, create a detailed implementation plan for the paper-to-notebook codebase. Use `CODEBASE_PREFS.md` and `DESIGN_DOC.md` as the authoritative sources for architecture, module ordering, readability standards, artifact organization, and validation requirements.

Ground all recommendations in the current repository state. Schemas and stage boundaries must come before prompt convenience. Use precise, technical language and complete sentences.

---

Based on my research, here is what I found:

## Current State
- [Key discovery about the current repository state, existing documents, prompt files, and implemented or missing pipeline modules]
- [Established contracts, artifacts, or interfaces that already exist and must be preserved]
- [Repository constraints that affect implementation order, testing, or local inference]

## Recommended Implementation Direction
- [Minimal vertical slice from PDF ingestion to validated `.ipynb` output]
- [How stage-wise JSON artifacts and provenance will be preserved]
- [How local retrieval and local inference will stay lightweight and inspectable]

## Architecture Decisions to Lock Early
- [Typed schemas for `ParsedPaper`, `PaperChunk`, `ConceptItem`, `LearnerProfile`, `LessonSection`, `NotebookPlan`, `NotebookCell`, `NotebookBatch`, and `ValidationReport`]
- [Module and file layout under `src/`, `tests/`, and `examples/`]
- [Artifact and log layout per run or per source paper]
- [Notebook assembly through `nbformat` and validation strategy]

## Risks and Mitigations
- [Risk: monolithic prompt or pipeline collapse; Mitigation: strict staged interfaces and validation gates]
- [Risk: provenance loss across stages; Mitigation: explicit metadata fields and stage-aware artifact directories]
- [Risk: notation drift or undefined symbols; Mitigation: notation inventory and validator checks]
- [Risk: tensor-shape mistakes; Mitigation: explicit tensor-shape sections and helper validation]
- [Risk: invalid or unreadable notebooks; Mitigation: `nbformat` assembly, cell-length limits, and smoke tests]
- [Risk: local inference instability on constrained hardware; Mitigation: short-context retrieval-first generation and lightweight backend choices]

## Open Questions
- [Technical uncertainty that blocks implementation order or contract finalization]
- [Decision needed about parser choice, retrieval boundaries, learner-profile schema, or validation depth]

Which sequencing best matches the repository’s immediate goal: a strict end-to-end MVP first, or a broader scaffold that defines all stages before deep implementation?

---

Plan requirements:
- Specify file paths, module names, function or class names, and interfaces to be added or modified.
- Keep the implementation order close to `config.py`, `schemas.py`, `parse_pdf.py`, `chunking.py`, `retrieve.py`, `concept_graph.py`, `learner_profile.py`, `planner.py`, `cell_generator.py`, `notebook_builder.py`, `validators.py`, `repair.py`, and `main.py`.
- State the explicit input and output contract for each stage object, and describe how each contract is validated.
- Treat JSON as the default serialization format between stages, with `.ipynb` used only for final notebook assembly.
- Require stage-aware artifact directories such as `parsed_paper`, `chunks`, `concept_graph`, `learner_profile`, `notebook_plan`, `cell_batches`, `validation_report`, and final notebook output.
- Align notebook writing with `nbformat`, not manual string concatenation.
- Keep retrieval lightweight and inspectable, with explicit provenance and section-local regeneration.
- Align local inference with Gemma 4 E2B Instruct through `llama.cpp` server or an equally lightweight local backend.
- Apply readability-first software engineering principles: explicit names, minimal hidden codepaths, explanatory comments, and self-contained model-specific files when model wrappers are added.
- Include configuration changes, reproducibility steps, and `dvc.yaml` or equivalent data-versioning work where relevant.
- Include test plans for schemas, chunking, metadata preservation, notebook writing and loading, notebook validation, pedagogical validators, tensor-shape helpers, config loading, golden-style plans, and notebook execution smoke tests.
- Ensure the plan delivers a minimal vertical slice before advanced improvements such as richer concept graphs, repair passes, or interactive notebook widgets.

Place your findings under:

`documents/logs/MM-DD-YYYY/plan/`

Example filename:

`documents/logs/MM-DD-YYYY/plan/plan-<kebab-case-topic>.md`
