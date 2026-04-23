---
date: 2026-04-20T17:41:42+07:00
researcher: Artificial Intelligence Agent
git_commit: c39218ea7046dbba2128aabee42108447ec1d854
branch: master
repository: ai-research-paper-to-notebook
topic: "Current status of the paper-to-notebook codebase"
tags: [research, paper-to-notebook, pedagogical-notebooks, local-inference]
status: complete
last_updated: 2026-04-20
last_updated_by: Artificial Intelligence Agent
---

# Research: Current status of the paper-to-notebook codebase

**Date**: 2026-04-20T17:41:42+07:00
**Researcher**: Artificial Intelligence Agent
**Git Commit**: `c39218ea7046dbba2128aabee42108447ec1d854`
**Branch**: `master`

## Research Question

Document the current status of the repository as it exists today, using the staged paper-to-notebook architecture as the frame of reference.

## Summary

The repository is no longer just a Phase 0 contract scaffold. In the current workspace, it contains implemented modules for every major pipeline stage described in the design documents: parsing, chunking, lexical retrieval, concept extraction, learner profiling, lesson planning, section-wise cell generation, notebook assembly, validation, and targeted repair. The CLI exposes each of these stages as separate commands, and the tests cover them extensively.

The important qualification is that the local workspace is ahead of the checked-in `HEAD`. `git status --short` shows modified and untracked stage modules and tests, so the present-day codebase is best described as a working staged local implementation in progress rather than a fully reflected committed history. This matters because `HEAD` alone understates the current codebase status.

At the pipeline level, the repository now matches the intended staged architecture more closely than the placeholder `dvc.yaml` suggests. However, the single-command `run` entry point is still explicitly unimplemented, DVC remains placeholder-only, and there are no committed sample run artifacts under `runs/`. The code supports stage-by-stage execution and validation, but not yet a committed end-to-end production path from one command.

## Detailed Findings

### Pipeline Stages

- **PDF ingestion and parsing**
  The parse stage is implemented through `parse_pdf_into_parsed_paper`, which calls `pymupdf4llm.to_markdown(..., page_chunks=True)` and normalizes the parser response into sectioned `ParsedPaper` output plus raw markdown and normalized page chunks (`src/parse_pdf.py:156`, `src/main.py:131`). Sections are derived from Markdown headers rather than from a PDF-native outline tree, and the parser currently leaves `authors=None` (`src/parse_pdf.py:181`).

- **Chunking and indexing**
  Parsed sections are split by character count with overlap and are emitted as `PaperChunk` records that preserve `section_id`, `section_title`, and page ranges while extracting equation markers, notation tokens, and figure references (`src/chunking.py:13`, `src/chunking.py:63`). The chunk stage writes `chunks.json` and updates `run_manifest.json` (`src/main.py:184`).

- **Concept extraction and dependency analysis**
  Concept extraction is implemented as an LLM-driven per-chunk stage with optional same-section supporting context retrieved by BM25 (`src/concept_graph.py:39`, `src/concept_graph.py:188`). Candidates are merged deterministically by normalized label and concept type, with synthetic prerequisite nodes created when prerequisite labels have no corresponding extracted concept (`src/concept_graph.py:104`).

- **Learner profile capture**
  The learner profile stage exists and is interactive. If `learner_profile.json` already exists for a run, it is reused; otherwise the CLI prompts for background, tensor preferences, derivation preference, depth, and pacing (`src/main.py:281`, `src/learner_profile.py:28`). The resulting profile is serialized to JSON.

- **Pedagogical lesson planning**
  Notebook planning is implemented through a structured JSON model call. The planner prompt includes the full typed learner profile and a compact concept graph, and the plan is validated for non-emptiness, duplicate titles, and unknown `source_chunk_ids` (`src/planner.py:56`, `src/planner.py:112`). Lesson section IDs from the model are normalized to deterministic `lesson_000` style IDs (`src/planner.py:49`).

- **Section-wise notebook cell generation**
  Cell generation is implemented one lesson section at a time. Retrieval for each lesson seeds from the plan’s explicit `source_chunk_ids`, then augments with lexical retrieval over all chunks (`src/cell_generator.py:86`). The generated batch is validated to require an introductory markdown cell, optional code, optional recap, and a maximum per-cell length (`src/cell_generator.py:141`, `src/cell_generator.py:184`).

- **Notebook assembly**
  Notebook assembly is implemented and uses `nbformat` object construction, not string concatenation. The assembly step checks for duplicate batches, unexpected batch section IDs, and missing planned sections before writing `final_notebook.ipynb` (`src/main.py:446`, `src/notebook_builder.py:73`). The notebook includes an overview cell summarizing the learner profile and notebook-level provenance metadata (`src/notebook_builder.py:22`, `src/notebook_builder.py:121`).

- **Validation and repair**
  Validation is implemented as a distinct stage with schema checks, overview checks, cell-length warnings, pedagogical-order validation, notation checks, tensor-shape checks, and an optional execution smoke test via `nbclient` (`src/validators.py:34`, `src/validators.py:249`). Repair is limited but real: it can rebuild a notebook from existing batches or regenerate selected sections and reassemble (`src/repair.py:23`).

### Data Contracts and Serialization

- **Typed schemas and JSON contracts between stages**
  The repository has explicit Pydantic contracts for `ParsedPaper`, `PaperChunk`, `ConceptItem`, `LearnerProfile`, `LessonSection`, `NotebookPlan`, `NotebookCell`, `NotebookBatch`, `ValidationReport`, and `RunManifest`, all built on a strict base model that forbids extra fields (`src/schemas.py:8`, `src/schemas.py:39`, `src/schemas.py:53`, `src/schemas.py:65`, `src/schemas.py:87`, `src/schemas.py:99`, `src/schemas.py:112`, `src/schemas.py:122`, `src/schemas.py:131`, `src/schemas.py:140`, `src/schemas.py:152`).

- **Final `.ipynb` construction strategy**
  Notebook construction aligns with `nbformat` and uses `nbformat.v4.new_notebook`, `new_markdown_cell`, and `new_code_cell` rather than manual JSON string assembly (`src/notebook_builder.py:73`).

- **Provenance and metadata preservation**
  Provenance is modeled in two layers:
  `StageProvenance` tracks stage name, timestamps, input artifact paths, output artifact path, code version, and optional model provenance (`src/schemas.py:22`).
  `RunManifest` tracks run-scoped artifact paths, PDF hash, params path, and active model provenance (`src/schemas.py:152`, `src/main.py:65`).
  Notebook-level metadata captures `paper_id`, `run_id`, model/backend, timestamp, pedagogical depth, and chunk provenance grouped by lesson section (`src/notebook_builder.py:22`).

- **Artifact organization**
  The artifact store creates a stage-aware run layout under `runs/<paper_slug>/<run_id>/` with fixed directories for parsed paper, chunks, concept graph, learner profile, notebook plan, cell batches, validation report, and final notebook (`src/artifact_store.py:12`, `src/artifact_store.py:35`).

### Retrieval and Inference

- **Retrieval strategy and ranking signals**
  Retrieval is lightweight and inspectable, matching the design preference. `PaperChunkRetriever` uses `rank_bm25` plus simple heuristics for preferred section ID, section-title overlap, notation overlap, equation presence for math-like queries, and boosts for title/abstract/introduction sections (`src/retrieve.py:40`, `src/retrieve.py:93`).

- **Local model backend and prompt-context policy**
  The model wrapper is kept in a single file, `src/models/gemma_4_e2b.py`, consistent with the “one model, one file” preference. The default path sends OpenAI-compatible chat-completions requests to a local `llama.cpp` server and expects schema-valid JSON responses (`src/models/gemma_4_e2b.py:69`, `src/models/gemma_4_e2b.py:96`). An optional Google Gen AI adapter exists for structured-output experiments, but the runtime defaults remain local (`src/models/gemma_4_e2b.py:139`, `src/config.py:11`, `params.yaml:1`).

- **Local runtime setup evidence**
  The repository includes a helper script for downloading `unsloth/gemma-4-E2B-it-GGUF` into `checkpoints/` and a sample `llama-server` launch command targeting `checkpoints/gemma-4-E2B-it-Q4_0.gguf` on `127.0.0.1:8080` with `--ctx-size 4096` (`download_gemma4.py:1`, `start-llama-server.txt:1`).

### Validation and Testing

- **Notebook validation approach**
  Validation includes schema validity, overview presence, pedagogical ordering, notation-consistency warnings, tensor-shape warnings, and an optional execution smoke test (`src/validators.py:34`, `src/validators.py:49`, `src/validators.py:91`, `src/validators.py:149`, `src/validators.py:192`, `src/validators.py:224`).

- **Pedagogical and notation checks**
  The code has concrete validation for notebook section order and missing notation symbols derived from `equations_to_unpack` and `tensor_shapes_to_state` (`src/validators.py:91`, `src/validators.py:149`). This is lighter than a full notation-inventory subsystem but is implemented.

- **Tensor-shape checks**
  Tensor-shape checking is present as a notebook validation pass that looks for expected tensor variables in section text (`src/validators.py:192`). There is no separate shape-helper module; the implementation is embedded in the validator logic.

- **Execution smoke tests**
  The validation stage can execute the notebook with `NotebookClient` in the run directory, using the configured timeout from `params.yaml` (`src/validators.py:224`, `src/main.py:591`, `params.yaml:9`).

- **Existing automated tests**
  The test suite is substantial and stage-oriented. It covers configuration loading, schema round-trips, parser behavior, chunking and retrieval, concept-graph merging, learner-profile prompting, planning, cell generation, notebook building, validators, artifact-store layout, CLI stage commands, and reference-pipeline flows (`pyproject.toml:43`, `tests/test_validators.py:125`, `tests/test_reference_pipeline.py:410`).

- **Verification run performed for this research**
  I ran:
  `pytest -q tests/test_config.py tests/test_schemas.py tests/test_parse_pdf.py tests/test_chunking.py tests/test_concept_graph.py tests/test_learner_profile.py tests/test_planner.py tests/test_cell_generator.py tests/test_notebook_builder.py tests/test_validators.py tests/test_artifact_store.py tests/test_reference_pipeline.py`
  Result: `77 passed, 1 warning in 0.80s`.

  I also ran:
  `pytest -q tests/test_cli.py`
  Result: `24 passed, 1 warning in 1.48s`.

## Code References

- `src/main.py:120` - CLI exposes `run`, `parse`, `chunk`, `concept`, `plan`, `generate`, `assemble`, and `validate`.
- `src/main.py:127` - Full `run` command is still a `NotImplementedError`.
- `src/main.py:281` - Planning stage reuses an existing learner profile or captures one interactively.
- `src/main.py:357` - Generation stage emits one JSON batch per lesson section.
- `src/main.py:446` - Assembly validates batch/plan alignment before building the notebook.
- `src/main.py:545` - Validation stage can trigger targeted repair when enabled.
- `src/schemas.py:39` - `ParsedPaper` contract.
- `src/schemas.py:53` - `PaperChunk` contract.
- `src/schemas.py:65` - `ConceptItem` contract.
- `src/schemas.py:112` - `NotebookPlan` contract.
- `src/schemas.py:131` - `NotebookBatch` contract.
- `src/schemas.py:140` - `ValidationReport` contract.
- `src/artifact_store.py:12` - Fixed stage directory and file naming conventions.
- `src/parse_pdf.py:156` - PDF parsing and section extraction.
- `src/chunking.py:63` - Chunk creation with overlap and metadata extraction.
- `src/retrieve.py:40` - BM25-based lexical retrieval with heuristics.
- `src/concept_graph.py:188` - Per-chunk concept extraction and merge pipeline.
- `src/planner.py:56` - Structured notebook planning from learner profile and concept graph.
- `src/cell_generator.py:141` - Per-section notebook batch generation.
- `src/notebook_builder.py:73` - Notebook assembly through `nbformat`.
- `src/validators.py:249` - Combined validation report generation.
- `src/repair.py:23` - Repair dispatch for rebuild or section regeneration.
- `src/models/gemma_4_e2b.py:69` - Single-file local model adapter plus optional cloud adapter.
- `dvc.yaml:1` - DVC stages are still placeholder shell commands, not wired to the implemented Python stages.
- `tests/test_reference_pipeline.py:410` - Reference tests cover concept, planning, generation, validation, and repair flows.
- `tests/test_cli.py:1` - CLI tests cover stage commands and artifact writing behavior.

## Alignment with Design Documents

The current workspace aligns with `documents/DESIGN_DOC.md` and `documents/CODEBASE_PREFS.md` on several important points:

- The implementation is staged rather than monolithic.
- JSON is the dominant intermediate serialization format.
- Notebook construction uses `nbformat`.
- Retrieval is lexical and inspectable.
- The default runtime shape is local Gemma via `llama.cpp`.
- Provenance is treated as a first-class concern in stage models, run manifests, and notebook metadata.
- Validation is not limited to notebook schema; it also includes pedagogical ordering and notation/tensor-shape checks.

The main partial or missing areas relative to the design documents are:

- The single-command full pipeline entry point is still not implemented (`src/main.py:127`).
- DVC orchestration is still placeholder-only despite the code having concrete stage implementations (`dvc.yaml:1`).
- There is no committed sample `runs/` artifact demonstrating an actual paper processed end to end.
- Learner capture is CLI-interactive only; there is no alternate non-interactive capture path in the current source tree.
- Notation handling and tensor-shape checks exist, but they are lighter-weight validators rather than a dedicated notation inventory or standalone shape-helper subsystem.
- The repository state documented here depends on uncommitted workspace changes, so the current implementation is ahead of the recorded commit history.

## Open Questions

- The repository’s authoritative committed status is ambiguous relative to the working tree because many core stage modules and tests are untracked or modified locally. This report reflects the current workspace, not only `HEAD`.
- The repository has a GitHub remote and is on `master`, but this report does not include permalinks because the observed codebase includes uncommitted local changes that may not exist on the remote.
- I did not execute a real PDF-to-notebook run against a live local `llama.cpp` server, so the runtime integration is documented from source code and tests rather than from a completed notebook-generation artifact.
