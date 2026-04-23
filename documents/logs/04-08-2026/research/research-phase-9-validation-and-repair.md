---
date: 2026-04-08 18:49:01 +07 +0700
researcher: TheMetaSetter
git_commit: c39218ea7046dbba2128aabee42108447ec1d854
branch: master
repository: ai-research-paper-to-notebook
topic: "Phase 9 validation and repair"
tags: [research, paper-to-notebook, pedagogical-notebooks, local-inference]
status: complete
last_updated: 2026-04-08
last_updated_by: TheMetaSetter
---

# Research: Phase 9 validation and repair

**Date**: 2026-04-08 18:49:01 +07 +0700  
**Researcher**: TheMetaSetter  
**Git Commit**: `c39218ea7046dbba2128aabee42108447ec1d854`  
**Branch**: `master`

## Research Question
What is the current repository state relevant to implementing Phase 9 validation and repair, and what constraints must the detailed planning artifact preserve?

## Summary
The repository now implements the staged pipeline through notebook assembly. `src/notebook_builder.py` and the `assemble` CLI command exist, and the full test suite currently passes. Phase 9 remains unimplemented: `src/validators.py` and `src/repair.py` do not exist, `validation_report/` exists only as a reserved run-directory location, and there is no notebook-level validation or repair command after assembly.

## Detailed Findings

### Pipeline Stages
- PDF ingestion and parsing are implemented in `src/parse_pdf.py` and exposed through `parse`.
- Chunking and retrieval are implemented in `src/chunking.py` and `src/retrieve.py`.
- Concept extraction, learner profile capture, and notebook planning are implemented in `src/concept_graph.py`, `src/learner_profile.py`, and `src/planner.py`.
- Section-wise cell generation is implemented in `src/cell_generator.py`.
- Notebook assembly is implemented in `src/notebook_builder.py` and exposed through `assemble` in `src/main.py`.
- Validation and repair are still absent as notebook-level stages.

### Data Contracts and Serialization
- Stage contracts are centralized in `src/schemas.py`.
- The repository already defines `ParsedPaper`, `PaperChunk`, `ConceptItem`, `LearnerProfile`, `LessonSection`, `NotebookPlan`, `NotebookCell`, `NotebookBatch`, and `ValidationReport`.
- Intermediate artifacts are serialized as JSON in run-stage directories.
- Final notebook output is written as `.ipynb` through `nbformat`, not manual JSON concatenation.

### Retrieval and Inference
- Retrieval remains lightweight and inspectable through lexical chunk retrieval.
- Model integration remains centered on `src/models/gemma_4_e2b.py`.
- Current validation logic is limited to stage-local checks inside planning and cell generation, not full assembled-notebook quality checks.

### Validation and Testing
- `src/cell_generator.py` validates notebook batches for intro-cell presence, code-example requirements, recap presence, and maximum per-cell line count.
- `src/planner.py` validates notebook plan structure before persistence.
- There is no `nbformat.validate` call for final notebooks in the main pipeline.
- There is no `nbclient` smoke execution stage after notebook assembly.
- There is no repair orchestration for schema, pedagogical, or execution failures.
- Current tests include notebook assembly coverage in `tests/test_notebook_builder.py` and CLI coverage for `assemble` in `tests/test_cli.py`.
- Repository baseline is currently green: `87 passed`.

## Code References
- `src/schemas.py` - defines `ValidationReport` and all stage contracts.
- `src/artifact_store.py` - reserves `validation_report/` in the standard run layout.
- `src/main.py` - stops at `assemble`; no validation or repair command exists.
- `src/notebook_builder.py` - assembles final `.ipynb` output through `nbformat`.
- `src/cell_generator.py` - contains local batch-level validation only.
- `tests/test_notebook_builder.py` - verifies notebook assembly ordering and metadata.
- `tests/test_cli.py` - verifies `assemble` behavior and notebook artifact persistence.

## Alignment with Design Documents
- The repository matches the staged architecture through notebook assembly.
- It matches the JSON-between-stages and `nbformat`-for-final-output requirements.
- It does not yet match the design goal for notebook-level schema validation, pedagogical ordering checks, notation consistency, tensor-shape consistency, smoke execution, or targeted repair.

## Open Questions
- Whether Phase 9 should treat notation and tensor-shape checks as notebook-text-only heuristics or combine notebook content with `NotebookPlan` and `NotebookBatch` provenance.
- Whether repair should write regenerated section batches in place or emit alternate repaired artifacts first and promote them only on success.
