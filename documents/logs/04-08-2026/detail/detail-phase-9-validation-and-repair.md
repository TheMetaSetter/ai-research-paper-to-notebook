# Detailed Plan: Phase 9 validation and repair

## Phase Summary
Phase 9 adds notebook-level validation and targeted repair after the existing `assemble` stage. The goal is to validate the final `.ipynb` for structural correctness and pedagogical quality, emit a persistent `ValidationReport`, and optionally repair only the failing notebook or section artifacts without re-running earlier pipeline stages.

## Phase 1: Validator contracts and report persistence
This phase establishes the validation-stage boundary and makes `ValidationReport` a real persisted artifact rather than a reserved schema.

### File-level edits
- Add `src/validators.py`.
- Update `src/main.py` with a validation CLI stage after `assemble`.
- Reuse `src/schemas.py` `ValidationReport`; extend it only if a minimal structured-issue field is necessary.
- Reuse `src/artifact_store.py` `validation_report/` location.
- Add `tests/test_validators.py`.

### Stage contracts
- Inputs: `run_manifest.json`, `notebook/final_notebook.ipynb`, `notebook_plan/notebook_plan.json`, `learner_profile/learner_profile.json`, `cell_batches/*.json`, and optional `parsed_paper/parsed_paper.json`.
- Output: `validation_report/validation_report.json`.
- Serialization: JSON for the report, `.ipynb` remains unchanged unless repair is triggered.
- Provenance fields: run id, notebook path, validation timestamp, check names, active model provenance when available, input artifact paths, and code version.

### Explicit edit content
- Implement a `ValidationIssue`-style internal structure in `src/validators.py` and translate it into `ValidationReport.errors`, `ValidationReport.warnings`, and `ValidationReport.checks_run`.
- Add a top-level validation entrypoint that loads the assembled notebook and supporting artifacts, runs checks, constructs a `ValidationReport`, and writes it to `validation_report/validation_report.json`.
- Add a CLI command in `src/main.py`, for example `validate(run_directory: Path)`, that always writes the report and prints the report path.

### Validation gates
- Validation must fail fast if the notebook artifact is missing.
- Validation must still write a report when checks fail.
- The report must be deterministic for the same notebook and supporting artifacts.

### Acceptance criteria
- Running the new CLI stage on a valid assembled notebook writes `validation_report/validation_report.json`.
- The report records `is_valid`, check names, errors, warnings, notebook path, and validation timestamp.
- No earlier stage artifacts are regenerated in this phase.

## Phase 2: Structural notebook validation
This phase adds notebook-level checks that do not depend on repair or model re-entry.

### File-level edits
- Expand `src/validators.py`.
- Add structural tests in `tests/test_validators.py`.
- Extend CLI coverage in `tests/test_cli.py`.

### Explicit checks
- `nbformat.validate` schema validity.
- Notebook contains cells and a learner-facing overview cell.
- Maximum cell-length checks on assembled notebook cells.
- Cell metadata presence checks where provenance is required for later validation.

### Validation gates
- Schema failures are errors.
- Oversized cells are warnings unless they make the notebook unreadable by defined threshold.
- Missing notebook file or unreadable notebook JSON is an error.

### Acceptance criteria
- Invalid notebook schema is captured in the report.
- Oversized cells produce warnings.
- A structurally valid notebook produces no structural errors.

## Phase 3: Pedagogical, notation, and tensor-shape checks
This phase adds cross-stage validation using the assembled notebook plus planning and batch provenance.

### File-level edits
- Expand `src/validators.py`.
- Reuse `src/notebook_builder.py` metadata and `NotebookPlan` ordering as validation inputs.
- Add focused tests in `tests/test_validators.py` and a golden-style reference case in `tests/test_reference_pipeline.py`.

### Explicit checks
- Pedagogical ordering against `NotebookPlan.lesson_sections`.
- Missing planned section detection in the final notebook.
- Notation consistency checks using notebook text and section or chunk provenance.
- Tensor-shape consistency checks using notebook text and `LessonSection.tensor_shapes_to_state` or batch provenance when available.

### Validation gates
- Section-order mismatches are errors.
- Undefined or drifting notation is at least a warning, and an error when it breaks section meaning.
- Missing required tensor-shape statements for sections that request them is at least a warning.

### Acceptance criteria
- A notebook with swapped section order fails validation.
- A notebook with missing notation definitions or missing tensor-shape grounding is flagged deterministically.
- The report distinguishes structural failures from pedagogical warnings.

## Phase 4: Execution smoke testing
This phase adds optional execution validation without turning Phase 9 into a mandatory heavy runtime.

### File-level edits
- Expand `src/validators.py`.
- Reuse config values from `src/config.py`.
- Add execution-focused tests in `tests/test_validators.py` with controlled failure stubs.

### Explicit edit content
- Add an `nbclient` smoke execution helper that uses `notebook_execution_timeout_seconds`.
- Gate smoke execution behind existing config such as `enable_notebook_execution_smoke_test`.
- Record execution failures in the report with clear messages tied to the failing validation check.

### Validation gates
- Smoke execution is skipped when disabled and still recorded in `checks_run`.
- Smoke execution failure does not prevent report writing.

### Acceptance criteria
- Successful smoke execution is recorded.
- Execution errors are captured in the report.
- Timeouts respect configured limits.

## Phase 5: Targeted repair and CLI integration
This phase introduces narrow repair behavior and keeps regeneration boundaries explicit.

### File-level edits
- Add `src/repair.py`.
- Update `src/main.py` validation CLI stage to optionally invoke repair.
- Extend `tests/test_cli.py` and `tests/test_reference_pipeline.py`.

### Repair boundaries
- If schema validation fails because notebook assembly output is malformed, rebuild the notebook from stored `cell_batches/*.json` and rerun structural validation.
- If pedagogical ordering or missing-section issues are detected, regenerate only affected section batches, then reassemble and rerun validation.
- If smoke execution fails in one code cell, isolate the failing section, request a local repair pass for that section only, then reassemble and rerun validation.
- Do not re-run parse, chunk, concept extraction, or notebook planning by default.

### Artifact outputs
- Always keep `validation_report/validation_report.json`.
- If repair writes a changed notebook or batch artifact, persist it under the existing run tree and update manifest paths only for the repaired outputs that become active.

### Acceptance criteria
- Repair is optional and gated by `enable_repair_pass`.
- A structural repair can rebuild from saved cell batches without regenerating earlier stages.
- A section-local repair touches only the affected batch or notebook outputs.

## Readability-first implementation guidance
- Use explicit names such as `validation_issues`, `ordered_notebook_cells`, `repaired_notebook_path`, and `failed_check_names`.
- Keep validator functions small and grouped by behavior rather than hidden behind dynamic registries.
- Add short comments only where report aggregation or repair routing is not self-explanatory.
- Keep all model-specific repair prompting inside `src/models/gemma_4_e2b.py` or `src/repair.py` orchestration that calls that model wrapper, preserving the `1 model - 1 file` rule.

## Test plan and validation steps
- Add `tests/test_validators.py` for schema validation, cell-length warnings, pedagogical ordering failures, notation or tensor-shape inconsistency detection, and smoke execution reporting.
- Extend `tests/test_cli.py` for the new validation command, manifest updates, missing-notebook failures, and optional repair flows.
- Extend `tests/test_reference_pipeline.py` with a deterministic golden validation-report case on an assembled notebook.
- Keep notebook write/load validation through `nbformat` and preserve current assembly tests in `tests/test_notebook_builder.py`.

## Measurable acceptance criteria
- The repo gains a CLI stage after `assemble` that validates the final notebook and writes `validation_report/validation_report.json`.
- `src/validators.py` and `src/repair.py` exist and are covered by focused Pytest cases.
- The validation stage can detect schema failures, section-order failures, oversized cells, notation or tensor-shape issues, and execution failures.
- Repair stays notebook-local or section-local and does not re-run earlier pipeline stages by default.
- The detailed implementation remains aligned with constrained local execution, JSON stage artifacts, `nbformat` notebook assembly, and explicit provenance.
