# Plan: Phase 9 validation and repair

## Current State
- The pipeline reaches final notebook assembly through `assemble` in `src/main.py`.
- `ValidationReport` already exists in `src/schemas.py`, but no code writes it.
- `validation_report/` is part of the artifact-store layout, but no stage uses it.
- Current validation exists only in local planning and cell-generation guards.

## Recommended Implementation Direction
- Add a notebook-level validation stage after `assemble`.
- Use the assembled notebook as the primary validation target, with `NotebookPlan`, `LearnerProfile`, and `NotebookBatch` artifacts as supporting context.
- Emit one deterministic JSON validation report per run under `validation_report/validation_report.json`.
- Keep repair targeted: rebuild notebook from saved cell specs for structural issues, regenerate only affected sections for pedagogical or content issues, and rerun only the necessary checks.

## Architecture Decisions to Lock Early
- `src/validators.py` owns notebook-level checks and report generation helpers.
- `src/repair.py` owns targeted remediation logic and should not absorb validator responsibilities.
- `src/main.py` gains a CLI stage after `assemble`, for example `validate`, that loads the final notebook and supporting artifacts, writes a `ValidationReport`, and optionally invokes repair when enabled.
- `ValidationReport` remains the primary persisted contract unless a minimal extension is needed for structured issues or repaired artifact paths.

## Risks and Mitigations
- Risk: validation logic duplicates earlier stage checks.
  Mitigation: notebook-level validators should focus on assembled output and cross-stage consistency.
- Risk: repair becomes a hidden full rerun.
  Mitigation: repair boundaries must explicitly forbid reparsing, rechunking, reconcepting, or replanning by default.
- Risk: smoke execution destabilizes local workflows.
  Mitigation: gate execution with config and honor `notebook_execution_timeout_seconds`.
- Risk: reports become non-actionable.
  Mitigation: require deterministic check names, severity, messages, and artifact references.

## Open Questions
- Whether to minimally extend `ValidationReport` for structured issue records or keep the current `errors` and `warnings` string lists in Phase 9.
- Whether repaired outputs should overwrite the notebook artifact or be written alongside the original and promoted by manifest update.

## Recommended Sequencing
1. Define validator/report boundaries and CLI inputs.
2. Implement structural notebook validation and report persistence.
3. Add pedagogical, notation, and tensor-shape checks.
4. Add optional smoke execution.
5. Add targeted repair orchestration.
6. Extend tests and reference coverage.
