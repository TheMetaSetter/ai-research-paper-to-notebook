# Structure: Phase 9 validation and repair

## Overview
Phase 9 adds the first notebook-level quality gate after assembly. It should validate the final `.ipynb`, emit a persistent `ValidationReport`, and optionally apply narrow repair actions without reopening earlier pipeline stages.

## Implementation Phases
1. [Validator contracts and report usage] - Introduce `src/validators.py`, confirm `ValidationReport` as the persisted stage contract, define validator inputs from `run_manifest.json`, assembled notebook output, `notebook_plan.json`, `learner_profile.json`, and `cell_batches/*.json`, and add JSON report writing under `validation_report/`.
2. [Structural notebook validation] - Add `nbformat` schema validation, notebook-presence checks, and maximum cell-length checks, with stable check names and severity mapping, plus tests for valid and invalid notebooks.
3. [Pedagogical and notation checks] - Add notebook-order checks against `NotebookPlan.lesson_sections`, notation consistency checks, and tensor-shape consistency checks using notebook content plus plan and batch provenance, with deterministic warnings or errors and targeted tests.
4. [Execution smoke testing] - Add optional `nbclient` smoke execution that respects config timeouts and records execution failures in the validation report without forcing repair by default.
5. [Targeted repair flow and CLI integration] - Introduce `src/repair.py` and a new CLI stage after `assemble` that writes `validation_report/validation_report.json`, optionally rebuilds the notebook or regenerates affected section batches, and updates run artifacts only for repaired outputs.
6. [Tests and reproducibility hooks] - Add `tests/test_validators.py`, extend CLI coverage, add a golden validation-report case, and preserve stage-aware run outputs and provenance fields for repaired artifacts.

## Stable Contracts Per Phase
- `ValidationReport` is the persisted output contract for the validation stage.
- The assembled notebook remains the primary validation input.
- `NotebookPlan`, `LearnerProfile`, and `NotebookBatch` remain supporting contracts and are not widened unless a minimal Phase 9 extension is required.
- Artifact outputs stay under the existing run layout, especially `notebook/` and `validation_report/`.
