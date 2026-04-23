# Product Completion Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current staged notebook-generation codebase into a usable v1 product that can reliably take one research paper plus one learner profile and produce a validated teaching notebook end to end.

**Architecture:** Keep the existing staged pipeline. Product completion should focus on closing orchestration, reproducibility, input UX, golden-path verification, and release readiness rather than redesigning the architecture. The current code already has most stage modules; the remaining work is to harden the pipeline into one coherent product surface.

**Tech Stack:** Python 3.11+, Typer CLI, Pydantic, nbformat, nbclient, PyMuPDF4LLM, rank-bm25, httpx, llama.cpp, pytest, DVC

---

## What “done” means for this product

The product should be considered complete for v1 when all of the following are true:

- A user can run one documented command and produce a final `.ipynb` from a paper plus learner profile.
- The pipeline writes deterministic stage artifacts under `runs/<paper_slug>/<run_id>/`.
- The product has one reference paper and one reference learner profile that pass the golden path locally.
- Validation catches notebook-structure issues, ordering issues, notation/tensor-shape issues, and execution failures.
- The repository documents how to download the model, start the local server, run the pipeline, and inspect artifacts.
- The committed history reflects the actual product state instead of relying on uncommitted workspace changes.

## Product strategy

Do not spend the next cycle improving model cleverness first. The current highest-leverage work is productization of the existing pipeline:

1. Make the golden path real.
2. Make the pipeline runnable from one command.
3. Make inputs and outputs reproducible.
4. Make failures inspectable and recoverable.
5. Make onboarding and docs clear enough that another engineer can run it without tribal knowledge.

## File map for the completion work

**Core code already in place**

- `src/main.py`
- `src/config.py`
- `src/schemas.py`
- `src/artifact_store.py`
- `src/parse_pdf.py`
- `src/chunking.py`
- `src/retrieve.py`
- `src/concept_graph.py`
- `src/learner_profile.py`
- `src/planner.py`
- `src/cell_generator.py`
- `src/notebook_builder.py`
- `src/validators.py`
- `src/repair.py`
- `src/models/gemma_4_e2b.py`

**Files that should be added or completed for product readiness**

- `README.md`
- `.env.example`
- `examples/sample_papers/<reference-paper>.pdf`
- `examples/sample_runs/` or one documented `runs/` snapshot strategy
- `tests/test_end_to_end_cli.py` or equivalent end-to-end smoke coverage
- `tests/test_run_command.py`
- `tests/test_dvc_pipeline.py` if DVC remains part of the product story

**Files that need alignment**

- `dvc.yaml`
- `params.yaml`
- `documents/DESIGN_DOC.md`
- `documents/IMPLEMENTATION_PLAN.md`
- `documents/CODEBASE_PREFS.md`

## Milestone 1: Freeze the real baseline

### Task 1: Reconcile “current workspace” with committed history

**Why this comes first**

Right now the repo’s actual implementation is ahead of `HEAD`. That makes every later decision ambiguous. Before finishing the product, the team needs one truthful baseline.

**Files**

- Review: `src/*.py`, `tests/*.py`, `.gitignore`
- Review: `documents/logs/04-20-2026/research/research-current-codebase-status.md`

- [ ] Confirm which untracked and modified files are part of the intended product baseline.
- [ ] Remove purely local noise from the baseline decision, especially `.DS_Store` and any accidental artifacts.
- [ ] Make one intentional “current pipeline baseline” commit so the repo history matches the implementation being discussed.
- [ ] Re-run the current passing tests after the baseline commit.

**Acceptance gate**

- `git status --short` is clean or intentionally clean enough.
- The repo has one commit that truthfully contains the current staged pipeline.

## Milestone 2: Make the golden path real

### Task 2: Define the v1 input contract

**Why this matters**

A product is not “a bunch of stages.” It is a repeatable transformation from known inputs to known outputs.

**Files**

- Modify: `README.md`
- Modify: `examples/sample_learner_profile.json`
- Create or verify: `examples/sample_papers/`
- Modify: `src/config.py`
- Modify: `src/learner_profile.py`

- [ ] Decide the supported v1 input modes:
  `paper2nb run <pdf> --params-path ... --learner-profile ...`
  and optionally interactive fallback only if no learner profile file is provided.
- [ ] Add a documented sample learner profile that is good enough for the golden path.
- [ ] Add one reference paper small enough to run locally on the target machine.
- [ ] Make the learner profile file path a first-class input to the full run, not only an interactive prompt inside the staged flow.

**Acceptance gate**

- One reference paper and one reference learner profile can be named in docs and tests as the official v1 example inputs.

### Task 3: Implement the full `run` orchestration

**Why this matters**

This is the biggest missing product gap. The architecture exists, but the user-facing end-to-end command does not.

**Files**

- Modify: `src/main.py`
- Test: `tests/test_run_command.py`
- Test: `tests/test_end_to_end_cli.py`

- [ ] Replace the `NotImplementedError` in `paper2nb run` with orchestration over parse, chunk, concept, plan, generate, assemble, and validate.
- [ ] Make `run` write one `run_manifest.json` and update it as stages complete.
- [ ] Make `run` stop cleanly on stage failure and report the failing artifact location.
- [ ] Make `run` optionally skip execution smoke tests or repair via existing `params.yaml` toggles.
- [ ] Keep the per-stage commands intact; `run` should compose them, not replace them.

**Acceptance gate**

- One command produces a final notebook and validation report for the golden-path example.

## Milestone 3: Make outputs reproducible and inspectable

### Task 4: Replace placeholder DVC with the actual stage graph

**Why this matters**

The docs promise reproducibility, but `dvc.yaml` still says “placeholder.” That gap weakens the product story.

**Files**

- Modify: `dvc.yaml`
- Modify: `params.yaml`
- Modify: `README.md`

- [ ] Decide whether DVC is truly in-scope for v1.
- [ ] If yes, wire `dvc.yaml` to the actual Python entry points and artifact directories.
- [ ] If no, remove DVC from the short-term product promise in docs and leave it as a clearly marked future item.
- [ ] Ensure the chosen path matches the README and implementation plan.

**Acceptance gate**

- There is no mismatch between docs and reality on reproducibility.

### Task 5: Add one golden reference run

**Why this matters**

A product needs a reference output that proves the system can complete its core job.

**Files**

- Create: `examples/sample_runs/README.md` or equivalent documentation
- Test: `tests/test_end_to_end_cli.py`
- Optionally create: one small committed reference artifact or a reproducible command that generates it

- [ ] Pick the smallest representative paper for a stable local run.
- [ ] Generate a reference notebook from it using the official sample learner profile.
- [ ] Decide what to commit:
  either a small golden notebook artifact,
  or a stable snapshot of intermediate JSON contracts,
  or only a documented reproducible command if artifacts are too large.
- [ ] Add a test that exercises the golden path at the highest level the repo can afford locally.

**Acceptance gate**

- Another engineer can reproduce the reference output by following the docs.

## Milestone 4: Harden quality gates

### Task 6: Tighten validation so it represents product quality, not only schema sanity

**Why this matters**

The current validators are a strong start, but product readiness depends on treating them as release gates.

**Files**

- Modify: `src/validators.py`
- Modify: `src/repair.py`
- Test: `tests/test_validators.py`
- Test: `tests/test_reference_pipeline.py`

- [ ] Keep schema, ordering, notation, tensor-shape, and execution checks as the required minimum.
- [ ] Add explicit pass/fail semantics in docs for what blocks release versus what is only a warning.
- [ ] Ensure repair behavior is deterministic enough to be trusted in local runs.
- [ ] Add at least one test where validation fails for a realistic pedagogical reason and repair is attempted.

**Acceptance gate**

- The team can answer: “What exact conditions make a notebook acceptable for v1?”

### Task 7: Close remaining test coverage gaps that matter for product completion

**Files**

- Create or modify: `tests/test_run_command.py`
- Create or modify: `tests/test_end_to_end_cli.py`
- Optionally create: `tests/test_repair_cli.py`

- [ ] Keep the fast unit and contract tests already present.
- [ ] Add coverage for the true `run` command.
- [ ] Add coverage for learner-profile-file input.
- [ ] Add coverage for one failed run path with a clear error message and artifact trail.
- [ ] Add coverage for one successful assembled notebook that also validates.

**Acceptance gate**

- Test coverage mirrors the actual product surface, not only stage internals.

## Milestone 5: Make the product usable by someone else

### Task 8: Write the real README

**Why this matters**

Without a usable README, the repo is still a research scaffold.

**Files**

- Create or modify: `README.md`
- Create or modify: `.env.example`
- Reference: `download_gemma4.py`
- Reference: `start-llama-server.txt`

- [ ] Add a 30-second product description.
- [ ] Add prerequisites for Apple Silicon local usage.
- [ ] Add exact steps to download Gemma GGUF and start `llama-server`.
- [ ] Add exact steps to run the golden path.
- [ ] Add a short explanation of the stage artifacts under `runs/`.
- [ ] Add a troubleshooting section for common failures: missing `pymupdf4llm`, missing `rank-bm25`, local model server unavailable, notebook execution failure.

**Acceptance gate**

- A new engineer can run the product locally without asking the author for missing steps.

### Task 9: Align docs so they stop contradicting current reality

**Files**

- Modify: `documents/DESIGN_DOC.md`
- Modify: `documents/IMPLEMENTATION_PLAN.md`
- Modify: `documents/CODEBASE_PREFS.md`

- [ ] Keep `DESIGN_DOC` as the architectural source of truth.
- [ ] Update `IMPLEMENTATION_PLAN` to reflect what is already built versus what remains.
- [ ] Ensure `CODEBASE_PREFS` still matches the actual repository boundaries and product expectations.
- [ ] Remove any doc language that still describes implemented features as future placeholders.

**Acceptance gate**

- The docs tell the truth about the current product and its next gaps.

## Milestone 6: Decide what not to do in v1

### Task 10: Explicitly cut scope

**Why this matters**

The fastest way to never finish this product is to keep treating it like a general notebook intelligence platform.

- [ ] Keep figure/table semantic reconstruction out of v1.
- [ ] Keep dense retrieval out of v1 unless lexical retrieval demonstrably fails on the golden path.
- [ ] Keep multi-paper synthesis out of v1.
- [ ] Keep cloud-first inference out of v1.
- [ ] Keep browser/agentic paper augmentation out of v1.

**Acceptance gate**

- The team can explain v1 in one sentence:
  “Local, staged, pedagogy-first generation of a validated notebook from one AI paper and one learner profile.”

## Recommended execution order

If I were finishing this product pragmatically, I would do it in this order:

1. Commit the current real baseline.
2. Implement `paper2nb run`.
3. Add learner-profile-file input.
4. Add one reference paper and one golden path.
5. Replace or de-scope placeholder DVC.
6. Harden release-grade validation and repair semantics.
7. Write the README and update docs.

## Release checklist

- [ ] `git status --short` is clean.
- [ ] `paper2nb run <reference-paper> --learner-profile <sample-profile>` works.
- [ ] Final notebook is created.
- [ ] Validation report is created.
- [ ] Golden-path tests pass.
- [ ] Docs match reality.
- [ ] Model download and server startup instructions are complete.

## Recommended first implementation slice

If only one slice should be tackled next, it should be:

**“Implement the real `run` command with learner-profile-file input and verify it on one reference paper.”**

That slice creates the first actual product boundary. Everything else becomes easier once that is real.
