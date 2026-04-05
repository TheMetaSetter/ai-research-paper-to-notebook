# Codebase Preferences for `paper-to-notebook`

## 1. Reference mechanisms and source code alignment

### For pipeline architecture
1. The codebase needs to follow the staged pipeline defined in `DESIGN_DOC.md`, namely: PDF ingestion, chunking and indexing, concept extraction, learner profile capture, pedagogical planning, section-wise cell generation, notebook assembly, and validation or repair.
2. Do not collapse the whole system into one monolithic script or one monolithic prompt. Each stage should have explicit inputs, outputs, and validation points.

### For notebook generation and validation
- Notebook construction needs to align with the `nbformat` notebook model described in `DESIGN_DOC.md`. Build `.ipynb` files through structured notebook objects, not through ad hoc string concatenation.
- Validation logic should align with the design target: schema validity, pedagogical section ordering, notation consistency, tensor-shape consistency where applicable, and notebook execution smoke tests.

### For local inference and retrieval
- Local inference integration should follow the design target in `DESIGN_DOC.md`: Gemma 4 E2B Instruct as the v1 model target, served locally through `llama.cpp` server or another equally lightweight local backend.
- Retrieval should stay lightweight and inspectable in v1. Prefer chunk-local lexical retrieval with explicit provenance over opaque retrieval abstractions.

---

## 2. Logging and task organization

- Place generated artifacts and logs under a stage-aware structure, with one directory per paper or run, so intermediate outputs can be inspected and regenerated without rerunning the whole pipeline.
- Organize intermediate artifacts by stage, such as `parsed_paper`, `chunks`, `concept_graph`, `learner_profile`, `notebook_plan`, `cell_batches`, `validation_report`, and final `.ipynb` output.
- Keep JSON as the default serialization format between stages. Use `.ipynb` only for the final assembled notebook artifact.

- Logs and artifacts should preserve provenance back to the source paper, including chunk ids, section titles, page references when available, model info, and generation timestamps.
- The organization of artifacts should make section-local regeneration easy. A failed cell-generation or validation step should not force reparsing the paper from the beginning.

---

## 3. Planning and codebase workflow order

- Plan the data contracts and stage boundaries before implementing generation details. In this project, schemas and pipeline boundaries come before prompt convenience.

- Traverse and grow the codebase using the repository structure targeted in `DESIGN_DOC.md`, especially the separation across `src/`, `tests/`, and `examples/`.
- The intended core module order should stay close to the design target: `config.py`, `schemas.py`, `parse_pdf.py`, `chunking.py`, `retrieve.py`, `concept_graph.py`, `learner_profile.py`, `planner.py`, `cell_generator.py`, `notebook_builder.py`, `validators.py`, `repair.py`, and `main.py`.
- If the repository structure changes in a meaningful way, update `DESIGN_DOC.md` so the architectural source of truth does not drift away from the implementation.

---

## 4. Model file organization and self-contained design

- Keep everything related to one particular model placed inside 1 single file for that model.

- Stick to `1 model - 1 file` rule.

- All inference logic and training logic of one model need to be placed in one single file of that model, in a way such that user can read the easiest.

- Core logic of one model, including inference and training logic, should be well-written and can be read from top-to-bottom in ONE SINGLE FILE of that model. The purpose is to make this self-contained.

- All calculations directly related to one model need to be placed inside the single file of that model, or more ideally, to be placed within Python classes of that model.

- All logic related to one model needs to be placed in one single file of that model.

---

## 5. Readability-first principles

This is important so I will repeat 3 times.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In research codebase like this one, READABILITY IS KING. So you can DO repeat yourself in a thoughtful way to make users read the code with ease. Variables should be named explicitly, with full words, even several words, readability is primordial.

- In this codebase, readability is king. So, DO repeat yourself in a thoughtful way.

- Stick to "least amount of codepaths" principle, which means configurations, models, pre-processing, all of that should be obvious to users. Reading should be obvious and configurations should be obvious.

- Write explanatory comments to support user reading code. Comments should be updated in parallel with code or implementations.

- Write explanatory comments to support user reading code. Comments should be updated in parallel with code or implementations.

---

## 6. Data versioning and reproducibility

- Each time we parse a paper, chunk it, plan a notebook, or generate notebook cells, we are creating a new derived artifact of the source paper and learner profile.

- So, please use data version control techniques, such as `dvc.yaml`, to track source papers, parsed representations, chunk collections, reference lesson plans, and sample notebook outputs, so history can be reproduced without effort.
- Reproducibility here means the user should be able to recover which paper, which learner profile, which retrieval inputs, and which model configuration produced one particular notebook.

---

## 7. Testing requirements

- Add small and minimalistic test cases to check for these things:
    - Validate typed schemas and JSON contracts between pipeline stages.
    - Test chunking logic, especially chunk overlap, metadata preservation, and section or page provenance.
    - TEST THE MECHANISM TO WRITE, LOAD, AND VALIDATE NOTEBOOKS. THIS IS SUPER IMPORTANT.
    - Test notebook builders to make sure the resulting `.ipynb` is structurally valid and keeps required metadata.
    - Test pedagogical validators to check required section components, notation coverage, and maximum cell-length constraints.
    - Test tensor-shape helper functions or shape-check primitives used in explanations and validation.
    - Test the functions or methods within classes that are used to initialize configurations from definition files, such as `.yaml` files or equivalent config objects.
    - Maintain at least a few golden-style tests for reference papers or reference lesson plans, so planning regressions are visible.
    - Design and implement simple test cases using Pytest

---

## 8. Ablation study friendliness

- Design the codebase such that components (e.g., modules, loss terms, preprocessing steps, augmentations) can be easily turned on or turned off without modifying core logic.

- Each component should be controllable via clear and explicit configuration (e.g., `.yaml`), avoiding hidden dependencies or implicit coupling.

- Avoid hard-coded interactions between components; instead, use modular design so that removing or disabling one component does not break the pipeline.

- Ensure that enabling or disabling components results in minimal changes in codepaths, so that ablation experiments are easy to run, compare, and reproduce.
