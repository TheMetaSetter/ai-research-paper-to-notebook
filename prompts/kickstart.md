Initialize the prompt set for this repository using `CODEBASE_PREFS.md` and `DESIGN_DOC.md` as the architectural source of truth.

Requirements:
- Use precise, technical, readability-first language with complete sentences.
- Reflect the staged pipeline defined in `DESIGN_DOC.md`: PDF ingestion, chunking and indexing, concept extraction, learner profile capture, pedagogical lesson planning, section-wise notebook cell generation, notebook assembly, and validation or repair.
- Do not collapse the workflow into one monolithic script, one monolithic prompt, or one one-shot notebook-generation pass. Each stage must have explicit inputs, outputs, and validation points.
- Treat pedagogy as primary. Prompts must require learner modeling and pedagogical planning before notebook-cell generation.
- Keep JSON as the default serialization format between stages. Treat `.ipynb` as the final assembled artifact only, written through `nbformat` rather than string concatenation.
- Require explicit typed contracts for stage objects such as `ParsedPaper`, `PaperChunk`, `ConceptItem`, `LearnerProfile`, `LessonSection`, `NotebookPlan`, `NotebookCell`, `NotebookBatch`, and `ValidationReport`.
- Preserve provenance in all stage outputs, including chunk ids, section titles, page references when available, model information, and generation timestamps.
- Require stage-aware artifact organization so one run or paper can be inspected and regenerated locally by stage, including directories such as `parsed_paper`, `chunks`, `concept_graph`, `learner_profile`, `notebook_plan`, `cell_batches`, `validation_report`, and final notebook output.
- Align local inference with the v1 target in `DESIGN_DOC.md`: Gemma 4 E2B Instruct through `llama.cpp` server or an equally lightweight local backend.
- Prefer lightweight, inspectable retrieval. Prompts should favor chunk-local lexical retrieval with explicit provenance over opaque retrieval abstractions.
- Require short-context, retrieval-first generation. Do not assume hidden future sections or hidden future notebook cells.
- Emphasize notebook pedagogy requirements: notation normalization, equation unpacking, tensor-shape walkthroughs, runnable toy code with explanatory comments, interpretation of outputs, and recap or exercise cells.
- Require specific implementation guidance with concrete file paths, module names, interfaces, schema changes, tests, configuration changes, validation procedures, and artifact-layout expectations.
- Emphasize `src/`, `tests/`, and `examples/` separation, with implementation order close to `config.py`, `schemas.py`, `parse_pdf.py`, `chunking.py`, `retrieve.py`, `concept_graph.py`, `learner_profile.py`, `planner.py`, `cell_generator.py`, `notebook_builder.py`, `validators.py`, `repair.py`, and `main.py`.
- Reinforce readability-first principles from `CODEBASE_PREFS.md`: explicit variable names, minimal hidden codepaths, thoughtful repetition when it improves clarity, and explanatory comments that are updated together with implementation.
- Preserve the `1 model - 1 file` rule whenever model-specific inference logic is introduced.
- Require tests for schema validation, chunk overlap and provenance, notebook writing and loading, notebook validation, pedagogical validators, tensor-shape helpers, configuration loading, and a few golden-style reference cases.
- Encourage reproducibility through stage-aware logs and data-versioned artifacts, including `dvc.yaml` or an equivalent reproducibility mechanism when planning implementation work.
