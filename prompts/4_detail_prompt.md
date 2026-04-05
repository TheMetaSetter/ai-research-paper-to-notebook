Based on the plan outline in `documents/logs/MM-DD-YYYY/structure/structure-<kebab-case-topic>.md`, write the full detailed plan content in Markdown. The plan must include phases, edits within each phase, explicit edit content, validation gates, and measurable acceptance criteria.

Your detailed plan must fit `CODEBASE_PREFS.md` and `DESIGN_DOC.md`. Use precise, technical language and complete sentences.

Required content for each phase:
- Phase summary tied to the pedagogy-first paper-to-notebook objective.
- File-level edits with precise paths under `src/`, `tests/`, `examples/`, configuration files, and reproducibility files such as `dvc.yaml` when relevant.
- Explicit stage contracts for `ParsedPaper`, `PaperChunk`, `ConceptItem`, `LearnerProfile`, `LessonSection`, `NotebookPlan`, `NotebookCell`, `NotebookBatch`, and `ValidationReport`.
- Clear stage inputs, outputs, JSON serialization format, provenance fields, and run-directory layout.
- Retrieval strategy details, including lexical retrieval, ranking signals, and section-local regeneration boundaries.
- Local inference integration details for Gemma 4 E2B Instruct through `llama.cpp` server or an equally lightweight backend.
- Prompting constraints for structure-sensitive stages, including short contexts, JSON-only planning outputs where appropriate, and explicit instructions not to assume hidden future notebook cells.
- Notebook assembly details through `nbformat`, including notebook metadata such as source paper title, generation timestamp, model info, chunk provenance, pedagogical depth, and project version.
- Validation and repair logic, including schema validity, pedagogical section ordering, notation consistency, tensor-shape consistency, maximum cell-length checks, and notebook execution smoke tests.
- Readability-first implementation guidance, including explicit variable names, minimal hidden codepaths, explanatory comments, and the `1 model - 1 file` rule for model-specific logic.
- Test plan and validation steps, including unit tests for schemas, chunk overlap, metadata preservation, notebook writing and loading, notebook validation, tensor-shape helpers, config loading, and at least a few golden-style reference cases.
- Acceptance criteria that are concrete, measurable, and aligned with constrained local execution and staged regeneration.

Write the detailed plan inside:
`documents/logs/MM-DD-YYYY/detail/detail-<kebab-case-topic>.md`
