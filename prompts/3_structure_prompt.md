Using the research note at `documents/logs/MM-DD-YYYY/research/research-<kebab-case-topic>.md` and the implementation plan at `documents/logs/MM-DD-YYYY/plan/plan-<kebab-case-topic>.md`, create an initial phase outline for the paper-to-notebook codebase.

The outline must reflect `CODEBASE_PREFS.md` and `DESIGN_DOC.md`. Preserve the staged architecture, keep schemas and stage boundaries ahead of prompt convenience, and keep the minimal vertical slice ahead of advanced pedagogy or repair features. Use precise, technical language and complete sentences.

Each phase should indicate:
- which pipeline stages it covers,
- which modules or files it introduces or changes,
- which contracts become stable in that phase,
- which validation and test work must ship with it,
- how readability-first and reproducibility requirements are preserved.

---

Here is the proposed plan structure:

## Overview
[One to two sentence summary grounded in pedagogy-first notebook generation, constrained local inference, and explicit stage contracts.]

## Implementation Phases
1. [Foundation and contracts] - [Configuration, schemas, run directory layout, provenance fields, and a thin executable entrypoint]
2. [Ingestion and retrieval foundation] - [PDF parsing, chunking, indexing, and lightweight retrieval with inspectable metadata]
3. [Pedagogical planning inputs] - [Concept extraction, learner profile capture, and lesson planning contracts]
4. [Section generation and notebook assembly] - [Section-wise notebook cell generation, `nbformat` assembly, and notebook metadata]
5. [Validation, repair, and reproducibility] - [Validators, smoke tests, repair flow, golden tests, and data-versioning support]

Does this phasing make sense? Should the order or granularity be adjusted before the detailed plan is written?

---

Write the outline inside a file named `structure-<kebab-case-topic>.md` under:

`documents/logs/MM-DD-YYYY/structure/`

Example filename:

`documents/logs/MM-DD-YYYY/structure/structure-<kebab-case-topic>.md`

Get feedback on the structure before writing the detailed content.
