## description: Document the paper-to-notebook repository as it exists today

# Research the Paper-to-Notebook Repository and Its Staged Pedagogical Generation Pipeline

You are tasked with conducting repository research for a system that converts an academic AI research paper into an interactive, self-contained Jupyter notebook that teaches the paper from first principles. The architectural source of truth is `DESIGN_DOC.md`, and the implementation preferences are defined in `CODEBASE_PREFS.md`.

## CRITICAL: YOUR ONLY JOB IS TO DOCUMENT AND EXPLAIN THE REPOSITORY AS IT EXISTS TODAY

- Do not propose optimizations, alternative architectures, or new features unless explicitly asked.
- Do not invent missing modules, stage outputs, or tests.
- Do not collapse multiple stages into one narrative if the repository separates them.
- Do not assume the full design is already implemented.
- If the implementation deviates from `DESIGN_DOC.md`, document the deviation clearly without prescribing a fix.
- You are creating a technical map of the current paper-to-notebook workflow, artifacts, contracts, and validation logic.

## Initial Setup

When this command is invoked, respond with:

I am ready to research the paper-to-notebook repository. Please provide your research question or area of interest, and I will analyze the staged pipeline, data contracts, artifact flow, local inference setup, and notebook validation procedures.

Then wait for the user's research query.

## Steps to follow after receiving the research query

1. Read directly mentioned files first.
   - If the user mentions specific files, read them fully before broad searches.
   - Build your initial understanding from the explicitly referenced files before exploring related modules.
2. Read the architectural context early.
   - Read `CODEBASE_PREFS.md` and `DESIGN_DOC.md` near the start.
   - Use them to interpret intended terminology, stage boundaries, and validation goals.
   - Do not treat design intent as implemented fact.
3. Analyze and decompose the research question.
   - Break the question into repository-specific research areas.
   - Map the question onto the staged pipeline: PDF ingestion, chunking and indexing, concept extraction, learner profile capture, pedagogical planning, section-wise cell generation, notebook assembly, and validation or repair.
   - Create a research plan using `update_plan` and track progress visibly.
4. Conduct comprehensive repository research.
   - Locate schema or contract definitions.
   - Locate modules or placeholders corresponding to `config.py`, `schemas.py`, `parse_pdf.py`, `chunking.py`, `retrieve.py`, `concept_graph.py`, `learner_profile.py`, `planner.py`, `cell_generator.py`, `notebook_builder.py`, `validators.py`, `repair.py`, and `main.py`.
   - Trace how parsed paper content becomes chunks, how chunks become pedagogical concepts or plans, how plans become notebook cells, and how cells become a final `.ipynb`.
   - Identify how learner information is captured and how it conditions planning or generation.
   - Identify how provenance is preserved, including chunk ids, section titles, page ranges, model identifiers, and timestamps.
   - Identify how logs and artifacts are organized per run or per source paper.
5. Verify implementation details against the intended contracts.
   - Stage contract expectations should include explicit inputs and outputs for `ParsedPaper`, `PaperChunk`, `ConceptItem`, `LearnerProfile`, `LessonSection`, `NotebookPlan`, `NotebookCell`, `NotebookBatch`, and `ValidationReport`.
   - Notebook construction should align with `nbformat` rather than string concatenation.
   - Intermediate serialization should prefer JSON, with `.ipynb` reserved for the final assembled notebook.
   - Retrieval should remain lightweight and inspectable, ideally chunk-local lexical retrieval with explicit provenance.
   - Local inference should align with the lightweight backend target in `DESIGN_DOC.md`, such as Gemma 4 E2B Instruct through `llama.cpp` server or an equivalent local backend.
   - If the repository deviates, document the deviation precisely.
6. Identify evidence of required quality controls.
   - Check for stage-local validation.
   - Check for notation normalization or notation inventory logic.
   - Check for tensor-shape explanation or tensor-shape validation helpers.
   - Check for notebook schema validation, pedagogical section-order checks, undefined-notation detection, code-execution smoke tests, and repair passes.
   - Check for tests covering schemas, chunking, notebook writing and loading, notebook validation, shape checks, and config loading.
   - Check for reproducibility mechanisms such as stage-aware logs, run directories, or `dvc.yaml`.
7. Synthesize findings.
   - Build a structured view of the current pipeline.
   - Connect file paths to stage responsibilities and actual data flow.
   - Distinguish clearly between implemented code, design intent, and missing components.
   - Highlight stage boundaries, validation points, and artifact locations.
8. Gather metadata for the research document.
   - Date: get current date and time.
   - Researcher: use your current identity or `Artificial Intelligence Agent`.
   - Git information: run `git rev-parse HEAD`, `git branch --show-current`, and `git config user.name`.
   - Create or reuse a date folder: `documents/logs/MM-DD-YYYY/`.
   - Put research notes under: `documents/logs/MM-DD-YYYY/research/`.
   - Use a descriptive filename such as `research-<lowercase-words-separated-by-hyphens>.md`.
9. Generate the research document using this format.
   ---
   date: [Current date and time with timezone in standard format]
   researcher: [Researcher name]
   git_commit: [Current commit hash]
   branch: [Current branch name]
   repository: [Repository name]
   topic: "[User's Question or Topic]"
   tags: [research, paper-to-notebook, pedagogical-notebooks, local-inference]
   status: complete
   last_updated: [Current date in YYYY-MM-DD format]
   last_updated_by: [Researcher name]
   ---

   # Research: [User's Question or Topic]

   **Date**: [Current date and time with timezone]
   **Researcher**: [Researcher name]
   **Git Commit**: [Current commit hash]
   **Branch**: [Current branch name]

   ## Research Question
   [Original user query]

   ## Summary
   [High-level documentation of the current paper-to-notebook implementation or scaffold.]

   ## Detailed Findings

   ### Pipeline Stages
   - PDF ingestion and parsing
   - Chunking and indexing
   - Concept extraction and dependency analysis
   - Learner profile capture
   - Pedagogical lesson planning
   - Section-wise notebook cell generation
   - Notebook assembly
   - Validation and repair

   ### Data Contracts and Serialization
   - Typed schemas and JSON contracts between stages
   - Final `.ipynb` construction strategy
   - Provenance and metadata preservation

   ### Retrieval and Inference
   - Retrieval strategy and ranking signals
   - Local model backend and prompt-context policy

   ### Validation and Testing
   - Notebook validation approach
   - Pedagogical and notation checks
   - Tensor-shape checks
   - Execution smoke tests
   - Existing or missing automated tests

   ## Code References
   - `path/to/file.py:123` - concise description
   - `path/to/file.py:456` - concise description

   ## Alignment with Design Documents
   [What matches `CODEBASE_PREFS.md` and `DESIGN_DOC.md`, and what remains absent or only partially implemented.]

   ## Open Questions
   [Ambiguities, missing files, or unresolved contracts that affect interpretation.]
10. Add repository permalinks if applicable.
    - If on the main branch or if the commit is pushed, generate repository permalinks.
    - Replace local file references with permalinks in the saved document when appropriate.
11. Sync and present findings.
    - Ensure the research note is saved under `documents/logs/MM-DD-YYYY/research/`.
    - Present a concise summary of findings to the user.
    - Ask whether they want deeper analysis of a specific stage, contract, or validation path.
12. Handle follow-up questions.
    - Append to the same research document.
    - Update front matter and add a follow-up section with a timestamp.
    - Perform additional repository research as needed.

## Important notes

- Use precise technical language and complete sentences.
- Distinguish clearly between source papers, intermediate JSON artifacts, validation reports, and final notebook output.
- Preserve the staged architecture in your explanation.
- Document current code and scaffolding, not aspirational behavior.
- Always call out stage inputs, outputs, validation gates, and provenance fields when they are defined.
- Follow the numbered steps exactly.
