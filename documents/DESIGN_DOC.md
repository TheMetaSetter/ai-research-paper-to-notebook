# Design Document: Paper-to-Notebook

## 1. Purpose

This project generates an **interactive, self-contained Jupyter notebook** from an academic AI research paper.

The notebook is not a plain summary. Its job is to teach the paper **from first principles**, adapting to the learner’s background before generating any notebook cells.

The output notebook should:
- explain the paper in careful pedagogical order,
- unpack equations and notation step by step,
- state tensor shapes explicitly,
- include runnable Python examples with explanatory comments,
- preserve traceability back to the source paper,
- remain readable, modular, and structurally valid as an `.ipynb` document.

This design treats the notebook as a **computational notebook** in the Jupyter sense: a document combining prose, code, and outputs in the `.ipynb` format defined by `nbformat`. Jupyter’s own documentation describes notebook files as `.ipynb` documents interpreted by `nbformat`, and the `nbformat` documentation states that the official notebook format is defined by a JSON schema. citeturn475205search3turn191814search2

---

## 2. Design Goals

### 2.1 Primary goal
Generate notebooks with **high pedagogical quality**, not merely syntactic completeness.

### 2.2 Secondary goals
- Run locally on constrained consumer hardware, especially Apple Silicon laptops.
- Be robust to long and mathematically dense papers.
- Produce deterministic intermediate artifacts that can be validated.
- Make notebook generation inspectable and debuggable.
- Support iterative improvement without redesigning the whole pipeline.

### 2.3 Non-goals for v1
- Full benchmark replication from the paper.
- Reliable extraction of every figure or table into executable form.
- End-to-end agentic browsing beyond the provided paper.
- One-shot generation of the entire notebook from the raw PDF.

---

## 3. Constraints

### 3.1 Hardware constraint
Target machine: Apple Silicon laptop with **16 GB RAM, but assume only about 8 GB is safely usable** for the model runtime plus Python process.

This strongly favors a **small local model + chunked generation pipeline** over a single very long-context pass. Google’s Gemma documentation notes that memory use grows with the context window because KV-cache consumption increases with prompt and generation length. Google also positions Gemma and llama.cpp as suitable for local execution, including Apple Silicon. citeturn475205search5turn475205search0turn475205search2

### 3.2 Model constraint
Use **Gemma 4 E2B Instruct** in a local format such as GGUF for v1.

Gemma 4 is available in E2B, E4B, 31B, and 26B A4B variants. The official Gemma documentation presents E2B and E4B as part of the Gemma 4 family, while Google’s Android Studio announcement lists **8 GB total RAM** as the recommended requirement for Gemma E2B. That makes E2B the sensible baseline for the current hardware target. citeturn328158search0turn475205search9turn328158search10

### 3.3 Quality constraint
The generated notebook must prefer:
- careful wording over verbosity,
- mathematical correctness over stylistic flourish,
- pedagogical ordering over paper ordering when the two differ,
- explicit assumptions over implicit jumps.

This is aligned with both the idea of **literate programming** introduced by Knuth and Jupyter-specific guidance emphasizing audience, narrative, documentation, and modular notebook structure. citeturn101724search2turn191814search0

---

## 4. Academic terminology used in this project

To keep the repo academically precise, the following terms should be used consistently:

- **Transformer**: the architecture introduced by Vaswani et al. in *Attention Is All You Need*. citeturn101724search0
- **Retrieval-Augmented Generation (RAG)**: the retrieval-conditioned generation framework formalized by Lewis et al. citeturn101724search1
- **Computational notebook**: the recognized medium combining code, narrative text, and outputs in notebook form. Jupyter documentation explicitly uses this term. citeturn475205search7turn191814search4
- **Literate programming**: the documentation-first programming perspective introduced by Knuth. citeturn101724search2

These terms are preferable to informal labels such as “paper explainer pipeline” or “interactive tutorial generator” when naming core architectural components.

---

## 5. System overview

The system should be implemented as a **multi-stage constrained generation pipeline**, not as a single monolithic prompt.

### 5.1 High-level stages
1. **PDF ingestion and parsing**
2. **Paper chunking and indexing**
3. **Concept extraction and dependency analysis**
4. **Learner profile capture**
5. **Pedagogical lesson planning**
6. **Section-wise notebook cell generation**
7. **Notebook assembly and schema validation**
8. **Quality checks and optional repair pass**

### 5.2 Why this architecture
A one-shot pipeline is fragile under tight memory and long context constraints. By contrast, a staged architecture:
- reduces memory pressure,
- permits structured validation after each step,
- improves provenance,
- allows section-local regeneration,
- makes pedagogical control practical.

This is also conceptually consistent with RAG: the model should generate each section while conditioned only on the learner profile and the most relevant paper chunks, rather than the entire paper every time. citeturn101724search1

---

## 6. Core design principle: pedagogy before rendering

The central design principle is:

> **Do not generate notebook cells until the system has modeled the learner and produced a pedagogical plan.**

This matters because paper order is often not teaching order. Research papers optimize for novelty, brevity, and comparison to prior work. A teaching notebook should instead optimize for:
- prerequisite order,
- notation normalization,
- progressive mathematical buildup,
- minimal hidden assumptions,
- repeated tensor-shape grounding.

The system should therefore build a **pedagogical dependency graph** before writing notebook cells.

---

## 7. Detailed architecture

### 7.1 Stage A — PDF ingestion

**Input:** PDF paper  
**Output:** sectioned Markdown plus metadata

Recommended first parser: **PyMuPDF4LLM**.

Rationale:
- lightweight,
- local,
- no GPU requirement,
- suitable for extracting layout-aware Markdown from PDFs.

PyMuPDF4LLM explicitly documents PDF-to-Markdown extraction for LLM and RAG workflows. citeturn475205search0turn475205search5

**Responsibilities**
- extract text into Markdown,
- preserve section headers,
- retain page references,
- keep figure/table captions when possible,
- ignore OCR-heavy figure semantics in v1.

**Output schema idea**
```json
{
  "paper_title": "...",
  "sections": [
    {
      "section_id": "sec_1",
      "title": "Introduction",
      "page_start": 1,
      "page_end": 2,
      "markdown": "..."
    }
  ]
}
```

### 7.2 Stage B — Chunking and indexing

**Input:** parsed sections  
**Output:** retrievable paper chunks

Each section should be divided into overlapping chunks sized for local inference. On the target machine, the effective prompt budget should remain conservative, so chunking is mandatory even though Gemma 4 advertises much larger maximum context windows. Google’s Gemma overview explicitly warns that memory increases dynamically with context length, while the model card advertises long contexts up to 256K depending on variant. Those are capability ceilings, not recommended operating points for this hardware. citeturn475205search5turn475205search1

**Recommended v1 retrieval**
- lexical retrieval with **BM25**,
- optional later upgrade to dense retrieval.

This is a deliberate simplification: for a single paper, high-quality lexical retrieval over clean sectioned chunks is often sufficient.

**Chunk metadata**
- chunk id,
- section id,
- section title,
- page range,
- text,
- equation markers,
- figure references,
- notation tokens.

### 7.3 Stage C — Concept extraction and dependency analysis

**Input:** chunks  
**Output:** concept graph / lesson primitives

The system should extract at least these object types:
- core claims,
- prerequisite concepts,
- definitions,
- equations,
- algorithms,
- architectural modules,
- loss functions,
- experimental claims,
- notation inventory,
- tensor-shape-relevant objects.

This stage is the bridge between paper parsing and teaching design.

**Key invariant**
A notebook section cannot be generated until all of its prerequisites are either:
1. already explained in previous sections, or  
2. reintroduced locally.

### 7.4 Stage D — Learner profile capture

**Input:** user responses  
**Output:** typed learner model

The system must ask the learner before generating cells.

Minimum profile fields:
- mathematics background,
- machine learning background,
- deep learning background,
- Python background,
- familiarity with tensors and linear algebra notation,
- preference for derivations versus intuition,
- desired depth,
- desired pacing.

**Example schema**
```json
{
  "math_level": "linear algebra + basic probability",
  "ml_level": "intermediate",
  "dl_level": "beginner",
  "python_level": "intermediate",
  "wants_tensor_shapes": true,
  "wants_derivations": true,
  "preferred_depth": "deep"
}
```

### 7.5 Stage E — Pedagogical lesson planning

**Input:** concept graph + learner profile  
**Output:** notebook plan

The model should not yet write prose-heavy cells. It should first produce a **lesson plan**.

Each planned section should include:
- title,
- teaching goal,
- prerequisite concepts,
- source chunk ids,
- key equations to unpack,
- likely misconceptions,
- tensor shapes to state,
- whether runnable code is needed,
- whether a recap or exercise is needed.

This design follows strong notebook-writing guidance. In *Ten Simple Rules for Writing and Sharing Computational Analyses in Jupyter Notebooks*, Rule et al. explicitly emphasize telling a story for an audience, documenting process and parameters, organizing the notebook into clear sections, and avoiding overlong cells. citeturn191814search0

### 7.6 Stage F — Section-wise cell generation

**Input:** one lesson section + retrieved chunks + learner profile  
**Output:** a validated batch of notebook cells

This is the main generation stage.

The model should generate cells for exactly **one pedagogical unit at a time**.

Each section should usually contain:
1. a section intro Markdown cell,
2. a notation normalization cell,
3. an equation unpacking cell,
4. a tensor-shape cell,
5. a runnable code cell,
6. a recap or exercise cell.

**Important rule**
Every code cell must be locally understandable. Code comments should explain **why** a step exists, not merely what the syntax does. This is directly aligned with notebook-writing guidance encouraging the separation of narrative Markdown from low-level code comments. citeturn191814search0

### 7.7 Stage G — Notebook assembly

**Input:** validated cell batches  
**Output:** `.ipynb`

Use `nbformat` to assemble the notebook, because the Jupyter project describes `.ipynb` as the notebook file format interpreted by `nbformat`, and `nbformat` provides the official JSON-schema-based definition of the document structure. citeturn475205search3turn191814search2

Notebook metadata should include:
- source paper title,
- generation timestamp,
- model info,
- chunk provenance,
- pedagogical depth,
- project version.

### 7.8 Stage H — Validation and repair

**Input:** notebook  
**Output:** validated or repaired notebook

Validation should include:
- JSON-schema validity via `nbformat`,
- required section ordering,
- absence of undefined notation,
- tensor-shape consistency checks,
- code execution smoke test,
- maximum cell-length checks,
- required pedagogical components present.

The “avoid long cells” recommendation is supported directly by Rule et al., who suggest that anything over roughly one page or 100 lines is too long. citeturn191814search0

---

## 8. Runtime and inference backend

### 8.1 Recommended backend
Use **llama.cpp server** as the local inference backend for v1.

Why:
- strong Apple Silicon support,
- OpenAI-compatible HTTP server,
- lightweight deployment,
- works with GGUF quantized models.

The llama.cpp repository explicitly documents a lightweight OpenAI API compatible server and notes fast execution on CPUs and Apple Silicon. Google’s Gemma run documentation also names llama.cpp as a fast local option. citeturn475205search2turn475205search0

### 8.2 Prompting style
Prefer:
- `system` role for behavioral constraints,
- JSON-only outputs for planning stages,
- low temperature for structure-sensitive steps,
- short per-call contexts,
- explicit “do not assume hidden future cells” instructions.

Gemma 4’s official prompt-formatting docs state that Gemma 4 has native `system` role support, and Google separately documents prompt-formatting guidance for Gemma 4. citeturn328158search3turn328158search2

### 8.3 Context policy
Even if the model supports long context windows, v1 should use a **retrieval-first short-context policy**:
- working context around 4K–8K for safety,
- retrieve a handful of relevant chunks per section,
- do not carry the whole paper through the entire generation process.

This policy is driven by local memory realities, not by the model’s advertised theoretical maximum. Google’s Gemma documentation explicitly notes that context-window memory consumption grows dynamically with prompt and generation length. citeturn475205search5

---

## 9. Data contracts

The project should rely on strict typed schemas between stages.

### 9.1 Why schemas matter
Schemas turn the system from “prompt spaghetti” into a debuggable pipeline.

They enable:
- validation,
- regeneration at one stage only,
- logging,
- error localization,
- unit testing.

### 9.2 Recommended contracts
- `ParsedPaper`
- `PaperChunk`
- `ConceptItem`
- `LearnerProfile`
- `LessonSection`
- `NotebookPlan`
- `NotebookCell`
- `NotebookBatch`
- `ValidationReport`

### 9.3 Serialization
Use JSON between stages; use `.ipynb` only at the final assembly step.

---

## 10. Pedagogical quality rubric

This project’s core differentiator is pedagogical quality. The generator should therefore be optimized against an explicit rubric.

### 10.1 Required properties of every major section
- **Context:** why this concept exists.
- **Definition:** what the object means.
- **Math:** exact equation or formal statement.
- **Shapes:** tensor dimensions where applicable.
- **Mechanics:** what computation occurs step by step.
- **Intuition:** why the computation is designed that way.
- **Example:** toy runnable code.
- **Check:** a short sanity test or reflective question.

### 10.2 Wording constraints
The generated explanations should:
- avoid vague pronouns when mathematics is involved,
- repeat symbol definitions when the distance from first use is large,
- prefer short explicit claims over dense compressed paragraphs,
- state assumptions before using them.

### 10.3 Mathematical explanation policy
For every important equation, the notebook should explain:
1. what each symbol denotes,  
2. the domain or shape of each object,  
3. what operation is being applied,  
4. why the equation is useful in the paper.

### 10.4 Tensor-shape policy
Every section involving neural computation should include explicit tensor-shape discussion.

Example style:
- `X ∈ R^{B × T × d_model}`
- `W_Q ∈ R^{d_model × d_k}`
- `Q = X W_Q ∈ R^{B × T × d_k}`

This should appear in Markdown before corresponding code.

---

## 11. Section template for notebook generation

Each pedagogical section should roughly follow this template:

1. **Section heading and goal**  
2. **Prerequisite reminder**  
3. **Notation table**  
4. **First-principles explanation**  
5. **Equation unpacking**  
6. **Tensor-shape walkthrough**  
7. **Toy code demonstration**  
8. **Interpretation of code output**  
9. **Connection back to the paper**  
10. **Recap / exercise / reflection**

This structure is consistent with notebook-writing recommendations centered on narrative structure and audience-aware organization. citeturn191814search0turn191814search4

---

## 12. Retrieval strategy

### 12.1 v1 strategy
Use lightweight retrieval over paper chunks.

Suggested ranking signals:
- lexical match to section goal,
- overlap with notation symbols,
- overlap with equation identifiers,
- same-section prior chunks,
- title and abstract boost.

### 12.2 Why not dense retrieval first
For a single-paper setting on constrained hardware, dense retrieval adds engineering and memory overhead with limited initial benefit.

### 12.3 Provenance
Every generated notebook section should preserve:
- source chunk ids,
- source section titles,
- optional page numbers.

This supports explainability and later debugging, which is one of the motivating benefits of RAG-style conditioning. citeturn101724search1

---

## 13. Failure modes and mitigations

### 13.1 Failure: mathematically plausible but paper-inconsistent explanation
**Mitigation:** section-local retrieval plus provenance display.

### 13.2 Failure: notation drift
**Mitigation:** explicit notation inventory and per-section notation normalization.

### 13.3 Failure: tensor-shape mistakes
**Mitigation:** deterministic shape checker for common operations.

### 13.4 Failure: overlong unreadable cells
**Mitigation:** enforce cell-length caps and split sections automatically. This is supported by Jupyter notebook guidance discouraging long cells. citeturn191814search0

### 13.5 Failure: invalid notebook JSON
**Mitigation:** always write through `nbformat` and validate against schema. citeturn191814search2

### 13.6 Failure: local model loses instruction fidelity
**Mitigation:** structured generation, low temperature, smaller tasks, repair passes.

---

## 14. Recommended repository structure

```text
paper-to-notebook/
  README.md
  DESIGN_DOC.md
  pyproject.toml
  .env.example
  src/
    config.py
    schemas.py
    parse_pdf.py
    chunking.py
    retrieve.py
    concept_graph.py
    learner_profile.py
    planner.py
    cell_generator.py
    notebook_builder.py
    validators.py
    repair.py
    main.py
  tests/
    test_schemas.py
    test_chunking.py
    test_notebook_validation.py
    test_shape_checks.py
  examples/
    sample_papers/
    sample_outputs/
```

---

## 15. Minimal implementation roadmap

### Phase 1 — working MVP
Deliver:
- PDF to Markdown parsing,
- chunking,
- learner questionnaire,
- notebook planning,
- section-wise cell generation,
- `.ipynb` writing,
- schema validation.

### Phase 2 — pedagogy improvements
Add:
- notation table generation,
- equation-by-equation unpacking mode,
- tensor-shape checker,
- recap and exercises,
- misconception prompts.

### Phase 3 — quality and usability
Add:
- automatic repair pass,
- per-section provenance display,
- optional figure-caption explanation,
- interactive widgets for selected examples,
- notebook execution smoke tests in CI.

---

## 16. Model selection policy

### v1 default
- **Gemma 4 E2B Instruct**, GGUF quantized, local inference.

### Reason
Google’s Gemma documentation presents E2B as an edge-oriented member of the Gemma 4 family, and current official ecosystem guidance indicates that E2B is the smallest practical Gemma 4 option for local constrained environments. Meanwhile, the long-context capacity of larger variants should not be mistaken for fit on an 8 GB usable-memory budget. citeturn328158search0turn328158search10turn475205search5

### Possible future upgrade path
- E4B for higher quality if memory allows,
- 26B A4B only on substantially larger-memory systems.

---

## 17. Testing strategy

### 17.1 Unit tests
- schema validation,
- chunk overlap correctness,
- notebook serialization,
- shape checker primitives.

### 17.2 Golden tests
Maintain a small set of reference papers and expected lesson plans.

### 17.3 Human evaluation
Evaluate notebooks on:
- correctness,
- pedagogical ordering,
- wording clarity,
- notation consistency,
- tensor-shape explicitness,
- code executability,
- source faithfulness.

### 17.4 Reproducibility note
Reproducibility and maintainability matter in notebooks. The Jupyter community literature emphasizes documenting dependencies, organizing notebooks clearly, and structuring them for re-use and sharing. citeturn191814search0turn101724search11

---

## 18. Implementation stance

This project should be built as a **pedagogy-first constrained generation system**.

That means:
- structured intermediate representations,
- section-local retrieval,
- strict schema validation,
- explicit notation and tensor-shape teaching,
- notebook assembly only at the final stage.

Under the current hardware constraint, the fastest route to a strong system is **not** “use more context.” It is “use better decomposition.” Google’s Gemma documentation explicitly warns that context-window memory cost grows with sequence length, while llama.cpp and Gemma’s local-run guidance make a small local deployment practical on Apple Silicon. citeturn475205search5turn475205search2turn475205search0

If executed this way, the project can produce notebooks that are not only valid and runnable, but genuinely useful for learning research papers from first principles.
