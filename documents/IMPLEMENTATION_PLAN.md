# Implementation Plan: `paper-to-notebook`

## 1. Purpose

This document turns the design in `DESIGN_DOC.md` into a phased implementation plan.

The target system reads an academic AI research paper and generates a **self-contained Jupyter notebook** that teaches the paper from first principles. The notebook must adapt to the learner profile before generating any notebook cells, explain mathematics and tensor shapes carefully, and remain structurally valid as an `.ipynb` artifact.

This implementation plan follows these codebase principles:

- staged pipeline, not one monolithic script,
- structured `.ipynb` construction through `nbformat`,
- section-local regeneration and provenance,
- one model, one file,
- readability-first naming and comments,
- DVC-friendly intermediate artifacts,
- small but meaningful Pytest coverage.

---

## 2. Non-negotiable engineering decisions

### 2.1 Default model and runtime

**Default model**: `Gemma 4 E2B Instruct Q4_0` in GGUF format.

**Default runtime**: `llama.cpp` server for local inference.

Why:

1. Google’s Gemma docs explicitly recommend local tools such as `llama.cpp`, MLX, and LiteRT-LM for desktop and Apple Silicon execution.
2. The Gemma docs also warn that memory grows with context-window use because the KV cache grows with prompt and generation length.
3. On a machine with 16 GB RAM but only about 8 GB practically usable, the implementation must prefer short working contexts and staged retrieval.

### 2.2 Where the Google Gen AI Python SDK fits

Use the **Google Gen AI Python SDK** (`google-genai`) as an **optional cloud adapter and structured-output reference implementation**, but **do not force it into the default local GGUF path**.

Reason:

- The official `google-genai` SDK supports the Gemini Developer API and Vertex AI.
- The official local Gemma runtime path for GGUF is better served through `llama.cpp`.
- For the local default path, a tiny `httpx` client against the OpenAI-compatible `llama-server` endpoint keeps the code honest, readable, and aligned with the official runtime guidance.

This split is important for clarity:

- **local Gemma GGUF inference**: `llama.cpp` + `httpx`
- **optional cloud evaluation or structured-output experiments**: `google-genai`

### 2.3 One model, one file

All model-specific logic for the default model lives in:

```text
src/models/gemma_4_e2b.py
```

That file should contain:

- model configuration defaults,
- prompt and generation helpers specific to Gemma 4 E2B,
- local llama.cpp request code,
- optional Google Gen AI cloud adapter,
- response parsing helpers,
- model metadata written to provenance.

Other pipeline stages must not know provider-specific details.

---

## 3. Repository layout

This layout follows your preferred module order and keeps the stage boundaries obvious.

```text
paper-to-notebook/
├── README.md
├── DESIGN_DOC.md
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── .env.example
├── examples/
│   ├── sample_learner_profile.json
│   ├── sample_params.yaml
│   └── sample_papers/
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── parse_pdf.py
│   ├── chunking.py
│   ├── retrieve.py
│   ├── concept_graph.py
│   ├── learner_profile.py
│   ├── planner.py
│   ├── cell_generator.py
│   ├── notebook_builder.py
│   ├── validators.py
│   ├── repair.py
│   ├── artifact_store.py
│   ├── main.py
│   └── models/
│       └── gemma_4_e2b.py
├── tests/
│   ├── test_config.py
│   ├── test_schemas.py
│   ├── test_chunking.py
│   ├── test_notebook_builder.py
│   ├── test_validators.py
│   ├── test_shape_helpers.py
│   └── test_reference_pipeline.py
└── runs/
    └── <paper_slug>/<run_id>/
```

---

## 4. Phase-by-phase implementation roadmap

## Phase 0 — Project foundation and contracts

### Goal

Establish schemas, configuration, directory conventions, and CLI scaffolding **before** implementing generation.

### Deliverables

- `pyproject.toml`
- `src/config.py`
- `src/schemas.py`
- `src/artifact_store.py`
- `src/main.py`
- `params.yaml`
- `tests/test_config.py`
- `tests/test_schemas.py`

### Why first

Your preferences explicitly put **data contracts and stage boundaries before prompt convenience**. This phase makes every later stage typed, inspectable, and reproducible.

### Recommended dependencies

```toml
[project]
name = "paper-to-notebook"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7",
  "pydantic-settings>=2.2",
  "typer>=0.12",
  "httpx>=0.27",
  "PyYAML>=6.0.2",
  "nbformat>=5.10",
  "nbclient>=0.10",
  "pymupdf4llm>=0.0.20",
  "rank-bm25>=0.2.2",
  "numpy>=1.26",
  "ipywidgets>=8.1",
  "matplotlib>=3.8",
  "google-genai>=1.0.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-cov>=5.0",
  "ruff>=0.4",
  "mypy>=1.10",
  "dvc>=3.50",
]

[project.scripts]
paper2nb = "src.main:app"
```

### Configuration design

Use `pydantic-settings` for environment-backed settings and a separate YAML file for run parameters.

```python
# src/config.py
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    """Environment-backed settings.

    These are process-level settings, secrets, and runtime endpoints.
    They should not change from paper to paper during one run.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPER2NB_",
        extra="ignore",
    )

    local_llama_server_base_url: str = "http://127.0.0.1:8080/v1"
    google_api_key: str | None = None
    use_google_cloud_adapter: bool = False


class RunParameters(BaseModel):
    """Run-specific parameters stored in YAML.

    These are intentionally explicit so ablations remain easy.
    """

    default_model_name: str = "gemma-4-e2b-q4_0"
    inference_backend: Literal["llama_cpp", "google_genai"] = "llama_cpp"
    chunk_size_characters: int = 2200
    chunk_overlap_characters: int = 250
    retrieval_top_k: int = 6
    maximum_generation_sections: int | None = None
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_max_output_tokens: int = 1800
    notebook_execution_timeout_seconds: int = 120
    enable_notebook_execution_smoke_test: bool = True
    enable_repair_pass: bool = True
    enable_widgets: bool = True


def load_run_parameters(yaml_path: str | Path) -> RunParameters:
    with open(yaml_path, "r", encoding="utf-8") as parameter_file:
        raw_parameter_dictionary = yaml.safe_load(parameter_file)
    return RunParameters.model_validate(raw_parameter_dictionary)
```

### Data contracts

Make schemas strict and readable.

```python
# src/schemas.py
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    page_start: int | None = None
    page_end: int | None = None
    markdown_text: str


class PaperChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    section_id: str
    section_title: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_text: str
    equation_markers: list[str] = []
    notation_tokens: list[str] = []
    figure_references: list[str] = []


class LearnerProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mathematics_background: str
    machine_learning_background: str
    deep_learning_background: str
    python_background: str
    tensor_familiarity: str
    wants_tensor_shapes: bool = True
    wants_derivations: bool = True
    preferred_depth: Literal["light", "medium", "deep"] = "deep"
    preferred_pacing: Literal["fast", "moderate", "slow"] = "moderate"


class ConceptNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concept_id: str
    label: str
    concept_type: Literal[
        "definition",
        "equation",
        "algorithm",
        "architecture_module",
        "loss_function",
        "claim",
        "experiment",
        "notation",
        "prerequisite",
    ]
    source_chunk_ids: list[str]
    prerequisites: list[str] = []
    tensor_shape_notes: list[str] = []


class PlannedNotebookSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    teaching_goal: str
    prerequisite_concepts: list[str]
    source_chunk_ids: list[str]
    equations_to_unpack: list[str]
    tensor_shapes_to_state: list[str]
    likely_misconceptions: list[str]
    requires_code_example: bool = True
    requires_recap: bool = True


class NotebookCellSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_type: Literal["markdown", "code"]
    source: str
    metadata: dict = {}


class GeneratedSectionCells(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    section_title: str
    cells: list[NotebookCellSpec]
```

### Artifact store

Make stage outputs inspectable and regenerable.

```python
# src/artifact_store.py
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


@dataclass
class RunArtifactStore:
    run_root_directory: Path

    @classmethod
    def create(cls, runs_root_directory: str | Path, paper_slug: str, run_id: str) -> "RunArtifactStore":
        run_root_directory = Path(runs_root_directory) / paper_slug / run_id
        run_root_directory.mkdir(parents=True, exist_ok=True)
        return cls(run_root_directory=run_root_directory)

    def stage_directory(self, stage_name: str) -> Path:
        stage_path = self.run_root_directory / stage_name
        stage_path.mkdir(parents=True, exist_ok=True)
        return stage_path

    def write_json(self, stage_name: str, file_name: str, payload: Any) -> Path:
        destination_path = self.stage_directory(stage_name) / file_name
        with open(destination_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2, ensure_ascii=False)
        return destination_path

    def write_text(self, stage_name: str, file_name: str, text: str) -> Path:
        destination_path = self.stage_directory(stage_name) / file_name
        destination_path.write_text(text, encoding="utf-8")
        return destination_path


def utc_timestamp_string() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
```

### CLI scaffolding

Use Typer subcommands, not a single opaque entry point.

```python
# src/main.py
from pathlib import Path

import typer

app = typer.Typer(help="Generate pedagogy-first notebooks from AI research papers.")


@app.command()
def run(
    pdf_path: Path,
    params_path: Path = Path("params.yaml"),
) -> None:
    """Run the full pipeline from PDF to validated notebook."""
    raise NotImplementedError


@app.command()
def parse(pdf_path: Path, params_path: Path = Path("params.yaml")) -> None:
    """Run only PDF parsing and emit stage artifacts."""
    raise NotImplementedError


@app.command()
def plan(run_directory: Path) -> None:
    """Run concept extraction, learner capture, and notebook planning."""
    raise NotImplementedError


@app.command()
def generate(run_directory: Path) -> None:
    """Generate notebook cells and assemble the notebook."""
    raise NotImplementedError
```

### Tests for Phase 0

```python
# tests/test_config.py
from src.config import RunParameters


def test_run_parameters_default_model_name_is_explicit() -> None:
    run_parameters = RunParameters()
    assert run_parameters.default_model_name == "gemma-4-e2b-q4_0"


def test_run_parameters_reject_unknown_keys() -> None:
    try:
        RunParameters.model_validate({"unknown_field": 1})
        assert False, "Validation should fail for unknown fields."
    except Exception:
        assert True
```

---

## Phase 1 — Model runtime integration

### Goal

Implement the model wrapper in exactly one file and keep the rest of the pipeline provider-agnostic.

### Deliverables

- `src/models/gemma_4_e2b.py`
- `tests/test_reference_pipeline.py` with a stubbed model call

### Design rules

- one model file,
- readable top-to-bottom flow,
- local default path first,
- optional Google cloud adapter second,
- structured outputs by schema whenever possible.

### Model file structure

```python
# src/models/gemma_4_e2b.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types as google_types
except Exception:  # pragma: no cover - optional dependency path remains explicit
    genai = None
    google_types = None


StructuredResponseType = TypeVar("StructuredResponseType", bound=BaseModel)


@dataclass
class Gemma4E2BModel:
    local_llama_server_base_url: str
    use_google_cloud_adapter: bool = False
    google_api_key: str | None = None
    local_model_name: str = "gemma-4-e2b-q4_0"
    local_temperature: float = 0.2
    local_max_output_tokens: int = 1800

    def generate_json(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: type[StructuredResponseType],
    ) -> StructuredResponseType:
        if self.use_google_cloud_adapter:
            return self._generate_json_with_google_genai(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        return self._generate_json_with_local_llama_cpp(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=response_model,
        )

    def _generate_json_with_local_llama_cpp(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: type[StructuredResponseType],
    ) -> StructuredResponseType:
        """Local default path.

        We deliberately use plain HTTP for clarity instead of layering another AI SDK
        on top of an OpenAI-compatible server.
        """
        request_payload = {
            "model": self.local_model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.local_temperature,
            "max_tokens": self.local_max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client(timeout=120.0) as http_client:
            response = http_client.post(
                f"{self.local_llama_server_base_url}/chat/completions",
                json=request_payload,
            )
            response.raise_for_status()
            response_payload = response.json()

        raw_text = response_payload["choices"][0]["message"]["content"]
        raw_dictionary = json.loads(raw_text)
        return response_model.model_validate(raw_dictionary)

    def _generate_json_with_google_genai(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: type[StructuredResponseType],
    ) -> StructuredResponseType:
        if genai is None or google_types is None:
            raise RuntimeError("google-genai is not installed but cloud adapter was requested.")
        if not self.google_api_key:
            raise RuntimeError("google_api_key is required for the Google Gen AI adapter.")

        client = genai.Client(api_key=self.google_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=google_types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_model,
                temperature=self.local_temperature,
                max_output_tokens=self.local_max_output_tokens,
            ),
        )
        return response_model.model_validate_json(response.text)
```

### Why this split is the cleanest choice

The official Google Gen AI SDK documents structured outputs through `response_schema`, including direct Pydantic support, which makes it a good optional adapter for schema-constrained generation. But the same official SDK is documented as supporting Gemini Developer API and Vertex AI, not local GGUF runtimes. That is why the local default path should remain `llama.cpp`-based. Do not hide that reality behind unnecessary abstraction.

### Local server command

Keep the runtime instructions explicit in the repo.

```bash
llama-server \
  --model /absolute/path/to/gemma-4-E2B-it-Q4_0.gguf \
  --port 8080 \
  --ctx-size 8192 \
  --n-gpu-layers 99
```

On Apple Silicon, start with `--ctx-size 4096` or `8192`, not huge contexts.

---

## Phase 2 — PDF ingestion

### Goal

Parse the paper into sectioned Markdown with page metadata.

### Deliverables

- `src/parse_pdf.py`
- `runs/<paper>/<run>/parsed_paper/`
- `tests/test_chunking.py` for provenance-preserving behavior

### Parsing rules

- first use `pymupdf4llm.to_markdown`,
- preserve section headers,
- preserve page references if available,
- keep figure captions and equation-looking fragments,
- avoid complex figure semantics in v1,
- keep raw Markdown as a saved artifact.

### Implementation

```python
# src/parse_pdf.py
from __future__ import annotations

import re
from pathlib import Path

import pymupdf4llm

from src.schemas import PaperSection


SECTION_HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_pdf_into_sections(pdf_path: str | Path) -> list[PaperSection]:
    markdown_text = pymupdf4llm.to_markdown(str(pdf_path))

    collected_sections: list[PaperSection] = []
    current_section_title = "Untitled Section"
    current_section_lines: list[str] = []
    current_section_index = 0

    for markdown_line in markdown_text.splitlines():
        header_match = SECTION_HEADER_PATTERN.match(markdown_line)
        if header_match is not None:
            if current_section_lines:
                collected_sections.append(
                    PaperSection(
                        section_id=f"section_{current_section_index}",
                        title=current_section_title,
                        markdown_text="\n".join(current_section_lines).strip(),
                    )
                )
                current_section_index += 1
                current_section_lines = []
            current_section_title = header_match.group(2).strip()
        else:
            current_section_lines.append(markdown_line)

    if current_section_lines:
        collected_sections.append(
            PaperSection(
                section_id=f"section_{current_section_index}",
                title=current_section_title,
                markdown_text="\n".join(current_section_lines).strip(),
            )
        )

    return collected_sections
```

---

## Phase 3 — Chunking and retrieval

### Goal

Create inspectable, provenance-rich chunks and a lightweight lexical retriever.

### Deliverables

- `src/chunking.py`
- `src/retrieve.py`
- `runs/<paper>/<run>/chunks/`
- `tests/test_chunking.py`

### Why lexical retrieval first

For a single paper, BM25 over clean chunks is often good enough and far easier to debug than dense retrieval. It also matches your preference for lightweight, inspectable retrieval in v1.

### Implementation

```python
# src/chunking.py
from __future__ import annotations

import re

from src.schemas import PaperChunk, PaperSection


EQUATION_PATTERN = re.compile(r"\\\(|\\\)|\\\[|\\\]|=|\\frac|\\sum|\\mathbb|\\mathbf")


def split_text_with_overlap(
    full_text: str,
    chunk_size_characters: int,
    chunk_overlap_characters: int,
) -> list[str]:
    text_fragments: list[str] = []
    start_index = 0

    while start_index < len(full_text):
        end_index = min(len(full_text), start_index + chunk_size_characters)
        text_fragments.append(full_text[start_index:end_index])
        if end_index == len(full_text):
            break
        start_index = end_index - chunk_overlap_characters

    return text_fragments


def build_chunks_from_sections(
    paper_sections: list[PaperSection],
    chunk_size_characters: int,
    chunk_overlap_characters: int,
) -> list[PaperChunk]:
    generated_chunks: list[PaperChunk] = []
    chunk_counter = 0

    for paper_section in paper_sections:
        section_fragments = split_text_with_overlap(
            full_text=paper_section.markdown_text,
            chunk_size_characters=chunk_size_characters,
            chunk_overlap_characters=chunk_overlap_characters,
        )

        for section_fragment in section_fragments:
            equation_markers = EQUATION_PATTERN.findall(section_fragment)
            generated_chunks.append(
                PaperChunk(
                    chunk_id=f"chunk_{chunk_counter}",
                    section_id=paper_section.section_id,
                    section_title=paper_section.title,
                    page_start=paper_section.page_start,
                    page_end=paper_section.page_end,
                    chunk_text=section_fragment,
                    equation_markers=equation_markers,
                )
            )
            chunk_counter += 1

    return generated_chunks
```

```python
# src/retrieve.py
from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.schemas import PaperChunk


class PaperChunkRetriever:
    def __init__(self, paper_chunks: list[PaperChunk]) -> None:
        self.paper_chunks = paper_chunks
        self.tokenized_chunk_texts = [paper_chunk.chunk_text.lower().split() for paper_chunk in paper_chunks]
        self.bm25_index = BM25Okapi(self.tokenized_chunk_texts)

    def search(self, query_text: str, top_k: int = 6) -> list[PaperChunk]:
        tokenized_query = query_text.lower().split()
        relevance_scores = self.bm25_index.get_scores(tokenized_query)
        chunk_score_pairs = list(zip(self.paper_chunks, relevance_scores, strict=False))
        sorted_pairs = sorted(chunk_score_pairs, key=lambda pair: pair[1], reverse=True)
        return [paper_chunk for paper_chunk, _ in sorted_pairs[:top_k]]
```

### Test example

```python
# tests/test_chunking.py
from src.chunking import split_text_with_overlap


def test_chunk_overlap_preserves_boundary_context() -> None:
    original_text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_text_with_overlap(
        full_text=original_text,
        chunk_size_characters=10,
        chunk_overlap_characters=3,
    )
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]
```

---

## Phase 4 — Concept extraction and dependency graph

### Goal

Extract teaching primitives from chunks before planning notebook sections.

### Deliverables

- `src/concept_graph.py`
- `runs/<paper>/<run>/concept_graph/`
- `tests/test_reference_pipeline.py` with a tiny golden concept graph

### Important design choice

Do **not** over-engineer the graph in v1. Use a simple typed list of `ConceptNode` objects plus prerequisite edges. A full graph database is unnecessary.

### Implementation approach

- retrieve chunks section by section,
- ask the model for structured concept extraction,
- merge duplicate concepts with deterministic heuristics,
- write `concept_nodes.json` and `concept_edges.json`.

```python
# src/concept_graph.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.models.gemma_4_e2b import Gemma4E2BModel
from src.schemas import ConceptNode, PaperChunk


class ConceptExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    concepts: list[ConceptNode]


CONCEPT_EXTRACTION_SYSTEM_INSTRUCTION = """
You extract teaching concepts from a research-paper chunk.
Return JSON only.
Use officially recognized academic terminology.
Identify prerequisites explicitly.
State tensor-shape-relevant information whenever present.
""".strip()


def extract_concepts_from_chunk(
    model: Gemma4E2BModel,
    paper_chunk: PaperChunk,
) -> list[ConceptNode]:
    user_prompt = f"""
Extract concepts from this paper chunk.

Chunk ID: {paper_chunk.chunk_id}
Section Title: {paper_chunk.section_title}
Text:
{paper_chunk.chunk_text}
""".strip()

    response = model.generate_json(
        system_instruction=CONCEPT_EXTRACTION_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        response_model=ConceptExtractionResponse,
    )
    return response.concepts
```

---

## Phase 5 — Learner profile capture

### Goal

Ask the learner for background knowledge **before any notebook cells are generated**.

### Deliverables

- `src/learner_profile.py`
- `runs/<paper>/<run>/learner_profile/`
- CLI flow for prompt-based learner capture

### Implementation

Keep this human-readable and explicit.

```python
# src/learner_profile.py
from __future__ import annotations

import json
from pathlib import Path

import typer

from src.schemas import LearnerProfile


def capture_learner_profile_interactively() -> LearnerProfile:
    return LearnerProfile(
        mathematics_background=typer.prompt("Mathematics background"),
        machine_learning_background=typer.prompt("Machine learning background"),
        deep_learning_background=typer.prompt("Deep learning background"),
        python_background=typer.prompt("Python background"),
        tensor_familiarity=typer.prompt("Tensor familiarity"),
        wants_tensor_shapes=typer.confirm("Do you want explicit tensor shapes?", default=True),
        wants_derivations=typer.confirm("Do you want mathematical derivations?", default=True),
        preferred_depth=typer.prompt("Depth: light / medium / deep", default="deep"),
        preferred_pacing=typer.prompt("Pacing: fast / moderate / slow", default="moderate"),
    )


def save_learner_profile(learner_profile: LearnerProfile, output_path: str | Path) -> None:
    Path(output_path).write_text(learner_profile.model_dump_json(indent=2), encoding="utf-8")
```

---

## Phase 6 — Pedagogical planning

### Goal

Generate a section plan before prose or code cells exist.

### Deliverables

- `src/planner.py`
- `runs/<paper>/<run>/notebook_plan/`
- golden test for one tiny paper fragment

### Planning rules

Each planned section must specify:

- teaching goal,
- prerequisite concepts,
- source chunk IDs,
- equations to unpack,
- tensor shapes to state,
- misconceptions,
- whether code is required,
- whether recap is required.

### Implementation

```python
# src/planner.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.models.gemma_4_e2b import Gemma4E2BModel
from src.schemas import LearnerProfile, PlannedNotebookSection


class NotebookPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    notebook_title: str
    audience_summary: str
    sections: list[PlannedNotebookSection]


PLANNER_SYSTEM_INSTRUCTION = """
You are designing a pedagogy-first computational notebook.
Return JSON only.
Order sections for learning, not for paper chronology.
Use careful wording.
Avoid unexplained notation jumps.
Keep section scope small enough for notebook readability.
""".strip()


def build_notebook_plan(
    model: Gemma4E2BModel,
    learner_profile: LearnerProfile,
    concept_nodes_json_text: str,
) -> NotebookPlanResponse:
    user_prompt = f"""
Learner profile:
{learner_profile.model_dump_json(indent=2)}

Concept graph:
{concept_nodes_json_text}
""".strip()

    return model.generate_json(
        system_instruction=PLANNER_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        response_model=NotebookPlanResponse,
    )
```

### Practical validator

Before accepting a plan, check:

- prerequisites of section `i` appear earlier or are marked reintroduced,
- section titles are unique,
- at least one source chunk exists per section,
- no section is oversized.

---

## Phase 7 — Section-wise cell generation

### Goal

Generate notebook cells one pedagogical section at a time.

### Deliverables

- `src/cell_generator.py`
- `runs/<paper>/<run>/cell_batches/`
- tests for cell schema and maximum cell lengths

### Required cell pattern for each section

A strong v1 default is:

1. section intro Markdown cell,
2. mathematical unpacking Markdown cell,
3. tensor-shape grounding Markdown cell when relevant,
4. runnable code cell,
5. recap or exercise Markdown cell.

### Why this structure

It aligns with notebook-writing guidance that discourages long cells and encourages explicit organization and narrative flow.

### Implementation

```python
# src/cell_generator.py
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.models.gemma_4_e2b import Gemma4E2BModel
from src.schemas import GeneratedSectionCells, LearnerProfile, PaperChunk, PlannedNotebookSection


class SectionCellGenerationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    section_id: str
    section_title: str
    cells: list


CELL_GENERATION_SYSTEM_INSTRUCTION = """
You generate notebook cells for one pedagogical section.
Return JSON only.
Explain from first principles.
Use careful wording.
Use explicit tensor shapes when relevant.
Python code must be self-contained and heavily commented.
Do not assume future cells exist.
""".strip()


def generate_cells_for_section(
    model: Gemma4E2BModel,
    learner_profile: LearnerProfile,
    planned_notebook_section: PlannedNotebookSection,
    retrieved_chunks: list[PaperChunk],
) -> GeneratedSectionCells:
    joined_chunk_text = "\n\n".join(
        f"[{paper_chunk.chunk_id}] {paper_chunk.section_title}\n{paper_chunk.chunk_text}"
        for paper_chunk in retrieved_chunks
    )

    user_prompt = f"""
Learner profile:
{learner_profile.model_dump_json(indent=2)}

Planned notebook section:
{planned_notebook_section.model_dump_json(indent=2)}

Retrieved source chunks:
{joined_chunk_text}
""".strip()

    return model.generate_json(
        system_instruction=CELL_GENERATION_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        response_model=GeneratedSectionCells,
    )
```

### Notebook interactivity

For interactive educational controls, use `ipywidgets`, especially `interact`, sliders, and simple selection widgets.

Example code snippet that the generator can emit:

```python
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits)
    exponential_values = np.exp(shifted_logits)
    return exponential_values / np.sum(exponential_values)


def visualize_temperature_effect(temperature: float = 1.0) -> None:
    raw_logits = np.array([2.0, 1.0, 0.1])
    probability_values = softmax(raw_logits / temperature)

    plt.figure(figsize=(6, 4))
    plt.bar(["token_1", "token_2", "token_3"], probability_values)
    plt.title(f"Softmax distribution at temperature = {temperature:.2f}")
    plt.ylabel("Probability")
    plt.show()


interact(visualize_temperature_effect, temperature=(0.2, 3.0, 0.1));
```

---

## Phase 8 — Notebook assembly

### Goal

Assemble validated cell batches into a real `.ipynb` notebook.

### Deliverables

- `src/notebook_builder.py`
- `tests/test_notebook_builder.py`

### Rules

- build notebook objects through `nbformat.v4` helpers,
- attach metadata for provenance,
- keep authorship and model info in notebook metadata,
- never concatenate raw JSON strings by hand.

### Implementation

```python
# src/notebook_builder.py
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat import NotebookNode

from src.schemas import GeneratedSectionCells


def build_notebook_object(
    notebook_title: str,
    audience_summary: str,
    generated_section_batches: list[GeneratedSectionCells],
    notebook_metadata: dict,
) -> NotebookNode:
    notebook_object = nbformat.v4.new_notebook()
    notebook_object.metadata.update(notebook_metadata)

    notebook_cells = [
        nbformat.v4.new_markdown_cell(
            source=(
                f"# {notebook_title}\n\n"
                f"**Audience**: {audience_summary}\n\n"
                f"This notebook was generated by a pedagogy-first staged pipeline."
            )
        )
    ]

    for generated_section_batch in generated_section_batches:
        for notebook_cell_spec in generated_section_batch.cells:
            if notebook_cell_spec.cell_type == "markdown":
                notebook_cells.append(
                    nbformat.v4.new_markdown_cell(
                        source=notebook_cell_spec.source,
                        metadata=notebook_cell_spec.metadata,
                    )
                )
            else:
                notebook_cells.append(
                    nbformat.v4.new_code_cell(
                        source=notebook_cell_spec.source,
                        metadata=notebook_cell_spec.metadata,
                    )
                )

    notebook_object.cells = notebook_cells
    return notebook_object


def write_notebook(notebook_object: NotebookNode, output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as output_file:
        nbformat.write(notebook_object, output_file)
```

### Notebook metadata recommendation

```python
notebook_metadata = {
    "authors": [{"name": "paper-to-notebook"}],
    "paper_to_notebook": {
        "source_paper": "attention_is_all_you_need.pdf",
        "model_name": "gemma-4-e2b-q4_0",
        "inference_backend": "llama_cpp",
        "run_id": "20260405T103000Z",
    },
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
}
```

---

## Phase 9 — Validation and repair

### Goal

Validate both notebook structure and pedagogical quality, then repair locally if needed.

### Deliverables

- `src/validators.py`
- `src/repair.py`
- `runs/<paper>/<run>/validation_report/`
- `tests/test_validators.py`
- notebook smoke test with `nbclient`

### Validation layers

1. **Schema validity** through `nbformat`
2. **Pedagogical ordering**
3. **Notation consistency**
4. **Tensor-shape consistency**
5. **Cell-length constraints**
6. **Execution smoke test** through `nbclient`

### Implementation

```python
# src/validators.py
from __future__ import annotations

from dataclasses import dataclass

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode


@dataclass
class ValidationIssue:
    severity: str
    message: str


def validate_notebook_schema(notebook_object: NotebookNode) -> list[ValidationIssue]:
    try:
        nbformat.validate(notebook_object)
        return []
    except Exception as validation_error:
        return [ValidationIssue(severity="error", message=str(validation_error))]


def validate_cell_lengths(notebook_object: NotebookNode, maximum_lines_per_cell: int = 100) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for cell_index, notebook_cell in enumerate(notebook_object.cells):
        line_count = len(notebook_cell.source.splitlines())
        if line_count > maximum_lines_per_cell:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    message=f"Cell {cell_index} has {line_count} lines, exceeding {maximum_lines_per_cell}.",
                )
            )
    return issues


def execute_notebook_smoke_test(notebook_object: NotebookNode, working_directory: str) -> list[ValidationIssue]:
    try:
        notebook_client = NotebookClient(
            notebook_object,
            timeout=120,
            kernel_name="python3",
            resources={"metadata": {"path": working_directory}},
        )
        notebook_client.execute()
        return []
    except Exception as execution_error:
        return [ValidationIssue(severity="error", message=str(execution_error))]
```

### Repair strategy

Repair should be **targeted**.

- If schema fails: rebuild notebook object from stored cell specs.
- If pedagogical rules fail: regenerate only the offending section.
- If execution fails: isolate failing code cell, request a local repair pass, and rerun smoke test.

Avoid re-running the entire pipeline.

---

## Phase 10 — DVC and reproducibility

### Goal

Track stage artifacts and make runs reproducible without recomputing everything blindly.

### Deliverables

- `dvc.yaml`
- `params.yaml`
- a reproducible sample run in `examples/`

### Stage structure

Each stage should write one inspectable output directory.

```yaml
# dvc.yaml
stages:
  parse_pdf:
    cmd: paper2nb parse examples/sample_papers/attention.pdf --params-path params.yaml
    deps:
      - examples/sample_papers/attention.pdf
      - src/parse_pdf.py
      - src/schemas.py
      - src/config.py
    params:
      - chunk_size_characters
      - chunk_overlap_characters
    outs:
      - runs/attention/sample_run/parsed_paper

  plan_notebook:
    cmd: paper2nb plan runs/attention/sample_run
    deps:
      - src/concept_graph.py
      - src/learner_profile.py
      - src/planner.py
      - src/models/gemma_4_e2b.py
    outs:
      - runs/attention/sample_run/concept_graph
      - runs/attention/sample_run/learner_profile
      - runs/attention/sample_run/notebook_plan

  generate_notebook:
    cmd: paper2nb generate runs/attention/sample_run
    deps:
      - src/cell_generator.py
      - src/notebook_builder.py
      - src/validators.py
      - src/repair.py
      - src/models/gemma_4_e2b.py
    outs:
      - runs/attention/sample_run/cell_batches
      - runs/attention/sample_run/validation_report
      - runs/attention/sample_run/final_notebook
```

### Provenance fields that must be saved

For every run, persist:

- paper path and checksum,
- learner profile,
- run parameters,
- model name,
- backend name,
- retrieved chunk IDs per section,
- generation timestamps,
- validation outcomes.

---

## Phase 11 — Test suite

### Goal

Build small, stable tests that catch structural regressions early.

### Minimum tests

1. `test_schemas.py`
   - validates JSON contracts between stages
2. `test_chunking.py`
   - overlap behavior
   - metadata preservation
3. `test_notebook_builder.py`
   - notebook write / load / validate loop
4. `test_validators.py`
   - long-cell warnings
   - schema failure capture
5. `test_shape_helpers.py`
   - tensor-shape helper consistency
6. `test_config.py`
   - YAML + environment-backed config loading
7. `test_reference_pipeline.py`
   - one tiny golden plan and one tiny generated notebook

### Example notebook round-trip test

```python
# tests/test_notebook_builder.py
import nbformat

from src.notebook_builder import build_notebook_object
from src.schemas import GeneratedSectionCells, NotebookCellSpec


def test_notebook_round_trip_is_valid(tmp_path) -> None:
    section_batch = GeneratedSectionCells(
        section_id="section_1",
        section_title="Toy section",
        cells=[
            NotebookCellSpec(cell_type="markdown", source="## Hello"),
            NotebookCellSpec(cell_type="code", source="x = 1\nprint(x)"),
        ],
    )

    notebook_object = build_notebook_object(
        notebook_title="Toy notebook",
        audience_summary="Test learner",
        generated_section_batches=[section_batch],
        notebook_metadata={},
    )

    output_path = tmp_path / "toy.ipynb"
    with open(output_path, "w", encoding="utf-8") as output_file:
        nbformat.write(notebook_object, output_file)

    reloaded_notebook = nbformat.read(output_path, as_version=4)
    nbformat.validate(reloaded_notebook)

    assert reloaded_notebook.cells[0].cell_type == "markdown"
    assert reloaded_notebook.cells[1].cell_type == "markdown"
    assert reloaded_notebook.cells[2].cell_type == "code"
```

---

## 5. Recommended implementation order inside the repo

Implement in this exact order so the codebase stays readable:

1. `config.py`
2. `schemas.py`
3. `artifact_store.py`
4. `main.py`
5. `models/gemma_4_e2b.py`
6. `parse_pdf.py`
7. `chunking.py`
8. `retrieve.py`
9. `concept_graph.py`
10. `learner_profile.py`
11. `planner.py`
12. `cell_generator.py`
13. `notebook_builder.py`
14. `validators.py`
15. `repair.py`
16. `tests/`
17. `dvc.yaml`

That order respects your preference that schemas and boundaries come before generation convenience.

---

## 6. Code style rules for this repo

### Naming

Use long, explicit names:

- `retrieved_source_chunks`
- `planned_notebook_section`
- `generated_section_batches`
- `notebook_execution_timeout_seconds`

Do not use cryptic short names unless they are universal mathematical symbols inside equations.

### Comments

Prefer comments that explain **why**:

```python
# We keep the local inference path as plain HTTP so the runtime contract stays visible.
# This avoids hiding llama.cpp-specific behavior behind a second AI SDK.
```

### Least number of codepaths

Keep two inference codepaths only:

1. local `llama.cpp` path,
2. optional `google-genai` cloud path.

Do not add Ollama, MLX, LiteRT-LM, and local Transformers adapters in v1. That would violate your readability-first rule.

---

## 7. First milestone definition

The first milestone is complete when the repo can:

1. parse one AI paper PDF into sections and chunks,
2. ask the learner for a profile,
3. extract a small concept graph,
4. build a pedagogical notebook plan,
5. generate at least three validated sections,
6. assemble a valid `.ipynb`,
7. execute a smoke test locally,
8. preserve stage artifacts and provenance.

Anything beyond that is phase-two polish.

---

## 8. Optional phase-two upgrades

Only after the first milestone works:

- dense retrieval,
- figure-caption-aware explanation cells,
- notation normalization against a paper-wide symbol table,
- richer `ipywidgets` interfaces,
- automatic study questions and answer reveals,
- optional cloud evaluation with `google-genai`,
- prompt ablation experiments through DVC params.

---

## 9. Final recommendation

Build the first version as a **pedagogy-first, retrieval-grounded, short-context pipeline**.

Do not optimize for novelty in the codebase. Optimize for readability, provenance, and local reliability. The strongest v1 is the one you can read top-to-bottom, debug section-by-section, and reproduce run-by-run.

---

## 10. References

- Google Gemma docs: model overview, model card, local run guidance
- Google Gen AI Python SDK docs
- Gemini structured-output docs
- PyMuPDF4LLM docs
- nbformat docs
- nbclient docs
- Typer docs
- Pydantic and pydantic-settings docs
- DVC docs
- ipywidgets docs
- Matplotlib docs
- Vaswani et al., *Attention Is All You Need*
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
- Knuth, *Literate Programming*
- Rule et al., *Ten Simple Rules for Writing and Sharing Computational Analyses in Jupyter Notebooks*
