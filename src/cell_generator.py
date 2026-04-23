from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.artifact_store import utc_timestamp_string
from src.models.gemma_4_e2b import Gemma4E2BModel
from src.retrieve import PaperChunkRetriever
from src.schemas import LearnerProfile, LessonSection, NotebookBatch, NotebookCell, PaperChunk, StageProvenance


MAXIMUM_CELL_LINES = 100

CELL_GENERATION_SYSTEM_INSTRUCTION = """
You generate notebook cells for exactly one pedagogical section.
Return JSON only.
Explain from first principles.
Use explicit tensor shapes when relevant.
Code must be self-contained and readable.
Code comments should explain why a step exists.
Do not assume future notebook cells exist.
""".strip()


class NotebookCellDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_type: str
    source: str
    metadata: dict[str, object] = Field(default_factory=dict)
    execution_intent: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)


class NotebookBatchDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_title: str
    cells: list[NotebookCellDraft]


def _compact_retrieved_chunks_text(retrieved_chunks: list[PaperChunk]) -> str:
    return "\n\n".join(
        (
            f"Chunk ID: {paper_chunk.chunk_id}\n"
            f"Section ID: {paper_chunk.section_id}\n"
            f"Section Title: {paper_chunk.section_title}\n"
            f"Text:\n{paper_chunk.chunk_text}"
        )
        for paper_chunk in retrieved_chunks
    )


def build_section_generation_query(
    learner_profile: LearnerProfile,
    lesson_section: LessonSection,
    retrieved_chunks: list[PaperChunk],
) -> str:
    return f"""
Generate a notebook cell batch for one lesson section.

Learner profile:
{learner_profile.model_dump_json(indent=2)}

Lesson section:
{lesson_section.model_dump_json(indent=2)}

Retrieved source chunks:
{_compact_retrieved_chunks_text(retrieved_chunks)}
""".strip()


def build_retrieval_query_from_lesson_section(lesson_section: LessonSection) -> str:
    query_parts = [
        lesson_section.title,
        lesson_section.teaching_goal,
        " ".join(lesson_section.prerequisite_concepts),
        " ".join(lesson_section.equations_to_unpack),
        " ".join(lesson_section.tensor_shapes_to_state),
        " ".join(lesson_section.likely_misconceptions),
    ]
    return "\n".join(query_part for query_part in query_parts if query_part.strip())


def select_retrieved_chunks_for_lesson_section(
    lesson_section: LessonSection,
    paper_chunks: list[PaperChunk],
    retrieval_top_k: int,
) -> list[PaperChunk]:
    chunks_by_id = {paper_chunk.chunk_id: paper_chunk for paper_chunk in paper_chunks}
    seeded_chunks: list[PaperChunk] = [
        chunks_by_id[source_chunk_id]
        for source_chunk_id in lesson_section.source_chunk_ids
        if source_chunk_id in chunks_by_id
    ]

    preferred_section_id = seeded_chunks[0].section_id if seeded_chunks else None
    lexical_retriever = PaperChunkRetriever(paper_chunks)
    retrieved_chunks = lexical_retriever.search(
        query_text=build_retrieval_query_from_lesson_section(lesson_section),
        top_k=max(1, retrieval_top_k),
        preferred_section_id=preferred_section_id,
    )

    deduplicated_chunks: list[PaperChunk] = []
    seen_chunk_ids: set[str] = set()
    for paper_chunk in seeded_chunks + retrieved_chunks:
        if paper_chunk.chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(paper_chunk.chunk_id)
        deduplicated_chunks.append(paper_chunk)
        if len(deduplicated_chunks) >= max(1, retrieval_top_k):
            break

    return deduplicated_chunks


def _normalize_notebook_cells(section_id: str, notebook_cell_drafts: list[NotebookCellDraft]) -> list[NotebookCell]:
    normalized_cells: list[NotebookCell] = []
    for cell_index, notebook_cell_draft in enumerate(notebook_cell_drafts):
        normalized_metadata = dict(notebook_cell_draft.metadata)
        normalized_metadata.setdefault("section_id", section_id)
        normalized_metadata.setdefault(
            "pedagogical_role",
            "code" if notebook_cell_draft.cell_type == "code" else f"markdown_{cell_index:03d}",
        )
        normalized_cells.append(
            NotebookCell(
                cell_id=f"cell_{section_id}_{cell_index:03d}",
                cell_type=notebook_cell_draft.cell_type,  # validated below
                source=notebook_cell_draft.source,
                metadata=normalized_metadata,
                execution_intent=notebook_cell_draft.execution_intent,
                source_chunk_ids=notebook_cell_draft.source_chunk_ids,
            )
        )
    return normalized_cells


def generate_notebook_batch_for_section(
    model: Gemma4E2BModel,
    learner_profile: LearnerProfile,
    lesson_section: LessonSection,
    retrieved_chunks: list[PaperChunk],
    input_artifact_paths: list[str],
    output_artifact_path: str,
    code_version: str,
) -> NotebookBatch:
    notebook_batch_draft = model.generate_json(
        system_instruction=CELL_GENERATION_SYSTEM_INSTRUCTION,
        user_prompt=build_section_generation_query(
            learner_profile=learner_profile,
            lesson_section=lesson_section,
            retrieved_chunks=retrieved_chunks,
        ),
        response_model=NotebookBatchDraft,
    )

    generated_at_utc = utc_timestamp_string()
    notebook_batch = NotebookBatch(
        section_id=lesson_section.section_id,
        section_title=notebook_batch_draft.section_title,
        cells=_normalize_notebook_cells(lesson_section.section_id, notebook_batch_draft.cells),
        batch_model_provenance=model.model_provenance(),
        generated_at_utc=generated_at_utc,
        stage_provenance=StageProvenance(
            stage_name="cell_generation",
            created_at_utc=generated_at_utc,
            input_artifact_paths=input_artifact_paths + [paper_chunk.chunk_id for paper_chunk in retrieved_chunks],
            output_artifact_path=output_artifact_path,
            code_version=code_version,
            model_provenance=model.model_provenance(),
        ),
    )

    validation_errors = validate_notebook_batch(notebook_batch=notebook_batch, lesson_section=lesson_section)
    if validation_errors:
        raise RuntimeError(f"Notebook batch validation failed: {'; '.join(validation_errors)}")

    return notebook_batch


def validate_notebook_batch(notebook_batch: NotebookBatch, lesson_section: LessonSection) -> list[str]:
    validation_errors: list[str] = []
    if not notebook_batch.cells:
        validation_errors.append(f"Notebook batch {notebook_batch.section_id} must contain at least one cell.")
        return validation_errors

    allowed_cell_types = {"markdown", "code"}
    if notebook_batch.cells[0].cell_type != "markdown":
        validation_errors.append(f"Notebook batch {notebook_batch.section_id} must start with a markdown intro cell.")

    has_code_cell = False
    has_recap_markdown_cell = False
    for notebook_cell in notebook_batch.cells:
        if notebook_cell.cell_type not in allowed_cell_types:
            validation_errors.append(
                f"Notebook batch {notebook_batch.section_id} contains unsupported cell type: {notebook_cell.cell_type}."
            )
        if len(notebook_cell.source.splitlines()) > MAXIMUM_CELL_LINES:
            validation_errors.append(
                f"Notebook batch {notebook_batch.section_id} contains overlong cell {notebook_cell.cell_id}."
            )
        if notebook_cell.cell_type == "code":
            has_code_cell = True
        if notebook_cell.cell_type == "markdown" and notebook_cell.metadata.get("pedagogical_role") in {
            "recap",
            "exercise",
            "recap_or_exercise",
        }:
            has_recap_markdown_cell = True

    if lesson_section.requires_code_example and not has_code_cell:
        validation_errors.append(f"Notebook batch {notebook_batch.section_id} is missing a required code cell.")
    if lesson_section.requires_recap and not has_recap_markdown_cell:
        validation_errors.append(f"Notebook batch {notebook_batch.section_id} is missing a required recap or exercise cell.")

    return validation_errors


def save_notebook_batch(notebook_batch: NotebookBatch, output_path: str | Path) -> Path:
    notebook_batch_path = Path(output_path)
    notebook_batch_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_batch_path.write_text(notebook_batch.model_dump_json(indent=2), encoding="utf-8")
    return notebook_batch_path
