from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from src.notebook_builder import build_notebook_metadata, build_notebook_object, load_notebook_batches, write_notebook
from src.schemas import LearnerProfile, ModelProvenance, NotebookBatch, NotebookPlan, ParsedPaper


def build_learner_profile() -> LearnerProfile:
    return LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
        preferred_depth="deep",
        preferred_pacing="moderate",
    )


def build_notebook_plan() -> NotebookPlan:
    return NotebookPlan(
        paper_id="attention-paper",
        learner_profile=build_learner_profile(),
        lesson_sections=[
            {
                "section_id": "lesson_001",
                "title": "Second topic",
                "teaching_goal": "Explain the second topic.",
                "source_chunk_ids": ["chunk_001"],
                "requires_code_example": True,
                "requires_recap": True,
            },
            {
                "section_id": "lesson_000",
                "title": "First topic",
                "teaching_goal": "Explain the first topic.",
                "source_chunk_ids": ["chunk_000"],
                "requires_code_example": True,
                "requires_recap": True,
            },
        ],
        created_at_utc="20260408T010203Z",
    )


def build_model_provenance() -> ModelProvenance:
    return ModelProvenance(
        inference_backend="llama_cpp",
        provider_name="llama.cpp",
        model_name="gemma-4-e2b-q4_0",
        temperature=0.2,
        max_output_tokens=1800,
    )


def build_parsed_paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="attention-paper",
        paper_title="Attention Is All You Need",
        source_pdf_path="/tmp/attention.pdf",
        source_pdf_sha256="abc123",
        parser_name="pymupdf4llm",
        parser_version="0.0.20",
        parsed_at_utc="20260408T010203Z",
        sections=[],
        abstract_text="",
    )


def build_notebook_batch(section_id: str, section_title: str, source_chunk_id: str) -> NotebookBatch:
    return NotebookBatch(
        section_id=section_id,
        section_title=section_title,
        cells=[
            {
                "cell_id": f"cell_{section_id}_000",
                "cell_type": "markdown",
                "source": f"## {section_title}",
                "metadata": {"pedagogical_role": "intro", "section_id": section_id},
                "source_chunk_ids": [source_chunk_id],
            },
            {
                "cell_id": f"cell_{section_id}_001",
                "cell_type": "code",
                "source": f"print('{section_title}')",
                "metadata": {"pedagogical_role": "code", "section_id": section_id},
                "execution_intent": "Run the section example.",
                "source_chunk_ids": [source_chunk_id],
            },
        ],
        generated_at_utc="20260408T010203Z",
    )


def test_load_notebook_batches_round_trips_saved_batches(tmp_path: Path) -> None:
    first_batch_path = tmp_path / "lesson_001.json"
    second_batch_path = tmp_path / "lesson_000.json"
    first_batch_path.write_text(
        build_notebook_batch("lesson_001", "Second topic", "chunk_001").model_dump_json(indent=2),
        encoding="utf-8",
    )
    second_batch_path.write_text(
        build_notebook_batch("lesson_000", "First topic", "chunk_000").model_dump_json(indent=2),
        encoding="utf-8",
    )

    loaded_batches = load_notebook_batches([first_batch_path, second_batch_path])

    assert [loaded_batch.section_id for loaded_batch in loaded_batches] == ["lesson_001", "lesson_000"]


def test_build_notebook_object_preserves_plan_order_not_filesystem_order() -> None:
    notebook_plan = build_notebook_plan()
    batch_for_second_section = build_notebook_batch("lesson_001", "Second topic", "chunk_001")
    batch_for_first_section = build_notebook_batch("lesson_000", "First topic", "chunk_000")
    ordered_batches = [batch_for_second_section, batch_for_first_section]

    notebook_metadata = build_notebook_metadata(
        notebook_plan=notebook_plan,
        run_id="20260408T010203Z",
        code_version="test-code-version",
        model_provenance=build_model_provenance(),
        generated_section_batches=ordered_batches,
        parsed_paper=build_parsed_paper(),
    )
    notebook_object = build_notebook_object(
        notebook_title="Attention Is All You Need",
        learner_profile=build_learner_profile(),
        generated_section_batches=ordered_batches,
        notebook_metadata=notebook_metadata,
    )

    assert notebook_object.cells[0].source.startswith("# Attention Is All You Need")
    assert notebook_object.cells[1].source == "## Second topic"
    assert notebook_object.cells[3].source == "## First topic"
    assert notebook_object.cells[2].metadata["pedagogical_role"] == "code"
    assert notebook_object.cells[2].execution_count is None
    assert notebook_object.cells[2].outputs == []


def test_build_notebook_metadata_includes_run_model_and_chunk_provenance_fields() -> None:
    metadata = build_notebook_metadata(
        notebook_plan=build_notebook_plan(),
        run_id="20260408T010203Z",
        code_version="test-code-version",
        model_provenance=build_model_provenance(),
        generated_section_batches=[
            build_notebook_batch("lesson_001", "Second topic", "chunk_001"),
            build_notebook_batch("lesson_000", "First topic", "chunk_000"),
        ],
        parsed_paper=build_parsed_paper(),
    )

    assert metadata["authors"] == [{"name": "paper-to-notebook"}]
    assert metadata["paper_to_notebook"]["paper_id"] == "attention-paper"
    assert metadata["paper_to_notebook"]["source_paper_title"] == "Attention Is All You Need"
    assert metadata["paper_to_notebook"]["run_id"] == "20260408T010203Z"
    assert metadata["paper_to_notebook"]["model_name"] == "gemma-4-e2b-q4_0"
    assert metadata["paper_to_notebook"]["inference_backend"] == "llama_cpp"
    assert metadata["paper_to_notebook"]["project_version_or_code_version"] == "test-code-version"
    assert metadata["paper_to_notebook"]["pedagogical_depth"] == "deep"
    assert metadata["paper_to_notebook"]["chunk_provenance_by_section"]["lesson_001"]["chunk_ids"] == ["chunk_001"]


def test_write_notebook_round_trips_through_nbformat(tmp_path: Path) -> None:
    notebook_object = build_notebook_object(
        notebook_title="Attention Is All You Need",
        learner_profile=build_learner_profile(),
        generated_section_batches=[
            build_notebook_batch("lesson_001", "Second topic", "chunk_001"),
            build_notebook_batch("lesson_000", "First topic", "chunk_000"),
        ],
        notebook_metadata=build_notebook_metadata(
            notebook_plan=build_notebook_plan(),
            run_id="20260408T010203Z",
            code_version="test-code-version",
            model_provenance=build_model_provenance(),
            generated_section_batches=[
                build_notebook_batch("lesson_001", "Second topic", "chunk_001"),
                build_notebook_batch("lesson_000", "First topic", "chunk_000"),
            ],
            parsed_paper=build_parsed_paper(),
        ),
    )

    notebook_path = write_notebook(notebook_object=notebook_object, output_path=tmp_path / "final_notebook.ipynb")
    reloaded_notebook = nbformat.read(notebook_path, as_version=4)

    assert notebook_path.is_file()
    assert reloaded_notebook.metadata["paper_to_notebook"]["paper_id"] == "attention-paper"
    assert reloaded_notebook.cells[1].source == "## Second topic"


def test_build_notebook_object_uses_learner_facing_overview_metadata() -> None:
    notebook_object = build_notebook_object(
        notebook_title="Attention Is All You Need",
        learner_profile=build_learner_profile(),
        generated_section_batches=[build_notebook_batch("lesson_001", "Second topic", "chunk_001")],
        notebook_metadata={},
    )

    overview_source = notebook_object.cells[0].source
    assert "**Pedagogical depth**: deep" in overview_source
    assert "**Math background**: Linear algebra" in overview_source


def test_duplicate_batch_detection_shape_for_assembly_logic() -> None:
    loaded_batches = [
        build_notebook_batch("lesson_001", "Second topic", "chunk_001"),
        build_notebook_batch("lesson_001", "Second topic duplicate", "chunk_002"),
    ]

    seen_section_ids: set[str] = set()
    with pytest.raises(RuntimeError, match="Duplicate notebook batch"):
        for loaded_batch in loaded_batches:
            if loaded_batch.section_id in seen_section_ids:
                raise RuntimeError(f"Duplicate notebook batch detected for section_id={loaded_batch.section_id}.")
            seen_section_ids.add(loaded_batch.section_id)
