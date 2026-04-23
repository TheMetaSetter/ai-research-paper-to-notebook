from __future__ import annotations

from pathlib import Path

import pytest

from src.cell_generator import (
    generate_notebook_batch_for_section,
    save_notebook_batch,
    select_retrieved_chunks_for_lesson_section,
    validate_notebook_batch,
)
from src.schemas import LearnerProfile, LessonSection, ModelProvenance, NotebookBatch, PaperChunk


def build_learner_profile() -> LearnerProfile:
    return LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrix multiplication",
        wants_tensor_shapes=True,
        wants_derivations=False,
        preferred_depth="deep",
        preferred_pacing="moderate",
    )


def build_lesson_section() -> LessonSection:
    return LessonSection(
        section_id="lesson_000",
        title="Attention from first principles",
        teaching_goal="Explain scaled dot-product attention.",
        prerequisite_concepts=["vectors"],
        source_chunk_ids=["chunk_00000"],
        equations_to_unpack=["softmax(QK^T)V"],
        tensor_shapes_to_state=["Q: B x T x d_k"],
        likely_misconceptions=["Attention is not recurrence."],
        requires_code_example=True,
        requires_recap=True,
    )


def build_paper_chunks() -> list[PaperChunk]:
    return [
        PaperChunk(
            chunk_id="chunk_00000",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Attention compares queries and keys.",
            equation_markers=["="],
            notation_tokens=["attention", "q", "k"],
            figure_references=[],
        ),
        PaperChunk(
            chunk_id="chunk_00001",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Softmax normalizes attention scores.",
            equation_markers=["softmax"],
            notation_tokens=["softmax"],
            figure_references=[],
        ),
    ]


class StubGenerationModel:
    def __init__(self) -> None:
        self.user_prompt: str | None = None

    def generate_json(self, system_instruction: str, user_prompt: str, response_model):
        self.user_prompt = user_prompt
        return response_model(
            section_title="Attention from first principles",
            cells=[
                {
                    "cell_type": "markdown",
                    "source": "## Attention from first principles",
                    "metadata": {"pedagogical_role": "intro"},
                    "source_chunk_ids": ["chunk_00000"],
                },
                {
                    "cell_type": "markdown",
                    "source": "The notation Q, K, and V defines how attention compares tokens.",
                    "metadata": {"pedagogical_role": "notation"},
                    "source_chunk_ids": ["chunk_00000"],
                },
                {
                    "cell_type": "code",
                    "source": "import numpy as np\n\n# Show why score normalization matters.\nprint(np.array([1.0, 2.0]))",
                    "metadata": {"pedagogical_role": "code"},
                    "execution_intent": "Demonstrate the local code example.",
                    "source_chunk_ids": ["chunk_00000", "chunk_00001"],
                },
                {
                    "cell_type": "markdown",
                    "source": "### Recap\nWhy does softmax help compare token scores?",
                    "metadata": {"pedagogical_role": "recap"},
                    "source_chunk_ids": ["chunk_00001"],
                },
            ],
        )

    def model_provenance(self) -> ModelProvenance:
        return ModelProvenance(
            inference_backend="llama_cpp",
            provider_name="llama.cpp",
            model_name="gemma-4-e2b-q4_0",
            temperature=0.2,
            max_output_tokens=1800,
        )


def test_generate_notebook_batch_for_section_maps_model_output() -> None:
    notebook_batch = generate_notebook_batch_for_section(
        model=StubGenerationModel(),
        learner_profile=build_learner_profile(),
        lesson_section=build_lesson_section(),
        retrieved_chunks=build_paper_chunks(),
        input_artifact_paths=["notebook_plan.json", "learner_profile.json", "chunks.json"],
        output_artifact_path="cell_batches/lesson_000.json",
        code_version="test-code-version",
    )

    assert notebook_batch.section_id == "lesson_000"
    assert notebook_batch.section_title == "Attention from first principles"
    assert [notebook_cell.cell_id for notebook_cell in notebook_batch.cells] == [
        "cell_lesson_000_000",
        "cell_lesson_000_001",
        "cell_lesson_000_002",
        "cell_lesson_000_003",
    ]
    assert notebook_batch.cells[0].metadata["section_id"] == "lesson_000"
    assert notebook_batch.stage_provenance is not None
    assert notebook_batch.stage_provenance.stage_name == "cell_generation"


def test_build_section_generation_query_includes_profile_section_and_chunks() -> None:
    generation_model = StubGenerationModel()
    lesson_section = build_lesson_section()
    retrieved_chunks = build_paper_chunks()

    generate_notebook_batch_for_section(
        model=generation_model,
        learner_profile=build_learner_profile(),
        lesson_section=lesson_section,
        retrieved_chunks=retrieved_chunks,
        input_artifact_paths=[],
        output_artifact_path="cell_batches/lesson_000.json",
        code_version="test-code-version",
    )

    assert generation_model.user_prompt is not None
    assert "Linear algebra" in generation_model.user_prompt
    assert lesson_section.title in generation_model.user_prompt
    assert "Chunk ID: chunk_00000" in generation_model.user_prompt
    assert "Softmax normalizes attention scores." in generation_model.user_prompt


def test_validate_notebook_batch_reports_batch_errors() -> None:
    overlong_source = "\n".join(["line"] * 101)
    invalid_batch = NotebookBatch(
        section_id="lesson_000",
        section_title="Attention from first principles",
        cells=[
            {
                "cell_id": "cell_lesson_000_000",
                "cell_type": "code",
                "source": overlong_source,
                "metadata": {"pedagogical_role": "code"},
                "source_chunk_ids": ["chunk_00000"],
            }
        ],
        generated_at_utc="20260405T120000Z",
    )

    validation_errors = validate_notebook_batch(invalid_batch, build_lesson_section())
    assert "Notebook batch lesson_000 must start with a markdown intro cell." in validation_errors
    assert "Notebook batch lesson_000 contains overlong cell cell_lesson_000_000." in validation_errors
    assert "Notebook batch lesson_000 is missing a required recap or exercise cell." in validation_errors


def test_validate_notebook_batch_requires_non_empty_batch() -> None:
    empty_batch = NotebookBatch(
        section_id="lesson_000",
        section_title="Attention from first principles",
        cells=[],
        generated_at_utc="20260405T120000Z",
    )
    assert validate_notebook_batch(empty_batch, build_lesson_section()) == [
        "Notebook batch lesson_000 must contain at least one cell."
    ]


def test_select_retrieved_chunks_seeds_explicit_source_chunks() -> None:
    class StubRetriever:
        def __init__(self, paper_chunks: list[PaperChunk]) -> None:
            self.paper_chunks = paper_chunks

        def search(self, query_text: str, top_k: int = 6, preferred_section_id: str | None = None) -> list[PaperChunk]:
            return list(self.paper_chunks)[:top_k]

    import src.cell_generator as cell_generator_module

    original_retriever = cell_generator_module.PaperChunkRetriever
    cell_generator_module.PaperChunkRetriever = StubRetriever
    try:
        selected_chunks = select_retrieved_chunks_for_lesson_section(
            lesson_section=build_lesson_section(),
            paper_chunks=build_paper_chunks(),
            retrieval_top_k=2,
        )
    finally:
        cell_generator_module.PaperChunkRetriever = original_retriever

    assert selected_chunks[0].chunk_id == "chunk_00000"
    assert len(selected_chunks) == 2


def test_save_notebook_batch_round_trips_json(tmp_path: Path) -> None:
    notebook_batch = generate_notebook_batch_for_section(
        model=StubGenerationModel(),
        learner_profile=build_learner_profile(),
        lesson_section=build_lesson_section(),
        retrieved_chunks=build_paper_chunks(),
        input_artifact_paths=[],
        output_artifact_path=str(tmp_path / "cell_batches" / "lesson_000.json"),
        code_version="test-code-version",
    )

    notebook_batch_path = save_notebook_batch(
        notebook_batch=notebook_batch,
        output_path=tmp_path / "cell_batches" / "lesson_000.json",
    )
    reloaded_notebook_batch = NotebookBatch.model_validate_json(notebook_batch_path.read_text(encoding="utf-8"))
    assert reloaded_notebook_batch == notebook_batch


def test_generate_notebook_batch_raises_for_invalid_model_output() -> None:
    class InvalidGenerationModel(StubGenerationModel):
        def generate_json(self, system_instruction: str, user_prompt: str, response_model):
            return response_model(section_title="Attention", cells=[])

    with pytest.raises(RuntimeError, match="Notebook batch validation failed"):
        generate_notebook_batch_for_section(
            model=InvalidGenerationModel(),
            learner_profile=build_learner_profile(),
            lesson_section=build_lesson_section(),
            retrieved_chunks=build_paper_chunks(),
            input_artifact_paths=[],
            output_artifact_path="cell_batches/lesson_000.json",
            code_version="test-code-version",
        )
