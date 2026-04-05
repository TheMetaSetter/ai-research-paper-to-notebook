from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from src.schemas import (
    ConceptItem,
    LearnerProfile,
    LessonSection,
    ModelProvenance,
    NotebookBatch,
    NotebookCell,
    NotebookPlan,
    PaperChunk,
    PaperSection,
    ParsedPaper,
    RunManifest,
    StageProvenance,
    ValidationReport,
)


def build_model_provenance() -> ModelProvenance:
    return ModelProvenance(
        inference_backend="llama_cpp",
        provider_name="llama.cpp",
        model_name="gemma-4-e2b-q4_0",
        temperature=0.2,
        max_output_tokens=1800,
    )


def build_stage_provenance() -> StageProvenance:
    return StageProvenance(
        stage_name="planning",
        created_at_utc="20260405T120000Z",
        input_artifact_paths=["runs/paper/run/chunks/chunks.json"],
        output_artifact_path="runs/paper/run/notebook_plan/notebook_plan.json",
        code_version="phase0-contracts",
        model_provenance=build_model_provenance(),
    )


def test_major_contracts_round_trip_through_json() -> None:
    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra and probability.",
        machine_learning_background="Intermediate supervised learning knowledge.",
        deep_learning_background="Beginner neural network knowledge.",
        python_background="Intermediate Python experience.",
        tensor_familiarity="Comfortable with matrix multiplication.",
    )
    lesson_section = LessonSection(
        section_id="lesson_1",
        title="Attention Overview",
        teaching_goal="Explain the high-level attention mechanism.",
        prerequisite_concepts=["vector spaces"],
        source_chunk_ids=["chunk_1"],
        equations_to_unpack=["softmax(QK^T / sqrt(d_k)) V"],
        tensor_shapes_to_state=["Q in R^{B x T x d_k}"],
        likely_misconceptions=["Attention is not the same as recurrence."],
    )
    notebook_cell = NotebookCell(
        cell_id="cell_1",
        cell_type="markdown",
        source="## Attention",
        metadata={"pedagogical_role": "heading", "order": 1},
        source_chunk_ids=["chunk_1"],
    )

    instances = [
        ParsedPaper(
            paper_id="paper_1",
            paper_title="Attention Is All You Need",
            source_pdf_path="/tmp/paper.pdf",
            source_pdf_sha256="abc123",
            parser_name="pymupdf4llm",
            parser_version="0.0.20",
            parsed_at_utc="20260405T120000Z",
            sections=[
                PaperSection(
                    section_id="sec_1",
                    title="Introduction",
                    page_start=1,
                    page_end=2,
                    markdown_text="Intro text.",
                )
            ],
            abstract_text="Abstract text.",
            authors=["Author One", "Author Two"],
            stage_provenance=build_stage_provenance(),
        ),
        PaperChunk(
            chunk_id="chunk_1",
            section_id="sec_1",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Chunk text.",
            equation_markers=["eq_1"],
            notation_tokens=["Q", "K", "V"],
            figure_references=["Figure 1"],
        ),
        ConceptItem(
            concept_id="concept_1",
            label="Scaled dot-product attention",
            concept_type="equation",
            source_chunk_ids=["chunk_1"],
            prerequisites=["concept_0"],
            equation_text="softmax(QK^T / sqrt(d_k))V",
            notation_symbols=["Q", "K", "V"],
            tensor_shape_notes=["Q: B x T x d_k"],
            confidence=0.9,
        ),
        learner_profile,
        lesson_section,
        NotebookPlan(
            paper_id="paper_1",
            learner_profile=learner_profile,
            lesson_sections=[lesson_section],
            planning_notes=["Begin with notation normalization."],
            planning_model_provenance=build_model_provenance(),
            created_at_utc="20260405T120000Z",
            stage_provenance=build_stage_provenance(),
        ),
        notebook_cell,
        NotebookBatch(
            section_id="lesson_1",
            section_title="Attention Overview",
            cells=[notebook_cell],
            batch_model_provenance=build_model_provenance(),
            generated_at_utc="20260405T120500Z",
            stage_provenance=build_stage_provenance(),
        ),
        ValidationReport(
            paper_id="paper_1",
            run_id="20260405T120000Z",
            is_valid=False,
            errors=["Missing recap cell."],
            warnings=["Cell length is close to threshold."],
            checks_run=["schema_validity", "pedagogical_ordering"],
            notebook_path="runs/paper_1/20260405T120000Z/notebook/final_notebook.ipynb",
            validated_at_utc="20260405T121000Z",
            stage_provenance=build_stage_provenance(),
        ),
        RunManifest(
            run_id="20260405T120000Z",
            paper_slug="attention-is-all-you-need",
            source_pdf_path="/tmp/paper.pdf",
            source_pdf_sha256="abc123",
            params_path="/tmp/params.yaml",
            run_parameters={"inference_backend": "llama_cpp"},
            planned_notebook_output_path="runs/paper/notebook/final_notebook.ipynb",
            active_model_provenance=build_model_provenance(),
            stage_artifact_paths={"parsed_paper": "runs/paper/parsed_paper/parsed_paper.json"},
            created_at_utc="20260405T120000Z",
        ),
    ]

    for instance in instances:
        serialized_payload = instance.model_dump(mode="json")
        reloaded_instance = type(instance).model_validate(serialized_payload)
        assert reloaded_instance == instance


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LearnerProfile.model_validate(
            {
                "mathematics_background": "Math",
                "machine_learning_background": "ML",
                "deep_learning_background": "DL",
                "python_background": "Python",
                "tensor_familiarity": "High",
                "unexpected": "value",
            }
        )


def test_provenance_fields_are_preserved_in_serialization() -> None:
    notebook_plan = NotebookPlan(
        paper_id="paper_1",
        learner_profile=LearnerProfile(
            mathematics_background="Math",
            machine_learning_background="ML",
            deep_learning_background="DL",
            python_background="Python",
            tensor_familiarity="High",
        ),
        lesson_sections=[],
        planning_notes=[],
        planning_model_provenance=build_model_provenance(),
        created_at_utc="20260405T120000Z",
        stage_provenance=build_stage_provenance(),
    )

    serialized_payload = notebook_plan.model_dump(mode="json")

    assert serialized_payload["planning_model_provenance"]["model_name"] == "gemma-4-e2b-q4_0"
    assert serialized_payload["stage_provenance"]["stage_name"] == "planning"


def test_notebook_metadata_fields_are_json_serializable() -> None:
    notebook_batch = NotebookBatch(
        section_id="lesson_1",
        section_title="Attention Overview",
        cells=[
            NotebookCell(
                cell_id="cell_1",
                cell_type="code",
                source="print('hello')",
                metadata={
                    "pedagogical_role": "toy_example",
                    "chunk_provenance": ["chunk_1", "chunk_2"],
                    "depth": {"mode": "deep"},
                },
                execution_intent="Demonstrate tensor shapes.",
                source_chunk_ids=["chunk_1", "chunk_2"],
            )
        ],
        batch_model_provenance=build_model_provenance(),
        generated_at_utc="20260405T120500Z",
    )

    serialized_payload = notebook_batch.model_dump(mode="json")

    json.dumps(serialized_payload)
    assert serialized_payload["cells"][0]["metadata"]["depth"]["mode"] == "deep"

