from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

import src.validators as validators_module
from src.notebook_builder import build_notebook_metadata, build_notebook_object
from src.schemas import LearnerProfile, NotebookBatch, NotebookPlan
from src.validators import (
    load_notebook_object,
    validate_assembled_notebook,
    validate_cell_lengths,
    validate_notebook_schema,
    validate_pedagogical_ordering,
)


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
                "section_id": "lesson_000",
                "title": "Attention basics",
                "teaching_goal": "Explain attention.",
                "equations_to_unpack": ["QK^T"],
                "tensor_shapes_to_state": ["Q: B x T x d_k"],
                "source_chunk_ids": ["chunk_000"],
                "requires_code_example": True,
                "requires_recap": True,
            },
            {
                "section_id": "lesson_001",
                "title": "Softmax intuition",
                "teaching_goal": "Explain softmax.",
                "source_chunk_ids": ["chunk_001"],
                "requires_code_example": True,
                "requires_recap": True,
            },
        ],
        created_at_utc="20260408T010203Z",
    )


def build_notebook_batches() -> list[NotebookBatch]:
    return [
        NotebookBatch(
            section_id="lesson_000",
            section_title="Attention basics",
            cells=[
                {
                    "cell_id": "cell_lesson_000_000",
                    "cell_type": "markdown",
                    "source": "## Attention basics\nWe use Q and K and note the shape Q: B x T x d_k.",
                    "metadata": {"section_id": "lesson_000", "pedagogical_role": "intro"},
                    "source_chunk_ids": ["chunk_000"],
                },
                {
                    "cell_id": "cell_lesson_000_001",
                    "cell_type": "code",
                    "source": "print('attention')",
                    "metadata": {"section_id": "lesson_000", "pedagogical_role": "code"},
                    "source_chunk_ids": ["chunk_000"],
                },
            ],
            generated_at_utc="20260408T010203Z",
        ),
        NotebookBatch(
            section_id="lesson_001",
            section_title="Softmax intuition",
            cells=[
                {
                    "cell_id": "cell_lesson_001_000",
                    "cell_type": "markdown",
                    "source": "## Softmax intuition\nSoftmax normalizes scores.",
                    "metadata": {"section_id": "lesson_001", "pedagogical_role": "intro"},
                    "source_chunk_ids": ["chunk_001"],
                }
            ],
            generated_at_utc="20260408T010203Z",
        ),
    ]


def build_sample_notebook_object() -> nbformat.NotebookNode:
    notebook_plan = build_notebook_plan()
    notebook_batches = build_notebook_batches()
    return build_notebook_object_from_batches(notebook_plan, notebook_batches)


def build_notebook_object_from_batches(
    notebook_plan: NotebookPlan,
    notebook_batches: list[NotebookBatch],
) -> nbformat.NotebookNode:
    return build_notebook_object(
        notebook_title="Attention Is All You Need",
        learner_profile=build_learner_profile(),
        generated_section_batches=notebook_batches,
        notebook_metadata=build_notebook_metadata(
            notebook_plan=notebook_plan,
            run_id="20260408T010203Z",
            code_version="test-code-version",
            model_provenance=None,
            generated_section_batches=notebook_batches,
            parsed_paper=None,
        ),
    )


def test_validate_notebook_schema_reports_invalid_notebook() -> None:
    invalid_notebook = {"cells": "not-a-list"}

    issues = validate_notebook_schema(invalid_notebook)  # type: ignore[arg-type]

    assert len(issues) == 1
    assert issues[0].check_name == "schema_validity"


def test_load_notebook_object_requires_existing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_notebook_object(tmp_path / "missing.ipynb")


def test_validate_cell_lengths_reports_overlong_cells() -> None:
    notebook_object = build_sample_notebook_object()
    notebook_object.cells[1].source = "\n".join(["line"] * 101)

    issues = validate_cell_lengths(notebook_object)

    assert len(issues) == 1
    assert issues[0].severity == "warning"
    assert issues[0].check_name == "cell_length"


def test_validate_pedagogical_ordering_reports_order_mismatch() -> None:
    notebook_object = build_sample_notebook_object()
    notebook_object.cells[1].metadata["section_id"] = "lesson_001"
    notebook_object.cells[3].metadata["section_id"] = "lesson_000"

    issues = validate_pedagogical_ordering(notebook_object, build_notebook_plan())

    assert any(validation_issue.check_name == "pedagogical_ordering" for validation_issue in issues)


def test_validate_assembled_notebook_reports_missing_section_and_warnings() -> None:
    notebook_plan = build_notebook_plan()
    notebook_batches = build_notebook_batches()[:1]
    notebook_object = build_notebook_object_from_batches(notebook_plan, notebook_batches)

    validation_report, validation_issues = validate_assembled_notebook(
        notebook_object=notebook_object,
        notebook_plan=notebook_plan,
        learner_profile=build_learner_profile(),
        generated_section_batches=notebook_batches,
        paper_id="attention-paper",
        run_id="20260408T010203Z",
        notebook_path="runs/paper/notebook/final_notebook.ipynb",
        input_artifact_paths=[],
        code_version="test-code-version",
        enable_smoke_test=False,
        execution_timeout_seconds=120,
        working_directory=".",
    )

    assert validation_report.is_valid is False
    assert any("missing planned section lesson_001" in error.lower() for error in validation_report.errors)
    assert any(validation_issue.check_name == "pedagogical_ordering" for validation_issue in validation_issues)


def test_validate_assembled_notebook_reports_notation_and_tensor_shape_warnings() -> None:
    notebook_plan = build_notebook_plan()
    notebook_batches = build_notebook_batches()
    notebook_batches[0].cells[0].source = "## Attention basics\nWe discuss attention without symbols."
    notebook_object = build_notebook_object_from_batches(notebook_plan, notebook_batches)

    validation_report, _ = validate_assembled_notebook(
        notebook_object=notebook_object,
        notebook_plan=notebook_plan,
        learner_profile=build_learner_profile(),
        generated_section_batches=notebook_batches,
        paper_id="attention-paper",
        run_id="20260408T010203Z",
        notebook_path="runs/paper/notebook/final_notebook.ipynb",
        input_artifact_paths=[],
        code_version="test-code-version",
        enable_smoke_test=False,
        execution_timeout_seconds=120,
        working_directory=".",
    )

    assert any("notation_consistency" in warning for warning in validation_report.warnings)
    assert any("tensor_shape_consistency" in warning for warning in validation_report.warnings)


def test_validate_assembled_notebook_records_smoke_test_success(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubNotebookClient:
        def __init__(self, notebook_object, timeout, kernel_name, resources) -> None:
            self.notebook_object = notebook_object

        def execute(self) -> None:
            return None

    monkeypatch.setattr(validators_module, "NotebookClient", StubNotebookClient)

    validation_report, _ = validate_assembled_notebook(
        notebook_object=build_sample_notebook_object(),
        notebook_plan=build_notebook_plan(),
        learner_profile=build_learner_profile(),
        generated_section_batches=build_notebook_batches(),
        paper_id="attention-paper",
        run_id="20260408T010203Z",
        notebook_path="runs/paper/notebook/final_notebook.ipynb",
        input_artifact_paths=[],
        code_version="test-code-version",
        enable_smoke_test=True,
        execution_timeout_seconds=120,
        working_directory=".",
    )

    assert "execution_smoke_test" in validation_report.checks_run
    assert not any("execution_smoke_test" in error for error in validation_report.errors)


def test_validate_assembled_notebook_records_smoke_test_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubNotebookClient:
        def __init__(self, notebook_object, timeout, kernel_name, resources) -> None:
            self.notebook_object = notebook_object

        def execute(self) -> None:
            raise RuntimeError("execution failed")

    monkeypatch.setattr(validators_module, "NotebookClient", StubNotebookClient)

    validation_report, validation_issues = validate_assembled_notebook(
        notebook_object=build_sample_notebook_object(),
        notebook_plan=build_notebook_plan(),
        learner_profile=build_learner_profile(),
        generated_section_batches=build_notebook_batches(),
        paper_id="attention-paper",
        run_id="20260408T010203Z",
        notebook_path="runs/paper/notebook/final_notebook.ipynb",
        input_artifact_paths=[],
        code_version="test-code-version",
        enable_smoke_test=True,
        execution_timeout_seconds=120,
        working_directory=".",
    )

    assert validation_report.is_valid is False
    assert any("execution_smoke_test" in error for error in validation_report.errors)
    assert any(validation_issue.repair_kind == "repair_execution" for validation_issue in validation_issues)
