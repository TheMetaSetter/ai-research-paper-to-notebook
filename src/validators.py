from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat import NotebookNode

from src.artifact_store import utc_timestamp_string
from src.schemas import LearnerProfile, NotebookBatch, NotebookPlan, StageProvenance, ValidationReport


MAXIMUM_NOTEBOOK_CELL_LINES = 100


@dataclass
class ValidationIssue:
    severity: str
    check_name: str
    message: str
    section_id: str | None = None
    repair_kind: str | None = None


def load_notebook_object(notebook_path: str | Path) -> NotebookNode:
    notebook_input_path = Path(notebook_path)
    if not notebook_input_path.is_file():
        raise FileNotFoundError(f"Notebook artifact was not found: {notebook_input_path}")
    return nbformat.read(notebook_input_path, as_version=4)


def validate_notebook_schema(notebook_object: NotebookNode) -> list[ValidationIssue]:
    try:
        nbformat.validate(notebook_object)
        return []
    except Exception as validation_error:
        return [
            ValidationIssue(
                severity="error",
                check_name="schema_validity",
                message=str(validation_error),
                repair_kind="rebuild_notebook",
            )
        ]


def validate_notebook_overview(notebook_object: NotebookNode) -> list[ValidationIssue]:
    if not notebook_object.cells:
        return [
            ValidationIssue(
                severity="error",
                check_name="notebook_overview",
                message="Notebook contains no cells.",
                repair_kind="rebuild_notebook",
            )
        ]

    first_cell = notebook_object.cells[0]
    if first_cell.cell_type != "markdown" or not first_cell.source.strip().startswith("# "):
        return [
            ValidationIssue(
                severity="error",
                check_name="notebook_overview",
                message="Notebook is missing the learner-facing overview markdown cell.",
                repair_kind="rebuild_notebook",
            )
        ]
    return []


def validate_cell_lengths(
    notebook_object: NotebookNode,
    maximum_lines_per_cell: int = MAXIMUM_NOTEBOOK_CELL_LINES,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for cell_index, notebook_cell in enumerate(notebook_object.cells):
        line_count = len(notebook_cell.source.splitlines())
        if line_count > maximum_lines_per_cell:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    check_name="cell_length",
                    message=f"Cell {cell_index} has {line_count} lines, exceeding {maximum_lines_per_cell}.",
                )
            )
    return issues


def validate_pedagogical_ordering(
    notebook_object: NotebookNode,
    notebook_plan: NotebookPlan,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    observed_section_order: list[str] = []
    seen_section_ids: set[str] = set()

    for notebook_cell in notebook_object.cells:
        section_id = notebook_cell.get("metadata", {}).get("section_id")
        if not isinstance(section_id, str) or section_id in seen_section_ids:
            continue
        seen_section_ids.add(section_id)
        observed_section_order.append(section_id)

    expected_section_order = [lesson_section.section_id for lesson_section in notebook_plan.lesson_sections]

    missing_section_ids = [section_id for section_id in expected_section_order if section_id not in seen_section_ids]
    for missing_section_id in missing_section_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                check_name="pedagogical_ordering",
                message=f"Notebook is missing planned section {missing_section_id}.",
                section_id=missing_section_id,
                repair_kind="rebuild_notebook",
            )
        )

    unexpected_section_ids = [section_id for section_id in observed_section_order if section_id not in expected_section_order]
    for unexpected_section_id in unexpected_section_ids:
        issues.append(
            ValidationIssue(
                severity="error",
                check_name="pedagogical_ordering",
                message=f"Notebook contains unexpected section {unexpected_section_id}.",
                section_id=unexpected_section_id,
                repair_kind="rebuild_notebook",
            )
        )

    filtered_observed_section_order = [section_id for section_id in observed_section_order if section_id in expected_section_order]
    if filtered_observed_section_order and filtered_observed_section_order != expected_section_order[: len(filtered_observed_section_order)]:
        issues.append(
            ValidationIssue(
                severity="error",
                check_name="pedagogical_ordering",
                message=(
                    "Notebook section order does not match the notebook plan. "
                    f"Expected {expected_section_order} but observed {filtered_observed_section_order}."
                ),
                repair_kind="rebuild_notebook",
            )
        )

    return issues


def validate_notation_consistency(
    notebook_object: NotebookNode,
    notebook_plan: NotebookPlan,
    generated_section_batches: list[NotebookBatch],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    section_text_by_id = _build_section_text_by_id(notebook_object)
    source_chunks_by_section = {
        notebook_batch.section_id: {
            source_chunk_id
            for notebook_cell in notebook_batch.cells
            for source_chunk_id in (notebook_cell.source_chunk_ids or [])
        }
        for notebook_batch in generated_section_batches
    }

    for lesson_section in notebook_plan.lesson_sections:
        required_notation_symbols = _extract_required_notation_symbols(lesson_section)
        if not required_notation_symbols:
            continue

        section_text = section_text_by_id.get(lesson_section.section_id, "")
        missing_symbols = [symbol for symbol in required_notation_symbols if symbol not in section_text]
        if not missing_symbols:
            continue

        chunk_context = sorted(source_chunks_by_section.get(lesson_section.section_id, set()))
        issues.append(
            ValidationIssue(
                severity="warning",
                check_name="notation_consistency",
                message=(
                    f"Section {lesson_section.section_id} is missing notation references for "
                    f"{', '.join(missing_symbols)}. Source chunk context: {chunk_context}."
                ),
                section_id=lesson_section.section_id,
                repair_kind="regenerate_section",
            )
        )

    return issues


def validate_tensor_shape_consistency(
    notebook_object: NotebookNode,
    notebook_plan: NotebookPlan,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    section_text_by_id = _build_section_text_by_id(notebook_object)

    for lesson_section in notebook_plan.lesson_sections:
        if not lesson_section.tensor_shapes_to_state:
            continue

        section_text = section_text_by_id.get(lesson_section.section_id, "")
        mentioned_tensor_variables = [tensor_variable for tensor_variable in _extract_tensor_variables(lesson_section) if tensor_variable in section_text]
        if mentioned_tensor_variables:
            continue

        issues.append(
            ValidationIssue(
                severity="warning",
                check_name="tensor_shape_consistency",
                message=(
                    f"Section {lesson_section.section_id} does not visibly ground the expected tensor shapes: "
                    f"{lesson_section.tensor_shapes_to_state}."
                ),
                section_id=lesson_section.section_id,
                repair_kind="regenerate_section",
            )
        )

    return issues


def execute_notebook_smoke_test(
    notebook_object: NotebookNode,
    working_directory: str,
    execution_timeout_seconds: int,
) -> list[ValidationIssue]:
    try:
        notebook_client = NotebookClient(
            notebook_object,
            timeout=execution_timeout_seconds,
            kernel_name="python3",
            resources={"metadata": {"path": working_directory}},
        )
        notebook_client.execute()
        return []
    except Exception as execution_error:
        return [
            ValidationIssue(
                severity="error",
                check_name="execution_smoke_test",
                message=str(execution_error),
                repair_kind="repair_execution",
            )
        ]


def validate_assembled_notebook(
    notebook_object: NotebookNode,
    notebook_plan: NotebookPlan,
    learner_profile: LearnerProfile,
    generated_section_batches: list[NotebookBatch],
    paper_id: str,
    run_id: str,
    notebook_path: str,
    input_artifact_paths: list[str],
    code_version: str,
    enable_smoke_test: bool,
    execution_timeout_seconds: int,
    working_directory: str,
) -> tuple[ValidationReport, list[ValidationIssue]]:
    validation_issues: list[ValidationIssue] = []
    check_names_run = [
        "schema_validity",
        "notebook_overview",
        "cell_length",
        "pedagogical_ordering",
        "notation_consistency",
        "tensor_shape_consistency",
    ]

    validation_issues.extend(validate_notebook_schema(notebook_object))
    validation_issues.extend(validate_notebook_overview(notebook_object))
    validation_issues.extend(validate_cell_lengths(notebook_object))
    validation_issues.extend(validate_pedagogical_ordering(notebook_object, notebook_plan))
    validation_issues.extend(validate_notation_consistency(notebook_object, notebook_plan, generated_section_batches))
    validation_issues.extend(validate_tensor_shape_consistency(notebook_object, notebook_plan))

    if enable_smoke_test:
        check_names_run.append("execution_smoke_test")
        validation_issues.extend(
            execute_notebook_smoke_test(
                notebook_object=nbformat.from_dict(notebook_object),
                working_directory=working_directory,
                execution_timeout_seconds=execution_timeout_seconds,
            )
        )
    else:
        check_names_run.append("execution_smoke_test_skipped")

    validated_at_utc = utc_timestamp_string()
    validation_report = ValidationReport(
        paper_id=paper_id,
        run_id=run_id,
        is_valid=not any(validation_issue.severity == "error" for validation_issue in validation_issues),
        errors=[
            _format_validation_issue(validation_issue)
            for validation_issue in validation_issues
            if validation_issue.severity == "error"
        ],
        warnings=[
            _format_validation_issue(validation_issue)
            for validation_issue in validation_issues
            if validation_issue.severity == "warning"
        ],
        checks_run=check_names_run,
        notebook_path=notebook_path,
        validated_at_utc=validated_at_utc,
        stage_provenance=StageProvenance(
            stage_name="validation_report",
            created_at_utc=validated_at_utc,
            input_artifact_paths=input_artifact_paths,
            output_artifact_path=None,
            code_version=code_version,
            model_provenance=None,
        ),
    )

    _ = learner_profile
    return validation_report, validation_issues


def _format_validation_issue(validation_issue: ValidationIssue) -> str:
    issue_prefix = f"[{validation_issue.check_name}]"
    if validation_issue.section_id is not None:
        return f"{issue_prefix} section={validation_issue.section_id}: {validation_issue.message}"
    return f"{issue_prefix} {validation_issue.message}"


def _build_section_text_by_id(notebook_object: NotebookNode) -> dict[str, str]:
    section_text_chunks: dict[str, list[str]] = {}
    for notebook_cell in notebook_object.cells:
        section_id = notebook_cell.get("metadata", {}).get("section_id")
        if not isinstance(section_id, str):
            continue
        section_text_chunks.setdefault(section_id, []).append(str(notebook_cell.source))
    return {section_id: "\n\n".join(text_chunks) for section_id, text_chunks in section_text_chunks.items()}


def _extract_required_notation_symbols(lesson_section) -> list[str]:
    notation_candidates = " ".join(lesson_section.equations_to_unpack + lesson_section.tensor_shapes_to_state)
    raw_notation_symbols = re.findall(r"[A-Z][A-Za-z0-9_]*", notation_candidates)
    normalized_symbols: set[str] = set()
    for raw_notation_symbol in raw_notation_symbols:
        if raw_notation_symbol.isupper() and len(raw_notation_symbol) > 1:
            normalized_symbols.update(raw_notation_symbol)
            continue
        normalized_symbols.add(raw_notation_symbol)
    return sorted(normalized_symbols)


def _extract_tensor_variables(lesson_section) -> list[str]:
    tensor_variables: list[str] = []
    for tensor_shape_statement in lesson_section.tensor_shapes_to_state:
        match = re.match(r"\s*([A-Za-z][A-Za-z0-9_]*)\s*:", tensor_shape_statement)
        if match:
            tensor_variables.append(match.group(1))
    return sorted(set(tensor_variables))
