from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.artifact_store import RunArtifactStore
from src.cell_generator import generate_notebook_batch_for_section, save_notebook_batch, select_retrieved_chunks_for_lesson_section
from src.config import EnvironmentSettings, RunParameters
from src.models.gemma_4_e2b import Gemma4E2BModel
from src.notebook_builder import build_notebook_metadata, build_notebook_object, write_notebook
from src.schemas import LearnerProfile, NotebookBatch, NotebookPlan, PaperChunk, ParsedPaper, RunManifest
from src.validators import ValidationIssue


@dataclass
class RepairOutcome:
    repaired: bool
    repaired_notebook_path: str | None = None
    repaired_batch_paths: list[str] | None = None


def attempt_targeted_repair(
    validation_issues: list[ValidationIssue],
    artifact_store: RunArtifactStore,
    run_manifest: RunManifest,
    notebook_plan: NotebookPlan,
    learner_profile: LearnerProfile,
    generated_section_batches: list[NotebookBatch],
    parsed_paper: ParsedPaper | None,
    environment_settings: EnvironmentSettings,
    run_parameters: RunParameters,
    code_version: str,
) -> RepairOutcome:
    if not validation_issues:
        return RepairOutcome(repaired=False)

    repair_kinds = {validation_issue.repair_kind for validation_issue in validation_issues if validation_issue.repair_kind}
    if "rebuild_notebook" in repair_kinds:
        repaired_notebook_path = rebuild_notebook_from_saved_batches(
            artifact_store=artifact_store,
            run_manifest=run_manifest,
            notebook_plan=notebook_plan,
            learner_profile=learner_profile,
            generated_section_batches=generated_section_batches,
            parsed_paper=parsed_paper,
            code_version=code_version,
        )
        return RepairOutcome(repaired=True, repaired_notebook_path=str(repaired_notebook_path), repaired_batch_paths=[])

    section_ids_to_regenerate = sorted(
        {
            validation_issue.section_id
            for validation_issue in validation_issues
            if validation_issue.repair_kind in {"regenerate_section", "repair_execution"} and validation_issue.section_id is not None
        }
    )
    if not section_ids_to_regenerate and "repair_execution" in repair_kinds:
        first_section_with_code = _first_section_with_code(generated_section_batches)
        if first_section_with_code is not None:
            section_ids_to_regenerate = [first_section_with_code]

    if not section_ids_to_regenerate:
        return RepairOutcome(repaired=False)

    repaired_batch_paths = regenerate_sections_and_reassemble(
        section_ids_to_regenerate=section_ids_to_regenerate,
        artifact_store=artifact_store,
        run_manifest=run_manifest,
        notebook_plan=notebook_plan,
        learner_profile=learner_profile,
        generated_section_batches=generated_section_batches,
        parsed_paper=parsed_paper,
        environment_settings=environment_settings,
        run_parameters=run_parameters,
        code_version=code_version,
    )
    if not repaired_batch_paths:
        return RepairOutcome(repaired=False)

    repaired_notebook_path = artifact_store.stage_directory("notebook") / "final_notebook.ipynb"
    return RepairOutcome(
        repaired=True,
        repaired_notebook_path=str(repaired_notebook_path),
        repaired_batch_paths=[str(repaired_batch_path) for repaired_batch_path in repaired_batch_paths],
    )


def rebuild_notebook_from_saved_batches(
    artifact_store: RunArtifactStore,
    run_manifest: RunManifest,
    notebook_plan: NotebookPlan,
    learner_profile: LearnerProfile,
    generated_section_batches: list[NotebookBatch],
    parsed_paper: ParsedPaper | None,
    code_version: str,
) -> Path:
    notebook_title = parsed_paper.paper_title if parsed_paper is not None else notebook_plan.paper_id
    notebook_metadata = build_notebook_metadata(
        notebook_plan=notebook_plan,
        run_id=run_manifest.run_id,
        code_version=code_version,
        model_provenance=run_manifest.active_model_provenance,
        generated_section_batches=generated_section_batches,
        parsed_paper=parsed_paper,
    )
    notebook_object = build_notebook_object(
        notebook_title=notebook_title,
        learner_profile=learner_profile,
        generated_section_batches=generated_section_batches,
        notebook_metadata=notebook_metadata,
    )
    return write_notebook(
        notebook_object=notebook_object,
        output_path=artifact_store.stage_directory("notebook") / "final_notebook.ipynb",
    )


def regenerate_sections_and_reassemble(
    section_ids_to_regenerate: list[str],
    artifact_store: RunArtifactStore,
    run_manifest: RunManifest,
    notebook_plan: NotebookPlan,
    learner_profile: LearnerProfile,
    generated_section_batches: list[NotebookBatch],
    parsed_paper: ParsedPaper | None,
    environment_settings: EnvironmentSettings,
    run_parameters: RunParameters,
    code_version: str,
) -> list[Path]:
    chunks_path_string = run_manifest.stage_artifact_paths.get("chunks")
    if not chunks_path_string:
        return []

    chunks_path = Path(chunks_path_string)
    if not chunks_path.is_file():
        return []

    paper_chunks = [PaperChunk.model_validate(chunk_payload) for chunk_payload in json.loads(chunks_path.read_text(encoding="utf-8"))]
    lesson_sections_by_id = {lesson_section.section_id: lesson_section for lesson_section in notebook_plan.lesson_sections}
    notebook_batches_by_id = {notebook_batch.section_id: notebook_batch for notebook_batch in generated_section_batches}
    generation_model = Gemma4E2BModel.from_settings(
        environment_settings=environment_settings,
        run_parameters=run_parameters,
    )

    repaired_batch_paths: list[Path] = []
    for section_id in section_ids_to_regenerate:
        lesson_section = lesson_sections_by_id.get(section_id)
        if lesson_section is None:
            continue

        retrieved_chunks = select_retrieved_chunks_for_lesson_section(
            lesson_section=lesson_section,
            paper_chunks=paper_chunks,
            retrieval_top_k=run_parameters.retrieval_top_k,
        )
        notebook_batch_path = artifact_store.stage_directory("cell_batches") / f"{lesson_section.section_id}.json"
        repaired_notebook_batch = generate_notebook_batch_for_section(
            model=generation_model,
            learner_profile=learner_profile,
            lesson_section=lesson_section,
            retrieved_chunks=retrieved_chunks,
            input_artifact_paths=[
                str(Path(run_manifest.stage_artifact_paths.get("notebook_plan", ""))),
                str(Path(run_manifest.stage_artifact_paths.get("learner_profile", ""))),
                str(chunks_path),
            ],
            output_artifact_path=str(notebook_batch_path),
            code_version=code_version,
        )
        notebook_batches_by_id[lesson_section.section_id] = repaired_notebook_batch
        repaired_batch_paths.append(save_notebook_batch(repaired_notebook_batch, notebook_batch_path))

    ordered_notebook_batches = [
        notebook_batches_by_id[lesson_section.section_id]
        for lesson_section in notebook_plan.lesson_sections
        if lesson_section.section_id in notebook_batches_by_id
    ]
    rebuild_notebook_from_saved_batches(
        artifact_store=artifact_store,
        run_manifest=run_manifest,
        notebook_plan=notebook_plan,
        learner_profile=learner_profile,
        generated_section_batches=ordered_notebook_batches,
        parsed_paper=parsed_paper,
        code_version=code_version,
    )
    return repaired_batch_paths


def _first_section_with_code(generated_section_batches: list[NotebookBatch]) -> str | None:
    for generated_section_batch in generated_section_batches:
        if any(notebook_cell.cell_type == "code" for notebook_cell in generated_section_batch.cells):
            return generated_section_batch.section_id
    return None
