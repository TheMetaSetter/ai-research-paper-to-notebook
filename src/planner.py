from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from src.artifact_store import utc_timestamp_string
from src.models.gemma_4_e2b import Gemma4E2BModel
from src.schemas import ConceptItem, LearnerProfile, LessonSection, NotebookPlan, StageProvenance


PLANNER_SYSTEM_INSTRUCTION = """
You are designing a pedagogy-first computational notebook.
Return JSON only.
Order sections for learning, not for paper chronology.
Keep each section small enough for notebook readability.
Use only source chunk IDs that appear in the provided concept graph.
Do not assume hidden future notebook cells.
Do not write prose-heavy notebook cells yet; produce a lesson plan only.
Avoid unexplained notation jumps and identify prerequisites explicitly.
""".strip()


class NotebookPlanDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_sections: list[LessonSection]
    planning_notes: list[str] = Field(default_factory=list)


def _compact_concept_graph_json(concept_items: list[ConceptItem]) -> str:
    concept_payload = [
        {
            "concept_id": concept_item.concept_id,
            "label": concept_item.label,
            "concept_type": concept_item.concept_type,
            "source_chunk_ids": concept_item.source_chunk_ids,
            "prerequisites": concept_item.prerequisites,
            "equation_text": concept_item.equation_text,
            "notation_symbols": concept_item.notation_symbols,
            "tensor_shape_notes": concept_item.tensor_shape_notes,
        }
        for concept_item in concept_items
    ]
    return json.dumps(concept_payload, ensure_ascii=False, separators=(",", ":"))


def _normalize_lesson_section_ids(lesson_sections: list[LessonSection]) -> list[LessonSection]:
    return [
        lesson_section.model_copy(update={"section_id": f"lesson_{section_index:03d}"})
        for section_index, lesson_section in enumerate(lesson_sections)
    ]


def build_notebook_plan(
    model: Gemma4E2BModel,
    learner_profile: LearnerProfile,
    concept_items: list[ConceptItem],
    paper_id: str,
    input_artifact_paths: list[str],
    output_artifact_path: str,
    code_version: str,
    maximum_generation_sections: int | None = None,
) -> NotebookPlan:
    user_prompt = f"""
Create a pedagogical notebook lesson plan from this learner profile and concept graph.
Use the exact source_chunk_ids from the concept graph.
Do not generate notebook cells.

Learner profile:
{learner_profile.model_dump_json(indent=2)}

Concept graph:
{_compact_concept_graph_json(concept_items)}
""".strip()

    plan_draft = model.generate_json(
        system_instruction=PLANNER_SYSTEM_INSTRUCTION,
        user_prompt=user_prompt,
        response_model=NotebookPlanDraft,
    )
    created_at_utc = utc_timestamp_string()
    notebook_plan = NotebookPlan(
        paper_id=paper_id,
        learner_profile=learner_profile,
        lesson_sections=_normalize_lesson_section_ids(plan_draft.lesson_sections),
        planning_notes=plan_draft.planning_notes,
        planning_model_provenance=model.model_provenance(),
        created_at_utc=created_at_utc,
        stage_provenance=StageProvenance(
            stage_name="notebook_plan",
            created_at_utc=created_at_utc,
            input_artifact_paths=input_artifact_paths,
            output_artifact_path=output_artifact_path,
            code_version=code_version,
            model_provenance=model.model_provenance(),
        ),
    )

    validation_errors = validate_notebook_plan(
        notebook_plan=notebook_plan,
        concept_items=concept_items,
        maximum_generation_sections=maximum_generation_sections,
    )
    if validation_errors:
        raise RuntimeError(f"Notebook plan validation failed: {'; '.join(validation_errors)}")

    return notebook_plan


def validate_notebook_plan(
    notebook_plan: NotebookPlan,
    concept_items: list[ConceptItem],
    maximum_generation_sections: int | None = None,
) -> list[str]:
    validation_errors: list[str] = []
    if not notebook_plan.lesson_sections:
        validation_errors.append("Notebook plan must contain at least one lesson section.")

    if maximum_generation_sections is not None and len(notebook_plan.lesson_sections) > maximum_generation_sections:
        validation_errors.append(
            f"Notebook plan has {len(notebook_plan.lesson_sections)} sections, "
            f"which exceeds maximum_generation_sections={maximum_generation_sections}."
        )

    section_titles = [lesson_section.title.strip().casefold() for lesson_section in notebook_plan.lesson_sections]
    duplicate_titles = sorted({section_title for section_title in section_titles if section_titles.count(section_title) > 1})
    for duplicate_title in duplicate_titles:
        validation_errors.append(f"Duplicate lesson section title: {duplicate_title}.")

    known_source_chunk_ids = {
        source_chunk_id
        for concept_item in concept_items
        for source_chunk_id in concept_item.source_chunk_ids
    }
    for lesson_section in notebook_plan.lesson_sections:
        if not lesson_section.source_chunk_ids:
            validation_errors.append(f"Lesson section {lesson_section.section_id} has no source chunk IDs.")
            continue

        unknown_source_chunk_ids = sorted(set(lesson_section.source_chunk_ids) - known_source_chunk_ids)
        if unknown_source_chunk_ids:
            validation_errors.append(
                f"Lesson section {lesson_section.section_id} references unknown source chunk IDs: "
                f"{', '.join(unknown_source_chunk_ids)}."
            )

    return validation_errors


def save_notebook_plan(notebook_plan: NotebookPlan, output_path: str | Path) -> Path:
    notebook_plan_path = Path(output_path)
    notebook_plan_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_plan_path.write_text(notebook_plan.model_dump_json(indent=2), encoding="utf-8")
    return notebook_plan_path
