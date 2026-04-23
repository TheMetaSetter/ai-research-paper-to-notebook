from __future__ import annotations

from pathlib import Path

import pytest

from src.planner import NotebookPlanDraft, build_notebook_plan, save_notebook_plan, validate_notebook_plan
from src.schemas import ConceptItem, LearnerProfile, ModelProvenance, NotebookPlan


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


def build_concept_items() -> list[ConceptItem]:
    return [
        ConceptItem(
            concept_id="concept_000",
            label="Scaled dot-product attention",
            concept_type="equation",
            source_chunk_ids=["chunk_00000"],
            prerequisites=[],
            equation_text="softmax(QK^T)V",
            notation_symbols=["Q", "K", "V"],
            tensor_shape_notes=["Q: B x T x d_k"],
            confidence=0.9,
        )
    ]


def build_model_provenance() -> ModelProvenance:
    return ModelProvenance(
        inference_backend="llama_cpp",
        provider_name="llama.cpp",
        model_name="gemma-4-e2b-q4_0",
        temperature=0.2,
        max_output_tokens=1800,
    )


class StubPlannerModel:
    def __init__(self) -> None:
        self.user_prompt: str | None = None

    def generate_json(self, system_instruction: str, user_prompt: str, response_model):
        self.user_prompt = user_prompt
        return response_model(
            lesson_sections=[
                {
                    "section_id": "model_supplied_id",
                    "title": "Attention from first principles",
                    "teaching_goal": "Explain why attention compares queries and keys.",
                    "prerequisite_concepts": ["concept_000"],
                    "source_chunk_ids": ["chunk_00000"],
                    "equations_to_unpack": ["softmax(QK^T)V"],
                    "tensor_shapes_to_state": ["Q: B x T x d_k"],
                    "likely_misconceptions": ["Attention is not recurrence."],
                    "requires_code_example": True,
                    "requires_recap": True,
                }
            ],
            planning_notes=["Use notation reminders before equations."],
        )

    def model_provenance(self) -> ModelProvenance:
        return build_model_provenance()


def test_build_notebook_plan_normalizes_ids_and_preserves_provenance() -> None:
    model = StubPlannerModel()
    concept_items = build_concept_items()
    learner_profile = build_learner_profile()

    notebook_plan = build_notebook_plan(
        model=model,
        learner_profile=learner_profile,
        concept_items=concept_items,
        paper_id="attention-paper",
        input_artifact_paths=["runs/paper/run/concept_graph/concepts.json", "runs/paper/run/learner_profile/learner_profile.json"],
        output_artifact_path="runs/paper/run/notebook_plan/notebook_plan.json",
        code_version="test-code-version",
        maximum_generation_sections=2,
    )

    assert notebook_plan.paper_id == "attention-paper"
    assert notebook_plan.learner_profile == learner_profile
    assert notebook_plan.lesson_sections[0].section_id == "lesson_000"
    assert notebook_plan.planning_model_provenance == build_model_provenance()
    assert notebook_plan.stage_provenance is not None
    assert notebook_plan.stage_provenance.stage_name == "notebook_plan"
    assert notebook_plan.stage_provenance.input_artifact_paths == [
        "runs/paper/run/concept_graph/concepts.json",
        "runs/paper/run/learner_profile/learner_profile.json",
    ]

    serialized_payload = notebook_plan.model_dump(mode="json")
    assert NotebookPlan.model_validate(serialized_payload) == notebook_plan


def test_build_notebook_plan_prompt_includes_learner_profile_and_compact_concepts() -> None:
    model = StubPlannerModel()

    build_notebook_plan(
        model=model,
        learner_profile=build_learner_profile(),
        concept_items=build_concept_items(),
        paper_id="paper",
        input_artifact_paths=[],
        output_artifact_path="notebook_plan.json",
        code_version="test-code-version",
    )

    assert model.user_prompt is not None
    assert "Linear algebra" in model.user_prompt
    assert "concept_000" in model.user_prompt
    assert "chunk_00000" in model.user_prompt
    assert "Do not generate notebook cells." in model.user_prompt


def test_validate_notebook_plan_reports_planning_errors() -> None:
    notebook_plan = NotebookPlan(
        paper_id="paper",
        learner_profile=build_learner_profile(),
        lesson_sections=[
            {
                "section_id": "lesson_000",
                "title": "Attention",
                "teaching_goal": "Explain attention.",
                "source_chunk_ids": ["chunk_00000"],
            },
            {
                "section_id": "lesson_001",
                "title": " attention ",
                "teaching_goal": "Repeat attention.",
                "source_chunk_ids": [],
            },
            {
                "section_id": "lesson_002",
                "title": "Unknown source",
                "teaching_goal": "Use an unknown chunk.",
                "source_chunk_ids": ["chunk_missing"],
            },
        ],
        created_at_utc="20260405T120000Z",
    )

    validation_errors = validate_notebook_plan(
        notebook_plan=notebook_plan,
        concept_items=build_concept_items(),
        maximum_generation_sections=2,
    )

    assert "Duplicate lesson section title: attention." in validation_errors
    assert "Lesson section lesson_001 has no source chunk IDs." in validation_errors
    assert "Lesson section lesson_002 references unknown source chunk IDs: chunk_missing." in validation_errors
    assert any("exceeds maximum_generation_sections=2" in validation_error for validation_error in validation_errors)


def test_validate_notebook_plan_rejects_empty_plan() -> None:
    notebook_plan = NotebookPlan(
        paper_id="paper",
        learner_profile=build_learner_profile(),
        lesson_sections=[],
        created_at_utc="20260405T120000Z",
    )

    assert validate_notebook_plan(notebook_plan=notebook_plan, concept_items=[]) == [
        "Notebook plan must contain at least one lesson section."
    ]


def test_build_notebook_plan_raises_for_invalid_model_plan() -> None:
    class InvalidPlannerModel(StubPlannerModel):
        def generate_json(self, system_instruction: str, user_prompt: str, response_model):
            return NotebookPlanDraft(lesson_sections=[], planning_notes=[])

    with pytest.raises(RuntimeError, match="Notebook plan validation failed"):
        build_notebook_plan(
            model=InvalidPlannerModel(),
            learner_profile=build_learner_profile(),
            concept_items=build_concept_items(),
            paper_id="paper",
            input_artifact_paths=[],
            output_artifact_path="notebook_plan.json",
            code_version="test-code-version",
        )


def test_save_notebook_plan_round_trips_json(tmp_path: Path) -> None:
    notebook_plan = build_notebook_plan(
        model=StubPlannerModel(),
        learner_profile=build_learner_profile(),
        concept_items=build_concept_items(),
        paper_id="paper",
        input_artifact_paths=[],
        output_artifact_path=str(tmp_path / "notebook_plan" / "notebook_plan.json"),
        code_version="test-code-version",
    )

    notebook_plan_path = save_notebook_plan(
        notebook_plan=notebook_plan,
        output_path=tmp_path / "notebook_plan" / "notebook_plan.json",
    )

    reloaded_notebook_plan = NotebookPlan.model_validate_json(notebook_plan_path.read_text(encoding="utf-8"))
    assert reloaded_notebook_plan == notebook_plan
