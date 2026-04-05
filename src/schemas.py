from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictSchemaModel(BaseModel):
    """Base model that forbids undeclared fields across stage contracts."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ModelProvenance(StrictSchemaModel):
    inference_backend: str
    provider_name: str
    model_name: str
    temperature: float
    max_output_tokens: int


class StageProvenance(StrictSchemaModel):
    stage_name: str
    created_at_utc: str
    input_artifact_paths: list[str] = Field(default_factory=list)
    output_artifact_path: str | None = None
    code_version: str
    model_provenance: ModelProvenance | None = None


class PaperSection(StrictSchemaModel):
    section_id: str
    title: str
    page_start: int | None = None
    page_end: int | None = None
    markdown_text: str


class ParsedPaper(StrictSchemaModel):
    paper_id: str
    paper_title: str
    source_pdf_path: str
    source_pdf_sha256: str
    parser_name: str
    parser_version: str
    parsed_at_utc: str
    sections: list[PaperSection]
    abstract_text: str
    authors: list[str] | None = None
    stage_provenance: StageProvenance | None = None


class PaperChunk(StrictSchemaModel):
    chunk_id: str
    section_id: str
    section_title: str
    page_start: int | None = None
    page_end: int | None = None
    chunk_text: str
    equation_markers: list[str] = Field(default_factory=list)
    notation_tokens: list[str] = Field(default_factory=list)
    figure_references: list[str] = Field(default_factory=list)


class ConceptItem(StrictSchemaModel):
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
    prerequisites: list[str] = Field(default_factory=list)
    equation_text: str | None = None
    notation_symbols: list[str] = Field(default_factory=list)
    tensor_shape_notes: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class LearnerProfile(StrictSchemaModel):
    mathematics_background: str
    machine_learning_background: str
    deep_learning_background: str
    python_background: str
    tensor_familiarity: str
    wants_tensor_shapes: bool = True
    wants_derivations: bool = True
    preferred_depth: Literal["light", "medium", "deep"] = "deep"
    preferred_pacing: Literal["fast", "moderate", "slow"] = "moderate"


class LessonSection(StrictSchemaModel):
    section_id: str
    title: str
    teaching_goal: str
    prerequisite_concepts: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    equations_to_unpack: list[str] = Field(default_factory=list)
    tensor_shapes_to_state: list[str] = Field(default_factory=list)
    likely_misconceptions: list[str] = Field(default_factory=list)
    requires_code_example: bool = True
    requires_recap: bool = True


class NotebookPlan(StrictSchemaModel):
    paper_id: str
    learner_profile: LearnerProfile
    lesson_sections: list[LessonSection]
    planning_notes: list[str] = Field(default_factory=list)
    planning_model_provenance: ModelProvenance | None = None
    created_at_utc: str
    stage_provenance: StageProvenance | None = None


class NotebookCell(StrictSchemaModel):
    cell_id: str
    cell_type: Literal["markdown", "code"]
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    execution_intent: str | None = None
    source_chunk_ids: list[str] | None = None


class NotebookBatch(StrictSchemaModel):
    section_id: str
    section_title: str
    cells: list[NotebookCell]
    batch_model_provenance: ModelProvenance | None = None
    generated_at_utc: str
    stage_provenance: StageProvenance | None = None


class ValidationReport(StrictSchemaModel):
    paper_id: str
    run_id: str
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    notebook_path: str | None = None
    validated_at_utc: str
    stage_provenance: StageProvenance | None = None


class RunManifest(StrictSchemaModel):
    run_id: str
    paper_slug: str
    source_pdf_path: str | None = None
    source_pdf_sha256: str | None = None
    params_path: str
    run_parameters: dict[str, Any]
    planned_notebook_output_path: str | None = None
    active_model_provenance: ModelProvenance | None = None
    stage_artifact_paths: dict[str, str] = Field(default_factory=dict)
    created_at_utc: str
