from __future__ import annotations

from dataclasses import dataclass

import nbformat
import pytest
from pydantic import BaseModel, ConfigDict

import src.concept_graph as concept_graph_module
from src.artifact_store import RunArtifactStore
from src.config import EnvironmentSettings, RunParameters
from src.cell_generator import generate_notebook_batch_for_section
from src.concept_graph import build_concept_graph
from src.models.gemma_4_e2b import Gemma4E2BModel
from src.notebook_builder import build_notebook_metadata, build_notebook_object
from src.planner import build_notebook_plan
from src.repair import rebuild_notebook_from_saved_batches
from src.schemas import ConceptItem, LearnerProfile, LessonSection, ModelProvenance, NotebookBatch, NotebookPlan, PaperChunk, RunManifest
from src.validators import validate_assembled_notebook


class StructuredTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_title: str
    chunk_ids: list[str]


@dataclass
class FakeHTTPResponse:
    payload: object
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"unexpected fake status code: {self.status_code}")

    def json(self) -> object:
        return self.payload


class FakeHTTPClient:
    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.request_url: str | None = None
        self.request_json: dict | None = None

    def __enter__(self) -> "FakeHTTPClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, json: dict) -> FakeHTTPResponse:
        self.request_url = url
        self.request_json = json
        return FakeHTTPResponse(payload=self.payload)


def test_local_llama_cpp_returns_valid_typed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHTTPClient(
        payload={
            "choices": [
                {
                    "message": {
                        "content": '{"lesson_title": "Attention Basics", "chunk_ids": ["chunk_1", "chunk_2"]}'
                    }
                }
            ]
        }
    )
    monkeypatch.setattr("src.models.gemma_4_e2b.httpx.Client", lambda timeout: fake_client)

    model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        local_model_name="gemma-4-e2b-q4_0",
    )

    response = model.generate_json(
        system_instruction="Return JSON only.",
        user_prompt="Generate a tiny lesson plan.",
        response_model=StructuredTestResponse,
    )

    assert response.lesson_title == "Attention Basics"
    assert response.chunk_ids == ["chunk_1", "chunk_2"]
    assert fake_client.request_url == "http://localhost:8080/v1/chat/completions"
    assert fake_client.request_json == {
        "model": "gemma-4-e2b-q4_0",
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Generate a tiny lesson plan."},
        ],
        "temperature": 0.2,
        "max_tokens": 1800,
        "response_format": {"type": "json_object"},
    }


def test_local_llama_cpp_accepts_fenced_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHTTPClient(
        payload={
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"lesson_title": "Attention Basics", "chunk_ids": ["chunk_1"]}\n```'
                    }
                }
            ]
        }
    )
    monkeypatch.setattr("src.models.gemma_4_e2b.httpx.Client", lambda timeout: fake_client)

    model = Gemma4E2BModel(local_llama_server_base_url="http://localhost:8080/v1")

    response = model.generate_json(
        system_instruction="Return JSON only.",
        user_prompt="Generate a tiny lesson plan.",
        response_model=StructuredTestResponse,
    )

    assert response.lesson_title == "Attention Basics"
    assert response.chunk_ids == ["chunk_1"]


def test_local_llama_cpp_raises_for_missing_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHTTPClient(payload={"choices": [{"message": {}}]})
    monkeypatch.setattr("src.models.gemma_4_e2b.httpx.Client", lambda timeout: fake_client)

    model = Gemma4E2BModel(local_llama_server_base_url="http://localhost:8080/v1")

    with pytest.raises(RuntimeError, match="missing message content"):
        model.generate_json(
            system_instruction="Return JSON only.",
            user_prompt="Generate a tiny lesson plan.",
            response_model=StructuredTestResponse,
        )


def test_local_llama_cpp_raises_for_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHTTPClient(payload={"choices": [{"message": {"content": "{not-json}"}}]})
    monkeypatch.setattr("src.models.gemma_4_e2b.httpx.Client", lambda timeout: fake_client)

    model = Gemma4E2BModel(local_llama_server_base_url="http://localhost:8080/v1")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        model.generate_json(
            system_instruction="Return JSON only.",
            user_prompt="Generate a tiny lesson plan.",
            response_model=StructuredTestResponse,
        )


def test_local_llama_cpp_raises_for_schema_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = FakeHTTPClient(payload={"choices": [{"message": {"content": '{"lesson_title": "Only title"}'}}]})
    monkeypatch.setattr("src.models.gemma_4_e2b.httpx.Client", lambda timeout: fake_client)

    model = Gemma4E2BModel(local_llama_server_base_url="http://localhost:8080/v1")

    with pytest.raises(RuntimeError, match="schema-invalid JSON"):
        model.generate_json(
            system_instruction="Return JSON only.",
            user_prompt="Generate a tiny lesson plan.",
            response_model=StructuredTestResponse,
        )


def test_google_adapter_requires_api_key() -> None:
    model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        use_google_cloud_adapter=True,
        google_api_key=None,
    )

    with pytest.raises(RuntimeError, match="requires google_api_key"):
        model.generate_json(
            system_instruction="Return JSON only.",
            user_prompt="Generate a tiny lesson plan.",
            response_model=StructuredTestResponse,
        )


def test_google_adapter_fails_when_imports_are_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.models.gemma_4_e2b.genai", None)
    monkeypatch.setattr("src.models.gemma_4_e2b.google_types", None)

    model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        use_google_cloud_adapter=True,
        google_api_key="test-key",
    )

    with pytest.raises(RuntimeError, match="adapter is unavailable"):
        model.generate_json(
            system_instruction="Return JSON only.",
            user_prompt="Generate a tiny lesson plan.",
            response_model=StructuredTestResponse,
        )


def test_google_adapter_returns_valid_typed_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeResponse:
        text = '{"lesson_title": "Attention Basics", "chunk_ids": ["chunk_1"]}'

    class FakeModelsAPI:
        def __init__(self) -> None:
            self.last_call: dict | None = None

        def generate_content(self, **kwargs) -> FakeResponse:
            self.last_call = kwargs
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = FakeModelsAPI()

    class FakeGenAI:
        Client = FakeClient

    class FakeGoogleTypes:
        GenerateContentConfig = FakeGenerateContentConfig

    monkeypatch.setattr("src.models.gemma_4_e2b.genai", FakeGenAI)
    monkeypatch.setattr("src.models.gemma_4_e2b.google_types", FakeGoogleTypes)

    model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        use_google_cloud_adapter=True,
        google_api_key="test-key",
        local_temperature=0.1,
        local_max_output_tokens=512,
    )

    response = model.generate_json(
        system_instruction="Return JSON only.",
        user_prompt="Generate a tiny lesson plan.",
        response_model=StructuredTestResponse,
    )

    assert response.lesson_title == "Attention Basics"
    assert response.chunk_ids == ["chunk_1"]


def test_google_adapter_accepts_fenced_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeGenerateContentConfig:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class FakeResponse:
        text = '```json\n{"lesson_title": "Attention Basics", "chunk_ids": ["chunk_1"]}\n```'

    class FakeModelsAPI:
        def generate_content(self, **kwargs) -> FakeResponse:
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = FakeModelsAPI()

    class FakeGenAI:
        Client = FakeClient

    class FakeGoogleTypes:
        GenerateContentConfig = FakeGenerateContentConfig

    monkeypatch.setattr("src.models.gemma_4_e2b.genai", FakeGenAI)
    monkeypatch.setattr("src.models.gemma_4_e2b.google_types", FakeGoogleTypes)

    model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        use_google_cloud_adapter=True,
        google_api_key="test-key",
    )

    response = model.generate_json(
        system_instruction="Return JSON only.",
        user_prompt="Generate a tiny lesson plan.",
        response_model=StructuredTestResponse,
    )

    assert response.lesson_title == "Attention Basics"
    assert response.chunk_ids == ["chunk_1"]


def test_notebook_assembly_golden_case_builds_structurally_valid_notebook() -> None:
    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
    )
    notebook_plan = NotebookPlan(
        paper_id="attention-paper",
        learner_profile=learner_profile,
        lesson_sections=[
            {
                "section_id": "lesson_000",
                "title": "Attention basics",
                "teaching_goal": "Explain scaled dot-product attention.",
                "source_chunk_ids": ["chunk_000"],
                "requires_code_example": True,
                "requires_recap": True,
            }
        ],
        created_at_utc="20260408T010203Z",
    )
    notebook_batch = NotebookBatch(
        section_id="lesson_000",
        section_title="Attention basics",
        cells=[
            {
                "cell_id": "cell_lesson_000_000",
                "cell_type": "markdown",
                "source": "## Attention basics",
                "metadata": {"pedagogical_role": "intro", "section_id": "lesson_000"},
                "source_chunk_ids": ["chunk_000"],
            },
            {
                "cell_id": "cell_lesson_000_001",
                "cell_type": "code",
                "source": "print('attention')",
                "metadata": {"pedagogical_role": "code", "section_id": "lesson_000"},
                "source_chunk_ids": ["chunk_000"],
            },
        ],
        generated_at_utc="20260408T010203Z",
    )

    notebook_object = build_notebook_object(
        notebook_title="Attention Is All You Need",
        learner_profile=learner_profile,
        generated_section_batches=[notebook_batch],
        notebook_metadata=build_notebook_metadata(
            notebook_plan=notebook_plan,
            run_id="20260408T010203Z",
            code_version="test-code-version",
            model_provenance=ModelProvenance(
                inference_backend="llama_cpp",
                provider_name="llama.cpp",
                model_name="gemma-4-e2b-q4_0",
                temperature=0.2,
                max_output_tokens=1800,
            ),
            generated_section_batches=[notebook_batch],
            parsed_paper=None,
        ),
    )

    nbformat.validate(notebook_object)
    assert notebook_object.cells[0].source.startswith("# Attention Is All You Need")
    assert notebook_object.metadata["paper_to_notebook"]["chunk_provenance_by_section"]["lesson_000"]["chunk_ids"] == [
        "chunk_000"
    ]


def test_model_provenance_reports_active_backend() -> None:
    local_model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        local_model_name="gemma-4-e2b-q4_0",
        local_temperature=0.2,
        local_max_output_tokens=1800,
    )
    cloud_model = Gemma4E2BModel(
        local_llama_server_base_url="http://localhost:8080/v1",
        use_google_cloud_adapter=True,
        google_api_key="test-key",
        local_model_name="gemma-4-e2b-q4_0",
        local_temperature=0.3,
        local_max_output_tokens=1024,
    )

    local_provenance = local_model.model_provenance()
    cloud_provenance = cloud_model.model_provenance()

    assert local_provenance.inference_backend == "llama_cpp"
    assert local_provenance.model_name == "gemma-4-e2b-q4_0"
    assert cloud_provenance.inference_backend == "google_genai"
    assert cloud_provenance.model_name == "gemini-2.5-flash"


def test_from_settings_uses_phase_zero_configuration_contracts() -> None:
    environment_settings = EnvironmentSettings(
        local_llama_server_base_url="http://localhost:9000/v1",
        google_api_key="test-key",
        use_google_cloud_adapter=True,
    )
    run_parameters = RunParameters(
        default_model_name="gemma-4-e2b-custom",
        generation_temperature=0.15,
        generation_max_output_tokens=900,
    )

    model = Gemma4E2BModel.from_settings(environment_settings, run_parameters)

    assert model.local_llama_server_base_url == "http://localhost:9000/v1"
    assert model.use_google_cloud_adapter is True
    assert model.google_api_key == "test-key"
    assert model.local_model_name == "gemma-4-e2b-custom"
    assert model.local_temperature == 0.15
    assert model.local_max_output_tokens == 900


def test_reference_concept_graph_pipeline_merges_overlapping_candidates() -> None:
    class StubModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate_json(self, system_instruction: str, user_prompt: str, response_model):
            self.calls.append(user_prompt)
            if "Primary Chunk ID: chunk_00000" in user_prompt:
                return response_model(
                    concepts=[
                        {
                            "label": "Scaled Dot-Product Attention",
                            "concept_type": "equation",
                            "source_chunk_ids": ["chunk_00000"],
                            "prerequisites": ["Softmax"],
                            "equation_text": "softmax(QK^T)V",
                            "notation_symbols": ["Q", "K"],
                            "tensor_shape_notes": ["Q: B x T x d_k"],
                            "confidence": 0.8,
                        }
                    ]
                )
            return response_model(
                concepts=[
                    {
                        "label": "Softmax",
                        "concept_type": "definition",
                        "source_chunk_ids": ["chunk_00001"],
                        "prerequisites": [],
                        "equation_text": None,
                        "notation_symbols": [],
                        "tensor_shape_notes": [],
                        "confidence": 0.7,
                    },
                    {
                        "label": "scaled dot-product attention",
                        "concept_type": "equation",
                        "source_chunk_ids": ["chunk_00001"],
                        "prerequisites": ["softmax"],
                        "equation_text": "softmax(QK^T / sqrt(d_k))V",
                        "notation_symbols": ["V"],
                        "tensor_shape_notes": ["V: B x T x d_v"],
                        "confidence": 0.9,
                    },
                ]
            )

    paper_chunks = [
        PaperChunk(
            chunk_id="chunk_00000",
            section_id="section_000",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            chunk_text="Attention uses Q and K.",
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
            chunk_text="Softmax normalizes scores.",
            equation_markers=[],
            notation_tokens=["softmax"],
            figure_references=[],
        ),
    ]

    original_support_builder = concept_graph_module.build_supporting_context_chunks
    concept_graph_module.build_supporting_context_chunks = lambda paper_chunks, primary_chunk, retrieval_top_k: []
    try:
        concept_items, concept_edges = build_concept_graph(
            model=StubModel(),
            paper_chunks=paper_chunks,
            retrieval_top_k=2,
        )
    finally:
        concept_graph_module.build_supporting_context_chunks = original_support_builder

    assert [concept_item.label for concept_item in concept_items] == ["Softmax", "Scaled Dot-Product Attention"]
    assert concept_items[1].source_chunk_ids == ["chunk_00000", "chunk_00001"]
    assert concept_items[1].prerequisites == ["concept_000"]
    assert concept_edges == [{"source_concept_id": "concept_000", "target_concept_id": "concept_001"}]


def test_reference_notebook_planning_pipeline_builds_one_section_plan() -> None:
    class StubPlannerModel:
        def generate_json(self, system_instruction: str, user_prompt: str, response_model):
            return response_model(
                lesson_sections=[
                    {
                        "section_id": "draft_id",
                        "title": "Attention from first principles",
                        "teaching_goal": "Explain why attention compares queries and keys.",
                        "prerequisite_concepts": ["concept_000"],
                        "source_chunk_ids": ["chunk_00000"],
                        "equations_to_unpack": ["softmax(QK^T / sqrt(d_k))V"],
                        "tensor_shapes_to_state": ["Q: B x T x d_k"],
                        "likely_misconceptions": ["Attention weights are not recurrence states."],
                        "requires_code_example": True,
                        "requires_recap": True,
                    }
                ],
                planning_notes=["Introduce notation before the equation."],
            )

        def model_provenance(self) -> ModelProvenance:
            return ModelProvenance(
                inference_backend="llama_cpp",
                provider_name="llama.cpp",
                model_name="gemma-4-e2b-q4_0",
                temperature=0.2,
                max_output_tokens=1800,
            )

    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
    )
    concept_items = [
        ConceptItem(
            concept_id="concept_000",
            label="Scaled dot-product attention",
            concept_type="equation",
            source_chunk_ids=["chunk_00000"],
            equation_text="softmax(QK^T / sqrt(d_k))V",
            notation_symbols=["Q", "K", "V"],
            tensor_shape_notes=["Q: B x T x d_k"],
        )
    ]

    notebook_plan = build_notebook_plan(
        model=StubPlannerModel(),
        learner_profile=learner_profile,
        concept_items=concept_items,
        paper_id="attention-paper",
        input_artifact_paths=["concepts.json", "learner_profile.json"],
        output_artifact_path="notebook_plan.json",
        code_version="test-code-version",
    )

    assert notebook_plan.paper_id == "attention-paper"
    assert [lesson_section.section_id for lesson_section in notebook_plan.lesson_sections] == ["lesson_000"]
    assert notebook_plan.lesson_sections[0].source_chunk_ids == ["chunk_00000"]
    assert notebook_plan.planning_notes == ["Introduce notation before the equation."]


def test_reference_cell_generation_pipeline_builds_one_section_batch() -> None:
    class StubGenerationModel:
        def generate_json(self, system_instruction: str, user_prompt: str, response_model):
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
                        "cell_type": "code",
                        "source": "print('attention')",
                        "metadata": {"pedagogical_role": "code"},
                        "execution_intent": "Run a tiny attention example.",
                        "source_chunk_ids": ["chunk_00000"],
                    },
                    {
                        "cell_type": "markdown",
                        "source": "### Recap\nWhat does attention compare?",
                        "metadata": {"pedagogical_role": "recap"},
                        "source_chunk_ids": ["chunk_00000"],
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

    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
    )
    lesson_section = LessonSection(
        section_id="lesson_000",
        title="Attention from first principles",
        teaching_goal="Explain scaled dot-product attention.",
        source_chunk_ids=["chunk_00000"],
        requires_code_example=True,
        requires_recap=True,
    )
    retrieved_chunks = [
        PaperChunk(
            chunk_id="chunk_00000",
            section_id="section_000",
            section_title="Introduction",
            chunk_text="Attention compares queries and keys.",
        )
    ]

    notebook_batch = generate_notebook_batch_for_section(
        model=StubGenerationModel(),
        learner_profile=learner_profile,
        lesson_section=lesson_section,
        retrieved_chunks=retrieved_chunks,
        input_artifact_paths=["notebook_plan.json", "learner_profile.json", "chunks.json"],
        output_artifact_path="cell_batches/lesson_000.json",
        code_version="test-code-version",
    )

    assert notebook_batch.section_id == "lesson_000"
    assert [notebook_cell.cell_id for notebook_cell in notebook_batch.cells] == [
        "cell_lesson_000_000",
        "cell_lesson_000_001",
        "cell_lesson_000_002",
    ]


def test_reference_validation_pipeline_produces_deterministic_report() -> None:
    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
    )
    notebook_plan = NotebookPlan(
        paper_id="attention-paper",
        learner_profile=learner_profile,
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
            }
        ],
        created_at_utc="20260408T010203Z",
    )
    notebook_batch = NotebookBatch(
        section_id="lesson_000",
        section_title="Attention basics",
        cells=[
            {
                "cell_id": "cell_lesson_000_000",
                "cell_type": "markdown",
                "source": "## Attention basics\nQ interacts with K and has shape Q: B x T x d_k.",
                "metadata": {"section_id": "lesson_000", "pedagogical_role": "intro"},
                "source_chunk_ids": ["chunk_000"],
            }
        ],
        generated_at_utc="20260408T010203Z",
    )
    notebook_object = build_notebook_object(
        notebook_title="Attention Is All You Need",
        learner_profile=learner_profile,
        generated_section_batches=[notebook_batch],
        notebook_metadata=build_notebook_metadata(
            notebook_plan=notebook_plan,
            run_id="20260408T010203Z",
            code_version="test-code-version",
            model_provenance=None,
            generated_section_batches=[notebook_batch],
            parsed_paper=None,
        ),
    )

    validation_report, validation_issues = validate_assembled_notebook(
        notebook_object=notebook_object,
        notebook_plan=notebook_plan,
        learner_profile=learner_profile,
        generated_section_batches=[notebook_batch],
        paper_id="attention-paper",
        run_id="20260408T010203Z",
        notebook_path="runs/paper/notebook/final_notebook.ipynb",
        input_artifact_paths=[],
        code_version="test-code-version",
        enable_smoke_test=False,
        execution_timeout_seconds=120,
        working_directory=".",
    )

    assert validation_report.is_valid is True
    assert validation_report.checks_run[-1] == "execution_smoke_test_skipped"
    assert validation_issues == []


def test_reference_repair_pipeline_rebuilds_notebook_from_saved_batches(tmp_path) -> None:
    artifact_store = RunArtifactStore.create(tmp_path / "runs", "paper", "20260408T010203Z")
    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
    )
    notebook_plan = NotebookPlan(
        paper_id="attention-paper",
        learner_profile=learner_profile,
        lesson_sections=[
            {
                "section_id": "lesson_000",
                "title": "Attention basics",
                "teaching_goal": "Explain attention.",
                "source_chunk_ids": ["chunk_000"],
                "requires_code_example": True,
                "requires_recap": True,
            }
        ],
        created_at_utc="20260408T010203Z",
    )
    notebook_batch = NotebookBatch(
        section_id="lesson_000",
        section_title="Attention basics",
        cells=[
            {
                "cell_id": "cell_lesson_000_000",
                "cell_type": "markdown",
                "source": "## Attention basics",
                "metadata": {"section_id": "lesson_000", "pedagogical_role": "intro"},
                "source_chunk_ids": ["chunk_000"],
            }
        ],
        generated_at_utc="20260408T010203Z",
    )
    run_manifest = RunManifest(
        run_id="20260408T010203Z",
        paper_slug="paper",
        params_path=str(tmp_path / "params.yaml"),
        run_parameters={},
        stage_artifact_paths={},
        created_at_utc="20260408T010203Z",
    )

    notebook_path = rebuild_notebook_from_saved_batches(
        artifact_store=artifact_store,
        run_manifest=run_manifest,
        notebook_plan=notebook_plan,
        learner_profile=learner_profile,
        generated_section_batches=[notebook_batch],
        parsed_paper=None,
        code_version="test-code-version",
    )

    rebuilt_notebook = nbformat.read(notebook_path, as_version=4)
    assert rebuilt_notebook.cells[1].source == "## Attention basics"
