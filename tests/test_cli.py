from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest
from typer.testing import CliRunner

from src.main import app
from src.schemas import (
    ConceptItem,
    LearnerProfile,
    ModelProvenance,
    NotebookBatch,
    NotebookPlan,
    PaperChunk,
    ParsedPaper,
    PaperSection,
    RunManifest,
    StageProvenance,
    ValidationReport,
)

runner = CliRunner()


def test_help_lists_expected_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "parse" in result.stdout
    assert "chunk" in result.stdout
    assert "concept" in result.stdout
    assert "plan" in result.stdout
    assert "generate" in result.stdout
    assert "assemble" in result.stdout
    assert "validate" in result.stdout


def test_parse_command_writes_phase_two_artifacts(tmp_path: Path, monkeypatch) -> None:
    pdf_path = tmp_path / "paper.pdf"
    params_path = tmp_path / "params.yaml"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    params_path.write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "chunk_size_characters: 2200",
                "chunk_overlap_characters: 250",
                "retrieval_top_k: 6",
                "generation_temperature: 0.2",
                "generation_max_output_tokens: 1800",
                "notebook_execution_timeout_seconds: 120",
                "enable_notebook_execution_smoke_test: true",
                "enable_repair_pass: true",
                "enable_widgets: true",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    def fake_parse_pdf_into_parsed_paper(pdf_path: Path, source_pdf_sha256: str, run_id: str, code_version: str | None = None):
        parsed_paper = ParsedPaper(
            paper_id="paper",
            paper_title="Paper Title",
            source_pdf_path=str(pdf_path),
            source_pdf_sha256=source_pdf_sha256,
            parser_name="pymupdf4llm",
            parser_version="0.0.20",
            parsed_at_utc=run_id,
            sections=[
                PaperSection(
                    section_id="section_000",
                    title="Paper Title",
                    page_start=1,
                    page_end=1,
                    markdown_text="Introduction text.",
                )
            ],
            abstract_text="",
            authors=None,
            stage_provenance=StageProvenance(
                stage_name="parsed_paper",
                created_at_utc=run_id,
                input_artifact_paths=[str(pdf_path)],
                output_artifact_path="parsed_paper.json",
                code_version=code_version or "unknown",
                model_provenance=None,
            ),
        )
        return parsed_paper, "# Paper Title\nIntroduction text.", [{"page_number": 1, "text": "# Paper Title\nIntroduction text."}]

    monkeypatch.setattr("src.main.parse_pdf_into_parsed_paper", fake_parse_pdf_into_parsed_paper)
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")

    result = runner.invoke(app, ["parse", str(pdf_path), "--params-path", str(params_path)])

    assert result.exit_code == 0
    assert (tmp_path / "runs").exists()

    run_directories = list((tmp_path / "runs" / "paper").iterdir())
    assert len(run_directories) == 1

    run_directory = run_directories[0]
    parsed_paper_path = run_directory / "parsed_paper" / "parsed_paper.json"
    raw_markdown_path = run_directory / "parsed_paper" / "raw_markdown.md"
    page_chunks_path = run_directory / "parsed_paper" / "page_chunks.json"
    run_manifest_path = run_directory / "run_manifest.json"

    assert parsed_paper_path.is_file()
    assert raw_markdown_path.is_file()
    assert page_chunks_path.is_file()
    assert run_manifest_path.is_file()

    run_manifest_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert run_manifest_payload["stage_artifact_paths"]["parsed_paper"].endswith("parsed_paper/parsed_paper.json")
    assert run_manifest_payload["stage_artifact_paths"]["parsed_paper_raw_markdown"].endswith("parsed_paper/raw_markdown.md")
    assert run_manifest_payload["stage_artifact_paths"]["parsed_paper_page_chunks"].endswith("parsed_paper/page_chunks.json")
    assert "Parsed paper artifact:" in result.stdout


def test_plan_command_validates_existing_run_directory(tmp_path: Path, monkeypatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plan", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)


class StubPlanningModel:
    def generate_json(self, system_instruction: str, user_prompt: str, response_model):
        return response_model(
            lesson_sections=[
                {
                    "section_id": "model_section",
                    "title": "Attention from first principles",
                    "teaching_goal": "Explain scaled dot-product attention.",
                    "prerequisite_concepts": ["concept_000"],
                    "source_chunk_ids": ["chunk_00000"],
                    "equations_to_unpack": ["softmax(QK^T)V"],
                    "tensor_shapes_to_state": ["Q: B x T x d_k"],
                    "likely_misconceptions": ["Attention is not recurrence."],
                    "requires_code_example": True,
                    "requires_recap": True,
                }
            ],
            planning_notes=["Start with notation."],
        )

    def model_provenance(self) -> ModelProvenance:
        return ModelProvenance(
            inference_backend="llama_cpp",
            provider_name="llama.cpp",
            model_name="gemma-4-e2b-q4_0",
            temperature=0.2,
            max_output_tokens=1800,
        )


def test_plan_command_writes_learner_profile_and_notebook_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "concept_graph").mkdir(parents=True)
    (run_directory / "parsed_paper").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    concept_graph_path = run_directory / "concept_graph" / "concepts.json"
    concept_graph_path.write_text(
        json.dumps(
            [
                ConceptItem(
                    concept_id="concept_000",
                    label="Scaled dot-product attention",
                    concept_type="equation",
                    source_chunk_ids=["chunk_00000"],
                    prerequisites=[],
                    equation_text="softmax(QK^T)V",
                    notation_symbols=["Q", "K"],
                    tensor_shape_notes=["Q: B x T x d_k"],
                    confidence=0.9,
                ).model_dump(mode="json")
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    parsed_paper = ParsedPaper(
        paper_id="parsed-paper-id",
        paper_title="Paper Title",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        parser_name="pymupdf4llm",
        parser_version="0.0.20",
        parsed_at_utc="20260405T120000Z",
        sections=[],
        abstract_text="",
    )
    parsed_paper_path = run_directory / "parsed_paper" / "parsed_paper.json"
    parsed_paper_path.write_text(parsed_paper.model_dump_json(indent=2), encoding="utf-8")
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={
            "concept_graph": str(concept_graph_path),
            "parsed_paper": str(parsed_paper_path),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "src.main.capture_learner_profile_interactively",
        lambda: LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrix multiplication",
            wants_tensor_shapes=True,
            wants_derivations=False,
            preferred_depth="deep",
            preferred_pacing="moderate",
        ),
    )
    monkeypatch.setattr("src.main.Gemma4E2BModel.from_settings", lambda environment_settings, run_parameters: StubPlanningModel())
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plan", str(run_directory)])

    assert result.exit_code == 0
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    assert learner_profile_path.is_file()
    assert notebook_plan_path.is_file()

    learner_profile = LearnerProfile.model_validate_json(learner_profile_path.read_text(encoding="utf-8"))
    assert learner_profile.mathematics_background == "Linear algebra"
    assert learner_profile.wants_derivations is False

    notebook_plan = NotebookPlan.model_validate_json(notebook_plan_path.read_text(encoding="utf-8"))
    assert notebook_plan.paper_id == "parsed-paper-id"
    assert notebook_plan.lesson_sections[0].section_id == "lesson_000"
    assert notebook_plan.planning_model_provenance is not None
    assert notebook_plan.stage_provenance is not None
    assert notebook_plan.stage_provenance.stage_name == "notebook_plan"

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["learner_profile"].endswith("learner_profile/learner_profile.json")
    assert manifest_payload["stage_artifact_paths"]["notebook_plan"].endswith("notebook_plan/notebook_plan.json")
    assert manifest_payload["active_model_provenance"]["model_name"] == "gemma-4-e2b-q4_0"
    assert "Learner profile artifact:" in result.stdout
    assert "Notebook plan artifact:" in result.stdout


def test_plan_command_reuses_existing_learner_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "concept_graph").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    concept_graph_path = run_directory / "concept_graph" / "concepts.json"
    concept_graph_path.write_text(
        json.dumps(
            [
                ConceptItem(
                    concept_id="concept_000",
                    label="Scaled dot-product attention",
                    concept_type="equation",
                    source_chunk_ids=["chunk_00000"],
                ).model_dump(mode="json")
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Existing math",
            machine_learning_background="Existing ML",
            deep_learning_background="Existing DL",
            python_background="Existing Python",
            tensor_familiarity="Existing tensor familiarity",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={
            "concept_graph": str(concept_graph_path),
            "learner_profile": str(learner_profile_path),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr(
        "src.main.capture_learner_profile_interactively",
        lambda: pytest.fail("The plan command should reuse the existing learner profile."),
    )
    monkeypatch.setattr("src.main.Gemma4E2BModel.from_settings", lambda environment_settings, run_parameters: StubPlanningModel())
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plan", str(run_directory)])

    assert result.exit_code == 0
    notebook_plan = NotebookPlan.model_validate_json((run_directory / "notebook_plan" / "notebook_plan.json").read_text(encoding="utf-8"))
    assert notebook_plan.learner_profile.mathematics_background == "Existing math"


def test_plan_command_requires_concept_graph_artifact_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plan", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_plan_command_requires_existing_concept_graph_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={"concept_graph": str(run_directory / "concept_graph" / "concepts.json")},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["plan", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


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


def test_generate_command_writes_cell_batches_and_updates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "chunks").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "retrieval_top_k: 2",
            ]
        ),
        encoding="utf-8",
    )
    chunks_path = run_directory / "chunks" / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
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
                ).model_dump(mode="json")
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrices",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(
        NotebookPlan(
            paper_id="paper",
            learner_profile=LearnerProfile(
                mathematics_background="Linear algebra",
                machine_learning_background="Intermediate ML",
                deep_learning_background="Beginner DL",
                python_background="Intermediate Python",
                tensor_familiarity="Comfortable with matrices",
            ),
            lesson_sections=[
                {
                    "section_id": "lesson_000",
                    "title": "Attention from first principles",
                    "teaching_goal": "Explain scaled dot-product attention.",
                    "source_chunk_ids": ["chunk_00000"],
                    "requires_code_example": True,
                    "requires_recap": True,
                }
            ],
            created_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={
            "chunks": str(chunks_path),
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("src.main.Gemma4E2BModel.from_settings", lambda environment_settings, run_parameters: StubGenerationModel())
    monkeypatch.setattr(
        "src.main.select_retrieved_chunks_for_lesson_section",
        lambda lesson_section, paper_chunks, retrieval_top_k: paper_chunks[:retrieval_top_k],
    )
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["generate", str(run_directory)])

    assert result.exit_code == 0
    cell_batch_path = run_directory / "cell_batches" / "lesson_000.json"
    assert cell_batch_path.is_file()

    notebook_batch = NotebookBatch.model_validate_json(cell_batch_path.read_text(encoding="utf-8"))
    assert notebook_batch.section_id == "lesson_000"
    assert notebook_batch.cells[0].cell_id == "cell_lesson_000_000"

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["cell_batches"].endswith("cell_batches")
    assert manifest_payload["stage_artifact_paths"]["cell_batch_lesson_000"].endswith("cell_batches/lesson_000.json")
    assert manifest_payload["active_model_provenance"]["model_name"] == "gemma-4-e2b-q4_0"
    assert "Cell batch artifact:" in result.stdout


def test_generate_command_requires_chunks_artifact_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["generate", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_generate_command_requires_notebook_plan_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={"chunks": str(run_directory / "chunks" / "chunks.json"), "learner_profile": str(run_directory / "learner_profile" / "learner_profile.json")},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["generate", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_generate_command_respects_maximum_generation_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "chunks").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "maximum_generation_sections: 1",
                "retrieval_top_k: 2",
            ]
        ),
        encoding="utf-8",
    )
    chunks_path = run_directory / "chunks" / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
                PaperChunk(
                    chunk_id="chunk_00000",
                    section_id="section_000",
                    section_title="Introduction",
                    chunk_text="Attention compares queries and keys.",
                ).model_dump(mode="json")
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrices",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(
        NotebookPlan(
            paper_id="paper",
            learner_profile=LearnerProfile(
                mathematics_background="Linear algebra",
                machine_learning_background="Intermediate ML",
                deep_learning_background="Beginner DL",
                python_background="Intermediate Python",
                tensor_familiarity="Comfortable with matrices",
            ),
            lesson_sections=[
                {
                    "section_id": "lesson_000",
                    "title": "First section",
                    "teaching_goal": "Explain attention.",
                    "source_chunk_ids": ["chunk_00000"],
                    "requires_code_example": True,
                    "requires_recap": True,
                },
                {
                    "section_id": "lesson_001",
                    "title": "Second section",
                    "teaching_goal": "Explain softmax.",
                    "source_chunk_ids": ["chunk_00000"],
                    "requires_code_example": True,
                    "requires_recap": True,
                },
            ],
            created_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={
            "chunks": str(chunks_path),
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("src.main.Gemma4E2BModel.from_settings", lambda environment_settings, run_parameters: StubGenerationModel())
    monkeypatch.setattr(
        "src.main.select_retrieved_chunks_for_lesson_section",
        lambda lesson_section, paper_chunks, retrieval_top_k: paper_chunks[:retrieval_top_k],
    )
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["generate", str(run_directory)])

    assert result.exit_code == 0
    assert (run_directory / "cell_batches" / "lesson_000.json").is_file()
    assert not (run_directory / "cell_batches" / "lesson_001.json").exists()


def test_assemble_command_writes_notebook_and_updates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "cell_batches").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (run_directory / "parsed_paper").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")

    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
        preferred_depth="deep",
        preferred_pacing="moderate",
    )
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(learner_profile.model_dump_json(indent=2), encoding="utf-8")

    notebook_plan = NotebookPlan(
        paper_id="paper",
        learner_profile=learner_profile,
        lesson_sections=[
            {
                "section_id": "lesson_001",
                "title": "Second section",
                "teaching_goal": "Explain the second section.",
                "source_chunk_ids": ["chunk_001"],
                "requires_code_example": True,
                "requires_recap": True,
            },
            {
                "section_id": "lesson_000",
                "title": "First section",
                "teaching_goal": "Explain the first section.",
                "source_chunk_ids": ["chunk_000"],
                "requires_code_example": True,
                "requires_recap": True,
            },
        ],
        created_at_utc="20260405T120000Z",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(notebook_plan.model_dump_json(indent=2), encoding="utf-8")

    parsed_paper_path = run_directory / "parsed_paper" / "parsed_paper.json"
    parsed_paper_path.write_text(
        ParsedPaper(
            paper_id="paper",
            paper_title="Paper Title",
            source_pdf_path="/tmp/paper.pdf",
            source_pdf_sha256="abc123",
            parser_name="pymupdf4llm",
            parser_version="0.0.20",
            parsed_at_utc="20260405T120000Z",
            sections=[],
            abstract_text="",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    (run_directory / "cell_batches" / "lesson_000.json").write_text(
        NotebookBatch(
            section_id="lesson_000",
            section_title="First section",
            cells=[
                {
                    "cell_id": "cell_lesson_000_000",
                    "cell_type": "markdown",
                    "source": "## First section",
                    "metadata": {"pedagogical_role": "intro", "section_id": "lesson_000"},
                    "source_chunk_ids": ["chunk_000"],
                }
            ],
            generated_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_directory / "cell_batches" / "lesson_001.json").write_text(
        NotebookBatch(
            section_id="lesson_001",
            section_title="Second section",
            cells=[
                {
                    "cell_id": "cell_lesson_001_000",
                    "cell_type": "markdown",
                    "source": "## Second section",
                    "metadata": {"pedagogical_role": "intro", "section_id": "lesson_001"},
                    "source_chunk_ids": ["chunk_001"],
                },
                {
                    "cell_id": "cell_lesson_001_001",
                    "cell_type": "code",
                    "source": "print('second')",
                    "metadata": {"pedagogical_role": "code", "section_id": "lesson_001"},
                    "source_chunk_ids": ["chunk_001"],
                },
            ],
            generated_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )

    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        active_model_provenance=ModelProvenance(
            inference_backend="llama_cpp",
            provider_name="llama.cpp",
            model_name="gemma-4-e2b-q4_0",
            temperature=0.2,
            max_output_tokens=1800,
        ),
        stage_artifact_paths={
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
            "parsed_paper": str(parsed_paper_path),
            "cell_batches": str(run_directory / "cell_batches"),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["assemble", str(run_directory)])

    assert result.exit_code == 0
    notebook_path = run_directory / "notebook" / "final_notebook.ipynb"
    assert notebook_path.is_file()

    notebook_payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook_payload["cells"][1]["source"] == ["## Second section"]
    assert notebook_payload["cells"][3]["source"] == ["## First section"]
    assert notebook_payload["metadata"]["paper_to_notebook"]["project_version_or_code_version"] == "test-code-version"

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["notebook"].endswith("notebook/final_notebook.ipynb")
    assert "Notebook artifact:" in result.stdout


def test_assemble_command_requires_notebook_plan_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={"learner_profile": str(run_directory / "learner_profile" / "learner_profile.json")},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["assemble", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_assemble_command_requires_cell_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrices",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(
        NotebookPlan(
            paper_id="paper",
            learner_profile=LearnerProfile(
                mathematics_background="Linear algebra",
                machine_learning_background="Intermediate ML",
                deep_learning_background="Beginner DL",
                python_background="Intermediate Python",
                tensor_familiarity="Comfortable with matrices",
            ),
            lesson_sections=[],
            created_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["assemble", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_assemble_command_fails_for_unknown_extra_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "cell_batches").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")

    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrices",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(
        NotebookPlan(
            paper_id="paper",
            learner_profile=LearnerProfile(
                mathematics_background="Linear algebra",
                machine_learning_background="Intermediate ML",
                deep_learning_background="Beginner DL",
                python_background="Intermediate Python",
                tensor_familiarity="Comfortable with matrices",
            ),
            lesson_sections=[
                {
                    "section_id": "lesson_000",
                    "title": "Expected section",
                    "teaching_goal": "Explain the expected section.",
                    "source_chunk_ids": ["chunk_000"],
                    "requires_code_example": True,
                    "requires_recap": True,
                }
            ],
            created_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (run_directory / "cell_batches" / "lesson_999.json").write_text(
        NotebookBatch(
            section_id="lesson_999",
            section_title="Unexpected section",
            cells=[
                {
                    "cell_id": "cell_lesson_999_000",
                    "cell_type": "markdown",
                    "source": "## Unexpected section",
                    "metadata": {"pedagogical_role": "intro", "section_id": "lesson_999"},
                }
            ],
            generated_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["assemble", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, RuntimeError)


def test_assemble_command_fails_for_missing_planned_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "cell_batches").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")

    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrices",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(
        NotebookPlan(
            paper_id="paper",
            learner_profile=LearnerProfile(
                mathematics_background="Linear algebra",
                machine_learning_background="Intermediate ML",
                deep_learning_background="Beginner DL",
                python_background="Intermediate Python",
                tensor_familiarity="Comfortable with matrices",
            ),
            lesson_sections=[
                {
                    "section_id": "lesson_000",
                    "title": "Expected section",
                    "teaching_goal": "Explain the expected section.",
                    "source_chunk_ids": ["chunk_000"],
                    "requires_code_example": True,
                    "requires_recap": True,
                }
            ],
            created_at_utc="20260405T120000Z",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["assemble", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_chunk_command_writes_chunks_and_updates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "parsed_paper").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "chunk_size_characters: 2200",
                "chunk_overlap_characters: 250",
                "retrieval_top_k: 6",
                "generation_temperature: 0.2",
                "generation_max_output_tokens: 1800",
                "notebook_execution_timeout_seconds: 120",
                "enable_notebook_execution_smoke_test: true",
                "enable_repair_pass: true",
                "enable_widgets: true",
            ]
        ),
        encoding="utf-8",
    )
    parsed_paper = ParsedPaper(
        paper_id="paper",
        paper_title="Paper Title",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        parser_name="pymupdf4llm",
        parser_version="0.0.20",
        parsed_at_utc="20260405T120000Z",
        sections=[
            PaperSection(
                section_id="section_000",
                title="Introduction",
                page_start=1,
                page_end=1,
                markdown_text="Q = XW_Q. Figure 1 shows attention.",
            )
        ],
        abstract_text="",
        authors=None,
        stage_provenance=StageProvenance(
            stage_name="parsed_paper",
            created_at_utc="20260405T120000Z",
            input_artifact_paths=["/tmp/paper.pdf"],
            output_artifact_path="parsed_paper.json",
            code_version="test-code-version",
            model_provenance=None,
        ),
    )
    parsed_paper_path = run_directory / "parsed_paper" / "parsed_paper.json"
    parsed_paper_path.write_text(parsed_paper.model_dump_json(indent=2), encoding="utf-8")

    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={"chunk_size_characters": 2200, "chunk_overlap_characters": 250},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={
            "parsed_paper": str(parsed_paper_path),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["chunk", str(run_directory)])

    assert result.exit_code == 0
    chunks_path = run_directory / "chunks" / "chunks.json"
    assert chunks_path.is_file()

    chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    assert chunks_payload[0]["chunk_id"] == "chunk_00000"
    assert chunks_payload[0]["section_id"] == "section_000"

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["chunks"].endswith("chunks/chunks.json")
    assert "Chunks artifact:" in result.stdout


def test_chunk_command_requires_parsed_paper_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["chunk", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_concept_command_writes_concept_graph_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "chunks").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "chunk_size_characters: 2200",
                "chunk_overlap_characters: 250",
                "retrieval_top_k: 6",
                "generation_temperature: 0.2",
                "generation_max_output_tokens: 1800",
                "notebook_execution_timeout_seconds: 120",
                "enable_notebook_execution_smoke_test: true",
                "enable_repair_pass: true",
                "enable_widgets: true",
            ]
        ),
        encoding="utf-8",
    )
    chunks_path = run_directory / "chunks" / "chunks.json"
    chunks_path.write_text(
        json.dumps(
            [
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
                ).model_dump(mode="json")
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={"retrieval_top_k": 6},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={
            "chunks": str(chunks_path),
            "run_manifest": str(run_directory / "run_manifest.json"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")

    monkeypatch.setattr("src.main.Gemma4E2BModel.from_settings", lambda environment_settings, run_parameters: object())
    monkeypatch.setattr(
        "src.main.build_concept_graph",
        lambda model, paper_chunks, retrieval_top_k: (
            [
                ConceptItem(
                    concept_id="concept_000",
                    label="Scaled dot-product attention",
                    concept_type="equation",
                    source_chunk_ids=["chunk_00000"],
                    prerequisites=[],
                    equation_text="softmax(QK^T)V",
                    notation_symbols=["Q", "K"],
                    tensor_shape_notes=["Q: B x T x d_k"],
                    confidence=0.9,
                )
            ],
            [],
        ),
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["concept", str(run_directory)])

    assert result.exit_code == 0
    concepts_path = run_directory / "concept_graph" / "concepts.json"
    concept_edges_path = run_directory / "concept_graph" / "concept_edges.json"
    assert concepts_path.is_file()
    assert concept_edges_path.is_file()

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["concept_graph"].endswith("concept_graph/concepts.json")
    assert manifest_payload["stage_artifact_paths"]["concept_graph_edges"].endswith("concept_graph/concept_edges.json")
    assert "Concept graph artifact:" in result.stdout


def test_concept_command_requires_chunks_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "default_model_name: gemma-4-e2b-q4_0\ninference_backend: llama_cpp\n",
        encoding="utf-8",
    )
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        source_pdf_path="/tmp/paper.pdf",
        source_pdf_sha256="abc123",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        planned_notebook_output_path=str(run_directory / "notebook" / "final_notebook.ipynb"),
        active_model_provenance=None,
        stage_artifact_paths={},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["concept", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def _write_validation_ready_run(tmp_path: Path) -> tuple[Path, Path]:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    (run_directory / "notebook").mkdir(parents=True)
    (run_directory / "cell_batches").mkdir(parents=True)
    (run_directory / "learner_profile").mkdir(parents=True)
    (run_directory / "notebook_plan").mkdir(parents=True)
    (tmp_path / "params.yaml").write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "enable_notebook_execution_smoke_test: false",
                "enable_repair_pass: false",
            ]
        ),
        encoding="utf-8",
    )

    learner_profile = LearnerProfile(
        mathematics_background="Linear algebra",
        machine_learning_background="Intermediate ML",
        deep_learning_background="Beginner DL",
        python_background="Intermediate Python",
        tensor_familiarity="Comfortable with matrices",
        preferred_depth="deep",
        preferred_pacing="moderate",
    )
    learner_profile_path = run_directory / "learner_profile" / "learner_profile.json"
    learner_profile_path.write_text(learner_profile.model_dump_json(indent=2), encoding="utf-8")

    notebook_plan = NotebookPlan(
        paper_id="paper",
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
        created_at_utc="20260405T120000Z",
    )
    notebook_plan_path = run_directory / "notebook_plan" / "notebook_plan.json"
    notebook_plan_path.write_text(notebook_plan.model_dump_json(indent=2), encoding="utf-8")

    notebook_batch = NotebookBatch(
        section_id="lesson_000",
        section_title="Attention basics",
        cells=[
            {
                "cell_id": "cell_lesson_000_000",
                "cell_type": "markdown",
                "source": "## Attention basics\nWe use Q and note Q: B x T x d_k.",
                "metadata": {"section_id": "lesson_000", "pedagogical_role": "intro"},
                "source_chunk_ids": ["chunk_000"],
            }
        ],
        generated_at_utc="20260405T120000Z",
    )
    cell_batch_path = run_directory / "cell_batches" / "lesson_000.json"
    cell_batch_path.write_text(notebook_batch.model_dump_json(indent=2), encoding="utf-8")

    notebook_object = nbformat.v4.new_notebook()
    notebook_object.cells = [
        nbformat.v4.new_markdown_cell("# Paper Title"),
        nbformat.v4.new_markdown_cell(
            "## Attention basics\nWe use Q and note Q: B x T x d_k.",
            metadata={"section_id": "lesson_000", "pedagogical_role": "intro"},
        ),
    ]
    notebook_path = run_directory / "notebook" / "final_notebook.ipynb"
    with notebook_path.open("w", encoding="utf-8") as output_file:
        nbformat.write(notebook_object, output_file)

    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        active_model_provenance=ModelProvenance(
            inference_backend="llama_cpp",
            provider_name="llama.cpp",
            model_name="gemma-4-e2b-q4_0",
            temperature=0.2,
            max_output_tokens=1800,
        ),
        stage_artifact_paths={
            "learner_profile": str(learner_profile_path),
            "notebook_plan": str(notebook_plan_path),
            "notebook": str(notebook_path),
            "cell_batches": str(run_directory / "cell_batches"),
        },
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    return run_directory, notebook_path


def test_validate_command_writes_validation_report_and_updates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory, _ = _write_validation_ready_run(tmp_path)
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["validate", str(run_directory)])

    assert result.exit_code == 0
    validation_report_path = run_directory / "validation_report" / "validation_report.json"
    assert validation_report_path.is_file()

    validation_report = ValidationReport.model_validate_json(validation_report_path.read_text(encoding="utf-8"))
    assert validation_report.notebook_path is not None
    assert "schema_validity" in validation_report.checks_run

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["validation_report"].endswith("validation_report/validation_report.json")
    assert "Validation report artifact:" in result.stdout


def test_validate_command_requires_notebook_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory = tmp_path / "runs" / "paper" / "20260405T120000Z"
    run_directory.mkdir(parents=True)
    (tmp_path / "params.yaml").write_text("default_model_name: gemma-4-e2b-q4_0\n", encoding="utf-8")
    run_manifest = RunManifest(
        run_id="20260405T120000Z",
        paper_slug="paper",
        params_path=str((tmp_path / "params.yaml").resolve()),
        run_parameters={},
        stage_artifact_paths={},
        created_at_utc="20260405T120000Z",
    )
    (run_directory / "run_manifest.json").write_text(run_manifest.model_dump_json(indent=2), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["validate", str(run_directory)])

    assert result.exit_code == 1
    assert isinstance(result.exception, FileNotFoundError)


def test_validate_command_with_repair_disabled_writes_failing_report_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_directory, notebook_path = _write_validation_ready_run(tmp_path)
    notebook_object = nbformat.read(notebook_path, as_version=4)
    notebook_object.cells[1].metadata["section_id"] = "lesson_999"
    with notebook_path.open("w", encoding="utf-8") as output_file:
        nbformat.write(notebook_object, output_file)

    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["validate", str(run_directory)])

    assert result.exit_code == 0
    validation_report = ValidationReport.model_validate_json(
        (run_directory / "validation_report" / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation_report.is_valid is False
    assert any("pedagogical_ordering" in error for error in validation_report.errors)


def test_validate_command_can_recover_with_repair_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_directory, notebook_path = _write_validation_ready_run(tmp_path)
    (tmp_path / "params.yaml").write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "enable_notebook_execution_smoke_test: false",
                "enable_repair_pass: true",
            ]
        ),
        encoding="utf-8",
    )

    validation_results = [
        (
            ValidationReport(
                paper_id="paper",
                run_id="20260405T120000Z",
                is_valid=False,
                errors=["[pedagogical_ordering] Notebook is missing planned section lesson_000."],
                warnings=[],
                checks_run=["schema_validity"],
                notebook_path=str(notebook_path),
                validated_at_utc="20260405T120001Z",
            ),
            [],
        ),
        (
            ValidationReport(
                paper_id="paper",
                run_id="20260405T120000Z",
                is_valid=True,
                errors=[],
                warnings=[],
                checks_run=["schema_validity"],
                notebook_path=str(notebook_path),
                validated_at_utc="20260405T120002Z",
            ),
            [],
        ),
    ]

    def fake_validate_assembled_notebook(**kwargs):
        return validation_results.pop(0)

    class FakeRepairOutcome:
        repaired = True
        repaired_notebook_path = str(notebook_path)
        repaired_batch_paths = []

    monkeypatch.setattr("src.main.validate_assembled_notebook", fake_validate_assembled_notebook)
    monkeypatch.setattr("src.main.attempt_targeted_repair", lambda **kwargs: FakeRepairOutcome())
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["validate", str(run_directory)])

    assert result.exit_code == 0
    validation_report = ValidationReport.model_validate_json(
        (run_directory / "validation_report" / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation_report.is_valid is True
