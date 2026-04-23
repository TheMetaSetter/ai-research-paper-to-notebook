from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.main import app
from src.schemas import LearnerProfile, ModelProvenance, ParsedPaper, PaperSection


runner = CliRunner()


def _write_run_parameters(parameter_path: Path) -> None:
    parameter_path.write_text(
        "\n".join(
            [
                "default_model_name: gemma-4-e2b-q4_0",
                "inference_backend: llama_cpp",
                "chunk_size_characters: 2200",
                "chunk_overlap_characters: 250",
                "retrieval_top_k: 2",
                "generation_temperature: 0.2",
                "generation_max_output_tokens: 1800",
                "enable_notebook_execution_smoke_test: false",
                "enable_repair_pass: false",
            ]
        ),
        encoding="utf-8",
    )


def _write_learner_profile(learner_profile_path: Path) -> None:
    learner_profile_path.write_text(
        LearnerProfile(
            mathematics_background="Linear algebra",
            machine_learning_background="Intermediate ML",
            deep_learning_background="Beginner DL",
            python_background="Intermediate Python",
            tensor_familiarity="Comfortable with matrices",
            wants_tensor_shapes=True,
            wants_derivations=False,
            preferred_depth="deep",
            preferred_pacing="moderate",
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )


class StubEndToEndModel:
    def generate_json(self, system_instruction: str, user_prompt: str, response_model):
        response_model_name = response_model.__name__
        if response_model_name == "ConceptExtractionResponse":
            return response_model(
                concepts=[
                    {
                        "label": "Scaled dot-product attention",
                        "concept_type": "equation",
                        "source_chunk_ids": ["chunk_00000"],
                        "prerequisites": [],
                        "equation_text": "softmax(QK^T)V",
                        "notation_symbols": ["Q", "K", "V"],
                        "tensor_shape_notes": ["Q: B x T x d_k"],
                        "confidence": 0.9,
                    }
                ]
            )
        if response_model_name == "NotebookPlanDraft":
            return response_model(
                lesson_sections=[
                    {
                        "section_id": "model_section",
                        "title": "Attention from first principles",
                        "teaching_goal": "Explain scaled dot-product attention.",
                        "prerequisite_concepts": [],
                        "source_chunk_ids": ["chunk_00000"],
                        "equations_to_unpack": ["softmax(QK^T)V"],
                        "tensor_shapes_to_state": ["Q: B x T x d_k"],
                        "likely_misconceptions": ["Attention is not recurrence."],
                        "requires_code_example": True,
                        "requires_recap": True,
                    }
                ],
                planning_notes=["Introduce notation before code."],
            )
        if response_model_name == "NotebookBatchDraft":
            return response_model(
                section_title="Attention from first principles",
                cells=[
                    {
                        "cell_type": "markdown",
                        "source": "## Attention from first principles\nQ has shape Q: B x T x d_k.",
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
                        "source": "### Recap\nQ, K, and V organize the attention computation.",
                        "metadata": {"pedagogical_role": "recap"},
                        "source_chunk_ids": ["chunk_00000"],
                    },
                ],
            )
        raise AssertionError(f"Unexpected response model: {response_model_name}")

    def model_provenance(self) -> ModelProvenance:
        return ModelProvenance(
            inference_backend="llama_cpp",
            provider_name="llama.cpp",
            model_name="gemma-4-e2b-q4_0",
            temperature=0.2,
            max_output_tokens=1800,
        )


def test_run_command_executes_full_pipeline_with_learner_profile_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    params_path = tmp_path / "params.yaml"
    learner_profile_path = tmp_path / "learner_profile.json"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    _write_run_parameters(params_path)
    _write_learner_profile(learner_profile_path)

    def fake_parse_pdf_into_parsed_paper(
        pdf_path: Path,
        source_pdf_sha256: str,
        run_id: str,
        code_version: str | None = None,
    ):
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
                    title="Method",
                    page_start=1,
                    page_end=1,
                    markdown_text="Attention uses Q, K, and V with Q: B x T x d_k.",
                )
            ],
            abstract_text="",
        )
        return parsed_paper, "# Paper Title\nAttention uses Q, K, and V.", [{"page_number": 1, "text": "Attention text."}]

    monkeypatch.setattr("src.main.parse_pdf_into_parsed_paper", fake_parse_pdf_into_parsed_paper)
    monkeypatch.setattr("src.main.Gemma4E2BModel.from_settings", lambda environment_settings, run_parameters: StubEndToEndModel())
    monkeypatch.setattr(
        "src.main.select_retrieved_chunks_for_lesson_section",
        lambda lesson_section, paper_chunks, retrieval_top_k: paper_chunks[:retrieval_top_k],
    )
    monkeypatch.setattr("src.main.capture_learner_profile_interactively", lambda: pytest.fail("run should use the learner profile file."))
    monkeypatch.setattr("src.main.utc_timestamp_string", lambda: "20260424T010203Z")
    monkeypatch.setattr("src.main._current_code_version", lambda: "test-code-version")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "run",
            str(pdf_path),
            "--params-path",
            str(params_path),
            "--learner-profile",
            str(learner_profile_path),
        ],
    )

    assert result.exit_code == 0
    run_directory = tmp_path / "runs" / "paper" / "20260424T010203Z"
    assert (run_directory / "parsed_paper" / "parsed_paper.json").is_file()
    assert (run_directory / "chunks" / "chunks.json").is_file()
    assert (run_directory / "concept_graph" / "concepts.json").is_file()
    assert (run_directory / "learner_profile" / "learner_profile.json").is_file()
    assert (run_directory / "notebook_plan" / "notebook_plan.json").is_file()
    assert (run_directory / "cell_batches" / "lesson_000.json").is_file()
    assert (run_directory / "notebook" / "final_notebook.ipynb").is_file()
    assert (run_directory / "validation_report" / "validation_report.json").is_file()

    saved_learner_profile = LearnerProfile.model_validate_json(
        (run_directory / "learner_profile" / "learner_profile.json").read_text(encoding="utf-8")
    )
    assert saved_learner_profile.mathematics_background == "Linear algebra"

    manifest_payload = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["stage_artifact_paths"]["validation_report"].endswith("validation_report/validation_report.json")
    assert manifest_payload["active_model_provenance"]["model_name"] == "gemma-4-e2b-q4_0"

    validation_payload = json.loads((run_directory / "validation_report" / "validation_report.json").read_text(encoding="utf-8"))
    assert validation_payload["is_valid"] is True
    assert "Run directory:" in result.stdout
    assert "Validation report artifact:" in result.stdout
