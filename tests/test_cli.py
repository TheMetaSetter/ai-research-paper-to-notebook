from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from src.main import app
from src.schemas import PaperChunk, ParsedPaper, PaperSection, RunManifest, StageProvenance

runner = CliRunner()


def test_help_lists_expected_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "parse" in result.stdout
    assert "chunk" in result.stdout
    assert "plan" in result.stdout
    assert "generate" in result.stdout


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
    assert isinstance(result.exception, NotImplementedError)


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
