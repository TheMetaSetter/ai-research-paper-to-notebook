from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from src.main import app

runner = CliRunner()


def test_help_lists_expected_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "parse" in result.stdout
    assert "plan" in result.stdout
    assert "generate" in result.stdout


def test_parse_command_validates_inputs_and_stops_after_phase_zero_context(tmp_path: Path, monkeypatch) -> None:
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

    result = runner.invoke(app, ["parse", str(pdf_path), "--params-path", str(params_path)])

    assert result.exit_code == 1
    assert isinstance(result.exception, NotImplementedError)
    assert (tmp_path / "runs").exists()


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

