from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config import load_environment_settings, load_run_parameters


def test_load_run_parameters_defaults_when_yaml_is_empty(tmp_path: Path) -> None:
    parameter_path = tmp_path / "params.yaml"
    parameter_path.write_text("{}", encoding="utf-8")

    run_parameters = load_run_parameters(parameter_path)

    assert run_parameters.default_model_name == "gemma-4-e2b-q4_0"
    assert run_parameters.inference_backend == "llama_cpp"
    assert run_parameters.retrieval_top_k == 6


def test_load_run_parameters_reads_yaml_overrides(tmp_path: Path) -> None:
    parameter_path = tmp_path / "params.yaml"
    parameter_path.write_text(
        "\n".join(
            [
                "default_model_name: custom-model",
                "inference_backend: google_genai",
                "chunk_size_characters: 1800",
                "enable_widgets: false",
            ]
        ),
        encoding="utf-8",
    )

    run_parameters = load_run_parameters(parameter_path)

    assert run_parameters.default_model_name == "custom-model"
    assert run_parameters.inference_backend == "google_genai"
    assert run_parameters.chunk_size_characters == 1800
    assert run_parameters.enable_widgets is False


def test_load_run_parameters_rejects_unknown_keys(tmp_path: Path) -> None:
    parameter_path = tmp_path / "params.yaml"
    parameter_path.write_text("unexpected_field: 1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_run_parameters(parameter_path)


def test_environment_settings_use_expected_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAPER2NB_LOCAL_LLAMA_SERVER_BASE_URL", "http://localhost:9000/v1")
    monkeypatch.setenv("PAPER2NB_GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("PAPER2NB_USE_GOOGLE_CLOUD_ADAPTER", "true")

    environment_settings = load_environment_settings()

    assert environment_settings.local_llama_server_base_url == "http://localhost:9000/v1"
    assert environment_settings.google_api_key == "test-key"
    assert environment_settings.use_google_cloud_adapter is True

