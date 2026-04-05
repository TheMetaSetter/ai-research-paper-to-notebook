from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    """Environment-backed runtime settings and secrets."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAPER2NB_",
        extra="ignore",
    )

    local_llama_server_base_url: str = "http://127.0.0.1:8080/v1"
    google_api_key: str | None = None
    use_google_cloud_adapter: bool = False


class RunParameters(BaseModel):
    """Explicit run parameters stored in YAML for reproducible executions."""

    model_config = ConfigDict(extra="forbid")

    default_model_name: str = "gemma-4-e2b-q4_0"
    inference_backend: Literal["llama_cpp", "google_genai"] = "llama_cpp"
    chunk_size_characters: int = Field(default=2200, ge=1)
    chunk_overlap_characters: int = Field(default=250, ge=0)
    retrieval_top_k: int = Field(default=6, ge=1)
    maximum_generation_sections: int | None = Field(default=None, ge=1)
    generation_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    generation_max_output_tokens: int = Field(default=1800, ge=1)
    notebook_execution_timeout_seconds: int = Field(default=120, ge=1)
    enable_notebook_execution_smoke_test: bool = True
    enable_repair_pass: bool = True
    enable_widgets: bool = True


def load_environment_settings() -> EnvironmentSettings:
    """Load environment-backed settings using the PAPER2NB_ prefix."""

    return EnvironmentSettings()


def load_run_parameters(yaml_path: str | Path) -> RunParameters:
    """Load YAML-backed run parameters."""

    parameter_path = Path(yaml_path)
    with parameter_path.open("r", encoding="utf-8") as parameter_file:
        raw_parameter_dictionary = yaml.safe_load(parameter_file) or {}

    if not isinstance(raw_parameter_dictionary, dict):
        raise ValueError(f"Run parameter file must contain a YAML mapping: {parameter_path}")

    return RunParameters.model_validate(raw_parameter_dictionary)

