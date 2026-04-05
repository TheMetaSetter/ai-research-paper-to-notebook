from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel, ConfigDict

from src.config import EnvironmentSettings, RunParameters
from src.models.gemma_4_e2b import Gemma4E2BModel


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
