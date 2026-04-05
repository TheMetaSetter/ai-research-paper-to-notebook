from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.config import EnvironmentSettings, RunParameters
from src.schemas import ModelProvenance

try:
    from google import genai
    from google.genai import types as google_types
except Exception:  # pragma: no cover - optional dependency path remains explicit
    genai = None
    google_types = None


StructuredResponseType = TypeVar("StructuredResponseType", bound=BaseModel)


def _extract_local_response_content(response_payload: Any) -> str:
    """Extract the first assistant message content from a chat-completions payload."""

    if not isinstance(response_payload, dict):
        raise RuntimeError("Local llama.cpp response payload is malformed: expected a JSON object.")

    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Local llama.cpp response payload is malformed: missing choices.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Local llama.cpp response payload is malformed: first choice is not an object.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Local llama.cpp response payload is malformed: missing message object.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Local llama.cpp response payload is malformed: missing message content.")

    return content


def _strip_optional_markdown_code_fence(raw_text: str) -> str:
    """Remove a top-level fenced code wrapper when the model returns fenced JSON."""

    cleaned_text = raw_text.strip()
    if not cleaned_text.startswith("```"):
        return cleaned_text

    lines = cleaned_text.splitlines()
    if not lines:
        return cleaned_text

    first_line = lines[0].strip().lower()
    last_line = lines[-1].strip()
    if not first_line.startswith("```") or last_line != "```":
        return cleaned_text

    fenced_body_lines = lines[1:-1]
    return "\n".join(fenced_body_lines).strip()


@dataclass
class Gemma4E2BModel:
    local_llama_server_base_url: str
    use_google_cloud_adapter: bool = False
    google_api_key: str | None = None
    local_model_name: str = "gemma-4-e2b-q4_0"
    local_temperature: float = 0.2
    local_max_output_tokens: int = 1800

    def generate_json(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: type[StructuredResponseType],
    ) -> StructuredResponseType:
        if self.use_google_cloud_adapter:
            return self._generate_json_with_google_genai(
                system_instruction=system_instruction,
                user_prompt=user_prompt,
                response_model=response_model,
            )
        return self._generate_json_with_local_llama_cpp(
            system_instruction=system_instruction,
            user_prompt=user_prompt,
            response_model=response_model,
        )

    def _generate_json_with_local_llama_cpp(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: type[StructuredResponseType],
    ) -> StructuredResponseType:
        """Send a structured JSON request to a local llama.cpp server."""

        request_payload = {
            "model": self.local_model_name,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.local_temperature,
            "max_tokens": self.local_max_output_tokens,
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client(timeout=120.0) as http_client:
                response = http_client.post(
                    f"{self.local_llama_server_base_url}/chat/completions",
                    json=request_payload,
                )
                response.raise_for_status()
                response_payload = response.json()
        except httpx.HTTPError as error:
            raise RuntimeError(f"Local llama.cpp request failed: {error}") from error

        raw_text = _extract_local_response_content(response_payload)
        cleaned_text = _strip_optional_markdown_code_fence(raw_text)

        try:
            raw_dictionary = json.loads(cleaned_text)
        except json.JSONDecodeError as error:
            raise RuntimeError(f"Local llama.cpp returned invalid JSON content: {error}") from error

        try:
            return response_model.model_validate(raw_dictionary)
        except ValidationError as error:
            raise RuntimeError(f"Local llama.cpp returned schema-invalid JSON: {error}") from error

    def _generate_json_with_google_genai(
        self,
        system_instruction: str,
        user_prompt: str,
        response_model: type[StructuredResponseType],
    ) -> StructuredResponseType:
        """Use the optional Google Gen AI adapter for schema-constrained output."""

        if not self.google_api_key:
            raise RuntimeError("Google Gen AI adapter requires google_api_key.")
        if genai is None or google_types is None:
            raise RuntimeError("Google Gen AI adapter is unavailable: google-genai is not installed.")

        client = genai.Client(api_key=self.google_api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=google_types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=response_model,
                temperature=self.local_temperature,
                max_output_tokens=self.local_max_output_tokens,
            ),
        )

        try:
            cleaned_text = _strip_optional_markdown_code_fence(response.text)
            return response_model.model_validate_json(cleaned_text)
        except ValidationError as error:
            raise RuntimeError(f"Google Gen AI adapter returned schema-invalid JSON: {error}") from error

    def model_provenance(self) -> ModelProvenance:
        provider_name = "google_genai" if self.use_google_cloud_adapter else "llama.cpp"
        inference_backend = "google_genai" if self.use_google_cloud_adapter else "llama_cpp"
        model_name = "gemini-2.5-flash" if self.use_google_cloud_adapter else self.local_model_name

        return ModelProvenance(
            inference_backend=inference_backend,
            provider_name=provider_name,
            model_name=model_name,
            temperature=self.local_temperature,
            max_output_tokens=self.local_max_output_tokens,
        )

    @classmethod
    def from_settings(
        cls,
        environment_settings: EnvironmentSettings,
        run_parameters: RunParameters,
    ) -> "Gemma4E2BModel":
        return cls(
            local_llama_server_base_url=environment_settings.local_llama_server_base_url,
            use_google_cloud_adapter=environment_settings.use_google_cloud_adapter,
            google_api_key=environment_settings.google_api_key,
            local_model_name=run_parameters.default_model_name,
            local_temperature=run_parameters.generation_temperature,
            local_max_output_tokens=run_parameters.generation_max_output_tokens,
        )
