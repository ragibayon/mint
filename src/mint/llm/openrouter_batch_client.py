from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from typing import Any, TypeVar

from pydantic import BaseModel
import requests

from mint.configs import DEFAULT_LLM_TIMEOUT_SECONDS

SchemaT = TypeVar("SchemaT", bound=BaseModel)


def strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(schema)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object" or "properties" in value:
                value["additionalProperties"] = False
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(normalized)
    return normalized


def structured_response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": strict_json_schema(schema.model_json_schema()),
        },
    }


def chat_request_body(
    *,
    model: str,
    messages: list[dict[str, str]],
    schema: type[BaseModel],
    temperature: float | None = 0,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": structured_response_format(schema),
    }
    if temperature is not None:
        body["temperature"] = temperature
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    return body


def batch_payload(
    *,
    model: str,
    requests_: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "endpoint": "/v1/chat/completions",
        "model": model,
        "completion_window": "24h",
        "requests": requests_,
    }


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise ValueError("Provider response message content is not text")
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        text = block.get("text")
        if isinstance(text, str):
            parts.append(text)
    if not parts:
        raise ValueError("Provider response contains no text blocks")
    return "".join(parts)


def parse_chat_response(body: dict[str, Any], schema: type[SchemaT]) -> SchemaT:
    choices = body.get("choices") or []
    if not choices:
        raise ValueError("Provider response contains no choices")
    message = choices[0].get("message") or {}
    parsed = message.get("parsed")
    if parsed is not None:
        if isinstance(parsed, str):
            return schema.model_validate_json(parsed)
        return schema.model_validate(parsed)
    return schema.model_validate_json(message_text(message.get("content")))


@dataclass
class OpenRouterBatchClient:
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        self.api_key = (self.api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        if not self.api_key:
            raise RuntimeError("Set OPENROUTER_API_KEY before using the batch API")
        self.base_url = self.base_url.rstrip("/")

    def request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = requests.request(
            method,
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout_seconds,
            **kwargs,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()[:2000]
            raise RuntimeError(
                f"OpenRouter HTTP {response.status_code}: {detail or response.reason}"
            ) from exc
        return response.json()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/batches", json=payload)

    def get(self, batch_id: str) -> dict[str, Any]:
        return self.request("GET", f"/batches/{batch_id}")

    def complete(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.request("POST", "/chat/completions", json=body)
