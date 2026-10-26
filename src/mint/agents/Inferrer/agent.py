from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
import requests

from ...configs import DEFAULT_LLM_TIMEOUT_SECONDS, PROMPT_VERSION
from ...llm.prompt_renderer import render_prompt
from .schemas.output import InferenceOutput

LOGGER = logging.getLogger(__name__)


class ContractInferrer:
    def __init__(
        self,
        model: str,
        *,
        settings: Any | None = None,
        prompts_dir: str | Path | None = None,
        env_file: str | Path | None = None,
    ) -> None:
        load_dotenv(dotenv_path=env_file, override=False)

        if settings is not None and settings.model != model:
            raise ValueError("settings.model must match model")

        self.settings = settings
        self._model_name = model
        self.prompts_dir = Path(
            prompts_dir or Path(__file__).parent / "prompts" / PROMPT_VERSION
        )
        self.max_attempts = int(os.getenv("INFERRER_MAX_ATTEMPTS", "3"))
        self.base_url = str(
            getattr(settings, "base_url", None)
            or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
        ).rstrip("/")
        self.timeout_seconds = float(
            getattr(settings, "timeout_seconds", None)
            or os.getenv("OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_LLM_TIMEOUT_SECONDS))
        )

    def infer(
        self,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:
        context = dict(data)
        context.setdefault("bug_context", None)

        system_prompt = render_prompt(
            self.prompts_dir,
            "system.j2",
            **context,
        )

        user_prompt = render_prompt(
            self.prompts_dir,
            "user.j2",
            **context,
        )

        payload = {
            "model": self.model_name,
            "stream": False,
            "think": False,
            "format": InferenceOutput.model_json_schema(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0},
        }
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                response = requests.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                response.raise_for_status()
                response_data = response.json()
                content = response_data["message"]["content"]
                if not str(content).strip():
                    if "format" in payload and attempt + 1 < self.max_attempts:
                        LOGGER.warning(
                            "Model %s returned empty structured output; retrying "
                            "with prompt-enforced JSON",
                            self.model_name,
                        )
                        payload.pop("format")
                        payload["messages"][-1]["content"] += (
                            "\n\nReturn only one JSON object with exactly this shape and "
                            "field names, without Markdown fences: "
                            '{"decision":"contracts_found","contracts":'
                            '[{"apis":["qualified.callable"],'
                            '"contract":"A self-contained requirement."}],'
                            '"rationale":"A concise evidence explanation."}. '
                            "Use decision no_contract or abstain with an empty "
                            "contracts array when appropriate."
                        )
                    raise ValueError("Ollama returned empty message content")
                output = InferenceOutput.model_validate_json(content)
                break
            except (KeyError, ValueError, requests.RequestException) as exc:
                last_error = exc
        else:
            raise RuntimeError(
                f"Ollama returned no valid contract response after "
                f"{self.max_attempts} attempts"
            ) from last_error

        metrics = {
            "model": response_data.get("model", self.model_name),
            "prompt_tokens": response_data.get("prompt_eval_count"),
            "completion_tokens": response_data.get("eval_count"),
            "total_duration_ns": response_data.get("total_duration"),
        }

        return {
            "output": output.model_dump(mode="json"),
            "metrics": metrics,
            "prompt": {
                "system": system_prompt,
                "user": user_prompt,
            },
        }

    def unload(self) -> None:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model_name, "keep_alive": 0},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    @property
    def model_name(self) -> str:
        return str(getattr(self.settings, "model", None) or self._model_name)
