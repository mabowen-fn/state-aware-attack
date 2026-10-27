"""OpenAI-compatible chat client for a local Ollama target model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stateaware.config import Settings


@dataclass(frozen=True, slots=True)
class TargetReply:
    text: str
    model: str
    latency_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    finish_reason: str | None


class OllamaTarget:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        try:
            from openai import OpenAI
        except ImportError as exc:  # Gives a clearer message if dependencies were not installed.
            raise RuntimeError("Missing dependency 'openai'; install with `uv pip install -e '.[dev]'`.") from exc
        self.client = OpenAI(
            base_url=f"{settings.target_base_url}/",
            api_key=settings.target_api_key,
            timeout=settings.target_timeout_seconds,
            max_retries=1,
        )

    def list_models(self) -> list[str]:
        response = self.client.models.list()
        return [model.id for model in response.data]

    def generate(self, messages: list[dict[str, str]]) -> TargetReply:
        import time

        started = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.settings.target_model,
            "messages": messages,
            "temperature": self.settings.target_temperature,
            "max_tokens": self.settings.target_max_tokens,
        }
        if self.settings.target_seed is not None:
            kwargs["seed"] = self.settings.target_seed
        result = self.client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - started
        choice = result.choices[0]
        content = choice.message.content
        text = content if isinstance(content, str) else ""
        usage = result.usage
        return TargetReply(
            text=text,
            model=result.model or self.settings.target_model,
            latency_seconds=latency,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            finish_reason=choice.finish_reason,
        )

    def close(self) -> None:
        self.client.close()
