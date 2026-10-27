"""HTTP client for the supplied HarmBench-compatible judge service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from stateaware.config import Settings


@dataclass(frozen=True, slots=True)
class JudgeResult:
    endpoint_mode: str
    asr: int | None
    probability: float | None
    version: str | None
    logit_yes: float | None = None
    logit_no: float | None = None
    logit_diff: float | None = None
    raw: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "endpoint_mode": self.endpoint_mode,
            "asr": self.asr,
            "probability": self.probability,
            "version": self.version,
            "logit_yes": self.logit_yes,
            "logit_no": self.logit_no,
            "logit_diff": self.logit_diff,
            "raw": self.raw,
        }


class JudgeError(RuntimeError):
    """Raised when the judge API fails or returns an invalid payload."""


class HarmBenchJudge:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(timeout=settings.judge_timeout_seconds)

    def health(self) -> dict[str, Any]:
        """The user's service contract specifies POST for /health."""
        try:
            response = self.client.post(self.settings.judge_health_url, json={})
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise JudgeError(f"POST health check failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise JudgeError("Health endpoint returned JSON that was not an object")
        return payload

    def score(self, behavior: str, generation: str, *, mode: str = "score") -> JudgeResult:
        if mode not in {"score", "full"}:
            raise ValueError("mode must be 'score' or 'full'")
        url = self.settings.judge_score_url if mode == "score" else self.settings.judge_score_full_url
        try:
            response = self.client.post(
                url,
                json={"behavior": behavior, "generation": generation},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise JudgeError(f"Judge {mode} request failed: {exc}") from exc
        if not isinstance(payload, dict):
            raise JudgeError("Judge returned JSON that was not an object")
        asr = payload.get("asr")
        probability = payload.get("probability")
        if asr not in (0, 1, None):
            raise JudgeError(f"Unexpected ASR value: {asr!r}")
        try:
            probability_value = float(probability) if probability is not None else None
        except (TypeError, ValueError) as exc:
            raise JudgeError(f"Unexpected probability value: {probability!r}") from exc
        return JudgeResult(
            endpoint_mode=mode,
            asr=asr,
            probability=probability_value,
            version=payload.get("version"),
            logit_yes=_as_optional_float(payload.get("logit_yes")),
            logit_no=_as_optional_float(payload.get("logit_no")),
            logit_diff=_as_optional_float(payload.get("logit_diff")),
            raw=payload,
        )

    def close(self) -> None:
        self.client.close()


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
