"""Environment-backed settings for target and judge clients."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else int(raw)


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else float(raw)


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    return None if raw is None or not raw.strip() else int(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    target_base_url: str
    target_api_key: str
    target_model: str
    target_temperature: float
    target_max_tokens: int
    target_timeout_seconds: float
    target_seed: int | None
    judge_score_url: str
    judge_score_full_url: str
    judge_health_url: str
    judge_timeout_seconds: float

    @classmethod
    def load(cls, env_file: str | Path = ".env") -> "Settings":
        load_dotenv(dotenv_path=env_file, override=False)
        return cls(
            target_base_url=os.getenv("TARGET_BASE_URL", "http://10.50.2.40:11434/v1").rstrip("/"),
            target_api_key=os.getenv("TARGET_API_KEY", "ollama"),
            target_model=os.getenv("TARGET_MODEL", "llama3.1:8b"),
            target_temperature=_float_env("TARGET_TEMPERATURE", 0.0),
            target_max_tokens=_int_env("TARGET_MAX_TOKENS", 512),
            target_timeout_seconds=_float_env("TARGET_TIMEOUT_SECONDS", 180.0),
            target_seed=_optional_int_env("TARGET_SEED") if "TARGET_SEED" in os.environ else 17,
            judge_score_url=os.getenv("JUDGE_SCORE_URL", "http://10.50.2.39:18080/score"),
            judge_score_full_url=os.getenv(
                "JUDGE_SCORE_FULL_URL", "http://10.50.2.39:18082/score_full"
            ),
            judge_health_url=os.getenv("JUDGE_HEALTH_URL", "http://10.50.2.39:18082/health"),
            judge_timeout_seconds=_float_env("JUDGE_TIMEOUT_SECONDS", 90.0),
        )
