"""JSONL loading and validation for fixed conversational conditions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ALLOWED_ROLES = {"system", "user", "assistant"}


class DatasetError(ValueError):
    """Raised when a case file does not follow the expected schema."""


@dataclass(frozen=True, slots=True)
class Turn:
    user: str


@dataclass(frozen=True, slots=True)
class Condition:
    name: str
    prefix_messages: tuple[dict[str, str], ...]
    turns: tuple[Turn, ...]


@dataclass(frozen=True, slots=True)
class Case:
    case_id: str
    behavior: str
    conditions: tuple[Condition, ...]
    metadata: dict[str, Any]


def _validate_message(message: Any, *, context: str) -> dict[str, str]:
    if not isinstance(message, dict):
        raise DatasetError(f"{context}: message must be an object")
    role = message.get("role")
    content = message.get("content")
    if role not in ALLOWED_ROLES:
        raise DatasetError(f"{context}: role must be one of {sorted(ALLOWED_ROLES)}")
    if not isinstance(content, str) or not content.strip():
        raise DatasetError(f"{context}: content must be a non-empty string")
    return {"role": role, "content": content}


def parse_case(raw: Any, *, line_number: int) -> Case:
    context = f"line {line_number}"
    if not isinstance(raw, dict):
        raise DatasetError(f"{context}: each JSONL line must be an object")
    case_id = raw.get("case_id")
    behavior = raw.get("behavior")
    raw_conditions = raw.get("conditions")
    if not isinstance(case_id, str) or not case_id.strip():
        raise DatasetError(f"{context}: case_id must be a non-empty string")
    if not isinstance(behavior, str) or not behavior.strip():
        raise DatasetError(f"{context}: behavior must be a non-empty string")
    if not isinstance(raw_conditions, dict) or not raw_conditions:
        raise DatasetError(f"{context}: conditions must be a non-empty object")

    conditions: list[Condition] = []
    for name, raw_condition in raw_conditions.items():
        if not isinstance(name, str) or not name.strip():
            raise DatasetError(f"{context}: condition names must be non-empty strings")
        if not isinstance(raw_condition, dict):
            raise DatasetError(f"{context}, condition {name}: expected an object")
        raw_prefix = raw_condition.get("prefix_messages", [])
        raw_turns = raw_condition.get("turns", [])
        if not isinstance(raw_prefix, list):
            raise DatasetError(f"{context}, condition {name}: prefix_messages must be a list")
        if not isinstance(raw_turns, list) or not raw_turns:
            raise DatasetError(f"{context}, condition {name}: turns must be a non-empty list")
        prefix = tuple(
            _validate_message(message, context=f"{context}, condition {name}, prefix")
            for message in raw_prefix
        )
        turns: list[Turn] = []
        for turn_index, turn in enumerate(raw_turns, start=1):
            if not isinstance(turn, dict):
                raise DatasetError(
                    f"{context}, condition {name}, turn {turn_index}: expected an object"
                )
            user = turn.get("user")
            if not isinstance(user, str) or not user.strip():
                raise DatasetError(
                    f"{context}, condition {name}, turn {turn_index}: user must be non-empty"
                )
            turns.append(Turn(user=user))
        conditions.append(Condition(name=name, prefix_messages=prefix, turns=tuple(turns)))

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise DatasetError(f"{context}: metadata must be an object if present")
    return Case(
        case_id=case_id,
        behavior=behavior,
        conditions=tuple(conditions),
        metadata=metadata,
    )


def load_cases(path: str | Path) -> list[Case]:
    case_path = Path(path)
    if not case_path.is_file():
        raise FileNotFoundError(f"Case file not found: {case_path}")
    cases: list[Case] = []
    seen_ids: set[str] = set()
    with case_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            case = parse_case(raw, line_number=line_number)
            if case.case_id in seen_ids:
                raise DatasetError(f"line {line_number}: duplicate case_id {case.case_id!r}")
            seen_ids.add(case.case_id)
            cases.append(case)
    if not cases:
        raise DatasetError(f"No cases found in {case_path}")
    return cases
