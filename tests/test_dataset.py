import json

import pytest

from stateaware.dataset import DatasetError, load_cases, parse_case


def test_parse_minimal_case() -> None:
    case = parse_case(
        {
            "case_id": "x",
            "behavior": "safe benchmark label",
            "conditions": {"direct": {"turns": [{"user": "Hello"}]}},
        },
        line_number=1,
    )
    assert case.case_id == "x"
    assert case.conditions[0].name == "direct"
    assert case.conditions[0].turns[0].user == "Hello"


def test_reject_invalid_role() -> None:
    with pytest.raises(DatasetError):
        parse_case(
            {
                "case_id": "x",
                "behavior": "label",
                "conditions": {
                    "bad": {
                        "prefix_messages": [{"role": "tool", "content": "hi"}],
                        "turns": [{"user": "hello"}],
                    }
                },
            },
            line_number=1,
        )


def test_load_smoke_cases(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    item = {
        "case_id": "one",
        "behavior": "safe",
        "conditions": {"direct": {"turns": [{"user": "hi"}]}},
    }
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    assert len(load_cases(path)) == 1


def test_reject_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    item = {
        "case_id": "one",
        "behavior": "safe",
        "conditions": {"direct": {"turns": [{"user": "hi"}]}},
    }
    line = json.dumps(item)
    path.write_text(line + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate case_id"):
        load_cases(path)
