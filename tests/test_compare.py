import json

from stateaware.compare import compare_conditions


def test_paired_bootstrap_comparison(tmp_path) -> None:
    payload = {
        "trajectories": [
            {"case_id": "a", "condition": "direct", "last_turn_asr": 0},
            {"case_id": "a", "condition": "candidate", "last_turn_asr": 1},
            {"case_id": "b", "condition": "direct", "last_turn_asr": 0},
            {"case_id": "b", "condition": "candidate", "last_turn_asr": 1},
            {"case_id": "c", "condition": "direct", "last_turn_asr": 1},
            {"case_id": "c", "condition": "candidate", "last_turn_asr": 1},
        ]
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = compare_conditions(
        path, "direct", "candidate", metric="last_turn_asr", bootstrap_samples=1000
    )
    assert result["paired_cases"] == 3
    assert result["mean_paired_difference_candidate_minus_baseline"] == 2 / 3
    assert result["paired_bootstrap_95pct_ci"][0] <= 2 / 3
    assert result["paired_bootstrap_95pct_ci"][1] >= 2 / 3
