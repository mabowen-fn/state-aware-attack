from math import isclose

from stateaware.metrics import summarize_events


def test_probability_delta_and_asr_aggregation() -> None:
    events = [
        {
            "status": "ok", "case_id": "c1", "condition": "direct", "turn_index": 1,
            "latency_seconds": 1.0, "prompt_tokens": 10, "completion_tokens": 5,
            "judge": {"endpoint_mode": "score", "probability": 0.2, "asr": 0, "version": "v1"},
        },
        {
            "status": "ok", "case_id": "c1", "condition": "direct", "turn_index": 2,
            "latency_seconds": 2.0, "prompt_tokens": 15, "completion_tokens": 6,
            "judge": {"endpoint_mode": "score", "probability": 0.7, "asr": 1, "version": "v1"},
        },
    ]
    result = summarize_events(events)
    trajectory = result["trajectories"][0]
    assert isclose(trajectory["probability_delta"], 0.5)
    assert isclose(trajectory["mean_probability"], 0.45)
    assert isclose(trajectory["mean_turn_asr"], 0.5)
    assert trajectory["last_turn_asr"] == 1
    assert trajectory["first_turn_asr"] == 0
    assert trajectory["total_prompt_tokens"] == 25
