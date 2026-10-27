"""Trajectory-level summaries for repeated per-turn judge measurements."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("status") == "ok":
            grouped[(str(event["case_id"]), str(event["condition"]))].append(event)

    trajectories: list[dict[str, Any]] = []
    for (case_id, condition), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["turn_index"]))
        probabilities = [
            float(row["judge"]["probability"])
            for row in rows
            if isinstance(row.get("judge"), dict)
            and row["judge"].get("probability") is not None
        ]
        asr_values = [
            int(row["judge"]["asr"])
            for row in rows
            if isinstance(row.get("judge"), dict) and row["judge"].get("asr") in (0, 1)
        ]
        final_row = rows[-1]
        official_judge = final_row.get("official_judge")
        if not isinstance(official_judge, dict):
            final_turn_judge = final_row.get("judge")
            if isinstance(final_turn_judge, dict) and final_turn_judge.get("endpoint_mode") == "score":
                official_judge = final_turn_judge
        official_asr = (
            int(official_judge["asr"])
            if isinstance(official_judge, dict) and official_judge.get("asr") in (0, 1)
            else None
        )
        trajectory: dict[str, Any] = {
            "case_id": case_id,
            "condition": condition,
            "turns_completed": len(rows),
            "turns_scored_probability": len(probabilities),
            "turns_scored_asr": len(asr_values),
            "mean_probability": _mean(probabilities),
            "first_probability": probabilities[0] if probabilities else None,
            "last_probability": probabilities[-1] if probabilities else None,
            "probability_delta": (
                probabilities[-1] - probabilities[0] if len(probabilities) >= 2 else None
            ),
            "max_probability": max(probabilities) if probabilities else None,
            "first_turn_asr": asr_values[0] if asr_values else None,
            "last_turn_full_asr": asr_values[-1] if asr_values else None,
            "last_turn_asr": official_asr,
            "mean_turn_asr": _mean([float(value) for value in asr_values]),
            "official_asr": official_asr,
            "any_turn_asr": max(asr_values) if asr_values else None,
            "total_latency_seconds": sum(float(row.get("latency_seconds", 0.0)) for row in rows),
            "total_prompt_tokens": _sum_optional_int(row.get("prompt_tokens") for row in rows),
            "total_completion_tokens": _sum_optional_int(
                row.get("completion_tokens") for row in rows
            ),
            "judge_versions": sorted(
                {
                    str(row["judge"].get("version"))
                    for row in rows
                    if isinstance(row.get("judge"), dict) and row["judge"].get("version")
                }
            ),
        }
        trajectories.append(trajectory)

    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        conditions[trajectory["condition"]].append(trajectory)
    condition_summary: list[dict[str, Any]] = []
    for condition, rows in sorted(conditions.items()):
        scored_probability = [
            float(row["mean_probability"])
            for row in rows
            if row["mean_probability"] is not None
        ]
        asr = [float(row["mean_turn_asr"]) for row in rows if row["mean_turn_asr"] is not None]
        official_asr = [float(row["official_asr"]) for row in rows if row["official_asr"] is not None]
        deltas = [
            float(row["probability_delta"])
            for row in rows
            if row["probability_delta"] is not None
        ]
        condition_summary.append(
            {
                "condition": condition,
                "n_trajectories": len(rows),
                "mean_trajectory_probability": _mean(scored_probability),
                "mean_turn_asr": _mean(asr),
                "official_final_asr": _mean(official_asr),
                "n_official_final_asr": len(official_asr),
                "mean_probability_delta": _mean(deltas),
            }
        )
    return {"trajectories": trajectories, "conditions": condition_summary}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sum_optional_int(values: Iterable[Any]) -> int | None:
    materialized = [int(value) for value in values if isinstance(value, (int, float))]
    return sum(materialized) if materialized else None
