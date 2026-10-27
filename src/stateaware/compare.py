"""Paired, case-level bootstrap comparisons from summary.json outputs."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


ALLOWED_METRICS = {
    "last_turn_asr",
    "mean_turn_asr",
    "last_probability",
    "mean_probability",
    "probability_delta",
    "max_probability",
}


def compare_conditions(
    summary_path: str | Path,
    baseline_name: str,
    candidate_name: str,
    metric: str = "last_turn_asr",
    bootstrap_samples: int = 10_000,
    seed: int = 17,
) -> dict[str, Any]:
    if metric not in ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(ALLOWED_METRICS)}")
    if bootstrap_samples < 100:
        raise ValueError("bootstrap_samples must be at least 100")
    data = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    by_condition: dict[str, dict[str, dict[str, Any]]] = {}
    for row in data.get("trajectories", []):
        condition = str(row.get("condition"))
        case_id = str(row.get("case_id"))
        by_condition.setdefault(condition, {})[case_id] = row
    if baseline_name not in by_condition:
        raise ValueError(f"Baseline condition {baseline_name!r} not found in summary")
    if candidate_name not in by_condition:
        raise ValueError(f"Candidate condition {candidate_name!r} not found in summary")

    base_rows = by_condition[baseline_name]
    cand_rows = by_condition[candidate_name]
    common_ids = sorted(set(base_rows).intersection(cand_rows))
    pairs: list[tuple[float, float]] = []
    excluded: list[str] = []
    for case_id in common_ids:
        base_value = base_rows[case_id].get(metric)
        cand_value = cand_rows[case_id].get(metric)
        if not isinstance(base_value, (int, float)) or not isinstance(cand_value, (int, float)):
            excluded.append(case_id)
            continue
        pairs.append((float(base_value), float(cand_value)))
    if not pairs:
        raise ValueError("No paired numeric observations found for the selected metric")

    differences = [candidate - baseline for baseline, candidate in pairs]
    observed = sum(differences) / len(differences)
    rng = random.Random(seed)
    n = len(differences)
    boot_means: list[float] = []
    for _ in range(bootstrap_samples):
        sample = [differences[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lower = _percentile(boot_means, 0.025)
    upper = _percentile(boot_means, 0.975)

    baseline_mean = sum(pair[0] for pair in pairs) / n
    candidate_mean = sum(pair[1] for pair in pairs) / n
    return {
        "baseline_condition": baseline_name,
        "candidate_condition": candidate_name,
        "metric": metric,
        "metric_definition": _metric_definition(metric),
        "paired_cases": n,
        "excluded_case_ids": excluded,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "mean_paired_difference_candidate_minus_baseline": observed,
        "paired_bootstrap_95pct_ci": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "seed": seed,
        "interpretation": "Positive differences favor the candidate only for metrics where higher means more of the measured score; assess whether that is desirable for the chosen metric.",
    }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot calculate percentile from empty values")
    pos = (len(sorted_values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = pos - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _metric_definition(metric: str) -> str:
    return {
        "last_turn_asr": "judge ASR on the final scored turn (primary binary endpoint)",
        "mean_turn_asr": "mean per-turn judge ASR within each case trajectory",
        "last_probability": "judge probability-like score on the final scored turn",
        "mean_probability": "mean judge probability-like score across turns",
        "probability_delta": "last observed score minus first observed score",
        "max_probability": "maximum judge probability-like score within the trajectory",
    }[metric]
