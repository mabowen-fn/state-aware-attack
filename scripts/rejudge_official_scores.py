"""Recover missing canonical /score judgments without rerunning target-model generations.

This reads an existing events.jsonl, retries the official HarmBench /score endpoint for
final-turn generations that lack an official_judge record, and writes a new run directory.
The source run is never modified.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateaware.config import Settings
from stateaware.judge import HarmBenchJudge, JudgeError
from stateaware.metrics import summarize_events


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Events file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_no} must contain a JSON object")
            rows.append(row)
    return rows


def rejudge(
    events: list[dict[str, Any]],
    settings: Settings,
    *,
    max_retries: int = 3,
    retry_delay_seconds: float = 1.0,
) -> dict[str, Any]:
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")
    # Determine final logged turn per case/condition; do not assume all conditions
    # have the same number of turns.
    final_indices: dict[tuple[str, str], int] = {}
    for i, event in enumerate(events):
        key = (str(event.get("case_id", "")), str(event.get("condition", "")))
        if not all(key):
            continue
        prior_i = final_indices.get(key)
        if prior_i is None or int(event.get("turn_index", 0)) >= int(events[prior_i].get("turn_index", 0)):
            final_indices[key] = i

    judge = HarmBenchJudge(settings)
    attempted = 0
    recovered = 0
    skipped_existing = 0
    not_repairable = 0
    failures: list[dict[str, Any]] = []
    try:
        for (case_id, condition), idx in sorted(final_indices.items()):
            event = events[idx]
            official = event.get("official_judge")
            if isinstance(official, dict) and official.get("asr") in (0, 1):
                skipped_existing += 1
                continue
            # A score-mode run may already have a canonical score in `judge`.
            current = event.get("judge")
            if (isinstance(current, dict)
                    and current.get("endpoint_mode") == "score"
                    and current.get("asr") in (0, 1)):
                event["official_judge"] = current
                event["official_judge_recovery"] = {
                    "status": "promoted_existing_score",
                    "attempts": 0,
                    "recovered_at_utc": _now(),
                }
                recovered += 1
                continue
            behavior = event.get("behavior")
            generation = event.get("generation")
            if (event.get("status") != "ok"
                    or not isinstance(behavior, str) or not behavior.strip()
                    or not isinstance(generation, str) or not generation.strip()):
                not_repairable += 1
                failures.append({
                    "case_id": case_id,
                    "condition": condition,
                    "reason": "final event lacks successful generation, behavior, or status",
                })
                continue

            attempted += 1
            last_error: str | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    result = judge.score(behavior=behavior, generation=generation, mode="score")
                    if result.asr not in (0, 1):
                        raise JudgeError(f"Canonical endpoint returned invalid/missing ASR: {result.asr!r}")
                    event["official_judge"] = result.to_dict()
                    event["official_judge_recovery"] = {
                        "status": "recovered",
                        "attempts": attempt,
                        "recovered_at_utc": _now(),
                        "endpoint": settings.judge_score_url,
                    }
                    recovered += 1
                    last_error = None
                    break
                except Exception as exc:  # record network and payload failures alike
                    last_error = f"{type(exc).__name__}: {exc}"
                    if attempt < max_retries:
                        time.sleep(retry_delay_seconds * attempt)
            if last_error is not None:
                event["official_judge_recovery"] = {
                    "status": "failed",
                    "attempts": max_retries,
                    "last_error": last_error,
                    "recovered_at_utc": _now(),
                    "endpoint": settings.judge_score_url,
                }
                failures.append({
                    "case_id": case_id,
                    "condition": condition,
                    "reason": last_error,
                })
    finally:
        judge.close()

    return {
        "final_trajectories": len(final_indices),
        "attempted_endpoint_calls": attempted,
        "recovered": recovered,
        "already_scored": skipped_existing,
        "not_repairable": not_repairable,
        "failed": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Existing run directory with events.jsonl")
    parser.add_argument("--out-dir", required=True, help="New output directory; source is not modified")
    parser.add_argument("--env-file", default=".env", help="Environment file for judge endpoint settings")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing files in out-dir")
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.out_dir)
    if source.resolve() == output.resolve():
        parser.error("--out-dir must differ from --input-dir; source runs are immutable")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        parser.error(f"Output directory {output} is not empty; choose a new path or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)

    events = _read_jsonl(source / "events.jsonl")
    settings = Settings.load(args.env_file)
    report = rejudge(
        events,
        settings,
        max_retries=args.max_retries,
        retry_delay_seconds=args.retry_delay_seconds,
    )
    finished_at = _now()

    with (output / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    summary = summarize_events(events)
    summary.update({
        "rejudged_at_utc": finished_at,
        "source_run": str(source.resolve()),
        "rejudge_report": report,
        "judge_mode": "rejudged_official_scores",
        "note": "Target generations reused from source run. Missing canonical /score results were retried; source run was not modified.",
        "successful_target_turns": sum(1 for e in events if e.get("status") == "ok"),
        "failed_target_turns": sum(1 for e in events if e.get("status") != "ok"),
    })
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    metadata_path = source / "run_metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "rejudged_at_utc": finished_at,
        "rejudged_from": str(source.resolve()),
        "rejudge_report": report,
        "canonical_judge_endpoint": settings.judge_score_url,
    })
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote repaired run under: {output.resolve()}")
    conditions = summary.get("conditions", [])
    print(json.dumps(conditions, indent=2, ensure_ascii=False))
    return 0 if report["failed"] == 0 and report["not_repairable"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
