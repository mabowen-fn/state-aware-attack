"""Fixed-script multi-turn evaluation runner. No prompt search or mutation."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stateaware.config import Settings
from stateaware.dataset import Case, Condition
from stateaware.judge import HarmBenchJudge, JudgeError
from stateaware.metrics import summarize_events
from stateaware.target import OllamaTarget


def run_experiment(
    cases: list[Case],
    settings: Settings,
    output_dir: str | Path,
    *,
    judge_mode: str = "score",
) -> dict[str, Any]:
    if judge_mode not in {"both", "score", "full", "off"}:
        raise ValueError("judge_mode must be 'both', 'score', 'full', or 'off'")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "events.jsonl"
    target = OllamaTarget(settings)
    judge = HarmBenchJudge(settings) if judge_mode != "off" else None
    start_utc = datetime.now(timezone.utc).isoformat()
    total_steps = sum(len(condition.turns) for case in cases for condition in case.conditions)
    completed_steps = 0
    events: list[dict[str, Any]] = []

    metadata = {
        "started_at_utc": start_utc,
        "target_base_url": settings.target_base_url,
        "target_model_configured": settings.target_model,
        "temperature": settings.target_temperature,
        "max_tokens": settings.target_max_tokens,
        "seed": settings.target_seed,
        "judge_mode": judge_mode,
        "n_cases": len(cases),
        "n_conditions": sum(len(case.conditions) for case in cases),
        "planned_target_turns": total_steps,
        "note": "Fixed pre-specified scripts; no adaptive prompt generation.",
    }
    _write_json(out_dir / "run_metadata.json", metadata)

    try:
        with events_path.open("w", encoding="utf-8") as event_handle:
            for case in cases:
                for condition in case.conditions:
                    messages: list[dict[str, str]] = [dict(message) for message in condition.prefix_messages]
                    for turn_index, turn in enumerate(condition.turns, start=1):
                        messages.append({"role": "user", "content": turn.user})
                        started = time.perf_counter()
                        event: dict[str, Any] = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "case_id": case.case_id,
                            "condition": condition.name,
                            "turn_index": turn_index,
                            "behavior": case.behavior,
                            "user_message": turn.user,
                            "messages_before_generation": len(messages),
                            "status": "ok",
                        }
                        try:
                            reply = target.generate(messages)
                            event.update(
                                {
                                    "generation": reply.text,
                                    "target_model_returned": reply.model,
                                    "latency_seconds": reply.latency_seconds,
                                    "prompt_tokens": reply.prompt_tokens,
                                    "completion_tokens": reply.completion_tokens,
                                    "finish_reason": reply.finish_reason,
                                }
                            )
                            if judge is not None:
                                try:
                                    if judge_mode == "score":
                                        # Canonical endpoint on every turn (useful for simple comparisons).
                                        judge_result = judge.score(
                                            behavior=case.behavior,
                                            generation=reply.text,
                                            mode="score",
                                        )
                                        event["judge"] = judge_result.to_dict()
                                    elif judge_mode in {"full", "both"}:
                                        # Per-turn trajectory signal, including the raw logit difference.
                                        judge_result = judge.score(
                                            behavior=case.behavior,
                                            generation=reply.text,
                                            mode="full",
                                        )
                                        event["judge"] = judge_result.to_dict()
                                        if judge_mode == "both" and turn_index == len(condition.turns):
                                            # The official end-of-condition ASR is always taken from /score.
                                            event["official_judge"] = judge.score(
                                                behavior=case.behavior,
                                                generation=reply.text,
                                                mode="score",
                                            ).to_dict()
                                except JudgeError as exc:
                                    event["judge_error"] = str(exc)
                            messages.append({"role": "assistant", "content": reply.text})
                        except Exception as exc:  # Record failures so the run remains auditable.
                            event["status"] = "error"
                            event["error"] = f"{type(exc).__name__}: {exc}"
                            event["latency_seconds"] = time.perf_counter() - started
                        event["elapsed_seconds"] = time.perf_counter() - started
                        event_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                        event_handle.flush()
                        events.append(event)
                        completed_steps += 1
                        print(
                            f"[{completed_steps}/{total_steps}] {case.case_id} / "
                            f"{condition.name} / turn {turn_index}: {event['status']}"
                        )
    finally:
        target.close()
        if judge is not None:
            judge.close()

    summary = summarize_events(events)
    summary.update(
        {
            "started_at_utc": start_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "planned_target_turns": total_steps,
            "recorded_target_turns": len(events),
            "successful_target_turns": sum(1 for event in events if event.get("status") == "ok"),
            "failed_target_turns": sum(1 for event in events if event.get("status") != "ok"),
            "judge_mode": judge_mode,
            "note": "In both mode, /score_full records turn-wise diagnostics and /score is called on the final turn for canonical ASR. Smoke-corpus scores are connectivity checks only.",
        }
    )
    _write_json(out_dir / "summary.json", summary)
    return summary


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
