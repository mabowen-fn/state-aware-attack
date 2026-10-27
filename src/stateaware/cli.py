"""Command-line interface for running and checking the research scaffold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from stateaware import __version__
from stateaware.compare import compare_conditions
from stateaware.config import Settings
from stateaware.dataset import DatasetError, load_cases
from stateaware.judge import HarmBenchJudge, JudgeError
from stateaware.runner import run_experiment
from stateaware.target import OllamaTarget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stateaware", description="State-aware RA/STAR evaluation")
    parser.add_argument("--version", action="version", version=f"state-aware-ra {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Check target model and judge health connectivity")
    doctor.add_argument("--env-file", default=".env", help="Environment file (default: .env)")

    run = subparsers.add_parser("run", help="Run fixed, pre-specified dialogue conditions")
    run.add_argument("--cases", required=True, help="Input JSONL case file")
    run.add_argument("--out", required=True, help="Output directory")
    run.add_argument(
        "--judge-mode",
        choices=("both", "score", "full", "off"),
        default="both",
        help="both=per-turn /score_full plus final /score; score=/score every turn; full=/score_full only; off=no judge calls",
    )
    run.add_argument("--env-file", default=".env", help="Environment file (default: .env)")

    compare = subparsers.add_parser("compare", help="Paired bootstrap comparison from summary.json")
    compare.add_argument("--summary", required=True, help="Path to summary.json")
    compare.add_argument("--baseline", required=True, help="Baseline condition name")
    compare.add_argument("--candidate", required=True, help="Candidate condition name")
    compare.add_argument(
        "--metric",
        choices=("last_turn_asr", "mean_turn_asr", "last_probability", "mean_probability", "probability_delta", "max_probability"),
        default="last_turn_asr",
    )
    compare.add_argument("--bootstrap-samples", type=int, default=10000)
    compare.add_argument("--seed", type=int, default=17)
    return parser


def _doctor(settings: Settings) -> int:
    problems: list[str] = []
    target = OllamaTarget(settings)
    judge = HarmBenchJudge(settings)
    try:
        try:
            model_ids = target.list_models()
            print(f"Target API: OK ({settings.target_base_url})")
            print(f"Models visible: {', '.join(model_ids) if model_ids else '(none returned)'}")
            if settings.target_model not in model_ids:
                problems.append(
                    f"Configured model {settings.target_model!r} not found in /v1/models response."
                )
        except Exception as exc:
            problems.append(f"Target API failed: {type(exc).__name__}: {exc}")

        try:
            health = judge.health()
            print(f"Judge health: OK ({settings.judge_health_url})")
            print(f"Judge health payload: {json.dumps(health, ensure_ascii=False)}")
        except JudgeError as exc:
            problems.append(str(exc))
    finally:
        target.close()
        judge.close()

    if problems:
        print("\nConnectivity problems:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("\nAll configured services are reachable.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            report = compare_conditions(
                args.summary,
                baseline_name=args.baseline,
                candidate_name=args.candidate,
                metric=args.metric,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
            )
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return 0

        settings = Settings.load(args.env_file)
        if args.command == "doctor":
            return _doctor(settings)
        if args.command == "run":
            cases = load_cases(args.cases)
            summary = run_experiment(cases, settings, args.out, judge_mode=args.judge_mode)
            print(f"\nWrote outputs under: {Path(args.out).resolve()}")
            print(json.dumps(summary.get("conditions", []), indent=2, ensure_ascii=False))
            return 0 if summary.get("failed_target_turns", 0) == 0 else 2
    except (DatasetError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        print(f"Runtime error: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
