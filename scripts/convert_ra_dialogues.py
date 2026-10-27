#!/usr/bin/env python3
"""Convert released Response-Attack dialogue JSON files to StateAware JSONL.

The converter does not generate or mutate attack prompts. It imports the
pre-generated round structure from the official RA release and adds a direct
condition so each imported item can be evaluated as a paired case.

Because the HarmBench judge needs the original benchmark behavior, use a
behavior map when the source JSON does not contain an explicit `behavior` or
`original_behavior` field. The map can be a CSV (recommended) or JSON object.

Expected RA source structure (per official generate_model_response.py):
{
  "rounds": [
    {"promptParts": [{"text": "..."}], "response": "..."},
    {"promptParts": [{"text": "final prompt ..."}]}
  ]
}
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable


VARIANT_PATTERN = re.compile(r"^(?P<base>.+)_(?P<variant>v2|v3)$", re.IGNORECASE)
BEHAVIOR_FIELDS = (
    "behavior",
    "original_behavior",
    "originalHarmfulQuery",  # field used by the official Response-Attack release
    "original_query",
    "original_goal",
)
BEHAVIOR_ID_FIELDS = ("BehaviorID", "behavior_id", "behaviorId")
MAP_KEY_FIELDS = (
    "filename",
    "file",
    "source_file",
    "case_id",
    "stem",
    "base_stem",
    "BehaviorID",
    "behavior_id",
    "behaviorId",
    "id",
)
MAP_BEHAVIOR_FIELDS = ("behavior", "Behavior", "original_behavior", "goal", "Goal")


class ConversionError(ValueError):
    """Raised when an RA source file cannot be safely converted."""


def _nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _source_behavior(raw: dict[str, Any]) -> str | None:
    """Only use explicit behavior metadata; do not guess from paraphrased rounds."""
    for key in BEHAVIOR_FIELDS:
        candidate = _nonempty_string(raw.get(key))
        if candidate:
            return candidate
    return None


def _behavior_id(raw: dict[str, Any]) -> str | None:
    for key in BEHAVIOR_ID_FIELDS:
        candidate = _nonempty_string(raw.get(key))
        if candidate:
            return candidate
    return None


def _base_stem(stem: str) -> str:
    match = VARIANT_PATTERN.match(stem)
    return match.group("base") if match else stem


def _variant(stem: str) -> str:
    match = VARIANT_PATTERN.match(stem)
    return match.group("variant").lower() if match else "base"


def _source_files(input_dir: Path, include_v2v3: bool) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"RA input directory does not exist: {input_dir}")
    files = sorted(path for path in input_dir.glob("*.json") if path.is_file())
    if not include_v2v3:
        files = [path for path in files if _variant(path.stem) == "base"]
    if not files:
        suffix = " (including _v2/_v3 variants)" if include_v2v3 else " (base .json files only)"
        raise ConversionError(f"No RA JSON files found in {input_dir}{suffix}.")
    return files


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConversionError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}") from exc


def _load_behavior_map(path: Path | None) -> dict[str, str]:
    """Load a CSV map or JSON key->behavior map.

    For CSV, accepted key columns include filename/stem/BehaviorID and accepted
    behavior columns include behavior/Behavior/original_behavior. Values are
    indexed by all non-empty key columns in each row.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Behavior map not found: {path}")

    mapping: dict[str, str] = {}
    if path.suffix.lower() == ".json":
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ConversionError("JSON behavior map must be an object mapping keys to behavior strings.")
        for key, value in payload.items():
            k = _nonempty_string(key)
            v = _nonempty_string(value)
            if k and v:
                mapping[k] = v
        return mapping

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ConversionError(f"Behavior map CSV has no header: {path}")
        key_fields = [field for field in MAP_KEY_FIELDS if field in reader.fieldnames]
        behavior_field = next((field for field in MAP_BEHAVIOR_FIELDS if field in reader.fieldnames), None)
        if not key_fields or behavior_field is None:
            raise ConversionError(
                "Behavior map CSV needs at least one key column such as filename, stem, or "
                "BehaviorID, plus a behavior column (behavior or Behavior). "
                f"Found columns: {reader.fieldnames}"
            )
        for line_no, row in enumerate(reader, start=2):
            behavior = _nonempty_string(row.get(behavior_field))
            if not behavior:
                continue
            for field in key_fields:
                key = _nonempty_string(row.get(field))
                if key:
                    mapping[key] = behavior
                    # File keys are also accepted without their suffix.
                    if key.lower().endswith(".json"):
                        mapping[Path(key).stem] = behavior
    return mapping


def _lookup_behavior(
    path: Path,
    raw: dict[str, Any],
    behavior_map: dict[str, str],
) -> str | None:
    keys: list[str] = [path.name, path.stem, _base_stem(path.stem)]
    source_id = _behavior_id(raw)
    if source_id:
        keys.append(source_id)
    for key in keys:
        candidate = behavior_map.get(key)
        if candidate:
            return candidate.strip()
    return _source_behavior(raw)


def _get_prompt(round_data: Any, *, path: Path, round_index: int) -> str:
    context = f"{path.name}: rounds[{round_index}]"
    if not isinstance(round_data, dict):
        raise ConversionError(f"{context} must be an object")

    parts = round_data.get("promptParts")
    if isinstance(parts, list) and parts:
        first = parts[0]
        if isinstance(first, dict):
            candidate = _nonempty_string(first.get("text"))
            if candidate:
                return candidate

    # A few mirrors/export formats use these names instead. The official
    # repository uses promptParts[0].text; these are conservative fallbacks.
    for key in ("prompt", "user"):
        candidate = _nonempty_string(round_data.get(key))
        if candidate:
            return candidate

    raise ConversionError(
        f"{context} has no non-empty promptParts[0].text (or fallback prompt/user field)"
    )


def _read_rounds(path: Path, raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ConversionError(f"{path.name}: top-level JSON value must be an object")
    rounds = raw.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise ConversionError(f"{path.name}: expected a non-empty 'rounds' list")
    # Validate every prompt now, before writing anything.
    for index, round_data in enumerate(rounds):
        _get_prompt(round_data, path=path, round_index=index)
    return rounds


def _make_template(files: Iterable[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for path in files:
        try:
            raw = _load_json(path)
            source_id = _behavior_id(raw) if isinstance(raw, dict) else None
            embedded = _source_behavior(raw) if isinstance(raw, dict) else None
        except ConversionError:
            source_id = None
            embedded = None
        rows.append(
            {
                "filename": path.name,
                "stem": path.stem,
                "base_stem": _base_stem(path.stem),
                "BehaviorID": source_id or "",
                "behavior": embedded or "",
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "stem", "base_stem", "BehaviorID", "behavior"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _convert_one(path: Path, behavior_map: dict[str, str]) -> dict[str, Any]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ConversionError(f"{path.name}: top-level JSON value must be an object")
    rounds = _read_rounds(path, raw)
    behavior = _lookup_behavior(path, raw, behavior_map)
    if not behavior:
        raise ConversionError(
            f"{path.name}: original benchmark behavior is missing. Fill `behavior` in a behavior-map "
            "CSV or add an explicit `behavior`/`original_behavior` field to the source file. "
            "Do not use a paraphrased primer as the judge behavior."
        )

    prefix: list[dict[str, str]] = []
    for index, round_data in enumerate(rounds[:-1]):
        prompt = _get_prompt(round_data, path=path, round_index=index)
        prefix.append({"role": "user", "content": prompt})
        response = _nonempty_string(round_data.get("response"))
        if response:
            prefix.append({"role": "assistant", "content": response})

    assistant_prefix_count = sum(1 for message in prefix if message["role"] == "assistant")
    if assistant_prefix_count == 0:
        raise ConversionError(
            f"{path.name}: no historical assistant response was found in rounds[:-1]; "
            "this does not appear to be a complete pre-generated RA dialogue."
        )

    final_prompt = _get_prompt(rounds[-1], path=path, round_index=len(rounds) - 1)
    base_id = _base_stem(path.stem)
    variant = _variant(path.stem)

    return {
        "case_id": path.stem,
        "behavior": behavior,
        "conditions": {
            "direct": {
                "prefix_messages": [],
                "turns": [{"user": behavior}],
            },
            "ra_fixed": {
                "prefix_messages": prefix,
                "turns": [{"user": final_prompt}],
            },
        },
        "metadata": {
            "split": "pilot",
            "source": "official-response-attack-dialogue",
            "source_file": path.name,
            "base_case_id": base_id,
            "ra_variant": variant,
            "source_round_count": len(rounds),
            "historical_assistant_messages": assistant_prefix_count,
            "note": "Pre-generated RA history; direct and RA conditions are paired by case_id.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert released RA dialogue JSON files to StateAware experiment JSONL."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory such as external/Response-Attack/data/dialogues/harmbench/dri",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ra_star_pilot.jsonl"),
        help="Output JSONL path (default: data/ra_star_pilot.jsonl)",
    )
    parser.add_argument(
        "--behavior-map",
        type=Path,
        help="CSV or JSON map from filename/stem/BehaviorID to original benchmark behavior",
    )
    parser.add_argument(
        "--make-map-template",
        type=Path,
        help="Write a CSV template for mapping RA files to original benchmark behaviors, then exit",
    )
    parser.add_argument(
        "--include-v2v3",
        action="store_true",
        help="Include _v2.json and _v3.json variants; base files only by default",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Deterministically sample this many files after sorting (seeded by --seed)",
    )
    parser.add_argument("--seed", type=int, default=17, help="Sampling seed (default: 17)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate all selected source files and behavior mappings without writing JSONL",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        files = _source_files(args.input_dir, args.include_v2v3)
        if args.sample_size is not None:
            if args.sample_size < 1:
                raise ConversionError("--sample-size must be at least 1")
            if args.sample_size > len(files):
                raise ConversionError(
                    f"--sample-size {args.sample_size} exceeds available files ({len(files)})"
                )
            rng = random.Random(args.seed)
            files = sorted(rng.sample(files, args.sample_size))

        if args.make_map_template:
            count = _make_template(files, args.make_map_template)
            print(f"Wrote behavior-map template with {count} rows: {args.make_map_template}")
            print("The official RA source field `originalHarmfulQuery` is detected automatically. Fill the behavior column only for source files that lack an explicit original behavior.")
            return 0

        behavior_map = _load_behavior_map(args.behavior_map)
        converted: list[dict[str, Any]] = []
        errors: list[str] = []
        for path in files:
            try:
                converted.append(_convert_one(path, behavior_map))
            except ConversionError as exc:
                errors.append(str(exc))

        if errors:
            print("Conversion validation failed; no JSONL was written:", file=sys.stderr)
            for error in errors[:30]:
                print(f"- {error}", file=sys.stderr)
            if len(errors) > 30:
                print(f"- ... and {len(errors) - 30} more errors", file=sys.stderr)
            print(
                "Hint: the official RA format uses `originalHarmfulQuery`, which is auto-detected. For exports without an explicit original behavior, generate a map template using --make-map-template, fill the exact original behavior, and rerun with --behavior-map.",
                file=sys.stderr,
            )
            return 2

        if args.check_only:
            print(f"Validated {len(converted)} RA dialogue(s); no output written.")
            return 0

        if args.output.exists() and not args.overwrite:
            raise ConversionError(
                f"Output already exists: {args.output}. Use --overwrite to replace it."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for record in converted:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

        print(f"Converted {len(converted)} dialogue(s) to {args.output}")
        print("Conditions per case: direct, ra_fixed")
        print("Note: --include-v2v3 treats variants as separate case_ids; use base files for the primary paired comparison.")
        return 0

    except (ConversionError, FileNotFoundError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
