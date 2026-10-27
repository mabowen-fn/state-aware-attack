#!/usr/bin/env python3
"""Extract pre-generation hidden-state vectors for a fixed RA/STAR evaluation.

This script is diagnostic only: it does not create, rewrite, rank, or optimize prompts.
It replays the exact recorded context up to each scored response and records the hidden
state at the last input token (the assistant-generation boundary), before the answer exists.

The feature model must match the evaluated target for target-specific claims. Ollama's
OpenAI-compatible API does not expose activations, so a Hugging Face checkpoint produces
cross-model exploratory features unless the experiment itself is rerun with that checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL row must be an object")
            rows.append(value)
    return rows


def message_hash(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_prompt_messages(
    case: dict[str, Any],
    condition_name: str,
    turn_index: int,
    event_index: dict[tuple[str, str, int], dict[str, Any]],
) -> list[dict[str, str]]:
    """Reconstruct all messages visible immediately before this turn's generation."""
    condition = case["conditions"][condition_name]
    turns = condition["turns"]
    if turn_index < 1 or turn_index > len(turns):
        raise ValueError(
            f"turn_index {turn_index} outside 1..{len(turns)} for "
            f"{case.get('case_id')}/{condition_name}"
        )

    messages = [
        {"role": item["role"], "content": item["content"]}
        for item in condition.get("prefix_messages", [])
    ]
    case_id = str(case["case_id"])
    for idx, turn in enumerate(turns[:turn_index], start=1):
        messages.append({"role": "user", "content": turn["user"]})
        if idx < turn_index:
            prior = event_index.get((case_id, condition_name, idx))
            if prior is None or prior.get("status") != "ok" or not prior.get("generation"):
                raise ValueError(
                    f"Missing successful prior generation for {case_id}/{condition_name}/turn {idx}"
                )
            messages.append({"role": "assistant", "content": str(prior["generation"])})
    return messages


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but PyTorch MPS is not available")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Input case JSONL")
    parser.add_argument("--events", type=Path, required=True, help="Recorded events.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="Output .npz feature file")
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Hugging Face model ID or local model directory (gated access may be required)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto"
    )
    parser.add_argument(
        "--label-source",
        choices=("official", "full"),
        default="official",
        help="Use /score official_judge labels (final turn only) or /score_full judge labels",
    )
    parser.add_argument("--condition", default=None, help="Optional condition filter, e.g. ra_fixed")
    parser.add_argument("--limit", type=int, default=None, help="Optional first-N eligible rows")
    parser.add_argument(
        "--trust-remote-code", action="store_true", help="Allow custom Hugging Face model code"
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset not found: {args.dataset}")
    if not args.events.is_file():
        parser.error(f"events file not found: {args.events}")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            "Missing research dependencies. Install with: "
            "uv pip install 'torch>=2.4' 'transformers>=4.45' 'accelerate>=1.0' 'numpy>=1.26'",
            file=sys.stderr,
        )
        print(f"Import error: {exc}", file=sys.stderr)
        return 2

    dataset_rows = read_jsonl(args.dataset)
    cases = {str(row.get("case_id")): row for row in dataset_rows}
    if len(cases) != len(dataset_rows):
        raise ValueError("Dataset contains duplicate case_id values")

    events = read_jsonl(args.events)
    event_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for event in events:
        try:
            key = (str(event["case_id"]), str(event["condition"]), int(event["turn_index"]))
        except (KeyError, TypeError, ValueError):
            continue
        event_index[key] = event

    eligible: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    skipped: dict[str, int] = {}
    for event in events:
        case_id = str(event.get("case_id", ""))
        condition_name = str(event.get("condition", ""))
        if args.condition and condition_name != args.condition:
            continue
        case = cases.get(case_id)
        if case is None:
            skipped["case_not_in_dataset"] = skipped.get("case_not_in_dataset", 0) + 1
            continue
        if event.get("status") != "ok" or not isinstance(event.get("generation"), str):
            skipped["target_event_failed"] = skipped.get("target_event_failed", 0) + 1
            continue
        label_obj = event.get("official_judge") if args.label_source == "official" else event.get("judge")
        if not isinstance(label_obj, dict) or label_obj.get("asr") not in (0, 1):
            skipped["label_missing"] = skipped.get("label_missing", 0) + 1
            continue
        if condition_name not in case.get("conditions", {}):
            skipped["condition_not_in_dataset"] = skipped.get("condition_not_in_dataset", 0) + 1
            continue
        eligible.append((case, event, label_obj))

    eligible.sort(key=lambda triple: (
        str(triple[1].get("case_id")), str(triple[1].get("condition")), int(triple[1].get("turn_index", 0))
    ))
    if args.limit is not None:
        eligible = eligible[: args.limit]
    if not eligible:
        raise ValueError(
            "No eligible scored events found. With --label-source official, events need an "
            "official_judge object from /score, usually on the final turn."
        )

    device = choose_device(torch, args.device)
    if args.dtype == "auto":
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]

    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=hf_token, trust_remote_code=args.trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=hf_token,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    probabilities: list[float] = []
    case_ids: list[str] = []
    conditions: list[str] = []
    turn_indices: list[int] = []
    prompt_lengths: list[int] = []
    prompt_hashes: list[str] = []
    target_model_names: set[str] = set()

    for row_no, (case, event, label_obj) in enumerate(eligible, start=1):
        case_id = str(event["case_id"])
        condition_name = str(event["condition"])
        turn_index = int(event["turn_index"])
        messages = make_prompt_messages(case, condition_name, turn_index, event_index)
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor([input_ids], dtype=torch.long)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(device)
        attention_mask = torch.ones_like(input_ids, device=device)

        with torch.inference_mode():
            result = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = result.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden_states despite output_hidden_states=True")
        # The final input position is the state used to predict the first generated token.
        # Stack embedding output + every transformer layer at that position.
        vector = torch.stack([layer[0, -1, :].detach().float().cpu() for layer in hidden_states])
        feature_rows.append(vector.numpy().astype(np.float16, copy=False))
        labels.append(int(label_obj["asr"]))
        probabilities.append(float(label_obj["probability"]) if label_obj.get("probability") is not None else np.nan)
        case_ids.append(case_id)
        conditions.append(condition_name)
        turn_indices.append(turn_index)
        prompt_lengths.append(int(input_ids.shape[-1]))
        prompt_hashes.append(message_hash(messages))
        if event.get("target_model_returned"):
            target_model_names.add(str(event["target_model_returned"]))
        print(f"[{row_no}/{len(eligible)}] extracted state: {case_id} / {condition_name} / turn {turn_index}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        features=np.stack(feature_rows, axis=0),
        labels=np.asarray(labels, dtype=np.int8),
        judge_probabilities=np.asarray(probabilities, dtype=np.float32),
        case_ids=np.asarray(case_ids, dtype="U128"),
        conditions=np.asarray(conditions, dtype="U64"),
        turn_indices=np.asarray(turn_indices, dtype=np.int16),
        prompt_lengths=np.asarray(prompt_lengths, dtype=np.int32),
        prompt_hashes=np.asarray(prompt_hashes, dtype="U64"),
    )

    target_names = sorted(target_model_names)
    metadata = {
        "model_requested": args.model,
        "model_config_name_or_path": getattr(model.config, "_name_or_path", None),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "n_examples": len(feature_rows),
        "feature_shape": list(np.stack(feature_rows, axis=0).shape),
        "label_source": args.label_source,
        "label_counts": {"0": int(sum(label == 0 for label in labels)), "1": int(sum(label == 1 for label in labels))},
        "conditions": sorted(set(conditions)),
        "logged_target_models": target_names,
        "same_model_identity_verified": False,
        "interpretation_warning": (
            "The recorded outcomes were generated by the model named in events.jsonl, often Ollama. "
            "This script extracts states from the separately loaded Hugging Face model. Unless the "
            "weights, tokenizer, chat template, and quantization path are proven equivalent, these "
            "features are exploratory cross-model correlates, not hidden states of the Ollama run. "
            "For a target-specific claim, regenerate and score the fixed conditions through the same "
            "instrumented checkpoint before analysis."
        ),
        "skipped_event_counts": skipped,
        "feature_definition": "All-layer hidden vectors at the last input token with add_generation_prompt=True, before the response is generated.",
        "input_dataset": str(args.dataset),
        "input_events": str(args.events),
    }
    metadata_path = args.out.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved features: {args.out}")
    print(f"Saved metadata: {metadata_path}")
    print("Label counts:", metadata["label_counts"])
    print("WARNING:", metadata["interpretation_warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
