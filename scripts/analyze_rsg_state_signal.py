#!/usr/bin/env python3
"""Run an exploratory, case-grouped test of whether pre-generation states predict ASR.

This script evaluates a fixed diagnostic question. It does not select or generate prompts.
By default it filters to `ra_fixed` to avoid the trivial confound that condition identity
alone separates direct queries from primed contexts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="Output .npz from extractor")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON summary path")
    parser.add_argument(
        "--condition", default="ra_fixed", help="Condition to analyze; default isolates ra_fixed"
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Hidden-state index to evaluate (default -1 = final layer; index 0 is embeddings)",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--c", type=float, default=0.1, help="L2 logistic-regression C")
    args = parser.parse_args()

    if not args.features.is_file():
        parser.error(f"features file not found: {args.features}")
    if args.folds < 2:
        parser.error("--folds must be >= 2")

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            balanced_accuracy_score,
            brier_score_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        parser.error(
            "Missing analysis dependencies. Install with: "
            "uv pip install 'numpy>=1.26' 'scikit-learn>=1.5'"
        )

    with np.load(args.features, allow_pickle=False) as payload:
        features = payload["features"].astype(np.float32)
        labels = payload["labels"].astype(np.int64)
        case_ids = payload["case_ids"].astype(str)
        conditions = payload["conditions"].astype(str)
        prompt_lengths = payload["prompt_lengths"].astype(np.float32)

    mask = np.ones(labels.shape[0], dtype=bool)
    if args.condition:
        mask &= conditions == args.condition
    features = features[mask]
    labels = labels[mask]
    case_ids = case_ids[mask]
    conditions = conditions[mask]
    prompt_lengths = prompt_lengths[mask]

    if features.ndim != 3:
        raise ValueError(f"Expected features shape [examples, layers, hidden], got {features.shape}")
    if features.shape[0] < args.folds * 2:
        raise ValueError(f"Only {features.shape[0]} examples after filtering; too few for {args.folds} folds")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"Need both ASR labels 0 and 1; got {np.unique(labels).tolist()}")
    if len(set(case_ids)) < args.folds:
        raise ValueError("Not enough distinct case IDs for grouped cross-validation")

    n_layers = features.shape[1]
    layer_index = args.layer_index if args.layer_index >= 0 else n_layers + args.layer_index
    if layer_index < 0 or layer_index >= n_layers:
        raise ValueError(f"layer-index {args.layer_index} invalid for {n_layers} stored states")

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, max_iter=2500, class_weight="balanced", random_state=args.seed),
    )
    x_state = features[:, layer_index, :]
    state_probs = cross_val_predict(
        probe, x_state, labels, groups=case_ids, cv=splitter, method="predict_proba"
    )[:, 1]
    state_preds = (state_probs >= 0.5).astype(np.int64)

    # Simple non-neural metadata baseline: prompt length only.
    x_length = prompt_lengths.reshape(-1, 1)
    length_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=args.c, max_iter=2500, class_weight="balanced", random_state=args.seed),
    )
    length_probs = cross_val_predict(
        length_model, x_length, labels, groups=case_ids, cv=splitter, method="predict_proba"
    )[:, 1]
    length_preds = (length_probs >= 0.5).astype(np.int64)

    def report_metrics(y: np.ndarray, probs: np.ndarray, preds: np.ndarray) -> dict[str, float]:
        return {
            "roc_auc": float(roc_auc_score(y, probs)),
            "average_precision": float(average_precision_score(y, probs)),
            "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            "brier_score": float(brier_score_loss(y, probs)),
        }

    state_metrics = report_metrics(labels, state_probs, state_preds)
    length_metrics = report_metrics(labels, length_probs, length_preds)
    prediction_rows = []
    for idx in range(len(labels)):
        prediction_rows.append({
            "case_id": str(case_ids[idx]),
            "condition": str(conditions[idx]),
            "label_asr": int(labels[idx]),
            "state_probe_probability": float(state_probs[idx]),
            "length_baseline_probability": float(length_probs[idx]),
            "prompt_tokens": int(prompt_lengths[idx]),
        })

    metadata_path = args.features.with_suffix(".metadata.json")
    source_metadata: dict[str, Any] = {}
    if metadata_path.exists():
        source_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = {
        "task": "exploratory prediction of official judge ASR from pre-generation hidden states",
        "condition_filter": args.condition,
        "n_examples": int(len(labels)),
        "n_cases": int(len(set(case_ids))),
        "positive_count": int(labels.sum()),
        "negative_count": int((labels == 0).sum()),
        "layer_index": int(layer_index),
        "feature_shape": list(features.shape),
        "cv": {"type": "StratifiedGroupKFold", "n_splits": args.folds, "group": "case_id", "seed": args.seed},
        "state_probe": state_metrics,
        "prompt_length_baseline": length_metrics,
        "interpretation_warning": (
            "Exploratory only. Cross-validation tests prediction, not causality or attack improvement. "
            "Do not use the best layer chosen on this same dataset as a confirmatory result. "
            "If feature_model and evaluated target are not the same checkpoint/tokenizer/template, "
            "the scores are cross-model correlations. This analysis does not establish that a state-aware "
            "attack beats RA."
        ),
        "source_feature_metadata": source_metadata,
        "predictions": prediction_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "condition_filter": args.condition,
        "n_examples": len(labels),
        "positive_count": int(labels.sum()),
        "state_probe": state_metrics,
        "prompt_length_baseline": length_metrics,
        "output": str(args.out),
    }, indent=2))
    print(summary["interpretation_warning"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
