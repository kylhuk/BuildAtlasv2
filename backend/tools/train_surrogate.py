from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.engine.surrogate import (
    TrainResult,
    evaluate_predictions,
    load_dataset_rows,
    load_model,
    resolve_snapshot_root,
)
from backend.engine.surrogate import (
    train as train_model,
)


def _dump(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def _collect_input(path: Path | None, inline: str | None) -> list[Mapping[str, Any]]:
    if inline and path:
        raise ValueError("provide either --input or --input-json, not both")
    raw: Any
    if inline:
        raw = json.loads(inline)
    elif path:
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError("no input provided")
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, Sequence):
        return [item for item in raw if isinstance(item, Mapping)]
    raise ValueError("input must be a JSON object or array of objects")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or infer with the EP-V4 surrogate baseline",
    )
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser(
        "train",
        help="train a new baseline surrogate",
    )
    train_parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="directory or snapshot root containing dataset.jsonl",
    )
    train_parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="directory to write data/models/ep-v4/<model_id>",
    )
    train_parser.add_argument(
        "--model-id",
        type=str,
        help="optional model identifier",
    )
    train_parser.add_argument(
        "--surrogate-backend",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="compute backend preference for surrogate training",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="evaluate an existing model against a snapshot",
    )
    eval_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="path to a model directory or model.json",
    )
    eval_parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="dataset snapshot root or parent folder",
    )

    infer_parser = subparsers.add_parser(
        "infer",
        help="run inference with a trained surrogate",
    )
    infer_parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="path to a model directory or model.json",
    )
    group = infer_parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-json",
        type=Path,
        help="path to JSON file containing row(s) to predict",
    )
    group.add_argument(
        "--input",
        type=str,
        help="inline JSON object or array to predict",
    )

    args = parser.parse_args(argv)
    if args.command == "train":
        return _handle_train(args)
    if args.command == "eval":
        return _handle_eval(args)
    if args.command == "infer":
        return _handle_infer(args)
    parser.error("command is required")


def _handle_train(args: argparse.Namespace) -> int:
    result: TrainResult = train_model(
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        model_id=args.model_id,
        compute_backend=args.surrogate_backend,
    )
    summary = {
        "model_id": result.model_id,
        "model_path": str(result.model_path),
        "metrics_path": str(result.metrics_path),
        "meta_path": str(result.meta_path),
        "dataset_snapshot_id": result.dataset_snapshot_id,
        "dataset_hash": result.dataset_hash,
        "row_count": result.row_count,
        "feature_schema_version": result.feature_schema_version,
        "compute_backend_preference": result.compute_backend_preference,
        "compute_backend_resolved": result.compute_backend_resolved,
        "compute_backend_fallback_reason": result.compute_backend_fallback_reason,
    }
    _dump(summary)
    return 0


def _handle_eval(args: argparse.Namespace) -> int:
    model = load_model(args.model_path)
    rows = load_dataset_rows(args.dataset_path)
    predictions = model.predict_many(rows)
    evaluation = evaluate_predictions(rows, predictions)
    dataset_root = resolve_snapshot_root(args.dataset_path)
    summary = {
        "model_id": model.model_id,
        "dataset_snapshot_id": model.dataset_snapshot_id,
        "evaluated_snapshot": str(dataset_root),
        "row_count": evaluation.row_count,
        "metric_mae": evaluation.metric_mae,
        "pass_probability": evaluation.pass_probability,
    }
    _dump(summary)
    return 0


def _handle_infer(args: argparse.Namespace) -> int:
    model = load_model(args.model_path)
    rows = _collect_input(args.input_json, args.input)
    predictions = model.predict_many(rows)
    _dump({"predictions": predictions})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
