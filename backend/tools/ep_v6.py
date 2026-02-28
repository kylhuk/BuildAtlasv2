from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from backend.engine.ep_v6 import ablation, mutation, novelty, signatures, surrogate

DEFAULT_MODEL_PATH = Path("data/ep_v6/models/interaction_surrogate.json")


def _print_json(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _parse_features(raw: Mapping[str, Any] | None) -> Mapping[str, float]:
    if not raw:
        return {}
    return {k: float(v) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="EP-V6 interaction evidence tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe")
    probe_parser.add_argument("--ruleset-id", required=True)
    probe_parser.add_argument("--scenario-id", required=True)
    probe_parser.add_argument("--skill-package-id", required=True)
    probe_parser.add_argument("--cache-dir", required=True)

    ablation_parser = subparsers.add_parser("ablation")
    ablation_parser.add_argument("--ruleset-id", required=True)
    ablation_parser.add_argument("--scenario-id", required=True)
    ablation_parser.add_argument("--skill-package-id", required=True)
    ablation_parser.add_argument("--out", required=True)

    train_parser = subparsers.add_parser("train-surrogate")
    train_parser.add_argument("--dataset", required=True)
    train_parser.add_argument("--out", required=True)

    infer_parser = subparsers.add_parser("infer-surrogate")
    infer_parser.add_argument("--model", required=True)
    infer_parser.add_argument("--input-json", required=True)

    mutate_parser = subparsers.add_parser("mutate")
    mutate_parser.add_argument("--input-json", required=True)
    mutate_parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))

    novelty_parser = subparsers.add_parser("enforce-novelty")
    novelty_parser.add_argument("--input-json", required=True)

    args = parser.parse_args()

    if args.command == "probe":
        signature = signatures.probe_signature(
            args.ruleset_id, args.scenario_id, args.skill_package_id, args.cache_dir
        )
        _print_json(signature)

    elif args.command == "ablation":
        signature = signatures.build_signature(
            args.ruleset_id, args.scenario_id, args.skill_package_id
        )
        rows = ablation.generate_ablation_rows(signature)
        ablation.write_ndjson(rows, args.out)
        _print_json({"rows": len(rows), "path": args.out})

    elif args.command == "train-surrogate":
        rows = surrogate.load_dataset(args.dataset)
        model = surrogate.train_surrogate(rows)
        surrogate.write_model(args.out, model)
        _print_json(
            {
                "model_path": args.out,
                "operator_count": len(model.get("operator_stats", {})),
            }
        )

    elif args.command == "infer-surrogate":
        model = surrogate.read_model(args.model)
        payload = json.loads(args.input_json)
        prediction = surrogate.infer(
            model,
            payload.get("operator", "support_remove"),
            _parse_features(payload.get("features")),
        )
        _print_json(prediction)

    elif args.command == "mutate":
        model = surrogate.read_model(args.model)
        payload = json.loads(args.input_json)
        candidate_features = _parse_features(payload.get("features"))
        constraints = payload.get("constraints", {})
        mutation_result = mutation.select_mutation(model, candidate_features, constraints)
        _print_json(
            {
                "candidate_id": payload.get("candidate_id"),
                **mutation_result,
            }
        )

    elif args.command == "enforce-novelty":
        payload = json.loads(args.input_json)
        result = novelty.enforce_quota(payload.get("scores", []), int(payload.get("quota", 0)))
        _print_json(result)


if __name__ == "__main__":
    main()
