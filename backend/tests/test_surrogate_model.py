"""Tests for the EP-V4 baseline surrogate model."""

from __future__ import annotations

import hashlib
import json
import math
import builtins
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pytest

import backend.engine.surrogate.model as surrogate_model

from backend.engine.surrogate.dataset import (
    FEATURE_ITEM_SLOT_COUNT,
    FEATURE_SCHEMA_VERSION,
    FEATURE_SIGNAL_KEYS,
    FEATURE_IDENTITY_TOKENS,
    FEATURE_IDENTITY_CROSS_TOKENS,
    build_dataset_snapshot,
)
from backend.engine.surrogate.model import (
    MODEL_BACKEND,
    MODEL_VERSION,
    TrainResult,
    evaluate_predictions,
    load_model,
    _apply_deterministic_balance,
    train,
)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _create_build(
    builds_root: Path,
    build_id: str,
    genome: Mapping[str, object],
    metrics: Mapping[str, object],
    build_details: Mapping[str, object] | None = None,
) -> None:
    build_dir = builds_root / build_id
    build_dir.mkdir(parents=True, exist_ok=True)
    _write_json(build_dir / "genome.json", genome)
    _write_json(build_dir / "metrics_raw.json", metrics)
    if build_details is not None:
        _write_json(build_dir / "build_details.json", build_details)


def _build_metrics(full_dps: float, max_hit: float, utility_score: float) -> dict[str, Any]:
    return {
        "alpha": {
            "metrics": {
                "full_dps": full_dps,
                "max_hit": max_hit,
                "utility_score": utility_score,
            },
            "defense": {
                "armour": 1000,
                "evasion": 500,
                "resists": {
                    "fire": 60,
                    "cold": 60,
                    "lightning": 60,
                    "chaos": 40,
                },
            },
            "resources": {"life": 5000, "mana": 1000},
            "reservation": {"reserved_percent": 50, "available_percent": 50},
            "attributes": {"strength": 120, "dexterity": 120, "intelligence": 120},
        }
    }


def _create_featured_snapshot(tmp_path: Path):
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome_base = {
        "schema_version": "v0",
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "featured-snapshot",
    }

    def _build_details(slot_count: int) -> dict[str, object]:
        slot_templates = [
            {
                "slot_id": f"slot_{index}",
                "base_type": "Fantasy Item",
                "contributions": {
                    "fire": 3,
                    "strength": 5,
                    "life": 10,
                    "energy_shield": 2,
                },
                "adjustable": True,
            }
            for index in range(slot_count)
        ]
        return {
            "items": {"slot_templates": slot_templates},
            "passives": {"node_ids": ["node-1"], "required_targets": ["target"]},
            "gems": {
                "groups": [
                    {"id": "damage", "group_type": "damage", "gems": ["arc"]},
                    {"id": "utility", "group_type": "utility", "gems": ["mana", "focus"]},
                ],
                "full_dps_group_id": "damage",
                "socket_plan": {
                    "assignments": [
                        {"group_id": "damage", "link_count": 4},
                        {"group_id": "utility", "link_count": 3},
                    ],
                    "main_link_requirement": 5,
                },
            },
        }

    for idx, slot_count in enumerate((1, 2, 3), start=1):
        metrics = {
            "alpha": {
                "metrics": {"full_dps": 1000.0 * slot_count, "max_hit": 1200, "utility_score": 1.0},
                "defense": {"armour": 1000, "evasion": 500},
                "resources": {"life": 5000, "mana": 1000},
                "reservation": {"reserved_percent": 50, "available_percent": 50},
                "attributes": {"strength": 120, "dexterity": 120, "intelligence": 120},
            }
        }
        _create_build(
            builds_root,
            f"featured-{idx}",
            {**genome_base, "seed": idx, "main_skill_package": "sunder"},
            metrics,
            build_details=_build_details(slot_count),
        )

    output_root = data_root / "datasets" / "ep-v4"
    return build_dataset_snapshot(data_root, output_root, "featured-snapshot")


def _create_token_snapshot(tmp_path: Path):
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome_base = {
        "schema_version": "v0",
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "token-snapshot",
    }

    def _build_details(main_gem: str, support_gem: str) -> dict[str, object]:
        return {
            "items": {
                "slot_templates": [
                    {
                        "slot_id": "weapon_2h",
                        "base_type": "Sword",
                        "contributions": {
                            "fire": 5,
                            "strength": 7,
                            "life": 12,
                            "energy_shield": 3,
                        },
                        "adjustable": True,
                    }
                ]
            },
            "passives": {
                "node_ids": ["node-1"],
                "required_targets": [],
            },
            "gems": {
                "groups": [
                    {
                        "id": "damage",
                        "group_type": "damage",
                        "gems": [main_gem, support_gem],
                    },
                    {
                        "id": "utility",
                        "group_type": "utility",
                        "gems": ["utility_support"],
                    },
                ],
                "full_dps_group_id": "damage",
                "socket_plan": {
                    "assignments": [
                        {"group_id": "damage", "slot_id": "weapon_2h", "link_count": 4},
                        {"group_id": "utility", "slot_id": "helmet", "link_count": 3},
                    ],
                    "main_link_requirement": 4,
                },
            },
        }

    def _token_metrics(full_dps: float) -> Mapping[str, object]:
        return {
            "alpha": {
                "metrics": {
                    "full_dps": full_dps,
                    "max_hit": 2200,
                    "utility_score": 1.0,
                },
                "defense": {
                    "armour": 1000,
                    "evasion": 500,
                },
                "resources": {
                    "life": 5000,
                    "mana": 1000,
                },
                "reservation": {
                    "reserved_percent": 55,
                    "available_percent": 45,
                },
                "attributes": {
                    "strength": 120,
                    "dexterity": 120,
                    "intelligence": 120,
                },
            }
        }

    _create_build(
        builds_root,
        "token-support-low",
        {**genome_base, "seed": 1, "main_skill_package": "sunder"},
        _token_metrics(1000.0),
        build_details=_build_details("Arcane", "Added Lightning"),
    )
    _create_build(
        builds_root,
        "token-support-high",
        {**genome_base, "seed": 2, "main_skill_package": "sunder"},
        _token_metrics(1300.0),
        build_details=_build_details("Arcane", "Controlled Destruction"),
    )

    output_root = data_root / "datasets" / "ep-v4"
    return build_dataset_snapshot(data_root, output_root, "token-snapshot")


def _create_snapshot(tmp_path: Path):
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome_base = {
        "schema_version": "v0",
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "snapshot",
    }

    _create_build(
        builds_root,
        "build-sunder-a",
        {**genome_base, "seed": 1, "main_skill_package": "sunder"},
        _build_metrics(2000, 2500, 1.2),
    )
    _create_build(
        builds_root,
        "build-sunder-b",
        {**genome_base, "seed": 2, "main_skill_package": "sunder"},
        _build_metrics(2200, 2600, 1.3),
    )
    _create_build(
        builds_root,
        "build-tornado",
        {**genome_base, "seed": 3, "main_skill_package": "tornado_shot"},
        _build_metrics(4000, 3800, 1.8),
    )

    output_root = data_root / "datasets" / "ep-v4"
    return build_dataset_snapshot(data_root, output_root, "snapshot-test")


def _create_gate_snapshot(tmp_path: Path, gate_labels: Sequence[bool | None]):
    data_root = tmp_path / "data"
    builds_root = data_root / "builds"
    genome_base = {
        "schema_version": "v0",
        "class": "Marauder",
        "ascendancy": "Chieftain",
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "gate-snapshot",
    }

    for idx, gate_pass in enumerate(gate_labels, start=1):
        metrics = _build_metrics(1500.0 + idx * 100.0, 2200, 1.0)
        scenario = cast(dict[str, Any], metrics["alpha"])
        scenario["gate_pass"] = gate_pass
        _create_build(
            builds_root,
            f"gate-build-{idx}",
            {**genome_base, "seed": idx, "main_skill_package": "sunder"},
            metrics,
        )

    output_root = data_root / "datasets" / "ep-v4"
    return build_dataset_snapshot(data_root, output_root, "gate-snapshot")


def _read_rows(snapshot_root: Path) -> list[Mapping[str, object]]:
    dataset_path = snapshot_root / "dataset.jsonl"
    return [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_meta(result: TrainResult) -> Mapping[str, object]:
    return json.loads(result.meta_path.read_text(encoding="utf-8"))



def _make_balancing_row(
    index: int,
    class_name: str,
    main_skill: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "build_id": f"{class_name}-{main_skill}-{index}",
        "scenario_id": f"scenario-{main_skill}",
        "class": class_name,
        "ascendancy": "Chieftain",
        "main_skill_package": main_skill,
        "defense_archetype": "armour",
        "budget_tier": "endgame",
        "profile_id": "pinnacle",
        "full_dps": 1000.0 + float(index),
        "max_hit": 1500.0 + float(index),
        "utility_score": 1.0,
    }
    for feature in FEATURE_SIGNAL_KEYS:
        row[feature] = float(index % 3)
    row[FEATURE_IDENTITY_TOKENS] = []
    row[FEATURE_IDENTITY_CROSS_TOKENS] = []
    return row


def _build_niche_rows(spec: Sequence[tuple[str, str, int]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    counter = 0
    for class_name, skill, count in spec:
        for _ in range(count):
            rows.append(_make_balancing_row(counter, class_name, skill))
            counter += 1
    return rows


def _write_dataset_snapshot_with_rows(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    snapshot_id: str,
) -> Path:
    snapshot_root = root / snapshot_id
    snapshot_root.mkdir(parents=True, exist_ok=True)
    dataset_path = snapshot_root / "dataset.jsonl"
    hasher = hashlib.sha256()
    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            hasher.update(line.encode("utf-8"))
            hasher.update(b"\n")
    manifest = {
        "snapshot_id": snapshot_id,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "row_count": len(rows),
        "dataset_hash": hasher.hexdigest(),
        "source_root_path": str(snapshot_root),
        "generated_at_utc": "2025-01-01T00:00:00Z",
    }
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot_root


def test_deterministic_balance_noop_for_small_dataset() -> None:
    rows = _build_niche_rows([("Marauder", "sunder", 40)])
    balanced, meta = surrogate_model._apply_deterministic_balance(rows)

    assert balanced == rows
    assert meta["applied"] is False
    assert "total rows" in meta["reason"]
    assert meta["input_row_count"] == 40
    assert meta["output_row_count"] == 40
    assert meta["eligible_niche_count"] == 1


def test_deterministic_balance_caps_dominant_niche() -> None:
    rows = _build_niche_rows(
        [
            ("Marauder", "sunder", 40),
            ("Ranger", "tornado_shot", 10),
            ("Witch", "essence_drain", 10),
        ]
    )
    balanced, meta = surrogate_model._apply_deterministic_balance(rows)

    assert meta["applied"] is True
    assert meta["cap_per_niche"] == 20
    assert meta["eligible_niche_count"] == 3
    assert meta["niche_counts_after"]["marauder|sunder"] == 20
    assert meta["output_row_count"] == 40
    assert meta["dropped_row_count"] == 20
    assert meta["retention_ratio"] == pytest.approx(40 / 60)
    assert "dominant share" in meta["reason"]


def test_deterministic_balance_deterministic_across_runs() -> None:
    rows = _build_niche_rows(
        [
            ("Marauder", "sunder", 40),
            ("Ranger", "tornado_shot", 10),
            ("Witch", "essence_drain", 10),
        ]
    )
    balanced_a, meta_a = surrogate_model._apply_deterministic_balance(rows)
    balanced_b, meta_b = surrogate_model._apply_deterministic_balance(rows)

    assert balanced_a == balanced_b
    assert meta_a == meta_b


def test_deterministic_balance_fallback_on_low_retention() -> None:
    rows = _build_niche_rows(
        [
            ("Marauder", "sunder", 50),
            ("Ranger", "tornado_shot", 5),
            ("Witch", "essence_drain", 5),
            ("Shadow", "blade_vortex", 5),
            ("Templar", "winter_orb", 5),
        ]
    )
    balanced, meta = surrogate_model._apply_deterministic_balance(rows)

    assert meta["applied"] is False
    assert meta["output_row_count"] == len(rows)
    assert meta["dropped_row_count"] == 0
    assert meta["retention_ratio"] == 1.0
    assert "retention ratio" in meta["reason"]
    assert meta["niche_counts_before"] == meta["niche_counts_after"]
    assert meta["eligible_niche_count"] == 5


def test_train_records_deterministic_balance_rows(tmp_path: Path) -> None:
    rows = _build_niche_rows(
        [
            ("Marauder", "sunder", 40),
            ("Ranger", "tornado_shot", 10),
            ("Witch", "essence_drain", 10),
        ]
    )
    snapshot_root = _write_dataset_snapshot_with_rows(
        tmp_path / "data", rows, "balanced-train"
    )
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot_root, output_root, model_id="balanced-train-model")

    meta = _read_meta(result)
    balance_meta = meta["deterministic_balance"]
    assert balance_meta["applied"] is True
    assert balance_meta["input_row_count"] == len(rows)
    assert balance_meta["output_row_count"] == 40
    assert balance_meta["niche_counts_after"]["marauder|sunder"] == 20

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["row_count"] == balance_meta["output_row_count"]
    assert result.row_count == metrics["row_count"]
    assert "dominant share" in balance_meta["reason"]


def test_train_creates_artifacts(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="test-model")

    model_dir = result.model_path.parent
    assert (model_dir / "model.json").exists()
    assert (model_dir / "metrics.json").exists()
    assert (model_dir / "model_meta.json").exists()

    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    assert meta["dataset_snapshot_id"] == snapshot.snapshot_id
    assert meta["dataset_row_count"] == snapshot.row_count


def test_eval_metrics_deterministic(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="eval-model")
    model = load_model(result.model_path.parent)

    rows = _read_rows(snapshot.dataset_path.parent)
    predictions = model.predict_many(rows)
    evaluation = evaluate_predictions(rows, predictions)

    metrics_data = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_data["row_count"] == evaluation.row_count

    for metric, value in evaluation.metric_mae.items():
        assert math.isclose(metrics_data["metric_mae"][metric], value, rel_tol=1e-9, abs_tol=0.0)
    for key, value in evaluation.pass_probability.items():
        assert math.isclose(metrics_data["pass_probability"][key], value, rel_tol=1e-9, abs_tol=0.0)


def test_predict_many_returns_expected_shape(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="infer-model")
    model = load_model(result.model_path.parent)

    inputs = [
        {"main_skill_package": "sunder", "full_dps": 0, "max_hit": 0, "utility_score": 0},
        {"main_skill_package": "unknown", "full_dps": 0, "max_hit": 0, "utility_score": 0},
    ]
    predictions = model.predict_many(inputs)
    again = model.predict_many(inputs)

    assert predictions == again
    assert len(predictions) == 2
    assert "metrics" in predictions[0]
    assert math.isclose(predictions[0]["metrics"]["full_dps"], 2100.0, rel_tol=1e-9)

    global_mean = (2000 + 2200 + 4000) / 3
    assert math.isclose(predictions[1]["metrics"]["full_dps"], global_mean, rel_tol=1e-9)
    assert 0.0 <= predictions[0]["pass_probability"] <= 1.0
    assert 0.0 <= predictions[1]["pass_probability"] <= 1.0


def test_feature_adjustments_change_predictions(tmp_path: Path) -> None:
    snapshot = _create_featured_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="feature-model")
    model = load_model(result.model_path.parent)

    base_row: dict[str, object] = {key: 0.0 for key in FEATURE_SIGNAL_KEYS}
    base_row.update(
        {
            "main_skill_package": "unknown",
            "full_dps": 0,
            "max_hit": 0,
            "utility_score": 0,
        }
    )
    low_feature = dict(base_row)
    low_feature[FEATURE_ITEM_SLOT_COUNT] = 1.0
    high_feature = dict(base_row)
    high_feature[FEATURE_ITEM_SLOT_COUNT] = 3.0

    predictions = model.predict_many([low_feature, high_feature])
    assert predictions[1]["metrics"]["full_dps"] > predictions[0]["metrics"]["full_dps"]


def test_token_effects_influence_predictions(tmp_path: Path) -> None:
    snapshot = _create_token_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="token-model")
    model = load_model(result.model_path.parent)

    base_row: dict[str, object] = {key: 0.0 for key in FEATURE_SIGNAL_KEYS}
    base_row["main_skill_package"] = "sunder"
    low_row = dict(base_row)
    low_row[FEATURE_IDENTITY_TOKENS] = ["main_gem:arcane", "main_support:added_lightning"]
    low_row[FEATURE_IDENTITY_CROSS_TOKENS] = [
        "main_gem:arcane|main_support:added_lightning",
        "base_type:sword|main_support:added_lightning",
        "main_gem:arcane|main_support:added_lightning|passive_node:node-1",
    ]
    high_row = dict(base_row)
    high_row[FEATURE_IDENTITY_TOKENS] = [
        "main_gem:arcane",
        "main_support:controlled_destruction",
    ]
    high_row[FEATURE_IDENTITY_CROSS_TOKENS] = [
        "main_gem:arcane|main_support:controlled_destruction",
        "base_type:sword|main_support:controlled_destruction",
        "main_gem:arcane|main_support:controlled_destruction|passive_node:node-1",
    ]

    predictions = model.predict_many([low_row, high_row])
    assert predictions[1]["metrics"]["full_dps"] > predictions[0]["metrics"]["full_dps"]
    assert model.identity_cross_token_effects["full_dps"]


def test_legacy_model_without_feature_metadata(tmp_path: Path) -> None:
    model_root = tmp_path / "models" / "legacy"
    model_root.mkdir(parents=True, exist_ok=True)
    model_path = model_root / "model.json"
    payload = {
        "model_id": "legacy",
        "dataset_snapshot_id": "legacy",
        "feature_schema_version": "v1",
        "global_metrics": {
            "full_dps": {"mean": 100.0, "std": 0.0, "min": 100.0, "max": 100.0, "count": 1},
            "max_hit": {"mean": 200.0, "std": 0.0, "min": 200.0, "max": 200.0, "count": 1},
            "utility_score": {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "count": 1},
        },
        "main_skill_metrics": {},
        "pass_metric": "full_dps",
        "backend": MODEL_BACKEND,
        "backend_version": MODEL_VERSION,
        "trained_at_utc": "2025-01-01T00:00:00Z",
    }
    model_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    model = load_model(model_path)
    assert model.feature_stats == {}
    assert model.feature_weights == {}
    assert model.identity_token_effects == {}
    assert model.identity_cross_token_effects == {}

    inputs = [
        {"main_skill_package": "unknown", "full_dps": 0, "max_hit": 0, "utility_score": 0},
    ]
    predictions = model.predict_many(inputs)
    assert predictions[0]["metrics"]["full_dps"] == 100.0


def test_train_cpu_backend_records_metadata(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="cpu-model",
        compute_backend="cpu",
    )

    assert result.compute_backend_preference == "cpu"
    assert result.compute_backend_resolved == "cpu"
    assert result.compute_backend_fallback_reason is None

    meta = json.loads(result.meta_path.read_text(encoding="utf-8"))
    assert meta["compute_backend_preference"] == "cpu"
    assert meta["compute_backend_resolved"] == "cpu"
    assert meta["compute_backend_fallback_reason"] is None

    model = load_model(result.model_path.parent)
    assert model.compute_backend == "cpu"


def test_train_cuda_backend_falls_back_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    monkeypatch.setattr(surrogate_model, "_torch_cuda_available", lambda: False)

    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="cuda-fallback-model",
        compute_backend="cuda",
    )

    assert result.compute_backend_preference == "cuda"
    assert result.compute_backend_resolved == "cpu"
    assert result.compute_backend_fallback_reason


def test_train_rejects_invalid_backend(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"

    with pytest.raises(ValueError, match="invalid compute backend"):
        train(
            snapshot.dataset_path.parent,
            output_root,
            model_id="bad-backend-model",
            compute_backend="invalid",
        )


def test_train_classifier_skips_when_gate_labels_missing(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="classifier-no-labels")

    meta = _read_meta(result)
    assert meta["classifier_enabled"] is True
    assert meta["classifier_status"] == "skipped"
    assert meta["classifier_skip_reason"] == "no gate_pass labels available"
    assert meta["classifier_train_samples"] == 0
    assert meta["classifier_label_distribution"] == {}


def test_train_classifier_skips_with_single_class_labels(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    snapshot = _create_gate_snapshot(tmp_path, [True, True, True])
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="classifier-single-class")

    meta = _read_meta(result)
    assert meta["classifier_status"] == "skipped"
    assert meta["classifier_skip_reason"] == "single-class gate_pass labels: 1=3"
    assert meta["classifier_train_samples"] == 3
    assert meta["classifier_label_distribution"] == {1: 3}


def test_train_classifier_runs_without_cv_when_low_count(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    snapshot = _create_gate_snapshot(tmp_path, [True, False, True])
    output_root = tmp_path / "models" / "ep-v4"
    result = train(snapshot.dataset_path.parent, output_root, model_id="classifier-low-count")

    meta = _read_meta(result)
    assert meta["classifier_status"] == "trained"
    assert meta["classifier_cv_status"] == "skipped"
    assert meta["classifier_cv_skip_reason"] == "insufficient labeled samples for requested cv"
    assert meta["classifier_cv_used_folds"] == 1
    assert meta["classifier_train_samples"] == 3
    assert meta["classifier_cv_requested_folds"] == 5
    assert meta["classifier_label_distribution"] == {0: 1, 1: 2}
    assert "classifier" in meta
    assert "classifier_metrics" in meta
    classifier_metrics = cast(Mapping[str, Any], meta["classifier_metrics"])
    assert classifier_metrics["train_samples"] == 3


def test_train_classifier_runs_cv_when_enough_labels(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    labels = [True, False, True, False, True, False]
    snapshot = _create_gate_snapshot(tmp_path, labels)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="classifier-happy",
    )

    meta = _read_meta(result)
    assert meta["classifier_status"] == "trained"
    assert meta["classifier_cv_status"] == "ran"
    assert meta["classifier_cv_used_folds"] == 3
    assert meta["classifier_cv_requested_folds"] == 5
    assert meta["classifier_train_samples"] == len(labels)
    assert meta["classifier_label_distribution"] == {0: 3, 1: 3}
    assert "classifier" in meta
    assert "classifier_metrics" in meta
    classifier_metrics = cast(Mapping[str, Any], meta["classifier_metrics"])
    assert classifier_metrics["train_samples"] == len(labels)
    assert "cv_mean" in classifier_metrics
    assert "cv_std" in classifier_metrics


def test_pass_probability_contract_with_classifier(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    labels = [True, False, True, False, True, False]
    snapshot = _create_gate_snapshot(tmp_path, labels)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="classifier-pass-prob",
    )

    model = load_model(result.model_path.parent)
    rows = _read_rows(snapshot.dataset_path.parent)
    predictions = model.predict_many(rows[:2])

    assert predictions
    for prediction in predictions:
        pass_prob = prediction["pass_probability"]
        assert isinstance(pass_prob, (float, int))
        assert 0.0 <= pass_prob <= 1.0


def test_train_classifier_handles_cross_val_score_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sklearn = pytest.importorskip("sklearn")
    snapshot = _create_gate_snapshot(tmp_path, [True, False, True, False])
    output_root = tmp_path / "models" / "ep-v4"

    def _fake_cross_val_score(*args, **kwargs):
        _ = (args, kwargs)
        raise ValueError("forced cv failure")

    monkeypatch.setattr(sklearn.model_selection, "cross_val_score", _fake_cross_val_score)

    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="classifier-cv-error",
    )

    meta = _read_meta(result)
    assert meta["classifier_status"] == "trained"
    assert meta["classifier_cv_status"] == "skipped"
    assert meta["classifier_cv_skip_reason"] == "cv error: ValueError"
    assert meta["classifier_cv_used_folds"] == 2
    assert "classifier" in meta
    assert "classifier_metrics" in meta
    classifier_metrics = cast(Mapping[str, Any], meta["classifier_metrics"])
    assert "cv_mean" not in classifier_metrics
    assert "cv_std" not in classifier_metrics


def test_train_classifier_marks_sklearn_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _create_gate_snapshot(tmp_path, [True, False, True, False])
    output_root = tmp_path / "models" / "ep-v4"

    original_import = builtins.__import__

    def _import_without_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("sklearn disabled for test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_without_sklearn)

    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="classifier-sklearn-missing",
    )

    meta = _read_meta(result)
    assert meta["classifier_enabled"] is True
    assert meta["classifier_status"] == "skipped"
    assert meta["classifier_skip_reason"] == "sklearn unavailable"
    assert meta["classifier_cv_status"] == "skipped"
    assert "classifier" not in meta
    assert "classifier_metrics" not in meta


def test_train_classifier_disabled_when_include_failures_false(tmp_path: Path) -> None:
    snapshot = _create_snapshot(tmp_path)
    output_root = tmp_path / "models" / "ep-v4"
    result = train(
        snapshot.dataset_path.parent,
        output_root,
        model_id="classifier-disabled",
        include_failures=False,
    )

    meta = _read_meta(result)
    assert meta["classifier_enabled"] is False
    assert meta["classifier_status"] == "disabled"
    assert meta["classifier_skip_reason"] == "include_failures disabled"
    assert meta["classifier_cv_requested_folds"] == 0
    assert "classifier" not in meta
    assert "classifier_metrics" not in meta
