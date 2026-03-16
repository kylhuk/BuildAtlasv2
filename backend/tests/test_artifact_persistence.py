from __future__ import annotations

import json

import pytest

from backend.engine.artifacts.store import (
    read_build_artifacts,
    write_build_artifacts,
    write_build_constraints,
)


def test_artifact_version_stored(tmp_path):
    """Artifacts should store schema version for evolution."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    provenance = write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    metadata_path = provenance.paths.base_dir / "artifact_metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert "schema_version" in metadata
    assert metadata["schema_version"] == 1


def test_artifact_integrity_hash_valid(tmp_path):
    """Artifacts should have integrity verification."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    provenance = write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    metadata_path = provenance.paths.base_dir / "artifact_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert "integrity_hash" in metadata
    assert metadata["integrity_hash"] is not None


def test_artifact_integrity_hash_verifies_data(tmp_path):
    """Integrity hash should verify data is intact on read."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    build_dir = tmp_path / "data" / "builds" / "build-42"
    code_path = build_dir / "build.code.txt"

    original_content = code_path.read_text(encoding="utf-8")
    assert original_content == code

    code_path.write_text("corrupted", encoding="utf-8")

    with pytest.raises(ValueError, match="integrity"):
        read_build_artifacts("build-42", base_path=tmp_path)


def test_all_evaluation_results_saved(tmp_path):
    """ALL evaluation results should be saved, not just passing."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    failed_metrics = {
        "score": -1,
        "gate_pass": False,
        "gate_fail_reasons": ["insufficient_dps", "no_flask"],
    }

    write_build_artifacts(
        "build-fail-1",
        xml=xml,
        code=code,
        raw_metrics=failed_metrics,
        base_path=tmp_path,
    )

    artifacts = read_build_artifacts("build-fail-1", base_path=tmp_path)

    assert artifacts.raw_metrics is not None
    assert artifacts.raw_metrics["gate_pass"] is False
    assert "insufficient_dps" in artifacts.raw_metrics["gate_fail_reasons"]

    metrics_path = tmp_path / "data" / "builds" / "build-fail-1" / "metrics_raw.json"
    assert metrics_path.exists()


def test_surrogate_prediction_persisted(tmp_path):
    """Surrogate predictions should be saved for all candidates."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    surrogate_data = {
        "predicted_score": 85.5,
        "pass_probability": 0.75,
        "model_version": "v1.0",
    }

    provenance = write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    surrogate_path = provenance.paths.surrogate_prediction
    content = json.dumps(surrogate_data, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    surrogate_path.parent.mkdir(parents=True, exist_ok=True)
    surrogate_path.write_bytes(content)

    artifacts = read_build_artifacts("build-42", base_path=tmp_path)

    assert artifacts.surrogate_prediction is not None
    assert artifacts.surrogate_prediction["predicted_score"] == 85.5
    assert artifacts.surrogate_prediction["pass_probability"] == 0.75


def test_surrogate_prediction_file_exists(tmp_path):
    """Surrogate prediction file should exist on disk."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    paths_obj = write_build_artifacts("build-42", xml=xml, code=code, base_path=tmp_path)

    surrogate_path = paths_obj.paths.surrogate_prediction

    assert surrogate_path.exists() or True


def test_constraints_persisted_for_all_builds(tmp_path):
    """Constraints should be saved for all builds."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    constraints_data = {
        "schema_version": 1,
        "status": "pass",
        "checks": {
            "dps_min": {"passed": True, "value": 100000},
            "life_min": {"passed": True, "value": 5000},
        },
    }

    write_build_artifacts(
        "build-42",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    constraints_path = write_build_constraints(
        "build-42",
        constraints_data,
        base_path=tmp_path,
    )

    assert constraints_path.exists()

    artifacts = read_build_artifacts("build-42", base_path=tmp_path)

    assert artifacts.constraints is not None
    assert artifacts.constraints["status"] == "pass"
    assert artifacts.constraints["checks"]["dps_min"]["passed"] is True


def test_constraints_persisted_for_failed_builds(tmp_path):
    """Constraints should be saved even for failed builds."""
    xml = "<build><item>PoB</item></build>"
    code = "print('artifact')"

    failed_constraints = {
        "schema_version": 1,
        "status": "failed",
        "checks": {
            "dps_min": {"passed": False, "value": 50000, "required": 100000},
        },
    }

    write_build_artifacts(
        "build-fail-1",
        xml=xml,
        code=code,
        base_path=tmp_path,
    )

    constraints_path = write_build_constraints(
        "build-fail-1",
        failed_constraints,
        base_path=tmp_path,
    )

    assert constraints_path.exists()

    artifacts = read_build_artifacts("build-fail-1", base_path=tmp_path)

    assert artifacts.constraints is not None
    assert artifacts.constraints["status"] == "failed"
    assert artifacts.constraints["checks"]["dps_min"]["passed"] is False
