import pytest
import torch
import numpy as np
from pathlib import Path
from backend.engine.surrogate.model import train, load_model, FEATURE_SIGNAL_KEYS


def test_slack_regressor_training(tmp_path):
    # Create a dummy dataset
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    snapshot_dir = dataset_dir / "v1"
    snapshot_dir.mkdir()

    dataset_file = snapshot_dir / "dataset.jsonl"
    manifest_file = snapshot_dir / "manifest.json"

    rows = []
    for i in range(100):
        row = {
            "build_id": f"build_{i}",
            "scenario_id": "scen_1",
            "profile_id": "prof_1",
            "main_skill_package": "skill_1",
            "class": "class_1",
            "gate_pass": i % 2 == 0,
            "min_gate_slack": float(i) / 10.0,
            "full_dps": 1000.0 + i,
            "max_hit": 500.0 + i,
        }
        # Add signal features
        for key in FEATURE_SIGNAL_KEYS:
            row[key] = float(np.random.randn())
        rows.append(row)

    import json

    with open(dataset_file, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    manifest = {
        "snapshot_id": "v1",
        "feature_schema_version": "1.0.0",
        "dataset_hash": "dummy",
        "row_count": 100,
    }
    with open(manifest_file, "w") as f:
        json.dump(manifest, f)

    # Train model
    output_dir = tmp_path / "models"
    result = train(
        dataset_path=dataset_dir,
        output_root=output_dir,
        model_id="test_model",
        include_failures=True,
    )

    assert result.model_id == "test_model"
    assert (output_dir / "test_model" / "model.json").exists()

    # Load model and predict
    model = load_model(output_dir / "test_model")
    assert model.slack_regressor is not None

    test_row = rows[0]
    predictions = model.predict_many([test_row])
    assert len(predictions) == 1
    assert "min_gate_slack" in predictions[0]
    assert isinstance(predictions[0]["min_gate_slack"], float)

    # Check if prediction is somewhat reasonable (not NaN)
    assert not np.isnan(predictions[0]["min_gate_slack"])


if __name__ == "__main__":
    pytest.main([__file__])
