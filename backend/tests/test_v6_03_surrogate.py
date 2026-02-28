from pathlib import Path

from backend.engine.ep_v6 import ablation, signatures, surrogate


def test_surrogate_train_and_infer(tmp_path: Path) -> None:
    signature = signatures.build_signature(
        "pob:local|scenarios:mapping@v0|prices:local",
        "mapping",
        "arc",
    )
    rows = ablation.generate_ablation_rows(signature)
    model = surrogate.train_surrogate(rows)

    assert model["metadata"]["version"] == "V6-03"
    assert model["operator_stats"]

    model_path = tmp_path / "interaction_surrogate.json"
    surrogate.write_model(model_path, model)
    loaded_model = surrogate.read_model(model_path)

    prediction = surrogate.infer(
        loaded_model,
        "support_remove",
        {"full_dps": 13000.0, "max_hit": 2100.0, "support_coverage": 4.5},
    )
    repeat = surrogate.infer(
        loaded_model,
        "support_remove",
        {"full_dps": 13000.0, "max_hit": 2100.0, "support_coverage": 4.5},
    )

    assert prediction == repeat
    assert prediction["operator"] == "support_remove"
    assert "prediction" in prediction
    assert prediction["uncertainty"] >= 0.0
