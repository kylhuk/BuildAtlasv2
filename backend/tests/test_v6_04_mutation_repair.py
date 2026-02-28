import json
from pathlib import Path

from backend.engine.ep_v6 import ablation, mutation, signatures, surrogate


def test_mutation_and_repair(tmp_path: Path) -> None:
    signature = signatures.build_signature(
        "pob:local|scenarios:mapping@v0|prices:local",
        "mapping",
        "arc",
    )
    rows = ablation.generate_ablation_rows(signature)
    model = surrogate.train_surrogate(rows)

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "ep_v6" / "candidate.json"
    payload = json.loads(fixture_path.read_text())
    result = mutation.select_mutation(model, payload["features"], payload["constraints"])

    assert result["selected_operator"] in mutation.DEFAULT_OPERATORS
    assert isinstance(result["scores"], dict)
    assert "repair_reasons" in result
    assert all(isinstance(reason, str) for reason in result["repair_reasons"])
