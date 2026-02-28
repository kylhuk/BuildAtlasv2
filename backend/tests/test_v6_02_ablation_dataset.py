import json
from pathlib import Path

from backend.engine.ep_v6 import ablation, signatures


def test_ablation_dataset_rows(tmp_path: Path) -> None:
    signature = signatures.build_signature(
        "pob:local|scenarios:mapping@v0|prices:local",
        "mapping",
        "arc",
    )
    rows = ablation.generate_ablation_rows(signature)

    assert len(rows) == len(ablation.OPERATOR_NAMES)
    for row in rows:
        assert row["operator"] in ablation.OPERATOR_NAMES
        assert "baseline" in row
        assert "variant" in row
        assert "delta" in row
        assert isinstance(row["metadata"], dict)

    output_path = tmp_path / "ablation.ndjson"
    ablation.write_ndjson(rows, output_path)
    lines = [line for line in output_path.read_text().splitlines() if line]
    assert len(lines) == len(rows)
    parsed = [json.loads(line) for line in lines]
    assert all(parsed[i]["operator"] == rows[i]["operator"] for i in range(len(rows)))
