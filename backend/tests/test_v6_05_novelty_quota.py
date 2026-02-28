import json
from pathlib import Path

from backend.engine.ep_v6 import novelty


def test_novelty_score_and_quota() -> None:
    distance = novelty.novelty_score([0.0, 1.0, 2.0], [0.0, 0.5, 1.5])
    assert distance > 0.0

    fixture = Path(__file__).resolve().parent / "fixtures" / "ep_v6" / "novelty_scores.json"
    payload = json.loads(fixture.read_text())
    result = novelty.enforce_quota(payload["scores"], payload["quota"])

    assert result["accepted_count"] == payload["quota"]
    assert result["rejected_count"] == len(payload["scores"]) - payload["quota"]
    assert result["accepted"] == sorted(payload["scores"], reverse=True)[: payload["quota"]]
