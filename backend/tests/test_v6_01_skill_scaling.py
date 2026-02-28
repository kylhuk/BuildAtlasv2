from pathlib import Path

from backend.engine.ep_v6 import signatures


def test_skill_signature_deterministic(tmp_path: Path) -> None:
    ruleset = "pob:local|scenarios:mapping@v0|prices:local"
    scenario = "mapping"
    skill_package = "arc"

    first_signature = signatures.build_signature(ruleset, scenario, skill_package)
    second_signature = signatures.build_signature(ruleset, scenario, skill_package)

    assert first_signature == second_signature
    assert len(first_signature["skill_probes"]) == len(signatures.SKILL_PROBES)

    cache_path = tmp_path / "cache"
    signatures.probe_signature(ruleset, scenario, skill_package, cache_path)
    cache = signatures.read_signature_cache(cache_path)

    assert first_signature["key"] in cache
    cached_metrics = cache[first_signature["key"]]["metrics"]
    assert cached_metrics["full_dps"] == first_signature["metrics"]["full_dps"]
