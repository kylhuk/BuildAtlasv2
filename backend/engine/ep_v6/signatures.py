from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

DEFAULT_METRIC_BASES: Mapping[str, float] = {
    "full_dps": 10000.0,
    "max_hit": 2000.0,
    "support_coverage": 4.0,
    "passive_clusters": 6.0,
    "item_tier": 5.0,
}

SKILL_PROBES: List[str] = [
    "support_primary",
    "support_secondary",
    "support_offensive",
]

PROBE_STEPS = 4


def cache_key(ruleset_id: str, scenario_id: str, skill_package_id: str) -> str:
    return f"{ruleset_id}|{scenario_id}|{skill_package_id}"


def _deterministic_offset(base: str, salt: str, modulus: int = 1000) -> float:
    digest = hashlib.sha256(f"{base}|{salt}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:6], "big")
    return (value % modulus) / 10.0


def build_signature(ruleset_id: str, scenario_id: str, skill_package_id: str) -> Dict[str, Any]:
    key = cache_key(ruleset_id, scenario_id, skill_package_id)
    metrics: Dict[str, float] = {}
    for metric, base in DEFAULT_METRIC_BASES.items():
        metrics[metric] = round(base + _deterministic_offset(key, metric, modulus=500), 2)

    skill_probes: List[Dict[str, Any]] = []
    for skill in SKILL_PROBES:
        probe_values: List[float] = []
        for step in range(PROBE_STEPS):
            delta = _deterministic_offset(key, f"{skill}|{step}", modulus=400)
            probe_values.append(round(300 * (step + 1) + delta, 2))
        skill_probes.append(
            {
                "skill_name": skill,
                "scaling": probe_values,
                "probe_score": round(_deterministic_offset(key, f"score|{skill}", modulus=120), 2),
            }
        )

    return {
        "ruleset_id": ruleset_id,
        "scenario_id": scenario_id,
        "skill_package_id": skill_package_id,
        "key": key,
        "probe_version": "V6-01",
        "metrics": metrics,
        "skill_probes": skill_probes,
    }


def probe_signature(
    ruleset_id: str,
    scenario_id: str,
    skill_package_id: str,
    cache_dir: Path | str | None = None,
) -> Dict[str, Any]:
    signature = build_signature(ruleset_id, scenario_id, skill_package_id)
    if cache_dir:
        cache = read_signature_cache(cache_dir)
        cache[signature["key"]] = signature
        write_signature_cache(cache_dir, cache)
    return signature


def read_signature_cache(cache_dir: Path | str) -> Dict[str, Any]:
    cache_file = Path(cache_dir) / "ep_v6_signatures.json"
    if not cache_file.exists():
        return {}
    with cache_file.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_signature_cache(cache_dir: Path | str, cache: Mapping[str, Any]) -> None:
    cache_path = Path(cache_dir) / "ep_v6_signatures.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
