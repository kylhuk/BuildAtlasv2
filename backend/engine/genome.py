"""Deterministic genome model + serialization helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence, TypeVar

T = TypeVar("T")
_SCHEMA_VERSION_V0 = "v0"


class GenomeError(ValueError):
    pass


class GenomeDeserializationError(GenomeError):
    pass


class UnsupportedGenomeVersionError(GenomeError):
    pass


@dataclass(frozen=True)
class GenomeV0:
    seed: int
    class_name: str
    ascendancy: str
    main_skill_package: str
    defense_archetype: str
    budget_tier: str
    profile_id: str
    schema_version: Literal[_SCHEMA_VERSION_V0] = field(default=_SCHEMA_VERSION_V0, init=False)


class DeterministicRng:
    """Simple RNG backed by :mod:`random.Random` with explicit seed."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def choice(self, values: Sequence[T]) -> T:
        if not values:
            raise ValueError("values must not be empty")
        return self._rng.choice(values)

    def randint(self, a: int, b: int) -> int:
        return self._rng.randint(a, b)


CLASS_ASCENDANCIES = {
    "marauder": ["Juggernaut", "Chieftain", "Berserker"],
    "duelist": ["Gladiator", "Champion", "Slayer"],
    "ranger": ["Deadeye", "Raider", "Pathfinder"],
    "shadow": ["Trickster", "Assassin", "Saboteur"],
    "witch": ["Elementalist", "Necromancer", "Occultist"],
    "templar": ["Guardian", "Inquisitor", "Hierophant"],
    "scion": ["Ascendant"],
}

MAIN_SKILL_PACKAGES = [
    "cyclone",
    "arc",
    "tornado_shot",
    "blade_vortex",
    "summon_raging_spirit",
    "essence_drain",
    "sunder",
    "frostbolt",
    "toxic_rain",
    "blade_flurry",
]

DEFENSE_ARCHETYPES = ["armour", "evasion", "energy_shield", "hybrid"]
BUDGET_TIERS = ["starter", "midgame", "endgame"]
PROFILE_IDS = ["alpha", "bravo", "charlie", "delta"]


def deterministic_genome_from_seed(seed: int) -> GenomeV0:
    """Create a minimal deterministic genome from a seed."""

    rng = DeterministicRng(seed)
    class_name = rng.choice(list(CLASS_ASCENDANCIES.keys()))
    ascendancy = rng.choice(CLASS_ASCENDANCIES[class_name])
    main_skill_package = rng.choice(MAIN_SKILL_PACKAGES)
    defense_archetype = rng.choice(DEFENSE_ARCHETYPES)
    budget_tier = rng.choice(BUDGET_TIERS)
    profile_id = rng.choice(PROFILE_IDS)
    return GenomeV0(
        seed=seed,
        class_name=class_name,
        ascendancy=ascendancy,
        main_skill_package=main_skill_package,
        defense_archetype=defense_archetype,
        budget_tier=budget_tier,
        profile_id=profile_id,
    )


def serialize_genome(genome: GenomeV0) -> str:
    """Serialize a genome payload to deterministic JSON text."""

    payload = {
        "schema_version": genome.schema_version,
        "seed": genome.seed,
        "class": genome.class_name,
        "ascendancy": genome.ascendancy,
        "main_skill_package": genome.main_skill_package,
        "defense_archetype": genome.defense_archetype,
        "budget_tier": genome.budget_tier,
        "profile_id": genome.profile_id,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def parse_genome(payload: str | Mapping[str, Any]) -> GenomeV0:
    """Parse a genome from JSON text or decoded payload."""

    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise GenomeDeserializationError("invalid genome JSON") from exc
    elif isinstance(payload, Mapping):
        data = payload
    else:
        raise GenomeDeserializationError("unsupported genome payload type")
    return parse_genome_payload(data)


def parse_genome_payload(payload: Mapping[str, Any]) -> GenomeV0:
    """Parse a genome payload that already looks like a JSON object."""

    version = payload.get("schema_version")
    if version is None:
        version = _SCHEMA_VERSION_V0
    if version != _SCHEMA_VERSION_V0:
        raise UnsupportedGenomeVersionError(f"unsupported genome schema_version={version}")

    seed = _parse_seed(payload.get("seed"))
    class_field = _parse_string(payload, ("class", "class_name"), "class")
    ascendancy = _parse_string(payload, ("ascendancy",), "ascendancy")
    main_skill_package = _parse_string(payload, ("main_skill_package",), "main_skill_package")
    defense_archetype = _parse_string(payload, ("defense_archetype",), "defense_archetype")
    budget_tier = _parse_string(payload, ("budget_tier",), "budget_tier")
    profile_id = _parse_string(payload, ("profile_id",), "profile_id")

    return GenomeV0(
        seed=seed,
        class_name=class_field,
        ascendancy=ascendancy,
        main_skill_package=main_skill_package,
        defense_archetype=defense_archetype,
        budget_tier=budget_tier,
        profile_id=profile_id,
    )


def _parse_string(payload: Mapping[str, Any], keys: Sequence[str], field_name: str) -> str:
    for key in keys:
        if key in payload:
            value = payload[key]
            if value is None:
                break
            if isinstance(value, str):
                return value
            return str(value)
    raise GenomeDeserializationError(f"missing or invalid field {field_name}")


def _parse_seed(value: Any) -> int:
    if value is None:
        raise GenomeDeserializationError("seed is required")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise GenomeDeserializationError("seed must be an integer") from exc


__all__ = [
    "GenomeV0",
    "DeterministicRng",
    "GenomeDeserializationError",
    "UnsupportedGenomeVersionError",
    "serialize_genome",
    "parse_genome",
    "parse_genome_payload",
    "deterministic_genome_from_seed",
]
