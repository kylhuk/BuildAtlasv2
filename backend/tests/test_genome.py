import pytest

from backend.engine.genome import (
    DeterministicRng,
    UnsupportedGenomeVersionError,
    deterministic_genome_from_seed,
    parse_genome,
    parse_genome_payload,
    serialize_genome,
)


def test_genome_serialization_round_trip() -> None:
    genome = deterministic_genome_from_seed(123)
    serialized = serialize_genome(genome)
    parsed = parse_genome(serialized)
    assert parsed == genome


def test_versionless_payload_parses_as_v0() -> None:
    payload = {
        "seed": 42,
        "class": "witch",
        "ascendancy": "Elementalist",
        "main_skill_package": "frostbolt",
        "defense_archetype": "energy_shield",
        "budget_tier": "midgame",
        "profile_id": "alpha",
    }
    genome = parse_genome(payload)
    assert genome.schema_version == "v0"
    assert genome.class_name == "witch"


def test_unsupported_schema_version() -> None:
    payload = {
        "schema_version": "v1",
        "seed": 1,
        "class": "marauder",
        "ascendancy": "Chieftain",
        "main_skill_package": "cyclone",
        "defense_archetype": "armour",
        "budget_tier": "starter",
        "profile_id": "alpha",
    }
    with pytest.raises(UnsupportedGenomeVersionError):
        parse_genome_payload(payload)


def test_rng_determinism() -> None:
    options = ["a", "b", "c"]
    rng_a = DeterministicRng(7)
    rng_b = DeterministicRng(7)
    ints_a = [rng_a.randint(0, 10) for _ in range(3)]
    ints_b = [rng_b.randint(0, 10) for _ in range(3)]
    assert ints_a == ints_b
    choices_a = [rng_a.choice(options) for _ in range(3)]
    choices_b = [rng_b.choice(options) for _ in range(3)]
    assert choices_a == choices_b


def test_deterministic_generator_repeats_given_seed() -> None:
    first = deterministic_genome_from_seed(99)
    second = deterministic_genome_from_seed(99)
    assert first == second
    assert first.seed == 99
