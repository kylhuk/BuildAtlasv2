"""Tests for ME-MAP-Elites archive integration into generation pipeline."""

from __future__ import annotations

from typing import Any, Mapping

import pytest

from backend.engine.archive import (
    ArchiveStore,
    descriptor_values_from_metrics,
    score_from_metrics,
)
from backend.engine.generation.runner import Candidate
from backend.engine.genome import GenomeV0


def _dummy_candidate(
    seed: int = 42,
    build_id: str = "test-build",
    main_skill: str = "arc",
    class_name: str = "Witch",
    ascendancy: str = "Occultist",
) -> Candidate:
    """Create a dummy candidate for testing."""
    genome = GenomeV0(
        seed=seed,
        class_name=class_name,
        ascendancy=ascendancy,
        main_skill_package=main_skill,
        defense_archetype="armour",
        budget_tier="starter",
        profile_id="alpha",
    )
    build_details_payload = {
        "identity": {"class_name": class_name, "ascendancy": ascendancy},
        "gems": {
            "groups": [
                {
                    "id": "weapon_main",
                    "name": "Arc Chain",
                    "group_type": "damage",
                    "gems": ["arc", "controlled_destruction_support"],
                }
            ],
            "socket_plan": {"assignments": []},
        },
    }
    return Candidate(
        seed=genome.seed,
        build_id=build_id,
        main_skill_package=genome.main_skill_package,
        class_name=genome.class_name,
        ascendancy=genome.ascendancy,
        budget_tier=genome.budget_tier,
        failures=[],
        metrics_payload={
            "mapping_t16": {
                "metrics": {"full_dps": 1500.0, "max_hit": 750.0},
                "defense": {"armour": 900.0, "resists": {"fire": 70.0}},
                "resources": {"life": 4200.0, "mana": 1300.0},
                "reservation": {"reserved_percent": 30.0, "available_percent": 70.0},
                "attributes": {"strength": 150.0, "dexterity": 120.0, "intelligence": 80.0},
            }
        },
        genome=genome,
        code_payload="{}",
        build_details_payload=build_details_payload,
    )


def test_archive_initialization():
    """Test that archive can be initialized with default axes."""
    archive = ArchiveStore()
    assert archive.axes is not None
    assert len(archive.axes) == 2
    assert archive.axes[0].metric_key == "full_dps"
    assert archive.axes[1].metric_key == "max_hit"
    assert archive.total_bins > 0


def test_archive_insert_verified_build():
    """Test inserting a verified build into archive."""
    archive = ArchiveStore()
    candidate = _dummy_candidate(seed=42, build_id="build-42")

    # Extract behavior characteristics from candidate metrics
    score = score_from_metrics(candidate.metrics_payload)
    descriptor = descriptor_values_from_metrics(candidate.metrics_payload, archive.axes)

    # Insert into archive
    inserted = archive.insert(
        candidate.build_id,
        score=score,
        descriptor=descriptor,
        metadata={
            "main_skill": candidate.main_skill_package,
            "class_name": candidate.class_name,
        },
    )

    assert inserted is True
    assert archive.metrics().bins_filled == 1
    assert archive.metrics().qd_score == score


def test_archive_replacement_on_higher_score():
    """Test that archive replaces lower-scoring builds with higher-scoring ones."""
    archive = ArchiveStore()

    # Insert first build
    candidate1 = _dummy_candidate(seed=42, build_id="build-1")
    score1 = score_from_metrics(candidate1.metrics_payload)
    descriptor1 = descriptor_values_from_metrics(candidate1.metrics_payload, archive.axes)
    assert archive.insert("build-1", score=score1, descriptor=descriptor1)

    # Insert second build with same descriptor but higher score
    candidate2 = _dummy_candidate(seed=43, build_id="build-2")
    metrics2 = {
        "mapping_t16": {
            "metrics": {"full_dps": 3000.0, "max_hit": 750.0},  # Higher DPS
            "defense": {"armour": 900.0, "resists": {"fire": 70.0}},
            "resources": {"life": 4200.0, "mana": 1300.0},
            "reservation": {"reserved_percent": 30.0, "available_percent": 70.0},
            "attributes": {"strength": 150.0, "dexterity": 120.0, "intelligence": 80.0},
        }
    }
    score2 = score_from_metrics(metrics2)
    descriptor2 = descriptor_values_from_metrics(metrics2, archive.axes)
    assert archive.insert("build-2", score=score2, descriptor=descriptor2)

    # Verify that build-2 replaced build-1
    assert archive.metrics().bins_filled == 1
    entry = archive.entry_for_bin(archive.entries()[0].bin_key)
    assert entry is not None
    assert entry.build_id == "build-2"
    assert entry.score == score2


def test_archive_multiple_bins():
    """Test that archive fills multiple bins with diverse builds."""
    archive = ArchiveStore()

    # Insert builds with different DPS values (should fill different damage bins)
    dps_values = [1000.0, 5000.0, 20000.0]
    for idx, dps in enumerate(dps_values):
        candidate = _dummy_candidate(seed=100 + idx, build_id=f"build-{idx}")
        metrics = {
            "mapping_t16": {
                "metrics": {"full_dps": dps, "max_hit": 750.0},
                "defense": {"armour": 900.0, "resists": {"fire": 70.0}},
                "resources": {"life": 4200.0, "mana": 1300.0},
                "reservation": {"reserved_percent": 30.0, "available_percent": 70.0},
                "attributes": {"strength": 150.0, "dexterity": 120.0, "intelligence": 80.0},
            }
        }
        score = score_from_metrics(metrics)
        descriptor = descriptor_values_from_metrics(metrics, archive.axes)
        archive.insert(f"build-{idx}", score=score, descriptor=descriptor)

    # Verify multiple bins are filled
    assert archive.metrics().bins_filled >= 2
    assert len(archive.entries()) >= 2


def test_archive_metrics_calculation():
    """Test that archive metrics are calculated correctly."""
    archive = ArchiveStore()

    # Insert 3 builds
    for idx in range(3):
        candidate = _dummy_candidate(seed=200 + idx, build_id=f"build-{idx}")
        metrics = {
            "mapping_t16": {
                "metrics": {"full_dps": 1000.0 * (idx + 1), "max_hit": 750.0},
                "defense": {"armour": 900.0, "resists": {"fire": 70.0}},
                "resources": {"life": 4200.0, "mana": 1300.0},
                "reservation": {"reserved_percent": 30.0, "available_percent": 70.0},
                "attributes": {"strength": 150.0, "dexterity": 120.0, "intelligence": 80.0},
            }
        }
        score = score_from_metrics(metrics)
        descriptor = descriptor_values_from_metrics(metrics, archive.axes)
        archive.insert(f"build-{idx}", score=score, descriptor=descriptor)

    metrics = archive.metrics()
    assert metrics.bins_filled > 0
    assert metrics.total_bins > 0
    assert 0.0 <= metrics.coverage <= 1.0
    assert metrics.qd_score > 0.0


def test_archive_descriptor_extraction():
    """Test that behavior characteristics are correctly extracted from metrics."""
    candidate = _dummy_candidate()
    archive = ArchiveStore()

    descriptor = descriptor_values_from_metrics(candidate.metrics_payload, archive.axes)

    # Should have one value per axis
    assert len(descriptor) == len(archive.axes)
    # Values should be numeric
    assert all(isinstance(v, float) for v in descriptor)
    # DPS should be positive
    assert descriptor[0] > 0.0


def test_archive_score_extraction():
    """Test that score is correctly extracted from metrics."""
    candidate = _dummy_candidate()

    score = score_from_metrics(candidate.metrics_payload)

    # Score should be the max full_dps across scenarios
    assert score == 1500.0


def test_archive_with_metadata():
    """Test that archive stores and retrieves metadata."""
    archive = ArchiveStore()
    candidate = _dummy_candidate()

    score = score_from_metrics(candidate.metrics_payload)
    descriptor = descriptor_values_from_metrics(candidate.metrics_payload, archive.axes)
    metadata = {
        "iteration": 1,
        "run_id": "test-run",
        "main_skill": candidate.main_skill_package,
        "class_name": candidate.class_name,
    }

    archive.insert(candidate.build_id, score=score, descriptor=descriptor, metadata=metadata)

    entry = archive.entry_for_bin(archive.entries()[0].bin_key)
    assert entry is not None
    assert entry.metadata["iteration"] == 1
    assert entry.metadata["run_id"] == "test-run"
    assert entry.metadata["main_skill"] == "arc"


def test_archive_entries_sorted():
    """Test that archive entries are returned in sorted order."""
    archive = ArchiveStore()

    # Insert builds in random order
    for idx in [2, 0, 1]:
        candidate = _dummy_candidate(seed=300 + idx, build_id=f"build-{idx}")
        metrics = {
            "mapping_t16": {
                "metrics": {"full_dps": 1000.0 + idx * 100, "max_hit": 750.0 + idx * 50},
                "defense": {"armour": 900.0, "resists": {"fire": 70.0}},
                "resources": {"life": 4200.0, "mana": 1300.0},
                "reservation": {"reserved_percent": 30.0, "available_percent": 70.0},
                "attributes": {"strength": 150.0, "dexterity": 120.0, "intelligence": 80.0},
            }
        }
        score = score_from_metrics(metrics)
        descriptor = descriptor_values_from_metrics(metrics, archive.axes)
        archive.insert(f"build-{idx}", score=score, descriptor=descriptor)

    entries = archive.entries()
    # Entries should be sorted by bin_key
    bin_keys = [entry.bin_key for entry in entries]
    assert bin_keys == sorted(bin_keys)
