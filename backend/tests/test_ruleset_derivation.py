from __future__ import annotations

from pathlib import Path

import pytest

from backend.engine.ruleset import (
    derive_ruleset_id,
    read_pob_commit,
    scenario_version_from_profile,
)


def test_read_pob_commit_reads_file(tmp_path: Path) -> None:
    version_file = tmp_path / "VERSION"
    version_file.write_text("deadbeef")
    assert read_pob_commit(version_file) == "deadbeef"


def test_scenario_version_from_profile_defaults_to_first_template() -> None:
    version_tag = scenario_version_from_profile("pinnacle")
    assert version_tag == "pinnacle@v0"


def test_scenario_version_from_profile_respects_scenario_id() -> None:
    version_tag = scenario_version_from_profile("delve", scenario_id="delve_tier_2")
    assert version_tag == "delve_tier_2@v0"


def test_scenario_version_from_profile_errors_when_missing_template() -> None:
    with pytest.raises(ValueError):
        scenario_version_from_profile("missing-profile")


def test_derive_ruleset_id_uses_parts() -> None:
    ruleset = derive_ruleset_id(
        pob_commit="abc123",
        scenario_version="pinnacle@v0",
        price_snapshot_id="sentinel-2026",
    )
    assert ruleset == "pob:abc123|scenarios:pinnacle@v0|prices:sentinel-2026"
