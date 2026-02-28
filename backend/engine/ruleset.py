from __future__ import annotations

from pathlib import Path

from backend.engine.artifacts.store import format_ruleset_id
from backend.engine.scenarios.loader import list_templates, scenario_version_tag

REPO_ROOT = Path(__file__).resolve().parents[2]
POB_VERSION_PATH = REPO_ROOT / "pob" / "VERSION"
DEFAULT_PRICE_SNAPSHOT_ID = "local"


def read_pob_commit(version_path: Path | str | None = None) -> str:
    path = Path(version_path) if version_path is not None else POB_VERSION_PATH
    return path.read_text(encoding="utf-8").strip()


def scenario_version_from_profile(profile_id: str, scenario_id: str | None = None) -> str:
    normalized = profile_id.strip()
    templates = [template for template in list_templates() if template.profile_id == normalized]
    if not templates:
        raise ValueError(f"no scenario templates found for profile_id={profile_id}")
    if scenario_id is not None:
        for template in templates:
            if template.scenario_id == scenario_id:
                return scenario_version_tag(template)
        raise ValueError(
            f"no scenario template for profile_id={profile_id} with scenario_id={scenario_id}"
        )
    return scenario_version_tag(templates[0])


def derive_ruleset_id(*, pob_commit: str, scenario_version: str, price_snapshot_id: str) -> str:
    return format_ruleset_id(pob_commit, scenario_version, price_snapshot_id)


__all__ = [
    "DEFAULT_PRICE_SNAPSHOT_ID",
    "derive_ruleset_id",
    "read_pob_commit",
    "scenario_version_from_profile",
]
