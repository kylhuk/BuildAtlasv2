from __future__ import annotations

import argparse

import pytest

from backend.app.api.models import BuildStatus
from backend.tools import reevaluate_builds


class _DummyRepo:
    def __init__(self) -> None:
        self.updated: list[tuple[str, str, bool]] = []
        self.rows: list[dict[str, object]] = []
        self.get_rows: dict[str, dict[str, object] | None] = {}

    def update_build_ruleset(
        self,
        build_id: str,
        *,
        ruleset_id: str | None = None,
        is_stale: bool | None = None,
    ) -> None:
        self.updated.append((build_id, str(ruleset_id), bool(is_stale)))

    def get_build(self, build_id: str):
        return self.get_rows.get(build_id)

    def list_builds(self, *, filters, sort_by, sort_dir, limit, offset):
        del sort_by, sort_dir, limit, offset
        assert filters.include_stale is True
        return self.rows


class _DummyEvaluator:
    def evaluate_build(self, build_id: str):
        if build_id == "broken":
            raise RuntimeError("boom")
        return BuildStatus.evaluated, [object(), object()]


def test_derive_target_ruleset_uses_explicit_override() -> None:
    args = argparse.Namespace(target_ruleset_id="explicit", profile_id=None)
    ruleset = reevaluate_builds._derive_target_ruleset(args, [{"profile_id": "pinnacle"}])
    assert ruleset == "explicit"


def test_derive_target_ruleset_from_profile(monkeypatch) -> None:
    monkeypatch.setattr(
        reevaluate_builds, "scenario_version_from_profile", lambda _profile: "pinnacle@v2"
    )
    monkeypatch.setattr(reevaluate_builds, "read_pob_commit", lambda: "abc123")
    args = argparse.Namespace(
        target_ruleset_id=None,
        profile_id="pinnacle",
        scenario_version=None,
        pob_commit=None,
        price_snapshot_id="snapshot-1",
    )
    ruleset = reevaluate_builds._derive_target_ruleset(args, [{"profile_id": "pinnacle"}])
    assert ruleset == "pob:abc123|scenarios:pinnacle@v2|prices:snapshot-1"


def test_derive_target_ruleset_requires_profile_for_mixed_builds() -> None:
    args = argparse.Namespace(
        target_ruleset_id=None,
        profile_id=None,
        scenario_version=None,
        pob_commit=None,
        price_snapshot_id="snapshot-1",
    )
    with pytest.raises(RuntimeError, match="profile_id must be provided"):
        reevaluate_builds._derive_target_ruleset(
            args,
            [{"profile_id": "pinnacle"}, {"profile_id": "mapping"}],
        )


def test_select_build_rows_deduplicates_and_keeps_include_stale() -> None:
    repo = _DummyRepo()
    repo.get_rows = {
        "build-1": {"build_id": "build-1", "profile_id": "pinnacle"},
        "missing": None,
    }
    repo.rows = [
        {"build_id": "build-1", "profile_id": "pinnacle"},
        {"build_id": "build-2", "profile_id": "pinnacle"},
    ]
    args = argparse.Namespace(
        build_ids=["build-1", "missing"],
        profile_id="pinnacle",
        ruleset_filter=None,
        limit=10,
        offset=0,
    )
    rows = reevaluate_builds._select_build_rows(repo, args)
    assert [row["build_id"] for row in rows] == ["build-1", "build-2"]


def test_evaluate_builds_updates_ruleset_and_collects_errors() -> None:
    repo = _DummyRepo()
    evaluator = _DummyEvaluator()
    builds = [
        {"build_id": "good"},
        {"build_id": "broken"},
        {"profile_id": "ignored-no-id"},
    ]
    results = reevaluate_builds._evaluate_builds(repo, evaluator, builds, "ruleset-next")

    assert repo.updated == [
        ("good", "ruleset-next", False),
        ("broken", "ruleset-next", False),
    ]
    assert results[0] == {"build_id": "good", "status": "evaluated", "scenarios": 2}
    assert results[1]["build_id"] == "broken"
    assert results[1]["status"] == "error"


def test_affected_profiles_uses_selected_profile_or_unique_build_profiles() -> None:
    args = argparse.Namespace(profile_id="delve")
    assert reevaluate_builds._affected_profiles(args, [{"profile_id": "pinnacle"}]) == ["delve"]

    args = argparse.Namespace(profile_id=None)
    profiles = reevaluate_builds._affected_profiles(
        args,
        [
            {"profile_id": "support"},
            {"profile_id": "pinnacle"},
            {"profile_id": "support"},
        ],
    )
    assert profiles == ["pinnacle", "support"]
