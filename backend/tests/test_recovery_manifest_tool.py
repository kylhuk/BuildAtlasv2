import json
from pathlib import Path

from backend.tools import recovery_manifest


def _run_tool(args, capsys):
    exit_code = recovery_manifest.main(args)
    captured = capsys.readouterr()
    return exit_code, json.loads(captured.out)


def _write_summary(run_dir: Path, data: dict):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(data), encoding="utf-8")


def test_verify_missing_run_summary(tmp_path, capsys):
    base_path = tmp_path / "runs"
    exit_code, payload = _run_tool(
        ["verify", "--run-id", "missing", "--base-path", str(base_path)],
        capsys,
    )
    assert exit_code == 1
    assert payload["summary"]["exists"] is False
    assert payload["errors"]
    assert "does not exist" in payload["errors"][0]


def test_summary_with_missing_artifact_path(tmp_path, capsys):
    base_path = tmp_path / "runs"
    run_id = "artifact-missing"
    run_dir = base_path / run_id
    summary_payload = {
        "run_id": run_id,
        "paths": {
            "archive": "archive.json",
            "missing": "missing.json",
        },
        "evaluation": {"successes": 2, "failures": 1, "errors": 0},
    }
    _write_summary(run_dir, summary_payload)
    (run_dir / "archive.json").write_text("archive data", encoding="utf-8")
    exit_code, payload = _run_tool(
        ["verify", "--run-id", run_id, "--base-path", str(base_path)],
        capsys,
    )
    assert exit_code == 0
    artifacts = payload["artifacts"]
    assert artifacts["present_count"] == 1
    assert artifacts["missing_count"] == 1
    assert artifacts["details"][0]["name"] == "archive"
    assert artifacts["details"][0]["exists"] is True
    assert artifacts["details"][1]["exists"] is False
    assert any("missing" in warning for warning in payload["warnings"])
    assert payload["outcome"]["verified"] == 2
    assert payload["outcome"]["failed"] == 1


def test_replay_dry_run(tmp_path, capsys):
    base_path = tmp_path / "runs"
    run_id = "dry-run"
    run_dir = base_path / run_id
    _write_summary(run_dir, {"run_id": run_id})
    exit_code, payload = _run_tool(
        [
            "replay",
            "--run-id",
            run_id,
            "--base-path",
            str(base_path),
            "--dry-run",
        ],
        capsys,
    )
    assert exit_code == 0
    assert payload["replay"]["dry_run"] is True
    assert payload["replay"]["request_path"] is None
    assert not any(run_dir.glob("replay_request_*.json"))


def test_replay_persists_marker(tmp_path, capsys):
    base_path = tmp_path / "runs"
    run_id = "persist-marker"
    run_dir = base_path / run_id
    _write_summary(run_dir, {"run_id": run_id})
    exit_code, payload = _run_tool(
        [
            "replay",
            "--run-id",
            run_id,
            "--base-path",
            str(base_path),
            "--no-dry-run",
            "--reason",
            "manual",
        ],
        capsys,
    )
    assert exit_code == 0
    request_path = payload["replay"]["request_path"]
    assert request_path
    marker = Path(request_path)
    assert marker.exists()
    marker_data = json.loads(marker.read_text(encoding="utf-8"))
    assert marker_data["dry_run"] is False
    assert marker_data["reason"] == "manual"
    assert marker_data["run_id"] == run_id
