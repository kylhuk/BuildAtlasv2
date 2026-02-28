from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from backend.app.settings import settings

REPLAY_MARKER_PREFIX = "replay_request"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    base_run_path = settings.data_path / "runs"
    parser = argparse.ArgumentParser(
        description="Inspect run manifest artifacts and optionally request a replay"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_arguments(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--run-id",
            required=True,
            help="Identifier of the run to inspect",
        )
        subparser.add_argument(
            "--base-path",
            type=Path,
            default=base_run_path,
            help="Base directory containing run subdirectories (default: data/runs)",
        )

    verify_parser = subparsers.add_parser(
        "verify", help="Report manifest/artifact availability"
    )
    _add_common_arguments(verify_parser)

    replay_parser = subparsers.add_parser(
        "replay", help="Record a replay request marker for the run"
    )
    _add_common_arguments(replay_parser)
    dry_run_group = replay_parser.add_mutually_exclusive_group()
    dry_run_group.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Only report; do not persist a replay marker",
    )
    dry_run_group.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Persist a replay request marker",
    )
    replay_parser.set_defaults(dry_run=True)
    replay_parser.add_argument(
        "--reason",
        default=None,
        help="Optional reason to record with the replay request",
    )
    return parser.parse_args(argv)


def _inspect_summary(
    run_dir: Path,
) -> tuple[dict[str, Any], Mapping[str, Any] | None, list[str], list[str]]:
    summary_path = run_dir / "summary.json"
    summary_info: dict[str, Any] = {
        "run_dir": str(run_dir),
        "path": str(summary_path),
        "run_dir_exists": run_dir.exists(),
        "exists": False,
        "parsed": False,
        "error": None,
    }
    errors: list[str] = []
    warnings: list[str] = []
    if not run_dir.exists():
        message = f"run directory {run_dir} does not exist"
        summary_info["error"] = message
        errors.append(message)
        return summary_info, None, errors, warnings
    if not summary_path.exists():
        message = f"summary manifest {summary_path} not found"
        summary_info["error"] = message
        errors.append(message)
        return summary_info, None, errors, warnings
    summary_info["exists"] = True
    try:
        content = summary_path.read_text(encoding="utf-8")
        payload = json.loads(content)
        if not isinstance(payload, Mapping):
            raise ValueError("summary manifest root is not a mapping")
        summary_info["parsed"] = True
        return summary_info, payload, errors, warnings
    except json.JSONDecodeError as exc:
        message = (
            f"summary manifest {summary_path} could not be parsed: {exc}"
        )
        summary_info["error"] = message
        errors.append(message)
    except (OSError, ValueError) as exc:
        message = f"summary manifest {summary_path} unavailable: {exc}"
        summary_info["error"] = message
        errors.append(message)
    return summary_info, None, errors, warnings


def _inspect_artifacts(
    summary_data: Mapping[str, Any] | None,
    run_dir: Path,
) -> tuple[dict[str, Any], list[str]]:
    details: list[dict[str, Any]] = []
    warnings: list[str] = []
    total = present = missing = 0
    if isinstance(summary_data, Mapping):
        candidate_paths = summary_data.get("paths")
        if isinstance(candidate_paths, Mapping):
            for name in sorted(candidate_paths):
                raw_value = candidate_paths.get(name)
                resolved_path: Path | None = None
                exists = False
                value_str = str(raw_value or "").strip()
                if value_str:
                    candidate_path = Path(value_str)
                    resolved_path = (
                        candidate_path
                        if candidate_path.is_absolute()
                        else run_dir / candidate_path
                    )
                    exists = resolved_path.exists()
                if exists:
                    present += 1
                else:
                    missing += 1
                    target = resolved_path or run_dir
                    warnings.append(f"artifact {name} missing at {target}")
                total += 1
                details.append(
                    {
                        "name": str(name),
                        "path": str(resolved_path) if resolved_path else "",
                        "exists": exists,
                    }
                )
    return (
        {
            "total": total,
            "present_count": present,
            "missing_count": missing,
            "details": details,
        },
        warnings,
    )


def _build_outcome_summary(summary_data: Mapping[str, Any] | None) -> dict[str, Any]:
    verified = None
    failed = None
    status = None
    if isinstance(summary_data, Mapping):
        evaluation = summary_data.get("evaluation")
        if isinstance(evaluation, Mapping):
            successes = evaluation.get("successes")
            if isinstance(successes, int):
                verified = successes
            failure_total = 0
            has_failure = False
            for key in ("failures", "errors"):
                value = evaluation.get(key)
                if isinstance(value, int):
                    failure_total += value
                    has_failure = True
            if has_failure:
                failed = failure_total
        status = summary_data.get("status")
    return {"verified": verified, "failed": failed, "status": status}


def _write_replay_marker(
    run_dir: Path,
    run_id: str,
    dry_run: bool,
    reason: str | None,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    marker_name = f"{REPLAY_MARKER_PREFIX}_{timestamp}.json"
    marker_path = run_dir / marker_name
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "reason": reason or "",
    }
    marker_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return marker_path


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    base_path = Path(args.base_path)
    run_dir = base_path / args.run_id
    summary_info, summary_data, summary_errors, summary_warnings = _inspect_summary(
        run_dir
    )
    artifact_report, artifact_warnings = _inspect_artifacts(summary_data, run_dir)
    errors = [*summary_errors]
    warnings = [*summary_warnings, *artifact_warnings]
    outcome = _build_outcome_summary(summary_data)
    replay_section: dict[str, Any] = {
        "supported": run_dir.exists(),
        "requested": args.command == "replay",
        "dry_run": False,
        "message": "no replay requested",
        "request_path": None,
    }
    if args.command == "replay":
        replay_section["dry_run"] = bool(getattr(args, "dry_run", True))
        if not run_dir.exists():
            error = f"run directory {run_dir} does not exist for replay"
            errors.append(error)
            replay_section["supported"] = False
            replay_section["message"] = "cannot record replay request"
        else:
            if replay_section["dry_run"]:
                replay_section["message"] = "dry run; replay marker not persisted"
            else:
                try:
                    marker = _write_replay_marker(
                        run_dir,
                        args.run_id,
                        False,
                        args.reason,
                    )
                    replay_section["request_path"] = str(marker)
                    replay_section["message"] = "replay request marker recorded"
                except OSError as exc:
                    errors.append(f"failed to persist replay marker: {exc}")
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "summary": summary_info,
        "artifacts": artifact_report,
        "outcome": outcome,
        "replay": replay_section,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
