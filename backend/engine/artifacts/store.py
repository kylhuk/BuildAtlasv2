from __future__ import annotations

import gzip
import json
import shutil
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Optional, Tuple, Union
from xml.sax.saxutils import escape, quoteattr

RULESET_PATTERN = re.compile(
    r"^pob:(?P<pob>[^|]+)\|scenarios:(?P<scenarios>[^|]+)\|prices:(?P<prices>[^|]+)$"
)

BUILD_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path
    build_xml: Path
    code: Path
    genome: Path
    scenarios_used: Path
    metrics_raw: Path
    build_details: Path
    surrogate_prediction: Path
    constraints: Path


@dataclass(frozen=True)
class ArtifactProvenance:
    build_id: str
    xml_hash: str | None
    paths: ArtifactPaths


@dataclass(frozen=True)
class BuildArtifacts:
    xml: str | None
    code: str
    genome: Optional[Any]
    scenarios_used: Optional[Any]
    raw_metrics: Optional[Any]
    build_details: Optional[Any]
    surrogate_prediction: Optional[Any]
    constraints: Optional[Any]


def _resolve_base(base_path: Optional[Union[str, Path]]) -> Path:
    candidate = Path(base_path) if base_path is not None else Path.cwd()
    return candidate.resolve()


def _validate_build_id(build_id: str) -> None:
    if not isinstance(build_id, str):
        raise ValueError("build_id must be a string")
    if not build_id:
        raise ValueError("build_id cannot be empty")
    if ".." in build_id:
        raise ValueError("build_id cannot contain ..")
    if not BUILD_ID_PATTERN.fullmatch(build_id):
        raise ValueError("build_id contains invalid characters")


def _safe_build_dir(build_id: str, base_path: Optional[Union[str, Path]] = None) -> Path:
    _validate_build_id(build_id)
    builds_root = _resolve_base(base_path) / "data" / "builds"
    builds_root_resolved = builds_root.resolve(strict=False)
    candidate_dir = (builds_root / build_id).resolve(strict=False)
    if not candidate_dir.is_relative_to(builds_root_resolved):
        raise ValueError("build_id resolves outside of the builds directory")
    return candidate_dir


def purge_build_artifacts(build_id: str, base_path: Optional[Union[str, Path]] = None) -> None:
    build_dir = _safe_build_dir(build_id, base_path)
    if not build_dir.exists():
        return
    shutil.rmtree(build_dir, ignore_errors=True)

def artifact_paths(build_id: str, base_path: Optional[Union[str, Path]] = None) -> ArtifactPaths:
    base_dir = _safe_build_dir(build_id, base_path)
    return ArtifactPaths(
        base_dir=base_dir,
        build_xml=base_dir / "build.xml.gz",
        code=base_dir / "build.code.txt",
        genome=base_dir / "genome.json",
        scenarios_used=base_dir / "scenarios_used.json",
        metrics_raw=base_dir / "metrics_raw.json",
        build_details=base_dir / "build_details.json",
        surrogate_prediction=base_dir / "surrogate_prediction.json",
        constraints=base_dir / "constraints.json",
    )


def _normalize_xml_bytes(xml: Union[str, bytes]) -> bytes:
    if isinstance(xml, bytes):
        text = xml.decode("utf-8")
    else:
        text = xml
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.encode("utf-8")


def _normalize_and_escape_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    if normalized.strip() == "":
        return ""
    return escape(normalized)


def _serialize_element(element: ET.Element) -> str:
    parts = [f"<{element.tag}"]
    if element.attrib:
        for name, value in sorted(element.attrib.items()):
            parts.append(f" {name}={quoteattr(value)}")
    parts.append(">")
    text_segment = _normalize_and_escape_text(element.text)
    if text_segment:
        parts.append(text_segment)
    for child in element:
        parts.append(_serialize_element(child))
        tail_segment = _normalize_and_escape_text(child.tail)
        if tail_segment:
            parts.append(tail_segment)
    parts.append(f"</{element.tag}>")
    return "".join(parts)


def _canonicalize_xml_bytes(xml_bytes: bytes) -> bytes:
    root = ET.fromstring(xml_bytes.decode("utf-8"))
    return _serialize_element(root).encode("utf-8")


def canonical_xml_hash(xml: Union[str, bytes]) -> str:
    xml_bytes = xml if isinstance(xml, bytes) else xml.encode("utf-8")
    try:
        canonical = _canonicalize_xml_bytes(xml_bytes)
    except (ET.ParseError, UnicodeDecodeError):
        canonical = _normalize_xml_bytes(xml_bytes)
    return sha256(canonical).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as handle:
        handle.write(data)
    tmp.replace(path)


def _atomic_write_gzip(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as writer:
            writer.write(data)
    tmp.replace(path)


def _write_optional_json(path: Path, payload: Any) -> None:
    if payload is None:
        return
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(path, content)


def write_build_artifacts(
    build_id: str,
    xml: Union[str, bytes, None],
    code: Union[str, bytes],
    genome: Optional[Any] = None,
    scenarios_used: Optional[Any] = None,
    raw_metrics: Optional[Any] = None,
    build_details: Optional[Any] = None,
    base_path: Optional[Union[str, Path]] = None,
) -> ArtifactProvenance:
    xml_bytes: bytes | None = None
    if xml is not None:
        xml_bytes = xml if isinstance(xml, bytes) else xml.encode("utf-8")
    hash_value = canonical_xml_hash(xml_bytes) if xml_bytes is not None else None
    paths = artifact_paths(build_id, base_path)

    if xml_bytes is not None:
        _atomic_write_gzip(paths.build_xml, xml_bytes)
    _atomic_write_bytes(paths.code, code if isinstance(code, bytes) else code.encode("utf-8"))

    _write_optional_json(paths.genome, genome)
    _write_optional_json(paths.scenarios_used, scenarios_used)
    _write_optional_json(paths.metrics_raw, raw_metrics)
    _write_optional_json(paths.build_details, build_details)

    return ArtifactProvenance(build_id=build_id, xml_hash=hash_value, paths=paths)


def write_build_constraints(
    build_id: str,
    payload: Any,
    base_path: Optional[Union[str, Path]] = None,
) -> Path:
    paths = artifact_paths(build_id, base_path)
    content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _atomic_write_bytes(paths.constraints, content)
    return paths.constraints



def _read_optional_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_build_artifacts(
    build_id: str, base_path: Optional[Union[str, Path]] = None
) -> BuildArtifacts:
    paths = artifact_paths(build_id, base_path)
    if not paths.base_dir.exists():
        raise FileNotFoundError(f"artifacts missing for {build_id}")
    xml_content: str | None = None
    if paths.build_xml.exists():
        with gzip.open(paths.build_xml, "rb") as handle:
            xml_content = handle.read().decode("utf-8")

    return BuildArtifacts(
        xml=xml_content,
        code=paths.code.read_text(encoding="utf-8"),
        genome=_read_optional_json(paths.genome),
        scenarios_used=_read_optional_json(paths.scenarios_used),
        raw_metrics=_read_optional_json(paths.metrics_raw),
        build_details=_read_optional_json(paths.build_details),
        surrogate_prediction=_read_optional_json(paths.surrogate_prediction),
        constraints=_read_optional_json(paths.constraints),
    )


def _validate_ruleset_token(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    candidate = value.strip()
    if not candidate:
        raise ValueError(f"{name} cannot be empty")
    if "|" in candidate:
        raise ValueError(f"{name} cannot contain '|'")
    return candidate


def format_ruleset_id(commit: str, scenarios: str, prices: str) -> str:
    commit_token = _validate_ruleset_token("commit", commit)
    scenarios_token = _validate_ruleset_token("scenarios", scenarios)
    prices_token = _validate_ruleset_token("prices", prices)
    return f"pob:{commit_token}|scenarios:{scenarios_token}|prices:{prices_token}"


def parse_ruleset_id(ruleset_id: str) -> Tuple[str, str, str]:
    match = RULESET_PATTERN.match(ruleset_id)
    if not match:
        raise ValueError("ruleset id must be in pob:<commit>|scenarios:<ver>|prices:<id> format")
    commit = _validate_ruleset_token("commit", match.group("pob"))
    scenarios = _validate_ruleset_token("scenarios", match.group("scenarios"))
    prices = _validate_ruleset_token("prices", match.group("prices"))
    return (commit, scenarios, prices)
