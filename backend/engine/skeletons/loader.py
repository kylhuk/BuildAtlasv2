from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from backend.app.settings import settings
from backend.engine.skeletons.schema import Skeleton


def _skeleton_dir() -> Path:
    """Return the directory where skeletons are stored."""
    return settings.data_path / "skeletons"


def _skeleton_paths() -> Sequence[Path]:
    """Return a sequence of paths to skeleton JSON files."""
    directory = _skeleton_dir()
    if not directory.exists():
        return ()
    return tuple(sorted(directory.glob("*.json")))


def load_skeleton_from_path(path: Path) -> Skeleton:
    """Load a skeleton from a JSON file path."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    skeleton = Skeleton.from_dict(data)
    skeleton.validate()
    return skeleton


def list_skeletons() -> Sequence[Skeleton]:
    """List all available skeletons."""
    return tuple(load_skeleton_from_path(path) for path in _skeleton_paths())


def load_skeleton(skeleton_id: str) -> Skeleton:
    """Load a skeleton by its ID."""
    path = _skeleton_dir() / f"{skeleton_id}.json"
    if not path.exists():
        raise ValueError(f"Skeleton not found: {skeleton_id}")
    return load_skeleton_from_path(path)


__all__ = [
    "list_skeletons",
    "load_skeleton",
    "load_skeleton_from_path",
]
