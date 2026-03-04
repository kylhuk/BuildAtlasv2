from __future__ import annotations

import logging

from backend.engine.evaluation.normalized import ResistSnapshot


def test_resist_snapshot_logs_unknown_resist_name(caplog) -> None:
    snapshot = ResistSnapshot(fire=75.0, cold=75.0, lightning=75.0, chaos=0.0)

    with caplog.at_level(logging.WARNING):
        value = snapshot.get("ward")

    assert value == 0.0
    assert "Unknown resist name requested: ward" in caplog.text
