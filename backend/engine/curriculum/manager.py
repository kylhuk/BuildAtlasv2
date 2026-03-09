"""Curriculum manager for batched evaluation recording."""

from __future__ import annotations

from typing import Any

from backend.engine.curriculum.scheduler import (
    CurriculumPhase,
    CurriculumScheduler,
    CurriculumState,
)
from backend.engine.scenarios.loader import ScenarioGateThresholds


class CurriculumManager:
    """Manages curriculum state with batching and persistence."""

    def __init__(self, enabled: bool = True, initial_phase: CurriculumPhase | None = None) -> None:
        """Initialize scheduler at given phase."""

        self.enabled = enabled
        if not enabled:
            # Disabled = UBER phase (100% thresholds, no tightening)
            self.scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.UBER)
        else:
            phase = initial_phase or CurriculumPhase.MAPPING
            self.scheduler = CurriculumScheduler(initial_phase=phase)

    @staticmethod
    def _restore_phase(phase_name: str | None) -> CurriculumPhase:
        """Resolve persisted phase names with backward-compatible fallbacks."""

        if not phase_name:
            return CurriculumPhase.MAPPING

        phase_name = str(phase_name).strip().upper()
        try:
            return CurriculumPhase[phase_name]
        except KeyError:
            # Keep migration path open for any legacy variants that may slip in.
            legacy_map = {
                "TUTORIAL": CurriculumPhase.MAPPING,
                "ACT_5": CurriculumPhase.MAPPING,
                "ACT5": CurriculumPhase.MAPPING,
                "ACT_10": CurriculumPhase.BOSSING,
                "ACT10": CurriculumPhase.BOSSING,
                "WHITE_MAPS": CurriculumPhase.PINNACLE,
                "YELLOW_MAPS": CurriculumPhase.PINNACLE,
                "RED_MAPS": CurriculumPhase.PINNACLE,
                "T16": CurriculumPhase.PINNACLE,
                "PINNACLE": CurriculumPhase.PINNACLE,
                "UBER": CurriculumPhase.UBER,
                "ZERO_GATES": CurriculumPhase.ZERO_GATES,
            }
            return legacy_map.get(phase_name, CurriculumPhase.MAPPING)

    def get_thresholds(self, base: ScenarioGateThresholds) -> ScenarioGateThresholds:
        """Get phase-adjusted thresholds. Call ONCE per iteration."""

        return self.scheduler.get_tightened_thresholds(base)

    def record_iteration(self, gate_passes: list[bool]) -> dict[str, Any]:
        """Record all gate results from an iteration.

        Returns transition info if phase changed.
        """

        if not self.enabled:
            return {"transitioned": False}

        old_phase = self.scheduler.current_phase

        # Record all evaluations first, then transition once at end.
        state = self.scheduler.state
        for passed in gate_passes:
            state = state.record_sample(passed)
        if state.should_transition:
            state = state.transition()
        self.scheduler.state = state

        new_phase = self.scheduler.current_phase

        return {
            "transitioned": new_phase != old_phase,
            "from_phase": old_phase.name if old_phase != new_phase else None,
            "to_phase": new_phase.name,
            "summary": self.scheduler.get_state_summary(),
        }

    def get_state(self) -> dict[str, Any]:
        """Serialize for persistence."""

        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "phase": self.scheduler.current_phase.name,
            "total_samples": self.scheduler.state.total_samples,
            "feasible_samples": self.scheduler.state.feasible_samples,
            "phase_samples": self.scheduler.state.phase_samples,
            "phase_feasible_samples": self.scheduler.state.phase_feasible_samples,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "CurriculumManager":
        """Restore from persisted state."""

        if not state.get("enabled", True):
            return cls(enabled=False)

        phase_name = state.get("phase")
        phase = cls._restore_phase(phase_name)

        manager = cls(enabled=True, initial_phase=phase)
        manager.scheduler.state = CurriculumState(
            current_phase=phase,
            total_samples=state.get("total_samples", 0),
            feasible_samples=state.get("feasible_samples", 0),
            phase_samples=state.get("phase_samples", 0),
            phase_feasible_samples=state.get("phase_feasible_samples", 0),
        )
        return manager
