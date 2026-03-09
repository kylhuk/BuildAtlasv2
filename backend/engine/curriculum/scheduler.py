"""Curriculum learning scheduler for progressive gate tightening."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from backend.engine.scenarios.loader import ScenarioGateThresholds, ScenarioReservationThreshold


class CurriculumPhase(Enum):
    """Legacy curriculum phases ordered by difficulty."""

    ZERO_GATES = auto()
    MAPPING = auto()
    BOSSING = auto()
    PINNACLE = auto()
    UBER = auto()

    @property
    def _ordered_phases(self) -> tuple["CurriculumPhase", ...]:
        """Return deterministic phase order for curriculum progression."""
        return (
            CurriculumPhase.ZERO_GATES,
            CurriculumPhase.MAPPING,
            CurriculumPhase.BOSSING,
            CurriculumPhase.PINNACLE,
            CurriculumPhase.UBER,
        )

    @property
    def phase_number(self) -> int:
        """Return phase number in the deterministic progression ladder."""
        return self._ordered_phases.index(self)

    def next_phase(self) -> "CurriculumPhase":
        """Get next phase in deterministic progression ladder."""
        ordered = list(self._ordered_phases)
        current_idx = ordered.index(self)
        if current_idx < len(ordered) - 1:
            return ordered[current_idx + 1]
        return CurriculumPhase.UBER

    @property
    def description(self) -> str:
        """Return human-readable phase description."""
        descriptions = {
            CurriculumPhase.ZERO_GATES: "Zero-gates entry phase (permissive)",
            CurriculumPhase.MAPPING: "Mapping gates (easier)",
            CurriculumPhase.BOSSING: "Bossing gates (medium)",
            CurriculumPhase.PINNACLE: "Pinnacle gates (hardest)",
            CurriculumPhase.UBER: "Uber gates (most extreme)",
        }
        return descriptions[self]


@dataclass(frozen=True)
class CurriculumGateConfig:
    """Gate configuration for a curriculum phase."""

    phase: CurriculumPhase
    resists_multiplier: float
    attributes_multiplier: float
    max_hit_multiplier: float
    full_dps_multiplier: float
    reservation_multiplier: float
    feasibility_target: float


# Phase configurations: progressively tighten gates
PHASE_CONFIGS = {
    CurriculumPhase.ZERO_GATES: CurriculumGateConfig(
        phase=CurriculumPhase.ZERO_GATES,
        resists_multiplier=1.00,
        attributes_multiplier=1.00,
        max_hit_multiplier=1.00,
        full_dps_multiplier=1.00,
        reservation_multiplier=1.00,
        feasibility_target=0.80,
    ),
    CurriculumPhase.MAPPING: CurriculumGateConfig(
        phase=CurriculumPhase.MAPPING,
        resists_multiplier=0.70,
        attributes_multiplier=0.70,
        max_hit_multiplier=0.70,
        full_dps_multiplier=0.70,
        reservation_multiplier=1.22,
        feasibility_target=0.60,
    ),
    CurriculumPhase.BOSSING: CurriculumGateConfig(
        phase=CurriculumPhase.BOSSING,
        resists_multiplier=0.80,  # 80% of original
        attributes_multiplier=0.95,
        max_hit_multiplier=0.80,
        full_dps_multiplier=0.80,
        reservation_multiplier=1.15,
        feasibility_target=0.50,
    ),
    CurriculumPhase.PINNACLE: CurriculumGateConfig(
        phase=CurriculumPhase.PINNACLE,
        resists_multiplier=1.00,
        attributes_multiplier=1.05,
        max_hit_multiplier=1.00,
        full_dps_multiplier=1.00,
        reservation_multiplier=1.00,
        feasibility_target=0.10,
    ),
    CurriculumPhase.UBER: CurriculumGateConfig(
        phase=CurriculumPhase.UBER,
        resists_multiplier=1.00,  # 100% of original (full constraints)
        attributes_multiplier=1.10,
        max_hit_multiplier=1.00,
        full_dps_multiplier=1.00,
        reservation_multiplier=1.00,
        feasibility_target=0.05,  # Target 5% feasibility
    ),
}

MIN_PHASE_SAMPLES = 20
MAX_PHASE_SAMPLES = 1000
FEASIBILITY_TRANSITION_THRESHOLD = 0.25  # Transition when 25% of builds pass gates


@dataclass(frozen=True)
class CurriculumState:
    """Current curriculum learning state."""

    current_phase: CurriculumPhase
    total_samples: int
    feasible_samples: int
    phase_samples: int
    phase_feasible_samples: int

    @property
    def overall_feasibility(self) -> float:
        """Overall feasibility rate (0.0-1.0)."""
        if self.total_samples == 0:
            return 0.0
        return self.feasible_samples / self.total_samples

    @property
    def phase_feasibility(self) -> float:
        """Feasibility rate for current phase (0.0-1.0)."""
        if self.phase_samples == 0:
            return 0.0
        return self.phase_feasible_samples / self.phase_samples

    @property
    def should_transition(self) -> bool:
        """Advance only when phase samples and feasibility criteria align.

        Do not advance if UBER is active or if we still lack the minimum
        number of samples. Transition immediately once the cap is reached
        (to prevent indefinite waiting) or when feasibility exceeds the
        configured threshold.
        """
        if self.current_phase == CurriculumPhase.UBER:
            return False
        if self.phase_samples < MIN_PHASE_SAMPLES:
            return False
        if self.phase_samples >= MAX_PHASE_SAMPLES:
            return True
        return self.phase_feasibility > FEASIBILITY_TRANSITION_THRESHOLD

    def next_phase(self) -> CurriculumPhase:
        """Get next phase in curriculum."""
        phases = list(self.current_phase._ordered_phases)
        current_idx = phases.index(self.current_phase)
        if current_idx < len(phases) - 1:
            return phases[current_idx + 1]
        return CurriculumPhase.UBER

    def transition(self) -> CurriculumState:
        """Create new state with transitioned phase."""
        if not self.should_transition:
            return self
        return CurriculumState(
            current_phase=self.next_phase(),
            total_samples=self.total_samples,
            feasible_samples=self.feasible_samples,
            phase_samples=0,
            phase_feasible_samples=0,
        )

    def record_sample(self, is_feasible: bool) -> CurriculumState:
        """Record a new sample evaluation."""
        return CurriculumState(
            current_phase=self.current_phase,
            total_samples=self.total_samples + 1,
            feasible_samples=self.feasible_samples + (1 if is_feasible else 0),
            phase_samples=self.phase_samples + 1,
            phase_feasible_samples=self.phase_feasible_samples + (1 if is_feasible else 0),
        )


class CurriculumScheduler:
    """Manages progressive gate tightening curriculum."""

    def __init__(self, initial_phase: CurriculumPhase = CurriculumPhase.MAPPING):
        """Initialize scheduler at given phase."""
        self.state = CurriculumState(
            current_phase=initial_phase,
            total_samples=0,
            feasible_samples=0,
            phase_samples=0,
            phase_feasible_samples=0,
        )

    @property
    def current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase."""
        return self.state.current_phase

    @property
    def current_config(self) -> CurriculumGateConfig:
        """Get gate configuration for current phase."""
        return PHASE_CONFIGS[self.current_phase]

    def get_tightened_thresholds(
        self, base_thresholds: ScenarioGateThresholds
    ) -> ScenarioGateThresholds:
        """Apply curriculum multipliers to base gate thresholds."""
        config = self.current_config

        # Apply multipliers to resist thresholds
        tightened_resists = {
            name: value * config.resists_multiplier
            for name, value in base_thresholds.resists.items()
        }

        # Apply multipliers to attribute thresholds
        tightened_attributes = {
            name: value * config.attributes_multiplier
            for name, value in base_thresholds.attributes.items()
        }

        # Apply multipliers to max hit and full dps
        tightened_max_hit = base_thresholds.min_max_hit * config.max_hit_multiplier
        tightened_full_dps = base_thresholds.min_full_dps * config.full_dps_multiplier

        # Apply multiplier to reservation (higher multiplier = more lenient)
        tightened_reservation = ScenarioReservationThreshold(
            max_percent=base_thresholds.reservation.max_percent * config.reservation_multiplier
        )

        return ScenarioGateThresholds(
            resists=tightened_resists,
            reservation=tightened_reservation,
            attributes=tightened_attributes,
            min_max_hit=tightened_max_hit,
            min_full_dps=tightened_full_dps,
        )

    def record_evaluation(self, gate_passed: bool) -> None:
        """Record gate evaluation result and check for phase transition."""
        self.state = self.state.record_sample(gate_passed)

        # Check if we should transition to next phase
        if self.state.should_transition:
            self.state = self.state.transition()

    def get_state_summary(self) -> dict:
        """Get human-readable state summary."""
        return {
            "current_phase": self.current_phase.name,
            "phase_number": self.current_phase.phase_number,
            "phase_description": self.current_phase.description,
            "total_samples": self.state.total_samples,
            "feasible_samples": self.state.feasible_samples,
            "overall_feasibility": f"{self.state.overall_feasibility:.2%}",
            "phase_samples": self.state.phase_samples,
            "phase_feasible_samples": self.state.phase_feasible_samples,
            "phase_feasibility": f"{self.state.phase_feasibility:.2%}",
            "should_transition": self.state.should_transition,
            "feasibility_target": f"{self.current_config.feasibility_target:.2%}",
        }


__all__ = [
    "CurriculumPhase",
    "CurriculumGateConfig",
    "CurriculumState",
    "CurriculumScheduler",
    "PHASE_CONFIGS",
    "FEASIBILITY_TRANSITION_THRESHOLD",
]
