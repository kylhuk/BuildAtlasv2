"""Curriculum learning module for progressive gate tightening."""

from backend.engine.curriculum.scheduler import (
    CurriculumGateConfig,
    CurriculumPhase,
    CurriculumScheduler,
    CurriculumState,
    FEASIBILITY_TRANSITION_THRESHOLD,
    PHASE_CONFIGS,
)

__all__ = [
    "CurriculumPhase",
    "CurriculumGateConfig",
    "CurriculumState",
    "CurriculumScheduler",
    "PHASE_CONFIGS",
    "FEASIBILITY_TRANSITION_THRESHOLD",
]
