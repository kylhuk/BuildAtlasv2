"""Curriculum learning module for progressive gate tightening."""

from backend.engine.curriculum.manager import CurriculumManager
from backend.engine.curriculum.scheduler import (
    FEASIBILITY_TRANSITION_THRESHOLD,
    PHASE_CONFIGS,
    CurriculumGateConfig,
    CurriculumPhase,
    CurriculumScheduler,
    CurriculumState,
)

__all__ = [
    "CurriculumPhase",
    "CurriculumGateConfig",
    "CurriculumState",
    "CurriculumScheduler",
    "PHASE_CONFIGS",
    "FEASIBILITY_TRANSITION_THRESHOLD",
    "CurriculumManager",
]
