"""Unit tests for CurriculumManager."""

import pytest

from backend.engine.curriculum import CurriculumManager, CurriculumPhase
from backend.engine.curriculum.scheduler import CurriculumState
from backend.engine.scenarios.loader import ScenarioGateThresholds, ScenarioReservationThreshold


def test_init_default():
    """Test default initialization starts at mapping phase."""
    mgr = CurriculumManager()
    assert mgr.enabled is True
    assert mgr.scheduler.current_phase == CurriculumPhase.MAPPING


def test_init_with_phase():
    """Test initialization with specific phase."""
    mgr = CurriculumManager(initial_phase=CurriculumPhase.BOSSING)
    assert mgr.scheduler.current_phase == CurriculumPhase.BOSSING


def test_init_with_zero_gates_phase():
    """Test initialization with zero-gates phase."""
    mgr = CurriculumManager(initial_phase=CurriculumPhase.ZERO_GATES)
    assert mgr.scheduler.current_phase == CurriculumPhase.ZERO_GATES


def test_init_disabled():
    """Test disabled mode pins to UBER phase."""
    mgr = CurriculumManager(enabled=False)
    assert mgr.enabled is False
    assert mgr.scheduler.current_phase == CurriculumPhase.UBER


def test_get_thresholds_mapping():
    """Test MAPPING phase applies 70% multiplier."""
    mgr = CurriculumManager(initial_phase=CurriculumPhase.MAPPING)
    base = ScenarioGateThresholds(
        resists={"fire": 75.0},
        reservation=ScenarioReservationThreshold(max_percent=99.0),
        attributes={},
        min_max_hit=3000.0,
        min_full_dps=0.0,
    )
    tightened = mgr.get_thresholds(base)
    assert tightened.min_max_hit == pytest.approx(3000.0 * 0.70)


def test_get_thresholds_uber():
    """Test UBER phase applies 100% multiplier (no tightening)."""
    mgr = CurriculumManager(initial_phase=CurriculumPhase.UBER)
    base = ScenarioGateThresholds(
        resists={"fire": 75.0},
        reservation=ScenarioReservationThreshold(max_percent=99.0),
        attributes={},
        min_max_hit=3000.0,
        min_full_dps=0.0,
    )
    tightened = mgr.get_thresholds(base)
    assert tightened.min_max_hit == pytest.approx(3000.0)
    assert tightened.reservation.max_percent == pytest.approx(99.0)
    assert tightened.resists["fire"] == pytest.approx(75.0)


def test_record_iteration_no_transition():
    """Test iteration recording without phase transition."""
    mgr = CurriculumManager()
    result = mgr.record_iteration([False] * 100)

    assert result["transitioned"] is False
    assert result["from_phase"] is None
    assert result["to_phase"] == "MAPPING"
    assert mgr.scheduler.current_phase == CurriculumPhase.MAPPING


def test_record_iteration_with_transition():
    """Test iteration recording triggers phase transition at >25%."""
    mgr = CurriculumManager()

    # First batch: 100 failures (0% pass rate)
    mgr.record_iteration([False] * 100)

    # Second batch: 34 passes (34/134 ~= 25.4% > 25% threshold)
    result = mgr.record_iteration([True] * 34)

    assert result["transitioned"] is True
    assert result["from_phase"] == "MAPPING"
    assert result["to_phase"] == "BOSSING"
    assert isinstance(result["summary"], dict)
    assert mgr.scheduler.current_phase == CurriculumPhase.BOSSING


def test_record_iteration_zero_gates_to_mapping():
    """Test zero-gates transition into Mapping when criteria are met."""
    mgr = CurriculumManager(initial_phase=CurriculumPhase.ZERO_GATES)

    result = mgr.record_iteration([True] * 20)

    assert result["transitioned"] is True
    assert result["from_phase"] == "ZERO_GATES"
    assert result["to_phase"] == "MAPPING"
    assert mgr.scheduler.current_phase == CurriculumPhase.MAPPING


def test_record_iteration_disabled():
    """Test disabled mode doesn't record."""
    mgr = CurriculumManager(enabled=False)
    result = mgr.record_iteration([True] * 100)
    assert result == {"transitioned": False}
    assert mgr.scheduler.current_phase == CurriculumPhase.UBER


def test_get_state():
    """Test state serialization."""
    mgr = CurriculumManager()

    # Keep phase feasibility exactly at the transition threshold (10%) so the
    # manager stays in the current phase.
    mgr.record_iteration([True] * 5 + [False] * 45)
    state = mgr.get_state()

    assert state["enabled"] is True
    assert state["phase"] == "MAPPING"
    assert state["total_samples"] == 50
    assert state["feasible_samples"] == 5
    assert state["phase_samples"] == 50
    assert state["phase_feasible_samples"] == 5


def test_from_state():
    """Test state deserialization."""
    state = {
        "enabled": True,
        "phase": "BOSSING",
        "total_samples": 100,
        "feasible_samples": 50,
        "phase_samples": 20,
        "phase_feasible_samples": 10,
    }
    mgr = CurriculumManager.from_state(state)

    assert mgr.enabled is True
    assert mgr.scheduler.current_phase == CurriculumPhase.BOSSING
    assert isinstance(mgr.scheduler.state, CurriculumState)
    assert mgr.scheduler.state.total_samples == 100
    assert mgr.scheduler.state.feasible_samples == 50
    assert mgr.scheduler.state.phase_samples == 20
    assert mgr.scheduler.state.phase_feasible_samples == 10


def test_from_state_disabled():
    """Test deserializing disabled state."""
    state = {"enabled": False}
    mgr = CurriculumManager.from_state(state)
    assert mgr.enabled is False
    assert mgr.scheduler.current_phase == CurriculumPhase.UBER


def test_from_state_with_legacy_phase_name():
    """Legacy phase names from persisted state should remain valid."""
    state = {
        "enabled": True,
        "phase": "MAPPING",
        "total_samples": 15,
        "feasible_samples": 3,
        "phase_samples": 15,
        "phase_feasible_samples": 3,
    }
    mgr = CurriculumManager.from_state(state)
    assert mgr.scheduler.current_phase == CurriculumPhase.MAPPING


def test_from_state_with_legacy_bossing_name():
    """Legacy BOSSING name should continue mapping to bossing."""
    state = {
        "enabled": True,
        "phase": "BOSSING",
        "total_samples": 20,
        "feasible_samples": 4,
        "phase_samples": 20,
        "phase_feasible_samples": 4,
    }
    mgr = CurriculumManager.from_state(state)
    assert mgr.scheduler.current_phase == CurriculumPhase.BOSSING
