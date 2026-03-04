"""Tests for curriculum learning scheduler."""

import pytest

from backend.engine.curriculum import (
    CurriculumPhase,
    CurriculumScheduler,
    CurriculumState,
    FEASIBILITY_TRANSITION_THRESHOLD,
    PHASE_CONFIGS,
)
from backend.engine.scenarios.loader import ScenarioGateThresholds, ScenarioReservationThreshold


class TestCurriculumPhase:
    """Test CurriculumPhase enum."""

    def test_phase_ordering(self):
        """Test phases are ordered correctly."""
        phases = list(CurriculumPhase)
        assert phases[0] == CurriculumPhase.MAPPING
        assert phases[1] == CurriculumPhase.BOSSING
        assert phases[2] == CurriculumPhase.PINNACLE
        assert phases[3] == CurriculumPhase.UBER

    def test_phase_numbers(self):
        """Test phase numbers are sequential."""
        assert CurriculumPhase.MAPPING.phase_number == 1
        assert CurriculumPhase.BOSSING.phase_number == 2
        assert CurriculumPhase.PINNACLE.phase_number == 3
        assert CurriculumPhase.UBER.phase_number == 4

    def test_phase_descriptions(self):
        """Test phase descriptions are meaningful."""
        assert "Mapping" in CurriculumPhase.MAPPING.description
        assert "Bossing" in CurriculumPhase.BOSSING.description
        assert "Pinnacle" in CurriculumPhase.PINNACLE.description
        assert "Uber" in CurriculumPhase.UBER.description


class TestPhaseConfigs:
    """Test phase configurations."""

    def test_all_phases_configured(self):
        """Test all phases have configurations."""
        for phase in CurriculumPhase:
            assert phase in PHASE_CONFIGS
            config = PHASE_CONFIGS[phase]
            assert config.phase == phase

    def test_multipliers_progression(self):
        """Test multipliers progress from easy to hard."""
        mapping = PHASE_CONFIGS[CurriculumPhase.MAPPING]
        bossing = PHASE_CONFIGS[CurriculumPhase.BOSSING]
        pinnacle = PHASE_CONFIGS[CurriculumPhase.PINNACLE]
        uber = PHASE_CONFIGS[CurriculumPhase.UBER]

        # Resist multipliers should increase (gates get tighter)
        assert mapping.resists_multiplier < bossing.resists_multiplier
        assert bossing.resists_multiplier < pinnacle.resists_multiplier
        assert pinnacle.resists_multiplier <= uber.resists_multiplier

        # Reservation multipliers should decrease (gates get tighter)
        assert mapping.reservation_multiplier > bossing.reservation_multiplier
        assert bossing.reservation_multiplier > pinnacle.reservation_multiplier
        assert pinnacle.reservation_multiplier >= uber.reservation_multiplier

    def test_feasibility_targets_decrease(self):
        """Test feasibility targets decrease as difficulty increases."""
        mapping = PHASE_CONFIGS[CurriculumPhase.MAPPING]
        bossing = PHASE_CONFIGS[CurriculumPhase.BOSSING]
        pinnacle = PHASE_CONFIGS[CurriculumPhase.PINNACLE]
        uber = PHASE_CONFIGS[CurriculumPhase.UBER]

        assert mapping.feasibility_target > bossing.feasibility_target
        assert bossing.feasibility_target > pinnacle.feasibility_target
        assert pinnacle.feasibility_target > uber.feasibility_target


class TestCurriculumState:
    """Test CurriculumState data class."""

    def test_initial_state(self):
        """Test initial state creation."""
        state = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=0,
            feasible_samples=0,
            phase_samples=0,
            phase_feasible_samples=0,
        )
        assert state.overall_feasibility == 0.0
        assert state.phase_feasibility == 0.0
        assert not state.should_transition

    def test_feasibility_calculation(self):
        """Test feasibility rate calculations."""
        state = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=10,
            feasible_samples=7,
            phase_samples=5,
            phase_feasible_samples=3,
        )
        assert state.overall_feasibility == 0.7
        assert state.phase_feasibility == 0.6

    def test_transition_criteria(self):
        """Test transition criteria (>10% feasibility)."""
        # Below threshold
        state_below = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=100,
            feasible_samples=10,
            phase_samples=100,
            phase_feasible_samples=10,
        )
        assert state_below.phase_feasibility == 0.1
        assert not state_below.should_transition

        # Above threshold
        state_above = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=100,
            feasible_samples=12,
            phase_samples=100,
            phase_feasible_samples=12,
        )
        assert state_above.phase_feasibility == 0.12
        assert state_above.should_transition

    def test_next_phase(self):
        """Test next phase progression."""
        state_mapping = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=0,
            feasible_samples=0,
            phase_samples=0,
            phase_feasible_samples=0,
        )
        assert state_mapping.next_phase() == CurriculumPhase.BOSSING

        state_bossing = CurriculumState(
            current_phase=CurriculumPhase.BOSSING,
            total_samples=0,
            feasible_samples=0,
            phase_samples=0,
            phase_feasible_samples=0,
        )
        assert state_bossing.next_phase() == CurriculumPhase.PINNACLE

        state_pinnacle = CurriculumState(
            current_phase=CurriculumPhase.PINNACLE,
            total_samples=0,
            feasible_samples=0,
            phase_samples=0,
            phase_feasible_samples=0,
        )
        assert state_pinnacle.next_phase() == CurriculumPhase.UBER

        state_uber = CurriculumState(
            current_phase=CurriculumPhase.UBER,
            total_samples=0,
            feasible_samples=0,
            phase_samples=0,
            phase_feasible_samples=0,
        )
        assert state_uber.next_phase() == CurriculumPhase.UBER  # Stays at UBER

    def test_transition_resets_phase_counters(self):
        """Test transition resets phase-specific counters."""
        state = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=100,
            feasible_samples=50,
            phase_samples=100,
            phase_feasible_samples=50,
        )
        transitioned = state.transition()

        assert transitioned.current_phase == CurriculumPhase.BOSSING
        assert transitioned.total_samples == 100
        assert transitioned.feasible_samples == 50
        assert transitioned.phase_samples == 0
        assert transitioned.phase_feasible_samples == 0

    def test_record_sample_feasible(self):
        """Test recording a feasible sample."""
        state = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=10,
            feasible_samples=5,
            phase_samples=10,
            phase_feasible_samples=5,
        )
        new_state = state.record_sample(is_feasible=True)

        assert new_state.total_samples == 11
        assert new_state.feasible_samples == 6
        assert new_state.phase_samples == 11
        assert new_state.phase_feasible_samples == 6

    def test_record_sample_infeasible(self):
        """Test recording an infeasible sample."""
        state = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=10,
            feasible_samples=5,
            phase_samples=10,
            phase_feasible_samples=5,
        )
        new_state = state.record_sample(is_feasible=False)

        assert new_state.total_samples == 11
        assert new_state.feasible_samples == 5
        assert new_state.phase_samples == 11
        assert new_state.phase_feasible_samples == 5


class TestCurriculumScheduler:
    """Test CurriculumScheduler class."""

    def test_initialization(self):
        """Test scheduler initialization."""
        scheduler = CurriculumScheduler()
        assert scheduler.current_phase == CurriculumPhase.MAPPING
        assert scheduler.state.total_samples == 0
        assert scheduler.state.feasible_samples == 0

    def test_initialization_with_phase(self):
        """Test scheduler initialization with specific phase."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.BOSSING)
        assert scheduler.current_phase == CurriculumPhase.BOSSING

    def test_current_config(self):
        """Test getting current phase configuration."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)
        config = scheduler.current_config
        assert config.phase == CurriculumPhase.MAPPING
        assert config.resists_multiplier == 0.70

    def test_get_tightened_thresholds_mapping(self):
        """Test threshold tightening for Mapping phase."""
        base_thresholds = ScenarioGateThresholds(
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            reservation=ScenarioReservationThreshold(max_percent=99.0),
            attributes={"strength": 100.0, "dexterity": 100.0, "intelligence": 100.0},
            min_max_hit=15000.0,
            min_full_dps=500000.0,
        )

        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)
        tightened = scheduler.get_tightened_thresholds(base_thresholds)

        # Mapping phase: 0.70 multiplier for resists
        assert tightened.resists["fire"] == pytest.approx(75.0 * 0.70)
        assert tightened.resists["cold"] == pytest.approx(75.0 * 0.70)

        # Mapping phase: 1.30 multiplier for reservation (more lenient)
        assert tightened.reservation.max_percent == pytest.approx(99.0 * 1.30)

        # Mapping phase: 0.70 multiplier for max_hit and full_dps
        assert tightened.min_max_hit == pytest.approx(15000.0 * 0.70)
        assert tightened.min_full_dps == pytest.approx(500000.0 * 0.70)

    def test_get_tightened_thresholds_uber(self):
        """Test threshold tightening for Uber phase (no tightening)."""
        base_thresholds = ScenarioGateThresholds(
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            reservation=ScenarioReservationThreshold(max_percent=99.0),
            attributes={"strength": 100.0, "dexterity": 100.0, "intelligence": 100.0},
            min_max_hit=15000.0,
            min_full_dps=500000.0,
        )

        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.UBER)
        tightened = scheduler.get_tightened_thresholds(base_thresholds)

        # Uber phase: 1.00 multiplier (no change)
        assert tightened.resists["fire"] == pytest.approx(75.0)
        assert tightened.reservation.max_percent == pytest.approx(99.0)
        assert tightened.min_max_hit == pytest.approx(15000.0)
        assert tightened.min_full_dps == pytest.approx(500000.0)

    def test_record_evaluation_feasible(self):
        """Test recording feasible evaluation."""
        scheduler = CurriculumScheduler()
        # Record one feasible sample - this will trigger transition (100% > 10%)
        scheduler.record_evaluation(gate_passed=True)

        # After transition, phase counters are reset
        assert scheduler.state.total_samples == 1
        assert scheduler.state.feasible_samples == 1
        # Phase counters are reset after transition
        assert scheduler.state.phase_samples == 0
        assert scheduler.state.phase_feasible_samples == 0

    def test_record_evaluation_infeasible(self):
        """Test recording infeasible evaluation."""
        scheduler = CurriculumScheduler()
        scheduler.record_evaluation(gate_passed=False)

        assert scheduler.state.total_samples == 1
        assert scheduler.state.feasible_samples == 0
        assert scheduler.state.phase_samples == 1
        assert scheduler.state.phase_feasible_samples == 0

    def test_phase_transition_on_threshold(self):
        """Test automatic phase transition when feasibility exceeds 10%."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)

        for _ in range(100):
            scheduler.record_evaluation(gate_passed=False)

        assert scheduler.current_phase == CurriculumPhase.MAPPING

        for _ in range(12):
            scheduler.record_evaluation(gate_passed=True)

        assert scheduler.current_phase == CurriculumPhase.BOSSING
        assert scheduler.state.phase_samples == 0
        assert scheduler.state.phase_feasible_samples == 0

    def test_full_curriculum_progression(self):
        """Test progression through all curriculum phases."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)

        phases_visited = [scheduler.current_phase]

        for phase_idx in range(3):
            for _ in range(100):
                scheduler.record_evaluation(gate_passed=False)
            for _ in range(12):
                scheduler.record_evaluation(gate_passed=True)

            phases_visited.append(scheduler.current_phase)

        assert phases_visited == [
            CurriculumPhase.MAPPING,
            CurriculumPhase.BOSSING,
            CurriculumPhase.PINNACLE,
            CurriculumPhase.UBER,
        ]

    def test_get_state_summary(self):
        """Test state summary generation."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.BOSSING)
        for _ in range(100):
            scheduler.record_evaluation(gate_passed=False)
        for _ in range(5):
            scheduler.record_evaluation(gate_passed=True)

        summary = scheduler.get_state_summary()

        assert summary["current_phase"] == "BOSSING"
        assert summary["phase_number"] == 2
        assert "Bossing" in summary["phase_description"]
        assert summary["total_samples"] == 105
        assert summary["feasible_samples"] == 5
        assert summary["phase_samples"] == 105
        assert summary["phase_feasible_samples"] == 5
        assert "4.76%" in summary["overall_feasibility"]
        assert "4.76%" in summary["phase_feasibility"]


class TestCurriculumIntegration:
    """Integration tests for curriculum learning."""

    def test_curriculum_with_realistic_scenario(self):
        """Test curriculum with realistic gate threshold scenario."""
        base_thresholds = ScenarioGateThresholds(
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            reservation=ScenarioReservationThreshold(max_percent=99.0),
            attributes={"strength": 100.0, "dexterity": 100.0, "intelligence": 100.0},
            min_max_hit=15000.0,
            min_full_dps=500000.0,
        )

        scheduler = CurriculumScheduler()

        # Verify each phase has progressively tighter gates
        phases_and_thresholds = []
        for phase in CurriculumPhase:
            scheduler_phase = CurriculumScheduler(initial_phase=phase)
            tightened = scheduler_phase.get_tightened_thresholds(base_thresholds)
            phases_and_thresholds.append((phase, tightened))

        # Verify resist requirements increase (gates tighten)
        for i in range(len(phases_and_thresholds) - 1):
            phase1, thresh1 = phases_and_thresholds[i]
            phase2, thresh2 = phases_and_thresholds[i + 1]
            assert thresh1.resists["fire"] <= thresh2.resists["fire"]

        # Verify reservation max_percent decreases (gates tighten)
        for i in range(len(phases_and_thresholds) - 1):
            phase1, thresh1 = phases_and_thresholds[i]
            phase2, thresh2 = phases_and_thresholds[i + 1]
            assert thresh1.reservation.max_percent >= thresh2.reservation.max_percent

    def test_feasibility_transition_threshold_constant(self):
        """Test that transition threshold is properly defined."""
        assert FEASIBILITY_TRANSITION_THRESHOLD == 0.10
