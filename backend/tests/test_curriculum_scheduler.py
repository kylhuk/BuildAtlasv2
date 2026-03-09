"""Tests for curriculum learning scheduler."""

import pytest

from backend.engine.curriculum import (
    FEASIBILITY_TRANSITION_THRESHOLD,
    PHASE_CONFIGS,
    CurriculumPhase,
    CurriculumScheduler,
    CurriculumState,
)
from backend.engine.scenarios.loader import ScenarioGateThresholds, ScenarioReservationThreshold


class TestCurriculumPhase:
    """Test CurriculumPhase enum."""

    def test_phase_ordering(self):
        """Test phases are ordered correctly."""
        phases = list(CurriculumPhase)
        assert phases[0] == CurriculumPhase.ZERO_GATES
        assert phases[1] == CurriculumPhase.MAPPING
        assert phases[2] == CurriculumPhase.BOSSING
        assert phases[3] == CurriculumPhase.PINNACLE
        assert phases[4] == CurriculumPhase.UBER

    def test_phase_numbers(self):
        """Test phase numbers are sequential."""
        assert CurriculumPhase.ZERO_GATES.phase_number == 0
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
        assert "Zero-gates" in CurriculumPhase.ZERO_GATES.description


class TestPhaseConfigs:
    """Test phase configurations."""

    def test_all_phases_configured(self):
        """Test all phases have configurations."""
        for phase in CurriculumPhase:
            assert phase in PHASE_CONFIGS
            config = PHASE_CONFIGS[phase]
            assert config.phase == phase

    def test_zero_gates_config_present(self):
        """Zero-gates phase keeps gates permissive."""
        config = PHASE_CONFIGS[CurriculumPhase.ZERO_GATES]
        assert config.resists_multiplier == 1.0
        assert config.attributes_multiplier == 1.0
        assert config.max_hit_multiplier == 1.0
        assert config.full_dps_multiplier == 1.0
        assert config.reservation_multiplier == 1.0

    def test_multipliers_progression(self):
        """Test multipliers progress from easy to hard."""
        progression = [
            CurriculumPhase.MAPPING,
            CurriculumPhase.BOSSING,
            CurriculumPhase.PINNACLE,
            CurriculumPhase.UBER,
        ]
        for first, second in zip(progression, progression[1:], strict=True):
            first_cfg = PHASE_CONFIGS[first]
            second_cfg = PHASE_CONFIGS[second]
            assert first_cfg.resists_multiplier <= second_cfg.resists_multiplier
            assert first_cfg.attributes_multiplier <= second_cfg.attributes_multiplier
            assert first_cfg.max_hit_multiplier <= second_cfg.max_hit_multiplier
            assert first_cfg.full_dps_multiplier <= second_cfg.full_dps_multiplier
            assert first_cfg.reservation_multiplier >= second_cfg.reservation_multiplier

    def test_multipliers_progression_from_zero_gates(self):
        """Zero-gates remains the bootstrap permissive phase."""
        assert PHASE_CONFIGS[CurriculumPhase.ZERO_GATES].resists_multiplier == 1.0
        assert PHASE_CONFIGS[CurriculumPhase.MAPPING].resists_multiplier < 1.0

    def test_feasibility_targets_decrease(self):
        """Test feasibility targets decrease as difficulty increases."""
        progression = [
            CurriculumPhase.MAPPING,
            CurriculumPhase.BOSSING,
            CurriculumPhase.PINNACLE,
            CurriculumPhase.UBER,
        ]
        targets = [PHASE_CONFIGS[phase].feasibility_target for phase in progression]
        for first, second in zip(targets, targets[1:], strict=True):
            assert first > second

    def test_phase_chain_property(self):
        """Test explicit ordered chain transitions."""
        expected_chain = (
            CurriculumPhase.ZERO_GATES,
            CurriculumPhase.MAPPING,
            CurriculumPhase.BOSSING,
            CurriculumPhase.PINNACLE,
            CurriculumPhase.UBER,
        )
        assert CurriculumPhase.ZERO_GATES._ordered_phases == expected_chain
        for current_phase, next_phase in zip(expected_chain, expected_chain[1:], strict=True):
            assert current_phase.next_phase() == next_phase


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
        """Test transition criteria (>25% feasibility)."""
        # Below 25% threshold - should NOT transition
        state_below = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=100,
            feasible_samples=24,
            phase_samples=100,
            phase_feasible_samples=24,
        )
        assert state_below.phase_feasibility == 0.24
        assert not state_below.should_transition

        # Above 25% threshold - should transition
        state_above = CurriculumState(
            current_phase=CurriculumPhase.MAPPING,
            total_samples=100,
            feasible_samples=26,
            phase_samples=100,
            phase_feasible_samples=26,
        )
        assert state_above.phase_feasibility == 0.26
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

    def test_initialization_zero_gates(self):
        """Test scheduler initialization with zero-gates phase."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.ZERO_GATES)
        assert scheduler.current_phase == CurriculumPhase.ZERO_GATES

    def test_current_config(self):
        """Test getting current phase configuration."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)
        config = scheduler.current_config
        assert config.phase == CurriculumPhase.MAPPING
        assert config.resists_multiplier == 0.70

    def test_get_tightened_thresholds_mapping(self):
        """Test threshold tightening for mapping phase."""
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

        # Mapping phase: 1.22 multiplier for reservation (more lenient)
        assert tightened.reservation.max_percent == pytest.approx(99.0 * 1.22)

        # Mapping phase: 0.70 multiplier for max_hit and full_dps
        assert tightened.min_max_hit == pytest.approx(15000.0 * 0.70)
        assert tightened.min_full_dps == pytest.approx(500000.0 * 0.70)

    def test_get_tightened_thresholds_zero_gates(self):
        """Test threshold tightening keeps zero-gates permissive."""
        base_thresholds = ScenarioGateThresholds(
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            reservation=ScenarioReservationThreshold(max_percent=99.0),
            attributes={"strength": 100.0, "dexterity": 100.0, "intelligence": 100.0},
            min_max_hit=15000.0,
            min_full_dps=500000.0,
        )

        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.ZERO_GATES)
        tightened = scheduler.get_tightened_thresholds(base_thresholds)

        assert tightened.resists == base_thresholds.resists
        assert tightened.attributes == base_thresholds.attributes
        assert tightened.reservation == base_thresholds.reservation
        assert tightened.min_max_hit == pytest.approx(15000.0)
        assert tightened.min_full_dps == pytest.approx(500000.0)

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
        """Test recording feasible evaluation respects MIN_PHASE_SAMPLES."""
        scheduler = CurriculumScheduler()
        # First feasible record should not transition before MIN_PHASE_SAMPLES
        scheduler.record_evaluation(gate_passed=True)

        assert scheduler.state.total_samples == 1
        assert scheduler.state.feasible_samples == 1
        assert scheduler.current_phase == CurriculumPhase.MAPPING
        assert scheduler.state.phase_samples == 1
        assert scheduler.state.phase_feasible_samples == 1

    def test_zero_gates_phase_transitions_to_mapping(self):
        """Zero-gates phase transitions deterministically to Mapping."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.ZERO_GATES)

        for _ in range(20):
            scheduler.record_evaluation(gate_passed=True)

        assert scheduler.current_phase == CurriculumPhase.MAPPING
        assert scheduler.state.total_samples == 20
        assert scheduler.state.phase_samples == 0
        assert scheduler.state.phase_feasible_samples == 0

        # Once MIN_PHASE_SAMPLES is reached with high feasibility, transition occurs
        for _ in range(20):
            scheduler.record_evaluation(gate_passed=True)

        assert scheduler.state.total_samples == 40
        assert scheduler.state.feasible_samples == 40
        assert scheduler.current_phase == CurriculumPhase.BOSSING
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

    def test_min_phase_samples_guardrail(self):
        """Ensure no transition before MIN_PHASE_SAMPLES."""
        scheduler = CurriculumScheduler()

        for _ in range(19):
            scheduler.record_evaluation(gate_passed=True)

        assert scheduler.state.total_samples == 19
        assert scheduler.state.phase_samples == 19
        assert scheduler.current_phase == CurriculumPhase.MAPPING
        assert scheduler.state.phase_feasible_samples == 19

    def test_max_phase_samples_guardrail(self):
        """Ensure forced transition after MAX_PHASE_SAMPLES."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)

        for _ in range(1000):
            scheduler.record_evaluation(gate_passed=False)

        assert scheduler.state.total_samples == 1000
        assert scheduler.current_phase == CurriculumPhase.BOSSING
        assert scheduler.state.phase_samples == 0
        assert scheduler.state.phase_feasible_samples == 0

    def test_phase_transition_on_threshold(self):
        """Test automatic phase transition when feasibility exceeds 25%."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)

        # First 100 builds fail
        for _ in range(100):
            scheduler.record_evaluation(gate_passed=False)

        assert scheduler.current_phase == CurriculumPhase.MAPPING

        # Need >25% feasibility to transition: 34/134 = 25.37%
        for _ in range(34):
            scheduler.record_evaluation(gate_passed=True)

        assert scheduler.current_phase == CurriculumPhase.BOSSING
        assert scheduler.state.phase_samples == 0
        assert scheduler.state.phase_feasible_samples == 0

    def test_full_curriculum_progression(self):
        """Test progression through all curriculum phases."""
        scheduler = CurriculumScheduler(initial_phase=CurriculumPhase.MAPPING)

        phases_visited = [scheduler.current_phase]

        for _ in range(8):
            # Need >25% pass rate to transition per phase
            for _ in range(100):
                scheduler.record_evaluation(gate_passed=False)
            for _ in range(34):  # 34/134 = 25.37% > 25% threshold
                scheduler.record_evaluation(gate_passed=True)

            phases_visited.append(scheduler.current_phase)

        assert phases_visited == [
            CurriculumPhase.MAPPING,
            CurriculumPhase.BOSSING,
            CurriculumPhase.PINNACLE,
            CurriculumPhase.UBER,
            CurriculumPhase.UBER,
            CurriculumPhase.UBER,
            CurriculumPhase.UBER,
            CurriculumPhase.UBER,
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

        # Verify each progression phase has progressively tighter gates.
        phases_and_thresholds = []
        for phase in CurriculumPhase.ZERO_GATES._ordered_phases[1:]:
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
        assert FEASIBILITY_TRANSITION_THRESHOLD == 0.25
