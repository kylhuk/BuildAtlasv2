from backend.engine.evaluation.gates import compute_gate_slacks
from backend.engine.evaluation.normalized import (
    NormalizedAttributes,
    NormalizedMetrics,
    NormalizedReservation,
    ResistSnapshot,
)
from backend.engine.scenarios.loader import ScenarioGateThresholds, ScenarioReservationThreshold


def test_compute_gate_slacks_passing():
    metrics = NormalizedMetrics(
        full_dps=1000000.0,
        max_hit=20000.0,
        armour=10000.0,
        evasion=5000.0,
        life=5000.0,
        mana=1000.0,
        utility_score=1.0,
        resists=ResistSnapshot(fire=75.0, cold=75.0, lightning=75.0, chaos=20.0),
        reservation=NormalizedReservation(reserved_percent=90.0, available_percent=100.0),
        attributes=NormalizedAttributes(strength=150.0, dexterity=150.0, intelligence=150.0),
        warnings=(),
    )

    thresholds = ScenarioGateThresholds(
        resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
        reservation=ScenarioReservationThreshold(max_percent=99.0),
        attributes={"strength": 100.0, "dexterity": 100.0, "intelligence": 100.0},
        min_max_hit=15000.0,
        min_full_dps=500000.0,
    )

    slacks = compute_gate_slacks(metrics, thresholds)

    assert slacks.resist_fire_slack == 0.0
    assert slacks.resist_chaos_slack == 20.0
    assert slacks.attr_strength_slack == 50.0
    assert slacks.max_hit_slack == 5000.0
    assert slacks.full_dps_slack == 500000.0
    assert slacks.reservation_slack == 9.0  # min(100-90, 99-90) = 9
    assert slacks.min_gate_slack == 0.0
    assert slacks.num_gate_violations == 0
    assert slacks.passes_all_gates is True


def test_compute_gate_slacks_failing():
    metrics = NormalizedMetrics(
        full_dps=400000.0,  # Fails (threshold 500k)
        max_hit=12000.0,  # Fails (threshold 15k)
        armour=0.0,
        evasion=0.0,
        life=0.0,
        mana=0.0,
        utility_score=0.0,
        resists=ResistSnapshot(
            fire=70.0, cold=75.0, lightning=75.0, chaos=-10.0
        ),  # Fire fails, Chaos fails
        reservation=NormalizedReservation(reserved_percent=95.0, available_percent=100.0),
        attributes=NormalizedAttributes(
            strength=80.0, dexterity=150.0, intelligence=150.0
        ),  # Str fails
        warnings=(),
    )

    thresholds = ScenarioGateThresholds(
        resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
        reservation=ScenarioReservationThreshold(max_percent=99.0),
        attributes={"strength": 100.0, "dexterity": 100.0, "intelligence": 100.0},
        min_max_hit=15000.0,
        min_full_dps=500000.0,
    )

    slacks = compute_gate_slacks(metrics, thresholds)

    assert slacks.resist_fire_slack == -5.0
    assert slacks.resist_chaos_slack == -10.0
    assert slacks.attr_strength_slack == -20.0
    assert slacks.max_hit_slack == -3000.0
    assert slacks.full_dps_slack == -100000.0
    assert slacks.num_gate_violations == 5
    assert slacks.min_gate_slack == -100000.0
    assert slacks.passes_all_gates is False


def test_compute_gate_slacks_reservation_limit():
    metrics = NormalizedMetrics(
        full_dps=1000000.0,
        max_hit=20000.0,
        armour=0.0,
        evasion=0.0,
        life=0.0,
        mana=0.0,
        utility_score=0.0,
        resists=ResistSnapshot(fire=75.0, cold=75.0, lightning=75.0, chaos=0.0),
        reservation=NormalizedReservation(
            reserved_percent=105.0, available_percent=100.0
        ),  # Infeasible
        attributes=NormalizedAttributes(strength=100.0, dexterity=100.0, intelligence=100.0),
        warnings=(),
    )
    thresholds = ScenarioGateThresholds(
        resists={},
        reservation=ScenarioReservationThreshold(max_percent=99.0),
        attributes={},
        min_max_hit=0.0,
        min_full_dps=0.0,
    )

    slacks = compute_gate_slacks(metrics, thresholds)
    # res_feasibility_slack = 100 - 105 = -5
    # res_limit_slack = 99 - 105 = -6
    # reservation_slack = min(-5, -6) = -6
    assert slacks.reservation_slack == -6.0
    assert slacks.num_gate_violations == 1
