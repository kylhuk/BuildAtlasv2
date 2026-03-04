import copy

import pytest

from backend.app.api.evaluator import adjust_metrics_for_profile
from backend.engine.evaluation import (
    GATE_REASON_FULL_DPS,
    GATE_REASON_MAX_HIT,
    GATE_REASON_RESERVATION,
    GATE_REASON_RESIST_FIRE,
    evaluate_gates,
    map_worker_output,
)
from backend.engine.scenarios.loader import (
    compose_ruleset_id,
    list_templates,
    load_template,
    scenario_version_tag,
)

DELVE_METRICS_PAYLOAD = {
    "metrics": {"full_dps": 3500.0, "max_hit": 4800.0, "utility_score": 3.0},
    "defense": {
        "armour": 4200.0,
        "evasion": 2600.0,
        "resists": {"fire": 80.0, "cold": 78.0, "lightning": 82.0, "chaos": 65.0},
    },
    "resources": {"life": 11200.0, "mana": 800.0},
    "reservation": {"reserved_percent": 68.0, "available_percent": 85.0},
    "attributes": {"strength": 190.0, "dexterity": 150.0, "intelligence": 140.0},
}

SUPPORT_METRICS_PAYLOAD = {
    "metrics": {"full_dps": 1500.0, "max_hit": 2400.0, "utility_score": 5.0},
    "defense": {
        "armour": 1500.0,
        "evasion": 900.0,
        "resists": {"fire": 65.0, "cold": 65.0, "lightning": 68.0, "chaos": 58.0},
    },
    "resources": {"life": 8000.0, "mana": 1600.0},
    "reservation": {"reserved_percent": 100.0, "available_percent": 110.0},
    "attributes": {"strength": 160.0, "dexterity": 140.0, "intelligence": 120.0},
}

PINNACLE_METRICS_PAYLOAD = {
    "metrics": {"full_dps": 6000.0, "max_hit": 6000.0, "utility_score": 1.0},
    "defense": {
        "armour": 4500.0,
        "evasion": 3000.0,
        "resists": {"fire": 90.0, "cold": 90.0, "lightning": 90.0, "chaos": 80.0},
    },
    "resources": {"life": 9500.0, "mana": 2500.0},
    "reservation": {"reserved_percent": 60.0, "available_percent": 90.0},
    "attributes": {"strength": 200.0, "dexterity": 200.0, "intelligence": 200.0},
}


EXPECTED_SCENARIOS = {
    "mapping_t16",
    "pinnacle",
    "uber_pinnacle",
    "delve_tier_1",
    "delve_tier_2",
    "delve_tier_3",
    "support_party",
}


def test_scenario_templates_and_version_helpers():
    templates = list_templates()
    assert len(templates) == len(EXPECTED_SCENARIOS)
    assert {template.scenario_id for template in templates} == EXPECTED_SCENARIOS

    mapping_template = load_template("mapping_t16", "v0")
    assert mapping_template.profile_id == "mapping"
    assert mapping_template.gate_thresholds.reservation.max_percent == 95.0

    assert scenario_version_tag(mapping_template) == "mapping_t16@v0"
    ruleset_id = compose_ruleset_id("abc123", mapping_template, "prices-demo")
    assert ruleset_id == "pob:abc123|scenarios:mapping_t16@v0|prices:prices-demo"


def test_normalized_mapping_handles_nested_payloads_and_defaults():
    payload = {
        "metrics": {"full_dps": 12345.6, "max_hit": 6200, "utility_score": 0.42},
        "defense": {
            "armour": 2500,
            "evasion": 1800,
            "resists": {"fire": 90, "cold": 92, "lightning": 88, "chaos": 67},
        },
        "resources": {"life": 8200, "mana": 1200},
        "reservation": {"reserved_percent": 75, "available_percent": 95},
        "attributes": {"strength": 210, "dexterity": 175, "intelligence": 140},
        "warnings": ["caps ok"],
    }
    metrics = map_worker_output(payload)
    assert metrics.full_dps == 12345.6
    assert metrics.armour == 2500
    assert metrics.resists.fire == 90
    assert metrics.attributes.strength == 210
    assert metrics.reservation.available_percent == 95
    assert metrics.warnings == ("caps ok",)

    defaults = map_worker_output({})
    assert defaults.full_dps == 0.0
    assert defaults.resists.cold == 0.0
    assert defaults.warnings == ()


def test_gate_evaluation_passes_and_reports_failure_reasons():
    template = load_template("pinnacle", "v0")
    base_payload = {
        "metrics": {"full_dps": 9000, "max_hit": template.gate_thresholds.min_max_hit + 600},
        "defense": {
            "armour": 3000,
            "resists": {
                "fire": template.gate_thresholds.resists["fire"] + 5,
                "cold": template.gate_thresholds.resists["cold"] + 3,
                "lightning": template.gate_thresholds.resists["lightning"] + 4,
                "chaos": template.gate_thresholds.resists["chaos"] + 6,
            },
        },
        "resources": {"life": 9500, "mana": 1400},
        "reservation": {"reserved_percent": 80, "available_percent": 100},
        "attributes": {
            "strength": template.gate_thresholds.attributes["strength"] + 20,
            "dexterity": template.gate_thresholds.attributes["dexterity"] + 10,
            "intelligence": template.gate_thresholds.attributes["intelligence"] + 10,
        },
    }
    metrics = map_worker_output(base_payload)
    evaluation = evaluate_gates(metrics, template.gate_thresholds)
    assert evaluation.gate_pass
    assert evaluation.gate_fail_reasons == ()

    failing_payload = copy.deepcopy(base_payload)
    failing_payload["defense"]["resists"]["fire"] = template.gate_thresholds.resists["fire"] - 5
    failing_payload["metrics"]["max_hit"] = template.gate_thresholds.min_max_hit - 200
    failing_payload["reservation"]["reserved_percent"] = (
        template.gate_thresholds.reservation.max_percent + 5
    )
    failure_metrics = map_worker_output(failing_payload)
    failure_evaluation = evaluate_gates(failure_metrics, template.gate_thresholds)
    assert not failure_evaluation.gate_pass
    failure_reasons = set(failure_evaluation.gate_fail_reasons)
    assert GATE_REASON_RESIST_FIRE in failure_reasons
    assert GATE_REASON_MAX_HIT in failure_reasons
    assert GATE_REASON_RESERVATION in failure_reasons

    slacks = failure_evaluation.gate_slacks
    assert slacks.resist_fire_slack < 0
    assert slacks.max_hit_slack < 0
    assert slacks.reservation_slack < 0
    assert slacks.num_gate_violations >= 3
    assert slacks.min_gate_slack == min(
        slacks.resist_fire_slack,
        slacks.resist_cold_slack,
        slacks.resist_lightning_slack,
        slacks.resist_chaos_slack,
        slacks.attr_strength_slack,
        slacks.attr_dexterity_slack,
        slacks.attr_intelligence_slack,
        slacks.max_hit_slack,
        slacks.full_dps_slack,
        slacks.reservation_slack,
    )


def test_uber_pinnacle_full_dps_gate():
    template = load_template("uber_pinnacle", "v0")
    assert template.gate_thresholds.min_full_dps == 2000000.0
    resists = template.gate_thresholds.resists
    attributes = template.gate_thresholds.attributes
    base_payload = {
        "metrics": {
            "full_dps": template.gate_thresholds.min_full_dps - 1,
            "max_hit": template.gate_thresholds.min_max_hit + 100,
        },
        "defense": {
            "armour": 12000,
            "resists": {key: value + 5 for key, value in resists.items()},
        },
        "resources": {"life": 13000, "mana": 2700},
        "reservation": {"reserved_percent": 70, "available_percent": 90},
        "attributes": {key: value + 5 for key, value in attributes.items()},
    }
    failure_evaluation = evaluate_gates(map_worker_output(base_payload), template.gate_thresholds)
    assert not failure_evaluation.gate_pass
    failure_reasons = set(failure_evaluation.gate_fail_reasons)
    assert GATE_REASON_FULL_DPS in failure_reasons

    passing_payload = copy.deepcopy(base_payload)
    passing_payload["metrics"]["full_dps"] = template.gate_thresholds.min_full_dps + 1000
    passing_evaluation = evaluate_gates(
        map_worker_output(passing_payload), template.gate_thresholds
    )
    assert passing_evaluation.gate_pass


def test_profile_survivability_score_for_delve():
    normalized = map_worker_output(DELVE_METRICS_PAYLOAD)
    adjusted = adjust_metrics_for_profile("delve", normalized)
    resist_total = (
        normalized.resists.fire
        + normalized.resists.cold
        + normalized.resists.lightning
        + normalized.resists.chaos
    )
    reservation_surplus = max(
        0.0,
        normalized.reservation.available_percent - normalized.reservation.reserved_percent,
    )
    defensive_stack = (
        normalized.life / 200.0
        + normalized.armour / 1200.0
        + normalized.evasion / 900.0
        + normalized.max_hit / 6000.0
    )
    expected = (
        defensive_stack
        + resist_total * 0.15
        + reservation_surplus * 0.4
        + normalized.utility_score * 0.1
    )
    assert adjusted.utility_score == pytest.approx(expected)


def test_profile_support_score_prioritizes_reservation_and_utility():
    normalized = map_worker_output(SUPPORT_METRICS_PAYLOAD)
    adjusted = adjust_metrics_for_profile("support", normalized)
    reserved = max(normalized.reservation.reserved_percent, 0.0)
    available = max(normalized.reservation.available_percent, 1.0)
    aura_bonus = (
        normalized.attributes.strength
        + normalized.attributes.dexterity
        + normalized.attributes.intelligence
    ) / 50.0
    reservation_surplus = max(0.0, available - reserved)
    reservation_efficiency = min(reserved / available, 1.0)
    expected = (
        aura_bonus
        + reservation_surplus * 1.2
        + reservation_efficiency * 5.0
        + normalized.mana / 250.0
        + normalized.utility_score * 0.2
    )
    assert adjusted.utility_score == pytest.approx(expected)


def test_profile_adjustment_is_noop_for_other_profiles():
    normalized = map_worker_output(PINNACLE_METRICS_PAYLOAD)
    adjusted = adjust_metrics_for_profile("mapping", normalized)
    assert adjusted is normalized
