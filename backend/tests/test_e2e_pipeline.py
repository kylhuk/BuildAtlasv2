"""
End-to-end tests for the feasibility-first pipeline.

Tests the full flow: Skeleton → CSP → Repair → Gate Evaluation
with mock PoB evaluation and gate slack computation.
"""
from __future__ import annotations

import pytest

from backend.engine.evaluation.gates import compute_gate_slacks, evaluate_gates
from backend.engine.evaluation.normalized import (
    NormalizedAttributes,
    NormalizedMetrics,
    NormalizedReservation,
    ResistSnapshot,
)
from backend.engine.repair.operators import (
    AttributeRepair,
    LifeRepair,
    ReservationRepair,
    ResistanceRepair,
)
from backend.engine.scenarios.loader import (
    ScenarioGateThresholds,
    ScenarioReservationThreshold,
    ScenarioTemplate,
)
from backend.engine.skeletons.schema import Skeleton
from backend.engine.validation.csp import BuildCSP

# ============================================================================
# FIXTURES: Skeleton, Templates, Build Data
# ============================================================================


@pytest.fixture
def test_skeleton() -> Skeleton:
    """Create a valid test skeleton."""
    return Skeleton(
        skeleton_id="test_cyclone_jugg",
        class_name="marauder",
        ascendancy="Juggernaut",
        main_skill="cyclone",
        skill_links=["cyclone", "Melee Physical Damage", "Fortify", "Increased Area of Effect"],
        aura_package=["Determination", "Defiance Banner"],
        defense_layer="armour",
        budget_tier="starter",
        target_gates={"full_dps": 1000000, "max_hit": 15000},
        required_uniques=[],
        tree_path="1,2,3,4,5",
    )


@pytest.fixture
def test_scenario_template() -> ScenarioTemplate:
    """Create a test scenario template with realistic gates."""
    return ScenarioTemplate(
        scenario_id="test_scenario_pinnacle",
        version="v0",
        profile_id="pinnacle",
        pob_config={},
        gate_thresholds=ScenarioGateThresholds(
            min_full_dps=500000.0,
            min_max_hit=10000.0,
            reservation=ScenarioReservationThreshold(max_percent=95.0),
            resists={"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            attributes={"strength": 100.0, "dexterity": 50.0, "intelligence": 50.0},
        ),
    )


@pytest.fixture
def passing_build_data() -> dict:
    """Create a build that passes all gates."""
    return {
        "build_id": "build_passing_001",
        "class_name": "marauder",
        "ascendancy": "Juggernaut",
        "main_skill_package": "cyclone",
        "main_skill": {"links": 6, "gems": ["Cyclone", "Melee Physical Damage", "Fortify"]},
        "items": [
            {
                "slot": "chest",
                "name": "Kaom's Heart",
                "requirements": {"strength": 150},
                "sockets": ["R", "R", "R"],
                "adjustable": False,
                "contributions": {"life": 500},
            },
            {
                "slot": "weapon",
                "name": "Rare Axe",
                "requirements": {"strength": 100},
                "sockets": ["R", "R"],
                "adjustable": True,
                "contributions": {"physical_damage": 200},
            },
        ],
        "attributes": {"strength": 200, "dexterity": 80, "intelligence": 60},
        "resistances": {"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 20.0},
        "reservation": 850.0,  # CSP expects float, not dict
        "total_mana": 1000.0,
        "passive_tree": {
            "allocated": [1, 2, 3, 4, 5],
            "adjacencies": {1: [2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4]},
            "start_nodes": [1],
        },
        "stats": {
            "full_dps": 1200000.0,
            "max_hit": 18000.0,
            "life": 5000.0,
            "armour": 8000.0,
            "evasion": 0.0,
            "mana": 1000.0,
            "utility_score": 1.0,
        },
    }


@pytest.fixture
def failing_build_data() -> dict:
    """Create a build that fails some gates."""
    return {
        "build_id": "build_failing_001",
        "class_name": "marauder",
        "ascendancy": "Juggernaut",
        "main_skill_package": "cyclone",
        "main_skill": {"links": 6, "gems": ["Cyclone", "Melee Physical Damage"]},
        "items": [
            {
                "slot": "chest",
                "name": "Rare Chest",
                "requirements": {"strength": 80},
                "sockets": ["R", "R", "R"],
                "adjustable": True,
                "contributions": {"life": 100},
            },
        ],
        "attributes": {"strength": 120, "dexterity": 40, "intelligence": 30},
        "resistances": {"fire": 60.0, "cold": 50.0, "lightning": 70.0, "chaos": -20.0},
        "reservation": 490.0,
        "total_mana": 500.0,
        "passive_tree": {
            "allocated": [1, 2],
            "adjacencies": {1: [2], 2: [1]},
            "start_nodes": [1],
        },
        "stats": {
            "full_dps": 300000.0,
            "max_hit": 8000.0,
            "life": 2000.0,
            "armour": 2000.0,
            "evasion": 0.0,
            "mana": 500.0,
            "utility_score": 0.5,
        },
    }


# ============================================================================
# TESTS: CSP Validation (Skeleton → CSP)
# ============================================================================


class TestCSPValidation:
    """Test constraint satisfaction problem validation."""

    def test_csp_valid_build(self, passing_build_data):
        """CSP should pass for a valid build."""
        csp = BuildCSP()
        is_valid, errors = csp.validate(passing_build_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_csp_invalid_links(self, passing_build_data):
        """CSP should fail if main skill doesn't have 6 links."""
        passing_build_data["main_skill"]["links"] = 5
        csp = BuildCSP()
        is_valid, errors = csp.validate(passing_build_data)

        assert is_valid is False
        assert any("6 links" in e for e in errors)

    def test_csp_invalid_item_requirements(self, passing_build_data):
        """CSP should fail if attributes don't meet item requirements."""
        passing_build_data["items"][0]["requirements"]["strength"] = 300
        csp = BuildCSP()
        is_valid, errors = csp.validate(passing_build_data)

        assert is_valid is False
        assert any("requires" in e.lower() for e in errors)

    def test_csp_invalid_mana_reservation(self, passing_build_data):
        """CSP should fail if reservation exceeds total mana."""
        passing_build_data["reservation"] = 1100.0
        csp = BuildCSP()
        is_valid, errors = csp.validate(passing_build_data)

        assert is_valid is False
        assert any("reservation" in e.lower() for e in errors)

    def test_csp_invalid_passive_connectivity(self, passing_build_data):
        """CSP should fail if passive tree is disconnected."""
        passing_build_data["passive_tree"]["allocated"] = [1, 2, 10]
        passing_build_data["passive_tree"]["adjacencies"][10] = []
        csp = BuildCSP()
        is_valid, errors = csp.validate(passing_build_data)

        assert is_valid is False
        assert any("connected" in e.lower() for e in errors)


# ============================================================================
# TESTS: Gate Evaluation (CSP → Gate Evaluation)
# ============================================================================


class TestGateEvaluation:
    """Test gate evaluation and slack computation."""

    def test_gate_evaluation_passing(self, test_scenario_template):
        """Gates should pass for a build meeting all thresholds."""
        metrics = NormalizedMetrics(
            full_dps=1200000.0,
            max_hit=18000.0,
            armour=8000.0,
            evasion=0.0,
            life=5000.0,
            mana=1000.0,
            utility_score=1.0,
            resists=ResistSnapshot(fire=75.0, cold=75.0, lightning=75.0, chaos=20.0),
            reservation=NormalizedReservation(reserved_percent=85.0, available_percent=100.0),
            attributes=NormalizedAttributes(strength=200.0, dexterity=80.0, intelligence=60.0),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)

        assert gate_eval.gate_pass is True
        assert len(gate_eval.gate_fail_reasons) == 0
        assert gate_eval.gate_slacks.passes_all_gates is True
        assert gate_eval.gate_slacks.num_gate_violations == 0

    def test_gate_evaluation_failing_dps(self, test_scenario_template):
        """Gates should fail if DPS is below threshold."""
        metrics = NormalizedMetrics(
            full_dps=300000.0,  # Below 500k
            max_hit=18000.0,
            armour=8000.0,
            evasion=0.0,
            life=5000.0,
            mana=1000.0,
            utility_score=1.0,
            resists=ResistSnapshot(fire=75.0, cold=75.0, lightning=75.0, chaos=20.0),
            reservation=NormalizedReservation(reserved_percent=85.0, available_percent=100.0),
            attributes=NormalizedAttributes(strength=200.0, dexterity=80.0, intelligence=60.0),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)

        assert gate_eval.gate_pass is False
        assert "full_dps_too_low" in gate_eval.gate_fail_reasons
        assert gate_eval.gate_slacks.full_dps_slack == -200000.0

    def test_gate_evaluation_failing_resists(self, test_scenario_template):
        """Gates should fail if resistances are below threshold."""
        metrics = NormalizedMetrics(
            full_dps=1200000.0,
            max_hit=18000.0,
            armour=8000.0,
            evasion=0.0,
            life=5000.0,
            mana=1000.0,
            utility_score=1.0,
            resists=ResistSnapshot(fire=60.0, cold=75.0, lightning=75.0, chaos=20.0),
            reservation=NormalizedReservation(reserved_percent=85.0, available_percent=100.0),
            attributes=NormalizedAttributes(strength=200.0, dexterity=80.0, intelligence=60.0),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)

        assert gate_eval.gate_pass is False
        assert "resist_fire_shortfall" in gate_eval.gate_fail_reasons
        assert gate_eval.gate_slacks.resist_fire_slack == -15.0

    def test_gate_evaluation_failing_attributes(self, test_scenario_template):
        """Gates should fail if attributes are below threshold."""
        metrics = NormalizedMetrics(
            full_dps=1200000.0,
            max_hit=18000.0,
            armour=8000.0,
            evasion=0.0,
            life=5000.0,
            mana=1000.0,
            utility_score=1.0,
            resists=ResistSnapshot(fire=75.0, cold=75.0, lightning=75.0, chaos=20.0),
            reservation=NormalizedReservation(reserved_percent=85.0, available_percent=100.0),
            attributes=NormalizedAttributes(strength=80.0, dexterity=80.0, intelligence=60.0),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)

        assert gate_eval.gate_pass is False
        assert "attributes_requirements" in gate_eval.gate_fail_reasons

    def test_gate_slack_computation_multiple_violations(self, test_scenario_template):
        """Slack computation should track multiple violations."""
        metrics = NormalizedMetrics(
            full_dps=300000.0,
            max_hit=8000.0,
            armour=0.0,
            evasion=0.0,
            life=0.0,
            mana=0.0,
            utility_score=0.0,
            resists=ResistSnapshot(fire=60.0, cold=50.0, lightning=75.0, chaos=-20.0),
            reservation=NormalizedReservation(reserved_percent=98.0, available_percent=100.0),
            attributes=NormalizedAttributes(strength=80.0, dexterity=40.0, intelligence=30.0),
            warnings=(),
        )

        slacks = compute_gate_slacks(metrics, test_scenario_template.gate_thresholds)

        assert slacks.num_gate_violations > 0
        assert slacks.full_dps_slack == -200000.0
        assert slacks.max_hit_slack == -2000.0
        assert slacks.resist_fire_slack == -15.0
        assert slacks.resist_chaos_slack == -20.0
        assert slacks.attr_strength_slack == -20.0


# ============================================================================
# TESTS: Repair Operators (Gate Failure → Repair)
# ============================================================================


class TestRepairOperators:
    """Test repair operators for fixing gate violations."""

    def test_resistance_repair_needed(self):
        """ResistanceRepair should detect when repair is needed."""
        build = {
            "resistances": {"fire": 60.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            "items": [
                {
                    "slot": "ring",
                    "adjustable": True,
                    "contributions": {"fire": 0},
                }
            ],
        }
        repair = ResistanceRepair()
        assert repair.needs_repair(build) is True

    def test_resistance_repair_apply(self):
        """ResistanceRepair should add resistance affixes."""
        build = {
            "resistances": {"fire": 60.0, "cold": 75.0, "lightning": 75.0, "chaos": 0.0},
            "items": [
                {
                    "slot": "ring",
                    "adjustable": True,
                    "contributions": {"fire": 0},
                }
            ],
        }
        repair = ResistanceRepair()
        repaired = repair.apply(build)

        assert "resistances" in repaired
        assert repaired["resistances"]["fire"] >= 60.0

    def test_life_repair_needed(self):
        """LifeRepair should detect when eHP is too low."""
        build = {
            "stats": {"ehp": 2000},
            "items": [
                {
                    "slot": "chest",
                    "adjustable": True,
                    "contributions": {"life": 0},
                }
            ],
        }
        repair = LifeRepair(ehp_threshold=3000)
        assert repair.needs_repair(build) is True

    def test_life_repair_apply(self):
        """LifeRepair should add life rolls."""
        build = {
            "stats": {"ehp": 2000, "life": 2000},
            "items": [
                {
                    "slot": "chest",
                    "adjustable": True,
                    "contributions": {"life": 0},
                }
            ],
        }
        repair = LifeRepair(ehp_threshold=3000)
        repaired = repair.apply(build)

        assert repaired["stats"]["life"] > 2000
        assert repaired["stats"]["ehp"] > 2000

    def test_attribute_repair_needed(self):
        """AttributeRepair should detect when attributes are insufficient."""
        build = {
            "attributes": {"strength": 80, "dexterity": 50, "intelligence": 50},
            "items": [
                {
                    "slot": "chest",
                    "requirements": {"strength": 150},
                    "adjustable": True,
                    "contributions": {"strength": 0},
                }
            ],
        }
        repair = AttributeRepair()
        assert repair.needs_repair(build) is True

    def test_attribute_repair_apply(self):
        """AttributeRepair should add attribute rolls."""
        build = {
            "attributes": {"strength": 80, "dexterity": 50, "intelligence": 50},
            "items": [
                {
                    "slot": "chest",
                    "requirements": {"strength": 150},
                    "adjustable": True,
                    "contributions": {"strength": 0},
                }
            ],
        }
        repair = AttributeRepair()
        repaired = repair.apply(build)

        assert repaired["attributes"]["strength"] >= 150

    def test_reservation_repair_needed(self):
        """ReservationRepair should detect when reservation exceeds total mana."""
        build = {
            "reservation": 1100.0,
            "total_mana": 1000.0,
            "gems": {"groups": []},
        }
        repair = ReservationRepair()
        assert repair.needs_repair(build) is True

    def test_reservation_repair_apply(self):
        """ReservationRepair should reduce reservation to total mana."""
        build = {
            "reservation": 1100.0,
            "total_mana": 1000.0,
            "gems": {"groups": []},
        }
        repair = ReservationRepair()
        repaired = repair.apply(build)

        assert repaired["reservation"] <= repaired["total_mana"]


# ============================================================================
# TESTS: End-to-End Pipeline
# ============================================================================


class TestE2EPipeline:
    """Test the complete feasibility-first pipeline."""

    def test_e2e_skeleton_to_csp_to_gates_passing(
        self, test_skeleton, test_scenario_template, passing_build_data
    ):
        """Full pipeline should succeed for a valid build."""
        # Step 1: Validate skeleton
        test_skeleton.validate()

        # Step 2: CSP validation
        csp = BuildCSP()
        is_valid, csp_errors = csp.validate(passing_build_data)
        assert is_valid is True, f"CSP validation failed: {csp_errors}"

        metrics = NormalizedMetrics(
            full_dps=passing_build_data["stats"]["full_dps"],
            max_hit=passing_build_data["stats"]["max_hit"],
            armour=passing_build_data["stats"]["armour"],
            evasion=passing_build_data["stats"]["evasion"],
            life=passing_build_data["stats"]["life"],
            mana=passing_build_data["stats"]["mana"],
            utility_score=passing_build_data["stats"]["utility_score"],
            resists=ResistSnapshot(
                fire=passing_build_data["resistances"]["fire"],
                cold=passing_build_data["resistances"]["cold"],
                lightning=passing_build_data["resistances"]["lightning"],
                chaos=passing_build_data["resistances"]["chaos"],
            ),
            reservation=NormalizedReservation(
                reserved_percent=85.0,
                available_percent=100.0,
            ),
            attributes=NormalizedAttributes(
                strength=passing_build_data["attributes"]["strength"],
                dexterity=passing_build_data["attributes"]["dexterity"],
                intelligence=passing_build_data["attributes"]["intelligence"],
            ),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)
        assert gate_eval.gate_pass is True
        assert gate_eval.gate_slacks.passes_all_gates is True

    def test_e2e_skeleton_to_csp_to_gates_failing_then_repair(
        self, test_skeleton, test_scenario_template, failing_build_data
    ):
        """Full pipeline should detect failures and allow repair."""
        test_skeleton.validate()

        csp = BuildCSP()
        is_valid, csp_errors = csp.validate(failing_build_data)
        assert is_valid is True

        metrics = NormalizedMetrics(
            full_dps=failing_build_data["stats"]["full_dps"],
            max_hit=failing_build_data["stats"]["max_hit"],
            armour=failing_build_data["stats"]["armour"],
            evasion=failing_build_data["stats"]["evasion"],
            life=failing_build_data["stats"]["life"],
            mana=failing_build_data["stats"]["mana"],
            utility_score=failing_build_data["stats"]["utility_score"],
            resists=ResistSnapshot(
                fire=failing_build_data["resistances"]["fire"],
                cold=failing_build_data["resistances"]["cold"],
                lightning=failing_build_data["resistances"]["lightning"],
                chaos=failing_build_data["resistances"]["chaos"],
            ),
            reservation=NormalizedReservation(
                reserved_percent=98.0,
                available_percent=100.0,
            ),
            attributes=NormalizedAttributes(
                strength=failing_build_data["attributes"]["strength"],
                dexterity=failing_build_data["attributes"]["dexterity"],
                intelligence=failing_build_data["attributes"]["intelligence"],
            ),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)
        assert gate_eval.gate_pass is False
        assert len(gate_eval.gate_fail_reasons) > 0

        repairs = [
            ResistanceRepair(),
            LifeRepair(ehp_threshold=3000),
            AttributeRepair(),
            ReservationRepair(),
        ]

        repaired_build = failing_build_data.copy()
        repairs_applied = 0
        for repair in repairs:
            if repair.needs_repair(repaired_build):
                repaired_build = repair.apply(repaired_build)
                repairs_applied += 1

        assert repairs_applied > 0

    def test_e2e_gate_slack_tracking(self, test_scenario_template):
        """Gate slack computation should track all violations."""
        metrics = NormalizedMetrics(
            full_dps=300000.0,
            max_hit=8000.0,
            armour=0.0,
            evasion=0.0,
            life=0.0,
            mana=0.0,
            utility_score=0.0,
            resists=ResistSnapshot(fire=60.0, cold=50.0, lightning=75.0, chaos=-20.0),
            reservation=NormalizedReservation(reserved_percent=98.0, available_percent=100.0),
            attributes=NormalizedAttributes(strength=80.0, dexterity=40.0, intelligence=30.0),
            warnings=(),
        )

        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)

        assert gate_eval.gate_slacks.num_gate_violations > 0
        assert gate_eval.gate_slacks.min_gate_slack == -200000.0
        assert gate_eval.gate_slacks.passes_all_gates is False

        slacks = gate_eval.gate_slacks
        assert slacks.full_dps_slack < 0
        assert slacks.max_hit_slack < 0
        assert slacks.resist_fire_slack < 0
        assert slacks.resist_chaos_slack < 0
        assert slacks.attr_strength_slack < 0

    def test_e2e_mock_pob_evaluation(self, test_scenario_template, passing_build_data):
        """Test pipeline with mocked PoB evaluation."""
        # Mock PoB worker response
        mock_pob_output = {
            "full_dps": 1200000.0,
            "max_hit": 18000.0,
            "life": 5000.0,
            "armour": 8000.0,
            "evasion": 0.0,
            "mana": 1000.0,
            "resists": {"fire": 75.0, "cold": 75.0, "lightning": 75.0, "chaos": 20.0},
            "attributes": {"strength": 200.0, "dexterity": 80.0, "intelligence": 60.0},
            "reservation": {"reserved_percent": 85.0, "available_percent": 100.0},
        }

        # Simulate PoB evaluation
        metrics = NormalizedMetrics(
            full_dps=mock_pob_output["full_dps"],
            max_hit=mock_pob_output["max_hit"],
            armour=mock_pob_output["armour"],
            evasion=mock_pob_output["evasion"],
            life=mock_pob_output["life"],
            mana=mock_pob_output["mana"],
            utility_score=1.0,
            resists=ResistSnapshot(
                fire=mock_pob_output["resists"]["fire"],
                cold=mock_pob_output["resists"]["cold"],
                lightning=mock_pob_output["resists"]["lightning"],
                chaos=mock_pob_output["resists"]["chaos"],
            ),
            reservation=NormalizedReservation(
                reserved_percent=mock_pob_output["reservation"]["reserved_percent"],
                available_percent=mock_pob_output["reservation"]["available_percent"],
            ),
            attributes=NormalizedAttributes(
                strength=mock_pob_output["attributes"]["strength"],
                dexterity=mock_pob_output["attributes"]["dexterity"],
                intelligence=mock_pob_output["attributes"]["intelligence"],
            ),
            warnings=(),
        )

        # Evaluate gates
        gate_eval = evaluate_gates(metrics, test_scenario_template.gate_thresholds)

        assert gate_eval.gate_pass is True
        assert gate_eval.gate_slacks.passes_all_gates is True

    def test_e2e_pipeline_with_multiple_scenarios(self, test_skeleton, passing_build_data):
        """Test pipeline evaluation across multiple scenarios."""
        scenarios = [
            ScenarioTemplate(
                scenario_id="scenario_easy",
                version="v0",
                profile_id="easy",
                pob_config={},
                gate_thresholds=ScenarioGateThresholds(
                    min_full_dps=300000.0,
                    min_max_hit=5000.0,
                    reservation=ScenarioReservationThreshold(max_percent=99.0),
                    resists={"fire": 50.0, "cold": 50.0, "lightning": 50.0, "chaos": 0.0},
                    attributes={"strength": 50.0, "dexterity": 50.0, "intelligence": 50.0},
                ),
            ),
            ScenarioTemplate(
                scenario_id="scenario_hard",
                version="v0",
                profile_id="hard",
                pob_config={},
                gate_thresholds=ScenarioGateThresholds(
                    min_full_dps=1500000.0,
                    min_max_hit=25000.0,
                    reservation=ScenarioReservationThreshold(max_percent=90.0),
                    resists={"fire": 85.0, "cold": 85.0, "lightning": 85.0, "chaos": 50.0},
                    attributes={"strength": 200.0, "dexterity": 150.0, "intelligence": 150.0},
                ),
            ),
        ]

        metrics = NormalizedMetrics(
            full_dps=passing_build_data["stats"]["full_dps"],
            max_hit=passing_build_data["stats"]["max_hit"],
            armour=passing_build_data["stats"]["armour"],
            evasion=passing_build_data["stats"]["evasion"],
            life=passing_build_data["stats"]["life"],
            mana=passing_build_data["stats"]["mana"],
            utility_score=passing_build_data["stats"]["utility_score"],
            resists=ResistSnapshot(
                fire=passing_build_data["resistances"]["fire"],
                cold=passing_build_data["resistances"]["cold"],
                lightning=passing_build_data["resistances"]["lightning"],
                chaos=passing_build_data["resistances"]["chaos"],
            ),
            reservation=NormalizedReservation(
                reserved_percent=85.0,
                available_percent=100.0,
            ),
            attributes=NormalizedAttributes(
                strength=passing_build_data["attributes"]["strength"],
                dexterity=passing_build_data["attributes"]["dexterity"],
                intelligence=passing_build_data["attributes"]["intelligence"],
            ),
            warnings=(),
        )

        results = {}
        for scenario in scenarios:
            gate_eval = evaluate_gates(metrics, scenario.gate_thresholds)
            results[scenario.scenario_id] = gate_eval.gate_pass

        # Build should pass easy scenario but fail hard scenario
        assert results["scenario_easy"] is True
        assert results["scenario_hard"] is False


# ============================================================================
# TESTS: Integration with Skeleton Expansion
# ============================================================================


class TestSkeletonExpansion:
    """Test skeleton expansion as part of the pipeline."""

    def test_skeleton_validation(self, test_skeleton):
        """Skeleton should validate successfully."""
        test_skeleton.validate()  # Should not raise

    def test_skeleton_to_dict(self, test_skeleton):
        """Skeleton should convert to dict."""
        skeleton_dict = test_skeleton.to_dict()

        assert skeleton_dict["skeleton_id"] == "test_cyclone_jugg"
        assert skeleton_dict["class_name"] == "marauder"
        assert skeleton_dict["ascendancy"] == "Juggernaut"
        assert skeleton_dict["main_skill"] == "cyclone"

    def test_skeleton_from_dict(self):
        """Skeleton should be created from dict."""
        data = {
            "skeleton_id": "test_id",
            "class_name": "marauder",
            "ascendancy": "Juggernaut",
            "main_skill": "cyclone",
            "skill_links": ["cyclone", "melee"],
            "aura_package": ["determination"],
            "defense_layer": "armour",
            "budget_tier": "starter",
            "target_gates": {"dps": 1000000},
            "required_uniques": [],
            "tree_path": "1,2,3",
        }

        skeleton = Skeleton.from_dict(data)
        assert skeleton.skeleton_id == "test_id"
        assert skeleton.class_name == "marauder"


# ============================================================================
# TESTS: Pipeline Failure Scenarios
# ============================================================================


class TestPipelineFailureScenarios:
    """Test error handling in the pipeline."""

    def test_invalid_skeleton_class(self):
        """Pipeline should reject invalid class."""
        with pytest.raises(ValueError, match="invalid class_name"):
            Skeleton(
                skeleton_id="test",
                class_name="invalid_class",
                ascendancy="Juggernaut",
                main_skill="cyclone",
                skill_links=[],
                aura_package=[],
                defense_layer="armour",
                budget_tier="starter",
                target_gates={},
                required_uniques=[],
                tree_path="1,2,3",
            ).validate()

    def test_invalid_skeleton_ascendancy(self):
        """Pipeline should reject invalid ascendancy."""
        with pytest.raises(ValueError, match="invalid ascendancy"):
            Skeleton(
                skeleton_id="test",
                class_name="marauder",
                ascendancy="InvalidAscendancy",
                main_skill="cyclone",
                skill_links=[],
                aura_package=[],
                defense_layer="armour",
                budget_tier="starter",
                target_gates={},
                required_uniques=[],
                tree_path="1,2,3",
            ).validate()

    def test_csp_multiple_errors(self, passing_build_data):
        """CSP should report multiple errors."""
        passing_build_data["main_skill"]["links"] = 4
        passing_build_data["items"][0]["requirements"]["strength"] = 500
        passing_build_data["reservation"] = 1100.0

        csp = BuildCSP()
        is_valid, errors = csp.validate(passing_build_data)

        assert is_valid is False
        assert len(errors) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
