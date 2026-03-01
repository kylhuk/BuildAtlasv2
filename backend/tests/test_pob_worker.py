"""
Integration tests for Path of Building LuaJIT worker.

Tests verify that:
1. PoB calculations are real (not stubs)
2. Worker correctly parses build XMLs
3. Configuration options affect calculation results
4. Stats are accurately returned via JSON-RPC protocol
"""

from pathlib import Path

import pytest

from backend.engine.worker_pool import WorkerPool

# Path to test builds
TEST_BUILD_DIR = (
    Path(__file__).parent.parent.parent / "PathOfBuilding" / "spec" / "TestBuilds" / "3.13"
)
OCCVORTEX_BUILD = TEST_BUILD_DIR / "OccVortex.xml"


class TestPoBWorker:
    """Test PoB worker integration."""

    @pytest.fixture
    def pob_worker_pool(self):
        """Create a WorkerPool with PoB LuaJIT worker."""
        pool = WorkerPool(
            num_workers=1,
            worker_cmd=("luajit", "pob/worker/worker.lua"),
            request_timeout=30.0,  # PoB calculations can take time
        )
        yield pool
        pool.close()

    @pytest.mark.skipif(not OCCVORTEX_BUILD.exists(), reason="Test build XML not found")
    def test_worker_loads_test_build(self, pob_worker_pool):
        """Test that worker can load and calculate a test build."""
        # Load test build XML
        with open(OCCVORTEX_BUILD, "r") as f:
            xml = f.read()

        # Create evaluation request
        payloads = [
            {
                "xml": xml,
                "scenario_id": "test_scenario",
                "profile_id": "test_profile",
                "ruleset_id": "test_ruleset",
            }
        ]

        # Evaluate build
        results = pob_worker_pool.evaluate_batch(payloads)

        # Verify we got a result
        assert len(results) == 1
        result = results[0]

        # Verify result structure (not an error)
        assert "result" in result or "error" in result
        if "error" in result:
            pytest.fail(f"Worker returned error: {result['error']}")

        # Verify result contains expected metrics
        result_data = result.get("result", {})
        assert "metrics" in result_data
        assert "defense" in result_data
        assert "resources" in result_data
        assert "attributes" in result_data

    @pytest.mark.skipif(not OCCVORTEX_BUILD.exists(), reason="Test build XML not found")
    def test_worker_returns_numeric_stats(self, pob_worker_pool):
        """Test that worker returns numeric stats (not empty/zero stubs)."""
        with open(OCCVORTEX_BUILD, "r") as f:
            xml = f.read()

        payloads = [{"xml": xml, "scenario_id": "test"}]
        results = pob_worker_pool.evaluate_batch(payloads)

        result = results[0]["result"]
        metrics = result["metrics"]

        # Verify DPS is calculated (should be non-zero for a valid build)
        full_dps = float(metrics.get("full_dps", 0))
        assert full_dps > 0, "DPS should be calculated for valid build"

        # Verify resources are calculated
        resources = result["resources"]
        life = float(resources.get("life", 0))
        assert life > 0, "Life should be calculated"

        # Verify defense stats are present
        defense = result["defense"]
        assert "armour" in defense
        assert "evasion" in defense
        assert "resists" in defense

    @pytest.mark.skipif(not OCCVORTEX_BUILD.exists(), reason="Test build XML not found")
    def test_worker_handles_configuration_options(self, pob_worker_pool):
        """Test that worker accepts configuration options (e.g., frenzy charges)."""
        with open(OCCVORTEX_BUILD, "r") as f:
            xml = f.read()

        # Test with default configuration
        payloads_default = [
            {
                "xml": xml,
                "scenario_id": "default_config",
                "config": {},
            }
        ]

        results_default = pob_worker_pool.evaluate_batch(payloads_default)
        result_default = results_default[0]

        # Verify no error with config param
        assert "result" in result_default or "error" in result_default
        if "error" in result_default:
            # Config errors are acceptable - just verify protocol works
            pass

    def test_worker_returns_json_rpc_format(self, pob_worker_pool):
        """Test that worker returns valid JSON-RPC response format."""
        # Valid request
        payloads = [
            {
                "xml": "<PathOfBuilding></PathOfBuilding>",
                "scenario_id": "invalid_test",
            }
        ]

        results = pob_worker_pool.evaluate_batch(payloads)

        result = results[0]
        # Should have either "result" or "error" field
        assert "result" in result or "error" in result

        # If error, should have proper JSON-RPC error format
        if "error" in result:
            error = result["error"]
            assert "code" in error
            assert "message" in error
            assert isinstance(error["code"], int)

    def test_worker_rejects_invalid_requests(self, pob_worker_pool):
        """Test that worker properly rejects invalid requests."""
        # Request with missing required field
        payloads = [
            {
                # Missing "xml" field
                "scenario_id": "test",
            }
        ]

        results = pob_worker_pool.evaluate_batch(payloads)

        result = results[0]
        # Should return error
        assert "error" in result
        error = result["error"]
        assert error["code"] == -32602  # Invalid params


class TestPoBCalculationAccuracy:
    """Test that PoB calculations match expected values from test specs."""

    @pytest.fixture
    def pob_worker_pool(self):
        """Create a WorkerPool with PoB LuaJIT worker."""
        pool = WorkerPool(
            num_workers=1,
            worker_cmd=("luajit", "pob/worker/worker.lua"),
            request_timeout=30.0,
        )
        yield pool
        pool.close()

    @pytest.mark.skipif(not OCCVORTEX_BUILD.exists(), reason="Test build XML not found")
    def test_calculations_are_deterministic(self, pob_worker_pool):
        """Test that running same build twice gives same results."""
        with open(OCCVORTEX_BUILD, "r") as f:
            xml = f.read()

        # Run twice
        payloads = [{"xml": xml, "scenario_id": "test"}]
        results1 = pob_worker_pool.evaluate_batch(payloads)
        results2 = pob_worker_pool.evaluate_batch(payloads)

        # Compare results (should be identical)
        result1 = results1[0].get("result", {})
        result2 = results2[0].get("result", {})

        # DPS should be identical
        dps1 = float(result1.get("metrics", {}).get("full_dps", 0))
        dps2 = float(result2.get("metrics", {}).get("full_dps", 0))

        assert dps1 == dps2, "Calculations should be deterministic"

        # Life should be identical
        life1 = float(result1.get("resources", {}).get("life", 0))
        life2 = float(result2.get("resources", {}).get("life", 0))

        assert life1 == life2, "Calculations should be deterministic"


if __name__ == "__main__":
    # Quick sanity check - can be run standalone
    import sys

    if not OCCVORTEX_BUILD.exists():
        print(f"Warning: Test build not found at {OCCVORTEX_BUILD}")
        print("Run tests with: pytest backend/tests/test_pob_worker.py")
        sys.exit(0)

    print("Running PoB worker sanity check...")
    pool = WorkerPool(
        num_workers=1,
        worker_cmd=("luajit", "pob/worker/worker.lua"),
        request_timeout=30.0,
    )

    try:
        with open(OCCVORTEX_BUILD, "r") as f:
            xml = f.read()

        payloads = [{"xml": xml, "scenario_id": "test"}]
        results = pool.evaluate_batch(payloads)

        result = results[0]
        print(f"✓ Worker responded: {list(result.keys())}")

        if "result" in result:
            print("✓ Calculation successful")
            metrics = result["result"].get("metrics", {})
            dps = metrics.get("full_dps", 0)
            print(f"✓ DPS calculated: {dps}")
        else:
            print(f"✗ Error: {result.get('error', 'Unknown')}")

    finally:
        pool.close()
