#!/bin/bash
# Simple test for PoB LuaJIT worker

set -e

WORKER_DIR="$(cd "$(dirname "$0")" && pwd)"
POB_ROOT="${WORKER_DIR}/../../PathOfBuilding"

# Use an existing build from the database (more likely to work with current PoB version)
TEST_BUILD="${WORKER_DIR}/../../data/builds/0004110d12bf4e0cb94591353ddcc7df/build.xml.gz"

echo "Testing PoB LuaJIT worker..."
echo "Test build: $TEST_BUILD"

if [ ! -f "$TEST_BUILD" ]; then
    echo "ERROR: Test build XML not found at $TEST_BUILD"
    exit 1
fi

# Read the XML file and encode as JSON
# Note: The build is gzipped, so we need to decompress it first
# Use Python to properly escape the XML for JSON (handles all special chars)
ENCODED_XML=$(python3 -c "import json,sys; print(json.dumps(__import__('gzip').decompress(open('$TEST_BUILD','rb').read()).decode('utf-8')))")

# Create JSON-RPC request
REQUEST="{\"id\":1,\"method\":\"evaluate\",\"params\":{\"xml\":$ENCODED_XML,\"scenario_id\":\"test_scenario\",\"profile_id\":\"test_profile\",\"ruleset_id\":\"test_ruleset\"}}"

# Run worker and send request
# Worker must be run from PathOfBuilding/src/ directory
cd "${POB_ROOT}/src" && echo "$REQUEST" | luajit "${WORKER_DIR}/pob_worker.lua"
