#!/bin/bash
# Run headless PoB worker from the correct directory
cd "$(dirname "$0")/../../PathOfBuilding/src"
exec lua "../../pob/worker/pob_headless_worker.lua" "$@"
