#!/bin/bash
cd ../../PathOfBuilding/src
# Run verify_calculations but inject the correct package path first
luajit -e "package.path = package.path .. ';./?.lua;../runtime/lua/?.lua;../runtime/lua/?/init.lua'" ../../backend/pob_worker/verify_calculations.lua
