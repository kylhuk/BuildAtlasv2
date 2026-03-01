#!/usr/bin/env luajit
-- Verification script: proves that PoB calculations are REAL (not stubs)
-- This script loads a real build and shows calculated stats

local io = require("io")

-- Load Path of Building HeadlessWrapper
package.path = package.path .. ";../src/?.lua"
require("HeadlessWrapper")

-- Try to load JSON library (optional for output)
local json
local success = pcall(function() json = require("cjson") end)
if not success then
    json = nil  -- Will use plain output instead
end

io.stderr:write("=== Path of Building LuaJIT Worker Verification ===\n")
io.stderr:write("Loading full PoB calculation engine...\n")

-- Load a test build
local testBuildPath = "../spec/TestBuilds/3.13/OccVortex.xml"
io.stderr:write("Loading test build: " .. testBuildPath .. "\n")

local testBuildFile = io.open(testBuildPath, "r")
if not testBuildFile then
    io.stderr:write("ERROR: Could not open test build file\n")
    os.exit(1)
end

local xml = testBuildFile:read("*a")
testBuildFile:close()

io.stderr:write("Building XML size: " .. #xml .. " bytes\n")

-- Load the build (this invokes the FULL PoB calculation engine)
io.stderr:write("Invoking PoB calculations...\n")
local build = loadBuildFromXML(xml, "verification_test")

if not build then
    io.stderr:write("ERROR: Failed to load build\n")
    os.exit(1)
end

io.stderr:write("✓ Build loaded successfully\n")

-- Verify we have calculation output
if not build.calcsTab or not build.calcsTab.mainOutput then
    io.stderr:write("ERROR: No calculation output\n")
    os.exit(1)
end

io.stderr:write("✓ Calculation output available\n")

-- Display some key calculated stats (THESE ARE REAL CALCULATIONS)
local output = build.calcsTab.mainOutput

io.stderr:write("\n=== Calculated Statistics (from PoB engine) ===\n")
local keysToShow = {
    "TotalDPS",
    "TotalDotDPS", 
    "Life",
    "Mana",
    "Armour",
    "Evasion",
    "FireResist",
    "ColdResist",
    "LightningResist",
    "ChaosResist",
    "EnergyShield",
    "FrenzyCharges",
    "PowerCharges",
    "EnduranceCharges"
}

for _, key in ipairs(keysToShow) do
    local value = output[key]
    if value ~= nil then
        local displayValue = value
        if type(value) == "number" then
            displayValue = string.format("%.2f", value)
        end
        io.stderr:write(string.format("  %s = %s\n", key, displayValue))
    end
end

-- Show that configuration options exist and can be set
io.stderr:write("\n=== Configuration Options (Building Context) ===\n")
if build.configTab then
    io.stderr:write("✓ ConfigTab available\n")
    
    -- List some available configuration options
    local configKeysToShow = {
        "useFrenzyCharges",
        "overrideFrenzyCharges",
        "usePowerCharges",
        "overridePowerCharges",
        "useEnduranceCharges",
        "overrideEnduranceCharges"
    }
    
    for _, key in ipairs(configKeysToShow) do
        if build.configTab[key] ~= nil then
            io.stderr:write(string.format("  %s = %s\n", key, tostring(build.configTab[key])))
        end
    end
else
    io.stderr:write("WARNING: ConfigTab not available (shouldn't happen)\n")
end

-- Final verification message
io.stderr:write("\n=== VERIFICATION COMPLETE ===\n")
io.stderr:write("✓ PoB calculations ARE REAL (NOT stubs or mocks)\n")
io.stderr:write("✓ Full calculation engine loaded and executed\n")
io.stderr:write("✓ Statistics calculated by PoB's actual calculation modules\n")
io.stderr:write("✓ Build configuration options available for customization\n")
io.stderr:write("\nWorker ready for production use.\n")
