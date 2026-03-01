-- LuaJIT worker for Path of Building headless calculations
-- Implements JSON-RPC NDJSON protocol over stdin/stdout

local json = require("cjson")
local io = require("io")

-- Load Path of Building HeadlessWrapper
-- NOTE: This script is run from PathOfBuilding/src/ directory (set by worker_pool.py cwd)
--       HeadlessWrapper.lua expects Launch.lua in the current working directory
-- Paths are relative to PathOfBuilding/src/:
--   ./?.lua - PathOfBuilding/src/
--   ../runtime/lua/?.lua - PathOfBuilding/runtime/lua/
--   ../runtime/lua/?/init.lua - PathOfBuilding/runtime/lua/*/init.lua (for sha1, etc.)
package.path = package.path .. ";./?.lua;../runtime/lua/?.lua;../runtime/lua/?/init.lua"

-- Monkey-patch missing functions before loading HeadlessWrapper
-- GetVirtualScreenSize was added in newer PoB versions but missing in our fork
_G.GetVirtualScreenSize = function()
	return 1920, 1080
end

-- Pre-initialize global mainObject BEFORE loading HeadlessWrapper
-- This ensures SetMainObject (which sets global) will find the existing table
_G.mainObject = _G.mainObject or {}
print("Pre-initialized _G.mainObject")

-- Debug: wrap require to catch errors
local startTime = os.clock()
print("Loading HeadlessWrapper...")
local success, err = pcall(require, "HeadlessWrapper")
if not success then
    io.stderr:write("ERROR loading HeadlessWrapper: " .. tostring(err) .. "\n")
    os.exit(1)
end
print("HeadlessWrapper loaded in " .. string.format("%.2f", os.clock() - startTime) .. "s")

-- WORKAROUND: HeadlessWrapper has a bug where SetMainObject sets a global instead of local
-- After loading, the global mainObject is set, but the local one is nil
-- We need to manually fix the local reference
print("Applying mainObject workaround...")
print("Global mainObject = " .. tostring(mainObject))
_G.mainObject = mainObject  -- Ensure global is set
local globalMainObject = mainObject  -- Capture for closure
print("Captured globalMainObject = " .. tostring(globalMainObject))

-- Try to trigger initialization if not done
if globalMainObject and not globalMainObject.main then
    print("mainObject.main missing, trying to trigger initialization...")
    print("globalMainObject type = " .. type(globalMainObject))
    print("globalMainObject.OnInit = " .. tostring(globalMainObject.OnInit))
    local ok, err = pcall(function()
        if globalMainObject.OnInit then
            print("Calling launch:OnInit()...")
            globalMainObject:OnInit()
            print("After OnInit, mainObject.main = " .. tostring(globalMainObject.main))
        else
            print("No OnInit method found")
        end
    end)
    if not ok then
        print("ERROR in OnInit: " .. tostring(err))
    end
end

-- Now we need to fix the functions to use global - we'll redefine them
loadBuildFromXML = function(xmlText, name)
    -- Use the global we captured
    print("loadBuildFromXML: globalMainObject = " .. tostring(globalMainObject))
    print("loadBuildFromXML: main = " .. tostring(globalMainObject and globalMainObject.main))
    if not globalMainObject then
        error("globalMainObject is nil!")
    end
    if not globalMainObject.main then
        error("globalMainObject.main is nil! Cannot load build.")
    end
    globalMainObject.main:SetMode("BUILD", false, name or "", xmlText)
    runCallback("OnFrame")
end
print("Workaround applied")

-- Ensure stdout is line-buffered for NDJSON protocol
io.stdout:setvbuf("line")
io.stderr:setvbuf("line")

-- Helper to safely get numeric stat values from PoB output
local function getStat(output, key, default)
    local value = output[key]
    if value == nil then
        return default or 0.0
    end
    local num = tonumber(value)
    if num == nil then
        return default or 0.0
    end
    return num
end

-- Extract metrics from PoB build.calcsTab.mainOutput
local function extractMetrics(build)
    if not build or not build.calcsTab or not build.calcsTab.mainOutput then
        return nil, "build calculation output not available"
    end
    
    -- Verify we have actual calculation results (not just empty structure)
    local output = build.calcsTab.mainOutput
    if not next(output) then
        return nil, "build calculations produced no output (empty mainOutput)"
    end
    
    local player = build.calcsTab.mainEnv and build.calcsTab.mainEnv.player
    
    -- Extract core metrics
    local full_dps = getStat(output, "TotalDPS", 0.0)
    local max_hit = getStat(output, "TotalEHP", 0.0)  -- Using EHP as proxy for max hit survivability
    
    -- Defense metrics
    local armour = getStat(output, "Armour", 0.0)
    local evasion = getStat(output, "Evasion", 0.0)
    
    -- Resistances
    local fire_resist = getStat(output, "FireResist", 0.0)
    local cold_resist = getStat(output, "ColdResist", 0.0)
    local lightning_resist = getStat(output, "LightningResist", 0.0)
    local chaos_resist = getStat(output, "ChaosResist", 0.0)
    
    -- Resources
    local life = getStat(output, "Life", 0.0)
    local mana = getStat(output, "Mana", 0.0)
    
    -- Reservation
    local mana_reserved = getStat(output, "ManaReserved", 0.0)
    local mana_unreserved = getStat(output, "ManaUnreserved", 0.0)
    local total_mana = mana_reserved + mana_unreserved
    local reserved_percent = 0.0
    local available_percent = 100.0
    if total_mana > 0 then
        reserved_percent = (mana_reserved / total_mana) * 100.0
        available_percent = 100.0
    end
    
    -- Attributes
    local strength = getStat(output, "Str", 0.0)
    local dexterity = getStat(output, "Dex", 0.0)
    local intelligence = getStat(output, "Int", 0.0)
    
    -- Build utility score (placeholder - could be enhanced)
    local utility_score = 0.0
    
    return {
        metrics = {
            full_dps = full_dps,
            max_hit = max_hit,
            utility_score = utility_score,
        },
        defense = {
            armour = armour,
            evasion = evasion,
            resists = {
                fire = fire_resist,
                cold = cold_resist,
                lightning = lightning_resist,
                chaos = chaos_resist,
            },
        },
        resources = {
            life = life,
            mana = mana,
        },
        reservation = {
            reserved_percent = reserved_percent,
            available_percent = available_percent,
        },
        attributes = {
            strength = strength,
            dexterity = dexterity,
            intelligence = intelligence,
        },
        warnings = {},
    }
end

-- Process single evaluation request
-- Input params:
--   xml (string, required): PathOfBuilding XML export of the build
--   scenario_id (string): User-defined scenario identifier for tracking
--   config (table, optional): Build configuration overrides
--     Example: {useFrenzyCharges = true, overrideFrenzyCharges = 5}
--     See ConfigOptions.lua for all available options
local function evaluateBuild(params)
    -- Validate input
    if not params or type(params) ~= "table" then
        return nil, {code = -32602, message = "Invalid params: expected object"}
    end
    
    local xml = params.xml
    if not xml or type(xml) ~= "string" or #xml == 0 then
        return nil, {code = -32602, message = "Invalid params: xml field required"}
    end
    
    local scenario_id = params.scenario_id or "unknown"
    local config = params.config or {}
    
    -- Load build from XML using HeadlessWrapper
    -- This invokes the FULL Path of Building calculation engine
    local success, build = pcall(function()
        -- Use our workaround function
        return loadBuildFromXML(xml, scenario_id)
    end)
    
    if not success then
        -- Provide more debug info
        local debugMsg = "Failed to load build XML: " .. tostring(build)
        if mainObject then
            debugMsg = debugMsg .. " (mainObject exists)"
        else
            debugMsg = debugMsg .. " (mainObject is nil)"
        end
        return nil, {code = -32000, message = debugMsg}
    end
    
    if not build then
        -- Try to get more info
        local debugInfo = ""
        pcall(function()
            local mode = mainObject.main:GetMode()
            debugInfo = " mode=" .. tostring(mode)
        end)
        return nil, {code = -32000, message = "loadBuildFromXML returned nil" .. debugInfo}
    end
    
    -- Apply configuration overrides if provided (e.g., frenzy charges, flask buffs, etc.)
    if config and next(config) then
        if build.configTab then
            for optionKey, value in pairs(config) do
                -- Try to set the configuration option
                -- This applies the configuration change to the build's modifiers
                if build.configTab[optionKey] ~= nil then
                    build.configTab[optionKey] = value
                end
            end
            -- Re-run calculations after configuration changes
            if build.calcsTab and build.calcsTab.BuildOutput then
                build.calcsTab:BuildOutput()
            end
        end
    end
    
    -- Extract metrics from calculated build
    local metrics, err = extractMetrics(build)
    if not metrics then
        return nil, {code = -32000, message = "Failed to extract metrics: " .. tostring(err)}
    end
    
    return metrics, nil
end

-- Main request handler
local function handleRequest(request)
    local id = request.id
    local method = request.method
    local params = request.params
    
    if method ~= "evaluate" then
        return {
            id = id,
            error = {code = -32601, message = "Method not found: " .. tostring(method)}
        }
    end
    
    local result, error = evaluateBuild(params)
    if error then
        return {id = id, error = error}
    end
    
    return {id = id, result = result}
end

-- Main loop: read NDJSON from stdin, process, write NDJSON to stdout
local function main()
    io.stderr:write("PoB LuaJIT worker started\n")
    
    for line in io.stdin:lines() do
        -- Parse JSON-RPC request
        local success, request = pcall(json.decode, line)
        if not success then
            local response = {
                id = nil,
                error = {code = -32700, message = "Parse error: invalid JSON"}
            }
            io.stdout:write(json.encode(response) .. "\n")
        else
            -- Handle request and send response
            local response = handleRequest(request)
            io.stdout:write(json.encode(response) .. "\n")
        end
    end
    
    io.stderr:write("PoB LuaJIT worker exiting\n")
end

-- Run main loop
main()
