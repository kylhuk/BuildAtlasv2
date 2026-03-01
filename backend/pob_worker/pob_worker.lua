-- PoB Worker: Properly integrated with HeadlessWrapper
-- Uses PoB's own module system correctly

local json = require("cjson")
local io = require("io")

-- ============================================================================
-- SETUP: Configure paths (relative to PathOfBuilding/src/)
-- ============================================================================

package.path = package.path .. ";./?.lua;../runtime/lua/?.lua;../runtime/lua/?/init.lua"

_G.GetVirtualScreenSize = function() return 1920, 1080 end
_G.GetScreenSize = function() return 1920, 1080 end
_G.GetScreenScale = function() return 1 end
_G.RenderInit = function() end
_G.ConExecute = function() end
_G.ConPrintf = function(...) end
_G.ConClear = function() end

launch = { }
_G.launch = launch

local mainObject = nil
local function SetMainObject(obj)
    mainObject = obj
    _G.mainObject = obj
end
SetMainObject(launch)

local success, err = pcall(require, "HeadlessWrapper")
if not success then 
    io.stderr:write("ERROR loading HeadlessWrapper: " .. tostring(err) .. "\n")
    os.exit(1)
end

local main = launch.main

-- Helper to safely get numeric stat values from PoB output
local function getStat(output, key, default)
    if not output then return default or 0.0 end
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

-- ============================================================================
-- EVALUATION FUNCTION
-- ============================================================================

local function evaluate_build(xml_text, scenario_id, profile_id, ruleset_id)
    if not main then
        return nil, {code = -32000, message = "main module not initialized"}
    end
    
    -- First switch to BUILD mode with the XML
    main:SetMode("BUILD", false, "worker_build", xml_text)
    
    -- Run multiple frames to process the build
    for i = 1, 5 do
        runCallback("OnFrame")
    end
    
    local buildMode = main.modes["BUILD"]
    if not buildMode then
        return nil, {code = -32000, message = "Build mode not loaded"}
    end
    
    if not buildMode.calcsTab then
        return nil, {code = -32000, message = "Calculation output not available (calcsTab nil)"}
    end
    
    if not buildMode.calcsTab.mainOutput then
        return nil, {code = -32000, message = "Calculation output not available (mainOutput nil)"}
    end
    
    local output = buildMode.calcsTab.mainOutput
    
    -- Extract metrics
    local full_dps = getStat(output, "TotalDPS", 0.0)
    local max_hit = getStat(output, "TotalEHP", 0.0)
    
    -- Simple utility score
    local utility_score = 0
    if full_dps > 0 and max_hit > 0 then
        local ratio = full_dps / max_hit
        utility_score = math.min(1.0, math.log(1 + ratio) / 10)
    end
    
    return {
        metrics = {
            full_dps = full_dps,
            max_hit = max_hit,
            utility_score = utility_score,
        },
        defense = {
            ehp = max_hit,
            ehp_percent = 100,
            armour = getStat(output, "Armour", 0.0),
            evasion = getStat(output, "Evasion", 0.0),
            resists = {
                fire = getStat(output, "FireResist", 0.0),
                cold = getStat(output, "ColdResist", 0.0),
                lightning = getStat(output, "LightningResist", 0.0),
                chaos = getStat(output, "ChaosResist", 0.0),
            }
        },
        resources = {
            life = getStat(output, "Life", 0.0),
            mana = getStat(output, "Mana", 0.0),
            mana_percent = 100,
        },
        reservation = {
            -- PoB natively has ManaUnreservedPercent
            available_percent = getStat(output, "ManaUnreservedPercent", 100.0),
            reserved_percent = 100.0 - getStat(output, "ManaUnreservedPercent", 100.0)
        },
        attributes = {
            class_name = buildMode.spec and buildMode.spec.curClassName or "Unknown",
            ascendancy_name = buildMode.spec and buildMode.spec.curAscendClassName or "None",
            strength = getStat(output, "Str", 0.0),
            dexterity = getStat(output, "Dex", 0.0),
            intelligence = getStat(output, "Int", 0.0)
        },
        skills = {},
    }
end

-- ============================================================================
-- JSON-RPC LOOP
-- ============================================================================

while true do
    local line = io.read("*l")
    if not line then break end
    
    -- Skip empty lines
    if line == "" then
        goto continue
    end
    
    local ok, request = pcall(json.decode, line)
    if not ok then
        io.stderr:write("JSON parse error: " .. tostring(request) .. "\n")
        goto continue
    end
    
    local id = request.id
    local method = request.method
    local params = request.params or {}
    
    if method == "evaluate" then
        local result, err = evaluate_build(
            params.xml,
            params.scenario_id,
            params.profile_id,
            params.ruleset_id
        )
        
        if result then
            print(json.encode({jsonrpc = "2.0", id = id, result = result}))
        else
            print(json.encode({jsonrpc = "2.0", id = id, error = err}))
        end
        
    elseif method == "ping" then
        print(json.encode({jsonrpc = "2.0", id = id, result = {status = "ok"}}))
    else
        print(json.encode({
            jsonrpc = "2.0", 
            id = id, 
            error = {code = -32601, message = "Method not found"}
        }))
    end
    
    ::continue::
end
