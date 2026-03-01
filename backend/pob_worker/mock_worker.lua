-- Simple Mock PoB Worker
-- Returns calculated metrics based on build parameters
-- This is a placeholder that lets the ML loop run

local json = require("cjson")

-- Configuration
local POWER_MULTIPLIER = 1.0  -- Scale factor for all calculations

-- Simple hash function for deterministic results
local function hash(str)
    local h = 0
    for i = 1, #str do
        h = (h * 31 + string.byte(str, i)) % 2147483647
    end
    return h
end

-- Calculate DPS based on class, ascendancy, and parameters
local function calculate_dps(class_name, ascendancy, seed)
    -- Base DPS by class (rough approximation)
    local class_dps = {
        ["Trickster"] = 800000,
        ["Hierophant"] = 750000,
        ["Deadeye"] = 900000,
        ["Slayer"] = 850000,
        ["Champion"] = 700000,
        ["Saboteur"] = 780000,
        ["Assassin"] = 820000,
        ["Necromancer"] = 680000,
        ["Occultist"] = 720000,
        ["Elementalist"] = 750000,
        ["Guardian"] = 600000,
        ["Inquisitor"] = 780000,
        ["Pathfinder"] = 850000,
        ["Juggernaut"] = 550000,
        ["Berserker"] = 950000,
        ["Chieftain"] = 700000,
        ["Raider"] = 880000,
        ["Ascendant"] = 650000,
    }
    
    local base = class_dps[class_name] or 500000
    
    -- Add ascendancy bonus
    if ascendancy and ascendancy ~= "" then
        base = base * 1.15
    end
    
    -- Add some variation based on seed
    local variation = 0.9 + (seed % 200) / 1000  -- 0.9 to 1.1
    
    return base * variation * POWER_MULTIPLIER
end

-- Calculate effective hit pool
local function calculate_ehp(class_name, ascendancy, seed)
    local base_ehp = {
        ["Trickster"] = 15000,
        ["Hierophant"] = 18000,
        ["Deadeye"] = 12000,
        ["Slayer"] = 20000,
        ["Champion"] = 25000,
        ["Saboteur"] = 14000,
        ["Assassin"] = 13000,
        ["Necromancer"] = 16000,
        ["Occultist"] = 15000,
        ["Elementalist"] = 14500,
        ["Guardian"] = 22000,
        ["Inquisitor"] = 17000,
        ["Pathfinder"] = 15500,
        ["Juggernaut"] = 35000,
        ["Berserker"] = 16000,
        ["Chieftain"] = 20000,
        ["Raider"] = 14000,
        ["Ascendant"] = 18000,
    }
    
    local base = base_ehp[class_name] or 15000
    
    if ascendancy and ascendancy ~= "" then
        base = base * 1.2
    end
    
    local variation = 0.85 + (seed % 300) / 1000
    
    return base * variation
end

-- Process a single build evaluation
local function evaluate_build(xml, scenario_id, profile_id, ruleset_id)
    -- Extract class and ascendancy from XML if possible
    -- Default to random if not found
    local class_name = "Trickster"
    local ascendancy = "Ghostshrouds"
    local seed = hash(xml)
    
    -- Try to parse class from XML
    local class_match = xml:match('class="([^"]+)"')
    local asc_match = xml:match('ascendancy="([^"]+)"')
    
    if class_match and class_match ~= "" then
        class_name = class_match
    end
    if asc_match and asc_match ~= "" then
        ascendancy = asc_match
    end
    
    -- Calculate metrics
    local full_dps = calculate_dps(class_name, ascendancy, seed)
    local max_hit = full_dps * 0.6  -- Approx max hit is 60% of DPS
    local ehp = calculate_ehp(class_name, ascendancy, seed)
    
    -- Utility score: balance of DPS and EHP
    local utility_score = math.sqrt(full_dps * ehp) / 1000000
    
    return {
        full_dps = full_dps,
        max_hit = max_hit,
        ehp = ehp,
        utility_score = utility_score,
        class_name = class_name,
        ascendancy = ascendancy,
    }
end

-- JSON-RPC handler
local function handle_request(req)
    local method = req.method
    local params = req.params or {}
    
    if method == "evaluate" then
        local xml = params.xml or ""
        local scenario_id = params.scenario_id or "default"
        local profile_id = params.profile_id or "pinnacle"
        local ruleset_id = params.ruleset_id or "standard"
        
        local result = evaluate_build(xml, scenario_id, profile_id, ruleset_id)
        
        return {
            jsonrpc = "2.0",
            id = req.id,
            result = {
                metrics = {
                    full_dps = result.full_dps,
                    max_hit = result.max_hit,
                    utility_score = result.utility_score,
                },
                defense = {
                    ehp = result.ehp,
                    ehp_percent = 100,
                    armour = 10000,
                    evasion = 10000,
                    resists = {
                        fire = 85,
                        cold = 85,
                        lightning = 85,
                        chaos = 75
                    }
                },
                resources = {
                    life = 5000,
                    mana = 100,
                    mana_percent = 100,
                },
                reservation = {
                    reserved_percent = 20,
                    available_percent = 80
                },
                attributes = {
                    class_name = result.class_name,
                    ascendancy_name = result.ascendancy,
                    strength = 200,
                    dexterity = 200,
                    intelligence = 200
                },
                calculation_time_ms = 10,
            }
        }
    elseif method == "ping" then
        return {
            jsonrpc = "2.0",
            id = req.id,
            result = { status = "ok" }
        }
    else
        return {
            jsonrpc = "2.0",
            id = req.id,
            error = {
                code = -32601,
                message = "Method not found: " .. method
            }
        }
    end
end

-- NDJSON main loop
local function main()
    -- Read lines from stdin
    local stdin = io.stdin
    local stdout = io.stdout
    
    -- Skip any startup messages, read JSON-RPC requests
    while true do
        local line = stdin:read("*l")
        if not line then
            break
        end
        
        if line ~= "" then
            local ok, req = pcall(json.decode, line)
            if not ok then
                io.stderr:write("JSON parse error: " .. tostring(req) .. "\n")
            else
                local resp = handle_request(req)
                local ok2, resp_str = pcall(json.encode, resp)
                if ok2 then
                    stdout:write(resp_str .. "\n")
                    stdout:flush()
                end
            end
        end
    end
end

-- Run
main()
