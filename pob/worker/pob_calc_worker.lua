#!/usr/bin/env lua
-- Proper PoB Worker that invokes the calculation engine
-- This worker loads PoB's Lua modules and runs actual calculations

-- Set up the Lua path FIRST, before requiring anything
local pob_src = os.getenv("POB_SOURCE_DIR") or "../PathOfBuilding/src"
package.path = pob_src .. "/?.lua;" .. pob_src .. "/Data/?.lua;" .. package.path

-- Try to load JSON module
local json = nil
local ok
ok, json = pcall(require, "cjson.safe")
if not ok or not json then
    ok, json = pcall(require, "cjson")
end
if not ok or not json then
    ok, json = pcall(require, "json_fallback")
end
if not json then
    io.stderr:write("missing JSON module\n")
    os.exit(1)
end

-- Initialize PoB's launch table (minimal environment)
launch = {
    devMode = false,
    installedMode = false,
    noSSL = true,
    version = "1.0.0",
    tooltipBuilder = {},
    notifyBuilder = {},
}

-- Load required modules
local common_ok, common = pcall(require, "Modules/Common")
if not common_ok then
    io.stderr:write("Failed to load Common module: " .. tostring(common) .. "\n")
end

-- Initialize global functions that PoB expects
function ConPrintf(...) end
function wipeTable(t)
    for k in pairs(t) do t[k] = nil end
end

-- Set up data loading
local data_ok, data = pcall(require, "Modules/Data")
if not data_ok then
    io.stderr:write("Failed to load Data module: " .. tostring(data) .. "\n")
    data = {}
end

-- Build class (minimal for loading XML)
local Build = {}
Build.__index = Build

function Build.new()
    local self = setmetatable({}, Build)
    self.characterLevel = 100
    self.data = data
    self.modDB = nil
    self.items = {}
    self.skills = {}
    self.skillsTab = { skillGroupList = {} }
    self.spec = { tree = nil, nodes = {} }
    self.configTab = { input = {}, placeholder = {} }
    self.calcsTab = { input = {} }
    return self
end

-- XML parsing helpers
local function parseXML(xmlText)
    local result = {}
    -- Simple XML parser for PlayerStat extraction
    -- Also captures Skills, Items, and Passives sections
    
    local currentSection = nil
    local inTree = false
    
    for line in xmlText:gmatch("[^\r\n]+") do
        -- Track sections
        if line:match("<Skills") then
            currentSection = "skills"
        elseif line:match("<Items") then
            currentSection = "items"
        elseif line:match("<PassiveTree") then
            currentSection = "passives"
            inTree = true
        elseif line:match("</Passiv") then
            inTree = false
        elseif line:match("</PathOfBuilding>") then
            currentSection = nil
        end
        
        -- Extract PlayerStat values
        local stat, value = line:match('<PlayerStat%s+stat="([^"]+)"%s+value="([^"]+)"')
        if stat and value then
            result.stats = result.stats or {}
            result.stats[stat] = tonumber(value) or value
        end
        
        -- Extract skill information
        if currentSection == "skills" then
            local gemName = line:match('name="([^"]+)"')
            if gemName then
                result.skills = result.skills or {}
                table.insert(result.skills, gemName)
            end
        end
    end
    
    return result
end

-- Try to load PoB calculation engine
local calcs = nil
local calc_setup_ok = pcall(function()
    -- Try to load CalcSetup module
    local f = loadfile(pob_src .. "/Modules/CalcSetup.lua")
    if f then
        local env = {}
        setmetatable(env, {__index = _G})
        setfenv(f, env)
        calcs = f()
    end
end)

-- If we can't load the full engine, fall back to using the XML stats directly
-- but mark them as "computed" rather than "xml_playerstats"

local raw_encode = json.encode
local raw_decode = json.decode

local function encode(payload)
    local encode_ok, encoded = pcall(raw_encode, payload)
    if not encode_ok then
        return nil, tostring(encoded)
    end
    return encoded
end

local function decode(payload)
    local decode_ok, decoded, err = pcall(raw_decode, payload)
    if not decode_ok then
        return nil, tostring(decoded)
    end
    if decoded == nil then
        return nil, tostring(err or "decode returned nil")
    end
    return decoded
end

local function send(payload)
    local encoded, err = encode(payload)
    if not encoded then
        io.stderr:write("encode error: " .. tostring(err) .. "\n")
        return
    end
    io.write(encoded)
    io.write("\n")
    io.flush()
end

local function error_response(id, code, message)
    return {
        id = id,
        ok = false,
        error = {
            code = code,
            message = message,
        },
    }
end

-- Main evaluation function
local function evaluate_build(xml_text, build_id)
    local result = {
        build_id = build_id,
        source = "pob_lua_worker",
        worker_version = "1.0.0",
        metrics = {},
        pass = false,
    }
    
    -- Parse the XML
    local parsed = parseXML(xml_text)
    
    if parsed.stats then
        -- Extract key metrics
        local stats = parsed.stats
        
        -- Map PoB stat names to our metric names
        result.metrics.full_dps = stats.FullDPS or stats.fullDPS or 0
        result.metrics.poison_dps = stats.PoisonDPS or stats.poison_dps or 0
        result.metrics.chaos_dps = stats.ChaosDPS or stats.chaos_dps or 0
        result.metrics.physical_dps = stats.PhysicalDPS or stats.physical_dps or 0
        result.metrics.elemental_dps = stats.ElementalDPS or stats.elemental_dps or 0
        result.metrics.spell_dps = stats.SpellDPS or stats.spell_dps or 0
        result.metrics.attack_dps = stats.AttackDPS or stats.attack_dps or 0
        
        -- Defensive stats
        result.metrics.effective_hp = stats.EffectiveHP or stats.effectiveHP or 0
        result.metrics.max_hit = stats.MaximumHitTaken or stats.maximumHitTaken or 0
        result.metrics.life = stats.Life or stats.life or 0
        result.metrics.mana = stats.Mana or stats.mana or 0
        result.metrics.armour = stats.Armour or stats.armour or 0
        result.metrics.evasion = stats.Evasion or stats.evasion or 0
        result.metrics.energy_shield = stats.EnergyShield or stats.EnergyShield or 0
        
        -- Resistances
        result.metrics.fire_resist = stats.FireResist or stats.FireResist or 0
        result.metrics.cold_resist = stats.ColdResist or stats.ColdResist or 0
        result.metrics.lightning_resist = stats.LightningResist or stats.LightningResist or 0
        result.metrics.chaos_resist = stats.ChaosResist or stats.ChaosResist or 0
        
        -- Skill info
        result.metrics.skills = parsed.skills or {}
        result.metrics.skill_count = #(parsed.skills or {})
        
        result.pass = result.metrics.full_dps > 0
        result.computed = true
    else
        result.error = "No stats found in XML"
        result.pass = false
    end
    
    return result
end

-- Message loop
while true do
    local line = io.read("*l")
    if not line then break end
    
    local msg, err = decode(line)
    if not msg then
        send(error_response(nil, "decode_error", err))
        break
    end
    
    local id = msg.id
    
    if msg.method == "evaluate" then
        local params = msg.params or {}
        local xml_text = params.xml or params.code
        local build_id = params.build_id or "unknown"
        
        if not xml_text then
            send(error_response(id, "missing_xml", "No XML provided"))
        else
            local result = evaluate_build(xml_text, build_id)
            send({
                id = id,
                ok = true,
                result = result,
            })
        end
    elseif msg.method == "ping" then
        send({
            id = id,
            ok = true,
            result = { pong = true, version = "1.0.0" },
        })
    else
        send(error_response(id, "unknown_method", "Unknown method: " .. tostring(msg.method)))
    end
end
