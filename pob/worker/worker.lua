local ok, json = pcall(require, "cjson.safe")
if not ok or not json then
    ok, json = pcall(require, "cjson")
end
if not ok or not json then
    ok, json = pcall(require, "pob.worker.json_fallback")
end
if not ok or not json then
    io.stderr:write("missing JSON module\n")
    os.exit(1)
end

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

local function validate_evaluate_params(params)
    if type(params) ~= "table" then
        return nil, "params must be a table"
    end
    local has_xml = params.xml ~= nil
    local has_code = params.code ~= nil
    if not has_xml and not has_code then
        return nil, "params require either xml or code"
    end
    return {has_xml = has_xml, has_code = has_code}
end

local function extract_xml_payload(params)
    if type(params.xml) == "string" and params.xml:find("<", 1, true) then
        return params.xml
    end
    if type(params.code) == "string" and params.code:find("<", 1, true) then
        return params.code
    end
    return nil, "worker requires PoB XML payload in params.xml (or XML text in params.code)"
end

local function parse_player_stats(xml_text)
    local stats = {}
    for stat, value in xml_text:gmatch('<PlayerStat%s+stat="([^"]+)"%s+value="([^"]+)"%s*/>') do
        local numeric = tonumber(value)
        if numeric ~= nil then
            stats[stat] = numeric
        end
    end
    for stat, value in xml_text:gmatch("<PlayerStat%s+stat='([^']+)'%s+value='([^']+)'%s*/>") do
        local numeric = tonumber(value)
        if numeric ~= nil then
            stats[stat] = numeric
        end
    end
    return stats
end

local function stat(stats, key, default)
    local value = stats[key]
    if value == nil then
        return default or 0
    end
    return value
end

local function evaluate_from_xml(xml_text)
    local warnings = {}
    local stats = parse_player_stats(xml_text)
    local stats_count = 0
    for _ in pairs(stats) do
        stats_count = stats_count + 1
    end
    if stats_count == 0 then
        return nil, "no PlayerStat values found in XML payload"
    end

    local full_dps = stat(stats, "FullDPS", 0)
    if full_dps == 0 then
        full_dps = stat(stats, "CombinedDPS", stat(stats, "WithDotDPS", stat(stats, "TotalDPS", 0)))
    end
    if full_dps == 0 then
        warnings[#warnings + 1] = "missing_full_dps"
    end

    local life = stat(stats, "Life", 0)
    local mana = stat(stats, "Mana", 0)
    local armour = stat(stats, "Armour", 0)
    local evasion = stat(stats, "Evasion", 0)
    local energy_shield = stat(stats, "EnergyShield", 0)
    local max_hit = stat(stats, "MaximumHitTaken", 0)
    if max_hit == 0 then
        max_hit = stat(stats, "TotalEHP", life + energy_shield)
    end

    local fire = stat(stats, "FireResist", 0)
    local cold = stat(stats, "ColdResist", 0)
    local lightning = stat(stats, "LightningResist", 0)
    local chaos = stat(stats, "ChaosResist", 0)
    local block = stat(stats, "BlockChance", 0)
    local spell_block = stat(stats, "SpellBlockChance", 0)
    local movement = stat(stats, "EffectiveMovementSpeedMod", stat(stats, "MovementSpeedMod", 1))

    local utility_score = movement * 10 + ((block + spell_block) * 0.5) + ((fire + cold + lightning + chaos) * 0.1)

    local life_unreserved = stat(stats, "LifeUnreservedPercent", 100)
    local mana_unreserved = stat(stats, "ManaUnreservedPercent", 100)
    local life_reserved = math.max(0, 100 - life_unreserved)
    local mana_reserved = math.max(0, 100 - mana_unreserved)

    local class_name = xml_text:match('className="([^"]+)"')
    local ascendancy = xml_text:match('ascendClassName="([^"]+)"')

    return {
        status = "ok",
        source = "pob_xml_playerstats",
        identity = {
            class_name = class_name or "unknown",
            ascendancy = ascendancy or "unknown",
        },
        metrics = {
            full_dps = full_dps,
            max_hit = max_hit,
            utility_score = utility_score,
        },
        defense = {
            armour = armour,
            evasion = evasion,
        },
        resources = {
            life = life,
            mana = mana,
        },
        reservation = {
            life_pct = life_reserved,
            mana_pct = mana_reserved,
            available_percent = mana_unreserved,
            reserved_percent = mana_reserved,
        },
        attributes = {
            str = stat(stats, "Str", 0),
            dex = stat(stats, "Dex", 0),
            int = stat(stats, "Int", 0),
        },
        resists = {
            fire = fire,
            cold = cold,
            lightning = lightning,
            chaos = chaos,
        },
        warnings = warnings,
    }
end

local function handle_request(request)
    local method = request.method
    if method == "ping" then
        return {
            id = request.id,
            ok = true,
            result = {
                protocol = "ndjson",
                version = "1.0",
                capabilities = {"ping", "evaluate"},
            },
        }
    elseif method == "evaluate" then
        local params_info, reason = validate_evaluate_params(request.params)
        if not params_info then
            return error_response(request.id, 1003, reason)
        end
        local xml_payload, xml_error = extract_xml_payload(request.params)
        if not xml_payload then
            return error_response(request.id, 1004, xml_error)
        end
        local evaluated, eval_error = evaluate_from_xml(xml_payload)
        if not evaluated then
            return error_response(request.id, 1005, eval_error)
        end
        local payload_type = params_info.has_xml and "xml" or "code"
        evaluated.payload = {
            type = payload_type,
            has_xml = params_info.has_xml,
            has_code = params_info.has_code,
        }
        return {
            id = request.id,
            ok = true,
            result = evaluated,
        }
    end
    return error_response(request.id, 1002, "unknown method")
end

while true do
    local line = io.read("*l")
    if not line then
        break
    end
    if line == "" then
        goto continue
    end

    local payload, err = decode(line)
    if not payload then
        send(error_response(nil, 1000, "invalid json: " .. tostring(err)))
        goto continue
    end

    local response = handle_request(payload)
    send(response)
    ::continue::
end
