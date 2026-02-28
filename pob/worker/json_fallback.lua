local json = {}
json.null = {}

local function utf8_from_codepoint(codepoint)
    if codepoint <= 0x7F then
        return string.char(codepoint)
    end
    if codepoint <= 0x7FF then
        local b1 = 0xC0 + math.floor(codepoint / 0x40)
        local b2 = 0x80 + (codepoint % 0x40)
        return string.char(b1, b2)
    end
    if codepoint <= 0xFFFF then
        local b1 = 0xE0 + math.floor(codepoint / 0x1000)
        local b2 = 0x80 + (math.floor(codepoint / 0x40) % 0x40)
        local b3 = 0x80 + (codepoint % 0x40)
        return string.char(b1, b2, b3)
    end
    local b1 = 0xF0 + math.floor(codepoint / 0x40000)
    local b2 = 0x80 + (math.floor(codepoint / 0x1000) % 0x40)
    local b3 = 0x80 + (math.floor(codepoint / 0x40) % 0x40)
    local b4 = 0x80 + (codepoint % 0x40)
    return string.char(b1, b2, b3, b4)
end

local function decode_error(message, position)
    return nil, message .. " at character " .. tostring(position)
end

local function decode_string(data, position)
    local pieces = {}
    local index = 1
    position = position + 1

    while true do
        local char = data:sub(position, position)
        if char == "" then
            return decode_error("unterminated string", position)
        end

        if char == "\"" then
            return table.concat(pieces), position + 1
        end

        if char == "\\" then
            local escape = data:sub(position + 1, position + 1)
            if escape == "\"" or escape == "\\" or escape == "/" then
                pieces[index] = escape
                index = index + 1
                position = position + 2
            elseif escape == "b" then
                pieces[index] = "\b"
                index = index + 1
                position = position + 2
            elseif escape == "f" then
                pieces[index] = "\f"
                index = index + 1
                position = position + 2
            elseif escape == "n" then
                pieces[index] = "\n"
                index = index + 1
                position = position + 2
            elseif escape == "r" then
                pieces[index] = "\r"
                index = index + 1
                position = position + 2
            elseif escape == "t" then
                pieces[index] = "\t"
                index = index + 1
                position = position + 2
            elseif escape == "u" then
                local hex = data:sub(position + 2, position + 5)
                if not hex:match("^[0-9a-fA-F][0-9a-fA-F][0-9a-fA-F][0-9a-fA-F]$") then
                    return decode_error("invalid unicode escape", position)
                end
                pieces[index] = utf8_from_codepoint(tonumber(hex, 16))
                index = index + 1
                position = position + 6
            else
                return decode_error("invalid escape character", position)
            end
        else
            pieces[index] = char
            index = index + 1
            position = position + 1
        end
    end
end

local function encode_string(value)
    local substitutions = {
        ["\\"] = "\\\\",
        ["\""] = "\\\"",
        ["\b"] = "\\b",
        ["\f"] = "\\f",
        ["\n"] = "\\n",
        ["\r"] = "\\r",
        ["\t"] = "\\t",
    }

    local escaped = value:gsub('[%z\1-\31\\"]', function(char)
        return substitutions[char] or string.format("\\u%04x", char:byte())
    end)
    return "\"" .. escaped .. "\""
end

local function skip_whitespace(data, position)
    while true do
        local char = data:sub(position, position)
        if char == " " or char == "\t" or char == "\n" or char == "\r" then
            position = position + 1
        else
            return position
        end
    end
end

local decode_value

local function decode_literal(data, position, literal, value)
    if data:sub(position, position + #literal - 1) ~= literal then
        return decode_error("invalid literal", position)
    end
    return value, position + #literal
end

local function decode_number(data, position)
    local number = data:match("^-?%d+%.?%d*[eE]?[+-]?%d*", position)
    if not number or number == "" then
        return decode_error("invalid number", position)
    end
    local value = tonumber(number)
    if value == nil then
        return decode_error("invalid number", position)
    end
    return value, position + #number
end

local function decode_array(data, position)
    local result = {}
    local index = 1
    position = position + 1
    position = skip_whitespace(data, position)

    if data:sub(position, position) == "]" then
        return result, position + 1
    end

    while true do
        local value, next_position = decode_value(data, position)
        if type(next_position) ~= "number" then
            return nil, next_position
        end
        result[index] = value
        index = index + 1
        position = skip_whitespace(data, next_position)

        local delimiter = data:sub(position, position)
        if delimiter == "," then
            position = skip_whitespace(data, position + 1)
        elseif delimiter == "]" then
            return result, position + 1
        else
            return decode_error("expected ',' or ']'", position)
        end
    end
end

local function decode_object(data, position)
    local result = {}
    position = position + 1
    position = skip_whitespace(data, position)

    if data:sub(position, position) == "}" then
        return result, position + 1
    end

    while true do
        if data:sub(position, position) ~= "\"" then
            return decode_error("expected string key", position)
        end

        local key, key_position = decode_string(data, position)
        if type(key_position) ~= "number" then
            return nil, key_position
        end

        position = skip_whitespace(data, key_position)
        if data:sub(position, position) ~= ":" then
            return decode_error("expected ':' after key", position)
        end

        position = skip_whitespace(data, position + 1)
        local value, value_position = decode_value(data, position)
        if type(value_position) ~= "number" then
            return nil, value_position
        end
        result[key] = value
        position = skip_whitespace(data, value_position)

        local delimiter = data:sub(position, position)
        if delimiter == "," then
            position = skip_whitespace(data, position + 1)
        elseif delimiter == "}" then
            return result, position + 1
        else
            return decode_error("expected ',' or '}'", position)
        end
    end
end

decode_value = function(data, position)
    position = skip_whitespace(data, position)
    local char = data:sub(position, position)

    if char == "\"" then
        return decode_string(data, position)
    elseif char == "{" then
        return decode_object(data, position)
    elseif char == "[" then
        return decode_array(data, position)
    elseif char == "t" then
        return decode_literal(data, position, "true", true)
    elseif char == "f" then
        return decode_literal(data, position, "false", false)
    elseif char == "n" then
        return decode_literal(data, position, "null", json.null)
    elseif char == "" then
        return decode_error("unexpected end of input", position)
    else
        return decode_number(data, position)
    end
end

function json.decode(data)
    if type(data) ~= "string" then
        return nil, "expected string"
    end

    local value, position = decode_value(data, 1)
    if type(position) ~= "number" then
        return nil, position
    end

    position = skip_whitespace(data, position)
    if position <= #data then
        return decode_error("trailing characters", position)
    end

    return value
end

local function is_array(value)
    if type(value) ~= "table" then
        return false, 0
    end

    local max_index = 0
    for key in pairs(value) do
        if type(key) ~= "number" or key <= 0 or key ~= math.floor(key) then
            return false, 0
        end
        if key > max_index then
            max_index = key
        end
    end

    for index = 1, max_index do
        if value[index] == nil then
            return false, 0
        end
    end

    return true, max_index
end

local function encode_value(value)
    local value_type = type(value)

    if value_type == "nil" then
        return "null"
    end
    if value == json.null then
        return "null"
    end
    if value_type == "string" then
        return encode_string(value)
    end
    if value_type == "number" then
        if value ~= value or value == math.huge or value == -math.huge then
            error("invalid number value")
        end
        return tostring(value)
    end
    if value_type == "boolean" then
        return value and "true" or "false"
    end
    if value_type == "table" then
        local array_like, max_index = is_array(value)
        local parts = {}
        if array_like then
            for index = 1, max_index do
                parts[#parts + 1] = encode_value(value[index])
            end
            return "[" .. table.concat(parts, ",") .. "]"
        end

        for key, item in pairs(value) do
            if type(key) ~= "string" then
                error("object keys must be strings")
            end
            parts[#parts + 1] = encode_string(key) .. ":" .. encode_value(item)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    end

    error("unsupported type for JSON encoding: " .. value_type)
end

function json.encode(value)
    return encode_value(value)
end

return json
