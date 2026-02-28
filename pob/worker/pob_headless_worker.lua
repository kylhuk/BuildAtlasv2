#!/usr/bin/env lua
-- Headless PoB Worker using the official HeadlessWrapper
-- This properly invokes PoB's calculation engine

-- Set up paths
local pob_src = os.getenv("POB_SOURCE_DIR") or "../../PathOfBuilding/src"
package.path = pob_src .. "/?.lua;" .. pob_src .. "/Modules/?.lua;" .. package.path

-- Override loadfile to search in PoB directories
local orig_loadfile = loadfile
_G.loadfile = function(filename)
    -- Try original first
    local f = orig_loadfile(filename)
    if f then return f end
    -- Try with pob_src prefix
    f = orig_loadfile(pob_src .. "/" .. filename)
    if f then return f end
    -- Try Modules/ prefix
    f = orig_loadfile(pob_src .. "/Modules/" .. filename)
    if f then return f end
    return nil, "file not found: " .. filename
end

-- Create stub modules that PoB expects
package.preload["UpdateCheck"] = function() return {} end

package.preload["jit"] = function()
    return {
        on = function() end,
        off = function() end,
        opt = { start = function() end },
        compile = function() end,
        start = function() end,
    }
end

-- bit module for Lua 5.4 compatibility
package.preload["bit"] = function()
    return {
        band = function(a, b) return a & b end,
        bor = function(a, b) return a | b end,
        bxor = function(a, b) return a ~ b end,
        bnot = function(a) return ~a end,
        lshift = function(a, n) return a << n end,
        rshift = function(a, n) return a >> n end,
        arshift = function(a, n) return math.floor(a / 2^n) end,
    }
end
_G.bit = package.preload["bit"]()

-- base64 module stub
package.preload["base64"] = function()
    local b64chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
    return {
        encode = function(data)
            return (data:gsub('.', function(x)
                local r,b='',x:byte()
                for i=8,1,-1 do r=r..(b%2^i-b%2^(i-1)>0 and '1' or '0') end
                return r;
            end):gsub('%d%d%d%d%d%d%d%d', function(x)
                if #x < 8 then return '' end
                local c=0
                for i=1,8 do c=c+(x:sub(i,i)=='1' and 2^(8-i) or 0) end
                return b64chars:sub(c+1,c+1)
            end)..({'==','='})[#data%3+1])
        end,
        decode = function(data)
            data = data:gsub('[^'..b64chars..'=]','')
            return (data:gsub('.', function(x)
                if x == '=' then return '' end
                local r,f='',(b64chars:find(x)-1)
                for i=6,1,-1 do r=r..(f%2^i-f%2^(i-1)>0 and '1' or '0') end
                return r;
            end):gsub('%d%d%d%d%d%d%d%d', function(x)
                if #x < 8 then return '' end
                local c=0
                for i=1,8 do c=c+(x:sub(i,i)=='1' and 2^(8-i) or 0) end
                return string.char(c)
            end))
        end,
    }
end

package.preload["xml"] = function()
    return { LoadXMLFile = function() return nil end }
end

-- Lua 5.4 compatibility: unpack is in table
if not unpack then unpack = table.unpack end

-- Set devMode to skip update checks
_G.devMode = true

-- Create launch object before loading HeadlessWrapper
launch = { devMode = true, installedMode = false, noSSL = true, version = "1.0.0" }
_G.launch = launch

-- Load the headless wrapper which sets up the PoB environment
dofile(pob_src .. "/HeadlessWrapper.lua")

-- The build module is now available in 'build' global
-- Continue with the worker loop...
