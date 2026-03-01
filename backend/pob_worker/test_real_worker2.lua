local json = require("cjson")
local io = require("io")

package.path = package.path .. ";./?.lua;../runtime/lua/?.lua;../runtime/lua/?/init.lua"

_G.GetVirtualScreenSize = function() return 1920, 1080 end
_G.GetScreenSize = function() return 1920, 1080 end
_G.GetScreenScale = function() return 1 end
_G.RenderInit = function() end
_G.ConExecute = function() end
_G.ConPrintf = function(...) print(string.format(...)) end
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
if not success then error(err) end

if launch.OnInit then
    launch:OnInit()
end

local xml_file = io.open("../../data/builds/0004110d12bf4e0cb94591353ddcc7df/build.xml.gz", "rb")
local zlib = require("zlib")
-- Actually zlib may not be available. We can just use the uncompressed python trick from before.
