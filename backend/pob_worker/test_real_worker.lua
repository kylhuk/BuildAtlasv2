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

local main = launch.main
main:SetMode("BUILD", false, "worker_build", "")
runCallback("OnFrame")

local buildMode = main.modes["BUILD"]
print("Build mode loaded:", buildMode ~= nil)
print("Build name:", buildMode.buildName)
print("calcsTab:", buildMode.calcsTab ~= nil)

if buildMode.calcsTab and buildMode.calcsTab.mainEnv then
    print("Has mainEnv")
else
    print("No mainEnv")
end

