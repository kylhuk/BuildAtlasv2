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

-- Hook ShowErrMsg to see why loading fails
function launch:ShowErrMsg(fmt, ...)
    print("LAUNCH ERROR:", string.format(fmt, ...))
end

local xml_text = [[<PathOfBuilding>
  <Build level="100" className="Templar" ascendClassName="Inquisitor" mainSocketGroup="1"/>
  <PlayerStat stat="FullDPS" value="773280"/>
  <PlayerStat stat="MaximumHitTaken" value="17388"/>
  <PlayerStat stat="Life" value="64440"/>
  <PlayerStat stat="Mana" value="19332"/>
  <PlayerStat stat="Armour" value="38664"/>
  <PlayerStat stat="Evasion" value="32220"/>
  <PlayerStat stat="EnergyShield" value="0"/>
  <PlayerStat stat="FireResist" value="90"/>
  <PlayerStat stat="ColdResist" value="90"/>
  <PlayerStat stat="LightningResist" value="90"/>
  <PlayerStat stat="ChaosResist" value="75"/>
  <PlayerStat stat="Str" value="200"/>
  <PlayerStat stat="Dex" value="170"/>
  <PlayerStat stat="Int" value="150"/>
  <PlayerStat stat="LifeUnreservedPercent" value="15"/>
  <PlayerStat stat="ManaUnreservedPercent" value="95"/>
  <PlayerStat stat="EffectiveMovementSpeedMod" value="1"/>
  <PlayerStat stat="BlockChance" value="0"/>
  <PlayerStat stat="SpellBlockChance" value="0"/>
</PathOfBuilding>]]

local main = launch.main

print("Loading build XML...")
main:SetMode("BUILD", false, "worker_build", xml_text)

-- Let it process
for i=1, 5 do
    runCallback("OnFrame")
end

local buildMode = main.modes["BUILD"]
print("Build mode loaded:", buildMode ~= nil)
print("calcsTab exists:", buildMode.calcsTab ~= nil)

if buildMode.calcsTab then
    print("mainOutput exists:", buildMode.calcsTab.mainOutput ~= nil)
    if buildMode.calcsTab.mainOutput then
        print("DPS:", buildMode.calcsTab.mainOutput.TotalDPS)
    end
end
