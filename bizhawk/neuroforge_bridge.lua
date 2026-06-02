-- neuroforge_bridge.lua
-- ---------------------------------------------------------------------------
-- NeuroForge <-> BizHawk vision-only bridge (Lua side-car), FILE TRANSPORT.
--
-- Newer EmuHawk uses the NLua engine, which does NOT bundle LuaSocket
-- (`require("socket")` fails). So this bridge carries the NeuroForge protocol
-- over atomically-renamed files using only Lua's standard io/os — always
-- available in NLua. (A socket transport exists on the Python side for builds
-- that do have LuaSocket; this file is the portable default.)
--
-- Load it in EmuHawk's Lua Console (Tools -> Lua Console -> Open Script) with a
-- ROM loaded, OR let the Python smoke test launch EmuHawk with --lua=<this>.
--
-- Wire framing (same as src/neuroforge/game/clients/protocol.py), one framed
-- message per file:
--   [ magic "NFB1" (4B) ][ type (1B) ][ payload length (4B, uint32 LE) ][ payload ]
--
--   Lua -> Python (dir l2p/): HELLO (JSON), FRAME (header + PNG bytes), BYE
--   Python -> Lua (dir p2l/): WELCOME (JSON), ACTION (button bitmask), RESET, CLOSE
--
-- VISION-ONLY INVARIANT: reads ONLY the screen (screenshot), writes ONLY
-- controller input. It MUST NOT call memory.* / mainmemory.* — enforced by
-- the test_lua_no_memory_reads unit test.
--
-- VERIFY-ON-YOUR-BIZHAWK (see docs/BIZHAWK_SETUP.md): client.screenshot(path)
-- writes a PNG; client.bufferwidth()/bufferheight() give native size;
-- joypad.set(buttons, 1) + emu.frameadvance() hold buttons and step a frame.
-- ---------------------------------------------------------------------------

local PROTOCOL_VERSION = 1
local CHANNELS = 3                 -- PNG screenshots decode to RGB on the Python side
local PIXEL_FORMAT = "png"
local FRAMESKIP = tonumber(os.getenv("NEUROFORGE_BRIDGE_FRAMESKIP")) or 1
local SPEED_PERCENT = tonumber(os.getenv("NEUROFORGE_BRIDGE_SPEED_PERCENT")) or 0
local NES_FPS = 60.0
local WAIT_TIMEOUT_S = 120

local TMP_DIR = os.getenv("TEMP") or os.getenv("TMP") or "."
local COMM_DIR = os.getenv("NEUROFORGE_BRIDGE_DIR") or (TMP_DIR .. "/neuroforge_bridge")
local L2P = COMM_DIR .. "/l2p"     -- emulator -> Python (we WRITE)
local P2L = COMM_DIR .. "/p2l"     -- Python -> emulator (we READ)
local SHOT_PATH = TMP_DIR .. "/neuroforge_frame.png"

local MSG = {
  HELLO = 1, FRAME = 2, BYE = 3,
  WELCOME = 16, ACTION = 17, RESET = 18, CLOSE = 19,
}
local MAGIC = "NFB1"
local BUTTONS = { "Up", "Down", "Left", "Right", "A", "B", "Start", "Select" }

local send_seq = 0
local recv_seq = 0

-- ── Little-endian integer (de)serialisation ─────────────────────────────────
local function le_bytes(value, nbytes)
  local out = {}
  value = math.floor(value)
  for i = 1, nbytes do
    out[i] = string.char(value % 256)
    value = math.floor(value / 256)
  end
  return table.concat(out)
end

local function le_read(str, offset, nbytes)
  local value, mult = 0, 1
  for i = 0, nbytes - 1 do
    value = value + string.byte(str, offset + i) * mult
    mult = mult * 256
  end
  return value
end

-- ── File helpers ────────────────────────────────────────────────────────────
local function read_file(path)
  local f = io.open(path, "rb")
  if not f then return nil end
  local data = f:read("*a")
  f:close()
  return data
end

local function write_file_atomic(dir, seq, data)
  local final = string.format("%s/%08d.msg", dir, seq)
  local tmp = string.format("%s/%08d.tmp", dir, seq)
  local f = assert(io.open(tmp, "wb"))
  f:write(data)
  f:close()
  os.remove(final)        -- no-op if absent; guards a rare leftover
  assert(os.rename(tmp, final))
end

-- ── Message framing over files ───────────────────────────────────────────────
local function configure_speed()
  local percent = math.floor(SPEED_PERCENT)
  if percent < 1 then return end
  pcall(function() emu.limitframerate(true) end)
  local ok, err = pcall(function() client.speedmode(percent) end)
  if ok then
    print("[neuroforge_bridge] speedmode=" .. tostring(percent) .. "%")
  else
    print("[neuroforge_bridge] client.speedmode failed: " .. tostring(err))
  end
end

local function send_message(msg_type, payload)
  payload = payload or ""
  local framed = MAGIC .. string.char(msg_type) .. le_bytes(#payload, 4) .. payload
  write_file_atomic(L2P, send_seq, framed)
  send_seq = send_seq + 1
end

local function recv_message()
  local path = string.format("%s/%08d.msg", P2L, recv_seq)
  local start = os.time()
  local data
  while true do
    data = read_file(path)
    if data and #data >= 9 then break end
    if os.time() - start > WAIT_TIMEOUT_S then
      error("timed out waiting for Python message " .. recv_seq)
    end
  end
  os.remove(path)
  recv_seq = recv_seq + 1
  if string.sub(data, 1, 4) ~= MAGIC then error("bad magic from Python") end
  local msg_type = string.byte(data, 5)
  local length = le_read(data, 6, 4)
  local payload = string.sub(data, 10, 9 + length)
  return msg_type, payload
end

-- ── Frame capture (PNG) ──────────────────────────────────────────────────────
local function send_frame()
  client.screenshot(SHOT_PATH)                       -- VERIFY: synchronous PNG write
  local png = read_file(SHOT_PATH)
  if not png then error("failed to read screenshot at " .. SHOT_PATH) end
  local frame_id = emu.framecount()
  local emu_time_us = math.floor((frame_id / NES_FPS) * 1000000)
  local payload = le_bytes(frame_id, 8) .. le_bytes(emu_time_us, 8) .. png
  send_message(MSG.FRAME, payload)
end

-- ── Action handling ───────────────────────────────────────────────────────────
local function decode_buttons(mask)
  local pressed = {
    Up = false, Down = false, Left = false, Right = false,
    A = false, B = false, Start = false, Select = false,
  }
  for i = 0, 7 do
    if math.floor(mask / (2 ^ i)) % 2 == 1 then
      pressed[BUTTONS[i + 1]] = true
    end
  end
  return pressed
end

local function apply_action_and_advance(buttons, passive)
  for _ = 1, FRAMESKIP do
    if not passive then
      joypad.set(buttons, 1)      -- re-apply each frame so held buttons persist
    end                            -- passive: leave the player's controller in control
    emu.frameadvance()
  end
end

-- ── Main ──────────────────────────────────────────────────────────────────────
local function main()
  configure_speed()

  -- Handshake: announce geometry/format, await WELCOME.
  local width = client.bufferwidth()
  local height = client.bufferheight()
  local hello = string.format(
    '{"version":%d,"width":%d,"height":%d,"channels":%d,"format":"%s"}',
    PROTOCOL_VERSION, width, height, CHANNELS, PIXEL_FORMAT)
  send_message(MSG.HELLO, hello)
  local wtype = recv_message()
  if wtype ~= MSG.WELCOME then error("expected WELCOME, got " .. tostring(wtype)) end

  -- Initial frame (consumed by the Python client's reset()).
  send_frame()

  while true do
    local msg_type, payload = recv_message()
    if msg_type == MSG.ACTION then
      local mask = string.byte(payload, 1) or 0
      local flags = string.byte(payload, 2) or 0
      local passive = (flags % 2) == 1        -- ACTION_FLAG_PASSIVE: human drives
      apply_action_and_advance(decode_buttons(mask), passive)
      send_frame()
    elseif msg_type == MSG.RESET then
      -- Empty payload => reboot the core. A non-empty payload is a raw savestate
      -- FILE PATH: load it as an ENVIRONMENT RESET (like resetting a Gym env to a
      -- start state) so training can begin inside a level. This is NOT memory
      -- reading -- the policy still only ever sees pixels (vision-only invariant).
      local loaded = false
      if payload and #payload > 0 then
        loaded = pcall(function() savestate.load(payload) end)
        if not loaded then
          print("[neuroforge_bridge] savestate.load failed: " .. payload .. " (rebooting core)")
        end
      end
      if not loaded then
        client.reboot_core()
      end
      emu.frameadvance()
      send_frame()
    elseif msg_type == MSG.CLOSE then
      break
    else
      error("unexpected message type: " .. tostring(msg_type))
    end
  end

  send_message(MSG.BYE)
end

local ok, err = pcall(main)
if not ok then
  print("[neuroforge_bridge] error: " .. tostring(err))
end
