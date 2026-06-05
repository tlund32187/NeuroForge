-- neuroforge_bridge.lua
-- ---------------------------------------------------------------------------
-- NeuroForge <-> BizHawk vision-only bridge (Lua side-car).
--
-- The protocol matches src/neuroforge/game/clients/protocol.py:
--   [ magic "NFB1" (4B) ][ type (1B) ][ payload length (4B LE) ][ payload ]
--
-- Transports:
--   * socket: preferred for live training. The bridge tries LuaSocket first
--     (including C:/BizHawk/Lua/socket/core.dll), then LuaCOM/MSWinsock.
--     Frame PNGs go through BizHawk's comm.socketServerScreenShot socket.
--   * file: portable fallback using atomically renamed message files and
--     client.screenshot(path).
--
-- The bridge reads screen pixels and writes controller input only.
-- ---------------------------------------------------------------------------

local PROTOCOL_VERSION = 1
local CHANNELS = 3
local PIXEL_FORMAT = "png"
local FRAMESKIP = tonumber(os.getenv("NEUROFORGE_BRIDGE_FRAMESKIP")) or 1
local SPEED_PERCENT = tonumber(os.getenv("NEUROFORGE_BRIDGE_SPEED_PERCENT")) or 0
local NES_FPS = 60.0
local WAIT_TIMEOUT_S = 120

local TMP_DIR = os.getenv("TEMP") or os.getenv("TMP") or "."
local COMM_DIR = os.getenv("NEUROFORGE_BRIDGE_DIR") or (TMP_DIR .. "/neuroforge_bridge")
local L2P = COMM_DIR .. "/l2p"
local P2L = COMM_DIR .. "/p2l"
local SHOT_PATH = TMP_DIR .. "/neuroforge_frame.png"

local BRIDGE_HOST = os.getenv("NEUROFORGE_BRIDGE_HOST") or "127.0.0.1"
local BRIDGE_PORT = tonumber(os.getenv("NEUROFORGE_BRIDGE_PORT")) or 0
local TRANSPORT_ENV = string.lower(os.getenv("NEUROFORGE_BRIDGE_TRANSPORT") or "")
local SOCKET_BACKEND_ENV = string.lower(os.getenv("NEUROFORGE_BRIDGE_SOCKET_BACKEND") or "auto")
local FRAME_CAPTURE_ENV = string.lower(os.getenv("NEUROFORGE_BRIDGE_FRAME_CAPTURE") or "")
local SCREENSHOT_HOST = os.getenv("NEUROFORGE_BRIDGE_SCREENSHOT_HOST") or BRIDGE_HOST
local SCREENSHOT_PORT = tonumber(os.getenv("NEUROFORGE_BRIDGE_SCREENSHOT_PORT")) or 0
local SCREENSHOT_TIMEOUT_MS = tonumber(os.getenv("NEUROFORGE_BRIDGE_SCREENSHOT_TIMEOUT_MS")) or 45000
local ERROR_PATH = os.getenv("NEUROFORGE_BRIDGE_ERROR_PATH") or ""
local TRANSPORT = TRANSPORT_ENV
if TRANSPORT == "" then
  TRANSPORT = BRIDGE_PORT > 0 and "socket" or "file"
end
local FRAME_CAPTURE = FRAME_CAPTURE_ENV

local MSG = {
  HELLO = 1, FRAME = 2, BYE = 3,
  WELCOME = 16, ACTION = 17, RESET = 18, CLOSE = 19,
}
local MAGIC = "NFB1"
local BUTTONS = { "Up", "Down", "Left", "Right", "A", "B", "Start", "Select" }

local send_seq = 0
local recv_seq = 0
local socket_conn = nil
local socket_backend = ""
local socket_recv_buf = ""
local screenshot_socket_ready = false

local function is_socket_transport()
  return TRANSPORT == "socket" or TRANSPORT == "luasocket" or TRANSPORT == "luacom"
end

if FRAME_CAPTURE_ENV == "" then
  FRAME_CAPTURE = (is_socket_transport() and SCREENSHOT_PORT > 0) and "socket" or "file"
end

-- Little-endian integer helpers ------------------------------------------------
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

-- File helpers -----------------------------------------------------------------
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
  os.remove(final)
  assert(os.rename(tmp, final))
end

local function write_error_log(message)
  if ERROR_PATH == "" then return end
  local f = io.open(ERROR_PATH, "ab")
  if not f then return end
  f:write(tostring(message) .. "\n")
  f:close()
end

-- Speed control ----------------------------------------------------------------
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

-- LuaSocket backend ------------------------------------------------------------
local function normalize_path(path)
  return string.gsub(path or "", "\\", "/")
end

local function append_package_path(field, pattern)
  if not package or not package[field] then return end
  if string.find(package[field], pattern, 1, true) then return end
  package[field] = package[field] .. ";" .. pattern
end

local function add_luasocket_search_paths()
  local roots = {
    os.getenv("NEUROFORGE_LUA_ROOT"),
    os.getenv("BIZHAWK_LUA_ROOT"),
    "C:/BizHawk/Lua",
  }
  for _, root in ipairs(roots) do
    if root and root ~= "" then
      local p = normalize_path(root)
      append_package_path("path", p .. "/?.lua")
      append_package_path("path", p .. "/?/init.lua")
      append_package_path("cpath", p .. "/?.dll")
      append_package_path("cpath", p .. "/?/?.dll")
      append_package_path("cpath", p .. "/?/core.dll")
    end
  end
end

local function require_luasocket()
  add_luasocket_search_paths()
  local ok, mod = pcall(require, "socket")
  if ok and type(mod) == "table" and mod.tcp then
    return mod
  end
  local first_err = tostring(mod)
  ok, mod = pcall(require, "socket.core")
  if ok and type(mod) == "table" and mod.tcp then
    return mod
  end
  error("LuaSocket unavailable: " .. first_err .. " / " .. tostring(mod))
end

local function connect_luasocket()
  local socket = require_luasocket()
  local tcp, err = socket.tcp()
  if not tcp then
    error("LuaSocket tcp() failed: " .. tostring(err))
  end
  if tcp.settimeout then tcp:settimeout(WAIT_TIMEOUT_S) end
  local ok, connect_err = tcp:connect(BRIDGE_HOST, BRIDGE_PORT)
  if not ok then
    error("LuaSocket connect failed: " .. tostring(connect_err))
  end
  if tcp.setoption then
    pcall(function() tcp:setoption("tcp-nodelay", true) end)
  end
  socket_conn = tcp
  socket_backend = "luasocket"
  print("[neuroforge_bridge] LuaSocket connected to " .. BRIDGE_HOST .. ":" .. tostring(BRIDGE_PORT))
end

local function luasocket_send_all(data)
  local pos = 1
  while pos <= #data do
    local sent, err, last = socket_conn:send(data, pos)
    if sent then
      pos = sent + 1
    elseif err == "timeout" and last and last >= pos then
      pos = last + 1
    else
      error("LuaSocket send failed: " .. tostring(err))
    end
  end
end

local function luasocket_recv_exact(n)
  if n <= 0 then return "" end
  local parts = {}
  local have = 0
  while have < n do
    local chunk, err, partial = socket_conn:receive(n - have)
    if chunk then
      parts[#parts + 1] = chunk
      have = have + #chunk
    elseif partial and #partial > 0 then
      parts[#parts + 1] = partial
      have = have + #partial
    else
      error("LuaSocket receive failed: " .. tostring(err))
    end
  end
  return table.concat(parts)
end

-- LuaCOM/MSWinsock backend -----------------------------------------------------
local function require_luacom()
  local ok, mod = pcall(require, "luacom")
  if ok and type(mod) == "table" and mod.CreateObject then
    return mod
  end
  if type(luacom) == "table" and luacom.CreateObject then
    return luacom
  end
  if type(luacom_CreateObject) == "function" then
    return { CreateObject = luacom_CreateObject }
  end
  error("LuaCOM unavailable: " .. tostring(mod))
end

local function connect_luacom()
  local lc = require_luacom()
  local ws = lc.CreateObject("MSWinsock.Winsock")
  if not ws then
    error("failed to create MSWinsock.Winsock")
  end
  ws:Connect(BRIDGE_HOST, BRIDGE_PORT)
  local start = os.time()
  while tonumber(ws.State) ~= 7 do
    if tonumber(ws.State) == 9 then
      error("MSWinsock connection failed")
    end
    if os.time() - start > WAIT_TIMEOUT_S then
      error("timed out connecting MSWinsock to " .. BRIDGE_HOST .. ":" .. tostring(BRIDGE_PORT))
    end
    emu.frameadvance()
  end
  socket_conn = ws
  socket_backend = "luacom"
  print("[neuroforge_bridge] LuaCOM connected to " .. BRIDGE_HOST .. ":" .. tostring(BRIDGE_PORT))
end

local function first_string(...)
  for i = 1, select("#", ...) do
    local value = select(i, ...)
    if type(value) == "string" then
      return value
    end
  end
  return nil
end

local function luacom_get_data(max_len)
  local attempts = {
    function() return socket_conn:GetData("", 8, max_len) end,
    function() return socket_conn:GetData("", max_len) end,
    function() return socket_conn:GetData("") end,
  }
  local last_err = nil
  for _, attempt in ipairs(attempts) do
    local ok, a, b, c, d = pcall(attempt)
    if ok then
      local data = first_string(a, b, c, d)
      if data then return data end
    else
      last_err = tostring(a)
    end
  end
  error("MSWinsock.GetData returned no string data: " .. tostring(last_err))
end

local function luacom_send_all(data)
  local ok, err = pcall(function() socket_conn:SendData(data) end)
  if not ok then
    error("MSWinsock.SendData failed: " .. tostring(err))
  end
end

local function luacom_recv_exact(n)
  if n <= 0 then return "" end
  local start = os.time()
  while #socket_recv_buf < n do
    local available = tonumber(socket_conn.BytesReceived) or 0
    if available > 0 then
      socket_recv_buf = socket_recv_buf .. luacom_get_data(available)
    else
      if os.time() - start > WAIT_TIMEOUT_S then
        error("timed out waiting for MSWinsock bytes")
      end
      emu.frameadvance()
    end
  end
  local out = string.sub(socket_recv_buf, 1, n)
  socket_recv_buf = string.sub(socket_recv_buf, n + 1)
  return out
end

local function connect_socket()
  if socket_conn then return end
  if BRIDGE_PORT <= 0 then
    error("socket transport requires NEUROFORGE_BRIDGE_PORT")
  end

  local wanted = SOCKET_BACKEND_ENV
  if TRANSPORT == "luasocket" or TRANSPORT == "luacom" then
    wanted = TRANSPORT
  end
  local first_err = nil
  if wanted == "auto" or wanted == "" or wanted == "luasocket" then
    local ok, err = pcall(connect_luasocket)
    if ok then return end
    first_err = tostring(err)
    if wanted == "luasocket" then error(first_err) end
    print("[neuroforge_bridge] LuaSocket unavailable, trying LuaCOM: " .. first_err)
  end
  if wanted == "auto" or wanted == "" or wanted == "luacom" then
    local ok, err = pcall(connect_luacom)
    if ok then return end
    if wanted == "luacom" then error(tostring(err)) end
    error("socket transport unavailable; LuaSocket: " .. tostring(first_err) .. "; LuaCOM: " .. tostring(err))
  end
  error("unknown socket backend: " .. tostring(wanted))
end

local function close_socket()
  local conn = socket_conn
  if not conn then return end
  socket_conn = nil
  if socket_backend == "luasocket" then
    pcall(function() conn:close() end)
  elseif socket_backend == "luacom" then
    pcall(function() conn:Close() end)
  end
end

-- Screenshot socket ------------------------------------------------------------
local function configure_screenshot_socket()
  if screenshot_socket_ready then return end
  if SCREENSHOT_PORT <= 0 then
    error("socket frame capture requires NEUROFORGE_BRIDGE_SCREENSHOT_PORT")
  end
  if comm == nil or comm.socketServerScreenShot == nil then
    error("BizHawk comm.socketServerScreenShot is unavailable")
  end
  if comm.socketServerSetIp ~= nil then
    pcall(function() comm.socketServerSetIp(SCREENSHOT_HOST) end)
  end
  if comm.socketServerSetPort ~= nil then
    pcall(function() comm.socketServerSetPort(SCREENSHOT_PORT) end)
  end
  if comm.socketServerSetTimeout ~= nil then
    pcall(function() comm.socketServerSetTimeout(SCREENSHOT_TIMEOUT_MS) end)
  end
  screenshot_socket_ready = true
end

local function send_screenshot_over_socket()
  configure_screenshot_socket()
  local result = comm.socketServerScreenShot()
  if comm.socketServerSuccessful ~= nil and not comm.socketServerSuccessful() then
    error("BizHawk screenshot socket send failed: " .. tostring(result))
  end
end

-- Message framing --------------------------------------------------------------
local function file_send_message(msg_type, payload)
  payload = payload or ""
  local framed = MAGIC .. string.char(msg_type) .. le_bytes(#payload, 4) .. payload
  write_file_atomic(L2P, send_seq, framed)
  send_seq = send_seq + 1
end

local function file_recv_message()
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

local function socket_send_message(msg_type, payload)
  payload = payload or ""
  local framed = MAGIC .. string.char(msg_type) .. le_bytes(#payload, 4) .. payload
  if socket_backend == "luasocket" then
    luasocket_send_all(framed)
  elseif socket_backend == "luacom" then
    luacom_send_all(framed)
  else
    error("socket transport is not connected")
  end
end

local function socket_recv_exact(n)
  if socket_backend == "luasocket" then
    return luasocket_recv_exact(n)
  elseif socket_backend == "luacom" then
    return luacom_recv_exact(n)
  end
  error("socket transport is not connected")
end

local function socket_recv_message()
  local header = socket_recv_exact(9)
  if string.sub(header, 1, 4) ~= MAGIC then error("bad magic from Python") end
  local msg_type = string.byte(header, 5)
  local length = le_read(header, 6, 4)
  local payload = socket_recv_exact(length)
  return msg_type, payload
end

local function send_message(msg_type, payload)
  if is_socket_transport() then
    socket_send_message(msg_type, payload)
  else
    file_send_message(msg_type, payload)
  end
end

local function recv_message()
  if is_socket_transport() then
    return socket_recv_message()
  end
  return file_recv_message()
end

-- Frame capture ----------------------------------------------------------------
local function send_frame()
  local frame_id = emu.framecount()
  local emu_time_us = math.floor((frame_id / NES_FPS) * 1000000)
  local prefix = le_bytes(frame_id, 8) .. le_bytes(emu_time_us, 8)
  if FRAME_CAPTURE == "socket" then
    configure_screenshot_socket()
    send_message(MSG.FRAME, prefix)
    send_screenshot_over_socket()
    return
  end
  client.screenshot(SHOT_PATH)
  local png = read_file(SHOT_PATH)
  if not png then error("failed to read screenshot at " .. SHOT_PATH) end
  local payload = prefix .. png
  send_message(MSG.FRAME, payload)
end

-- Action handling --------------------------------------------------------------
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
      joypad.set(buttons, 1)
    end
    emu.frameadvance()
  end
end

-- Main -------------------------------------------------------------------------
local function main()
  if is_socket_transport() then
    connect_socket()
  else
    print("[neuroforge_bridge] file transport: " .. COMM_DIR)
  end
  configure_speed()

  local width = client.bufferwidth()
  local height = client.bufferheight()
  local hello = string.format(
    '{"version":%d,"width":%d,"height":%d,"channels":%d,"format":"%s"}',
    PROTOCOL_VERSION, width, height, CHANNELS, PIXEL_FORMAT)
  send_message(MSG.HELLO, hello)
  local wtype = recv_message()
  if wtype ~= MSG.WELCOME then error("expected WELCOME, got " .. tostring(wtype)) end

  send_frame()

  while true do
    local msg_type, payload = recv_message()
    if msg_type == MSG.ACTION then
      local mask = string.byte(payload, 1) or 0
      local flags = string.byte(payload, 2) or 0
      local passive = (flags % 2) == 1
      apply_action_and_advance(decode_buttons(mask), passive)
      send_frame()
    elseif msg_type == MSG.RESET then
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
  local msg = "[neuroforge_bridge] error: " .. tostring(err)
  print(msg)
  write_error_log(msg)
end
close_socket()
