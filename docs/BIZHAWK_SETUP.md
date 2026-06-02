# BizHawk Bridge Setup (Phase 0)

The BizHawk bridge lets a NeuroForge brain see a game **by vision only** —
screen pixels in, controller buttons out, **no game-memory reads**.

> First phase of the roadmap toward playing SMB3 by vision. Everything here is
> exercised in CI with fakes; the only step needing a real emulator is the
> smoke run below — and there's a one-click way to do it.

## Quick start (one click)

1. **Run and Debug** panel → pick **"NeuroForge: BizHawk smoke test"** → press ▶.
2. EmuHawk opens, loads SMB3, and the demo brain taps **Start** past the title
   then **runs right**.
3. The terminal counts frames and saves screenshots (what the brain sees) to
   `artifacts/smoke/`.

Paths are auto-filled in [scripts/smoke_bizhawk.py](../scripts/smoke_bizhawk.py)
(`C:\BizHawk\EmuHawk.exe` + the SMB3 ROM). Edit the CONFIG block there if yours differ.

## Prerequisites

- **BizHawk / EmuHawk 2.9+** (Windows).
- A **legally-obtained ROM**. ROMs are not included.
- The NeuroForge venv with the `vision` extra (for PNG decoding):
  `pip install -e ".[torch,vision]"`.

No LuaSocket needed — see *Transport* below.

## Transport: file-based by default

Newer EmuHawk uses the **NLua** engine, which does **not** bundle LuaSocket
(`require("socket")` fails). So the bridge's default transport moves messages
over **atomically-renamed files** in a scratch folder
(`%TEMP%\neuroforge_bridge\`), using only Lua's standard `io`/`os`. It is the
portable, dependency-free path — slower than a socket, which is fine for
bring-up (Phase 6 optimises transport).

A `socket` transport also exists (`BizHawkClientConfig(transport="socket")`) for
EmuHawk builds that *do* have LuaSocket, but **file is the default and what the
smoke test uses**.

## How it works (lock-step)

```
Python (BizHawkClient)                         EmuHawk + neuroforge_bridge.lua
  make comm dirs, launch EmuHawk ───────────▶  --lua runs the bridge
  read HELLO  ◀───────── l2p/0 (geometry) ──── write HELLO {w,h,channels,png}
  write WELCOME ──────── p2l/0 (ack) ────────▶ read WELCOME
  reset() read FRAME ◀── l2p/1 (PNG) ───────── write FRAME (initial screenshot)
  step(action) write ACTION ─ p2l/N ─────────▶ read ACTION → joypad.set → frameadvance
  read FRAME  ◀───────── l2p/N+1 (PNG) ──────── write FRAME (post-action screenshot)
  close() write CLOSE ──────────────────────▶  read CLOSE → BYE → done
```

Exactly one action per emulated step. Frames are delivered at **native
resolution** (the HUD must stay readable for Phase 1); downsampling for the
network happens later in the policy's preprocessor.

## Manual run (without the smoke script)

1. Start Python first so it creates the comm dirs and waits:

   ```python
   from neuroforge.game import BizHawkClient, BizHawkClientConfig, VisionOnlyGameLoop
   from neuroforge.contracts.game import ControllerAction

   client = BizHawkClient(BizHawkClientConfig())  # NES native 256x240x3, file transport
   class HoldRight:
       def act(self, obs): return ControllerAction(right=True)
   loop = VisionOnlyGameLoop(client=client, policy=HoldRight())
   for _ in loop.run(max_steps=600):
       pass
   loop.close()
   ```

2. In EmuHawk: load the ROM, then **Tools → Lua Console → Open Script** and run
   [bizhawk/neuroforge_bridge.lua](../bizhawk/neuroforge_bridge.lua). It reads the
   comm dir from `NEUROFORGE_BRIDGE_DIR` (set automatically when Python launches
   EmuHawk) or falls back to `%TEMP%\neuroforge_bridge`.

## Verify-on-your-BizHawk

These Lua APIs vary by EmuHawk version — the spots are commented in
[neuroforge_bridge.lua](../bizhawk/neuroforge_bridge.lua):

- `client.screenshot(path)` writes the current frame as PNG synchronously.
- `client.bufferwidth()/bufferheight()` return the native frame size (NES = 256×240).
- `joypad.set(buttons, 1)` + `emu.frameadvance()` hold buttons and step a frame.
- `RESET` with an empty payload calls `client.reboot_core()`; with a savestate
  path it calls `savestate.load(path)` (Phase 4 — see *Training from a savestate*).

## Training from a savestate (Phase 4)

Booting from the ROM drops the brain at the **title/map screens, where there is
no reward to learn from** — so it can only flail. Training instead begins from a
**savestate placed at the start of a level**, where the vision reward (forward
scroll progress, score, staying alive) is dense and learnable. Each episode
reloads the savestate, so the brain gets many fresh attempts and a death/stall
ends the attempt and resets.

Loading a savestate is an **environment reset** (like resetting a Gym env to a
start state) — it is **not** memory reading, and the policy still only ever sees
pixels. The vision-only invariant holds (`savestate.load` is not a `memory.*`
API; the no-RAM test still passes).

**One-time setup:**

1. Open EmuHawk, load the SMB3 ROM, and play until Mario is standing at the very
   **start of a level** (e.g. World 1-1, just after it loads).
2. **File → Save State → Save State As…** and save to the path in
   `SAVESTATE_PATHS` in [scripts/train_smb3.py](../scripts/train_smb3.py)
   (default `C:\BizHawk\States\smb3_level1.State`).
3. Run **"NeuroForge: Train SMB3"** (▶). It reloads that savestate each episode
   and learns to make in-level progress. Add more paths to `SAVESTATE_PATHS` to
   train through a **curriculum** of increasingly hard levels — the brain
   advances a stage once it consistently clears most of the current one.

If the savestate file is missing the script still runs (booting the ROM) but
prints a warning; it won't learn much until you create one.

## Troubleshooting

- **`module 'socket' not found`** in the Lua console — expected on NLua builds;
  it only matters for the `socket` transport. The default `file` transport
  doesn't use sockets, so make sure you're on the current Lua script.
- **`timed out waiting for emulator message`** — the Lua script isn't running or
  can't see the comm dir. Confirm EmuHawk loaded the script (no
  `[neuroforge_bridge] error:` line in the Lua console), and that
  `%TEMP%\neuroforge_bridge\l2p` is filling with `.msg` files while it runs.
- **`MISSING` path in the smoke terminal** — fix the CONFIG block in
  [scripts/smoke_bizhawk.py](../scripts/smoke_bizhawk.py).
- **Nothing moves** — confirm the ROM is actually running (not paused). The demo
  policy taps Start for the first ~2s before running right.

## Performance note (Phase 6)

The file transport + full-resolution PNG per frame is for correctness, not
60 fps. Transport optimization (raw downsample, shared memory) is a later phase.
For live training, [scripts/train_smb3.py](../scripts/train_smb3.py) launches
BizHawk with `NEUROFORGE_BRIDGE_SPEED_PERCENT=400`, matching the menu's
**Speed 400%** setting (4x). With the default `FRAMESKIP=4`, that targets about
60 decisions/sec when PNG capture and file transport can keep up.

Use these knobs separately:

- `NEUROFORGE_BRIDGE_SPEED_PERCENT`: emulator throttle target percent (`400` = 4x).
- `NEUROFORGE_BRIDGE_FRAMESKIP`: action repeat / captured-frame cadence.

Raise `NEUROFORGE_BRIDGE_FRAMESKIP` to trade decision granularity for speed
when transport or model processing becomes the bottleneck.
