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

No external Lua install is needed. The live scripts use the bridge's socket
transport; the Lua side tries BizHawk's bundled LuaSocket files first, then
LuaCOM/MSWinsock, with a file fallback still available.

## Transport: socket for live runs, file fallback

Live training and smoke scripts now use `BizHawkClientConfig(transport="socket")`.
Python binds a localhost control socket and a dedicated screenshot socket,
launches EmuHawk, then passes the endpoints through `NEUROFORGE_BRIDGE_HOST`,
`NEUROFORGE_BRIDGE_PORT`, `NEUROFORGE_BRIDGE_SCREENSHOT_HOST`,
`NEUROFORGE_BRIDGE_SCREENSHOT_PORT`, and `NEUROFORGE_BRIDGE_TRANSPORT=socket`.

On the Lua side, [neuroforge_bridge.lua](../scripts/bizhawk/neuroforge_bridge.lua) tries:

- **LuaSocket** via `require("socket")`, then `require("socket.core")`.
  This covers installs where BizHawk has `C:\BizHawk\Lua\socket\core.dll` but
  not the pure-Lua `socket.lua` loader.
- **LuaCOM/MSWinsock** via `luacom.CreateObject("MSWinsock.Winsock")`.
- **BizHawk screenshot socket** via `comm.socketServerScreenShot()` for PNG
  frame bytes, so live socket mode does not write a temp screenshot file.
- **File transport** when `NEUROFORGE_BRIDGE_TRANSPORT=file` or no socket port
  is provided.

The file fallback moves messages over atomically-renamed files in
`%TEMP%\neuroforge_bridge\` and captures screenshots through
`client.screenshot(path)`. It is slower but remains useful for manual bring-up
on machines where socket setup is unavailable.

## How it works (lock-step socket path)

```
Python (BizHawkClient)                         EmuHawk + neuroforge_bridge.lua
  bind TCP, launch EmuHawk ------------------>  --lua runs the bridge
  accept socket <-----------------------------  LuaSocket/LuaCOM connects
  read HELLO  <-------------------------------  write HELLO {w,h,channels,png}
  write WELCOME ----------------------------->  read WELCOME
  reset() read FRAME metadata <---------------  write FRAME metadata
  read PNG over screenshot socket <-----------  comm.socketServerScreenShot()
  step(action) write ACTION ----------------->  read ACTION -> joypad.set -> frameadvance
  read FRAME metadata + PNG <----------------  metadata + screenshot socket
  close() write CLOSE ----------------------->  read CLOSE -> BYE -> done
```

Exactly one action per emulated step. Frames are delivered at **native
resolution** (the HUD must stay readable for Phase 1); downsampling for the
network happens later in the policy's preprocessor.

## Manual run (without the smoke script)

1. Start Python first so it creates the comm dirs and waits:

   ```python
   from neuroforge.environments.games.smb3 import BizHawkClient, BizHawkClientConfig, VisionOnlyGameLoop
   from neuroforge.contracts.applications.games import ControllerAction

   client = BizHawkClient(BizHawkClientConfig(transport="file"))
   class HoldRight:
       def act(self, obs): return ControllerAction(right=True)
   loop = VisionOnlyGameLoop(client=client, policy=HoldRight())
   for _ in loop.run(max_steps=600):
       pass
   loop.close()
   ```

2. In EmuHawk: load the ROM, then **Tools → Lua Console → Open Script** and run
   [scripts/bizhawk/neuroforge_bridge.lua](../scripts/bizhawk/neuroforge_bridge.lua). It reads the
   comm dir from `NEUROFORGE_BRIDGE_DIR` or falls back to
   `%TEMP%\neuroforge_bridge`.

## Verify-on-your-BizHawk

These Lua APIs vary by EmuHawk version — the spots are commented in
[neuroforge_bridge.lua](../scripts/bizhawk/neuroforge_bridge.lua):

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

## Evaluating a learned checkpoint

Training reward is noisy because the policy is still plastic and exploratory.
Use [scripts/evaluate_smb3_checkpoint.py](../scripts/evaluate_smb3_checkpoint.py)
to load `artifacts/smb3_policy.pt` read-only, freeze R-STDP updates, freeze the
learned perception stack, and measure episode-level progress without overwriting
the checkpoint.

Quick smoke:

```powershell
$env:NEUROFORGE_EVAL_EPISODES='1'
$env:NEUROFORGE_EVAL_FRAMES='120'
python scripts/evaluate_smb3_checkpoint.py
```

Optional random-policy comparison:

```powershell
$env:NEUROFORGE_EVAL_RANDOM_BASELINE='1'
python scripts/evaluate_smb3_checkpoint.py
```

The most useful trend metric is `max_x_progress` in the emitted
`training_trial` events. Reward still includes the action-energy terms, so it is
good for judging button economy; `max_x_progress` is the cleaner learning signal.

## VS Code live workflow

The Run/Debug dropdown has SMB3 buttons for the main live workflows:

- `NeuroForge: Train SMB3` continues online R-STDP training and writes
  `artifacts/smb3_policy.pt`.
- `NeuroForge: Evaluate SMB3 Checkpoint` compares that checkpoint against a
  random baseline with stochastic decode.
- `NeuroForge: Evolve SMB3` runs live neuroevolution and writes an evolution
  JSON checkpoint under `artifacts/runs/evolve_*/evolution/checkpoint.json`.
- `NeuroForge: Full SMB3 Suite` runs evolution, applies the best genome's
  checkpoint-compatible control/learning genes to training, then evaluates the
  resulting policy.
- `NeuroForge: Fresh Architecture SMB3 Suite` runs full phenotype evolution,
  warm-starts the evolved architecture from the trained `smb3_policy.pt` by
  copying overlapping weights, trains it into `artifacts/smb3_arch_policy.pt`,
  then evaluates that architecture against a random policy with the same shape.

The suite deliberately uses the compatible genome handoff by default. Full
evolved phenotypes can change network shape, so the fresh-architecture lane uses
partial warm-start migration: overlapping checkpoint weights are copied into the
new shape and new capacity starts from its evolved initialization.

Full phenotype SMB3 evolution can now alter the policy depth and pathways:
`n_hidden_layers` chooses a one- to three-layer hidden stack, `hidden_fanin`
controls sparse inter-hidden connectivity, and `input_to_motor_skip` adds an
optional direct sensorimotor shortcut. The default phenotype remains the old
single hidden layer, so existing checkpoints and compatible handoff behavior do
not change unless the full-architecture lane is selected.

## Transfer and retention

Warm-starting an evolved or cross-game brain preserves learned knowledge at the
start of the next run, but it does not by itself prevent catastrophic forgetting.
For transfer runs, set `NEUROFORGE_SMB3_RESUME_CHECKPOINT` to the old policy
checkpoint and set `NEUROFORGE_SMB3_CONSOLIDATION_STRENGTH` to a small positive
value (for example `0.001` to `0.01`). The R-STDP trainer will anchor weights
loaded from that checkpoint and gently pull them back toward the old values
during new training. For partial architecture warm-starts, only the copied
overlap is anchored; newly added neurons/connections remain free to specialize.

This is the first retention layer for the future SMB3 -> SMB1 workflow. The
next validation step should evaluate both games after transfer: new-game
progress measures adaptation, while the old-game checkpoint eval measures how
much ability was retained.

## Troubleshooting

- **`module 'socket' not found`** in the Lua console — the bridge next tries
  `socket.core`, which can load from `C:\BizHawk\Lua\socket\core.dll`. If that
  also fails, socket mode falls through to LuaCOM/MSWinsock.
- **`LuaCOM unavailable`** after a socket failure — run with
  `BizHawkClientConfig(transport="file")` to use the fallback while checking the
  BizHawk Lua install.
- **`timed out waiting for emulator message`** — the Lua script is not running,
  did not connect to Python, or is using the wrong transport. Confirm EmuHawk
  loaded the script and there is no `[neuroforge_bridge] error:` line in the Lua
  console.
- **`MISSING` path in the smoke terminal** — fix the CONFIG block in
  [scripts/smoke_bizhawk.py](../scripts/smoke_bizhawk.py).
- **Nothing moves** — confirm the ROM is actually running (not paused). The demo
  policy taps Start for the first ~2s before running right.

## Performance note (Phase 6)

Full-resolution PNG per frame is for correctness, not 60 fps. The socket
transport removes both the message-file round-trip and the temp screenshot file,
but PNG capture/encoding and the model step can still be bottlenecks.
For live training, [scripts/train_smb3.py](../scripts/train_smb3.py) launches
BizHawk with `NEUROFORGE_BRIDGE_SPEED_PERCENT=400`, matching the menu's
**Speed 400%** setting (4x). With the default `FRAMESKIP=4`, that targets about
60 decisions/sec when PNG capture and model processing can keep up.

Use these knobs separately:

- `NEUROFORGE_BRIDGE_SPEED_PERCENT`: emulator throttle target percent (`400` = 4x).
- `NEUROFORGE_BRIDGE_FRAMESKIP`: action repeat / captured-frame cadence.

Raise `NEUROFORGE_BRIDGE_FRAMESKIP` to trade decision granularity for speed
when transport or model processing becomes the bottleneck.
