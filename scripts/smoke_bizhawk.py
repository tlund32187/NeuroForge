"""One-click BizHawk bridge smoke test.

What it does, hands-off:
  1. Binds a local server in Python.
  2. Launches EmuHawk with the ROM and the NeuroForge Lua bridge.
  3. Holds RIGHT + B every frame (so Mario runs right) for a few hundred frames.
  4. Saves the first few frames it receives to artifacts/smoke/ as PNGs, so you
     can SEE exactly what the brain sees.

You should NOT need to type any commands: press the Run/play button on the
"NeuroForge: BizHawk smoke test" configuration in the Run and Debug panel.

If your paths differ from the auto-detected ones below, edit the three values
in the CONFIG block.
"""

from __future__ import annotations

import sys
from pathlib import Path

from neuroforge.contracts.applications.games import ControllerAction, GameObservation
from neuroforge.environments.games.clients import BizHawkConnectionError
from neuroforge.environments.games.clients.bizhawk.launcher import EmuHawkLauncher
from neuroforge.environments.games.smb3 import (
    BizHawkClient,
    BizHawkClientConfig,
    VisionOnlyGameLoop,
)
from neuroforge.environments.games.smb3.hud import SMB3HudConfig, SMB3HudExtractor

#
EMUHAWK_PATH = r"C:\BizHawk\EmuHawk.exe"
ROM_PATH = r"C:\BizHawk\ROM\Super Mario Bros. 3 (USA) (Rev 1)\Super Mario Bros. 3 (USA) (Rev 1).nes"  # noqa: E501
PORT = 8650
# MANUAL_MODE=True: the bridge does NOT inject buttons - YOU play with the
# keyboard/controller while we record frames + metrics. Most reliable way to
# reach an in-level state for HUD calibration. Set False to use the auto-demo.
MANUAL_MODE = True
FRAMES = 1400         # auto-demo frame budget (~25s of play)
MANUAL_FRAMES = 4000  # manual mode: give yourself time to walk into a level
# Frames whose received screenshot gets saved (title, map, in-level gameplay...).
SAVE_AT = frozenset({1, 90, 200, 400, 600, 800, 1000, 1200, 1399})
MANUAL_SAVE_EVERY = 200  # in manual mode, also save a frame every N frames
#

_REPO = Path(__file__).resolve().parents[1]
LUA_SCRIPT = _REPO / "bizhawk" / "neuroforge_bridge.lua"
OUT_DIR = _REPO / "artifacts" / "smoke"


class DemoPolicy:
    """A scripted demo 'brain' (not a learned policy) that walks into a level.

    SMB3 world map: from the start panel the first level node is UP, so the
    sequence is: Start past the title -> tap Up to reach the level node -> tap A
    to enter -> run right inside the level. Timing is approximate; it just needs
    to get the bridge into gameplay so x_progress and in-level frames appear.
    """

    def __init__(self) -> None:
        self._frame = -1

    def act(self, observation: GameObservation) -> ControllerAction:  # noqa: ARG002
        self._frame += 1
        n = self._frame
        if n < 90:
            # Title screen: tap Start (press ~5 frames, release ~5).
            return ControllerAction(start=(n // 5) % 2 == 0)
        if n < 170:
            # World map: tap Up to walk onto the level node above the start.
            return ControllerAction(up=(n // 6) % 2 == 0)
        if n < 240:
            # On the level node: tap A to enter the level.
            return ControllerAction(a=(n // 5) % 2 == 0)
        # Inside the level: run right with periodic jumps so it looks alive.
        return ControllerAction(right=True, b=True, a=(n // 8) % 4 == 0)


def _save_frame_png(observation: GameObservation, index: int) -> None:
    """Best-effort: re-encode a received frame to a viewable PNG."""
    try:
        import torch
        from torchvision.io import write_png  # pyright: ignore[reportMissingTypeStubs]

        frame = observation.frame
        tensor = (
            torch.frombuffer(bytearray(frame.data), dtype=torch.uint8)
            .reshape(frame.height, frame.width, frame.channels)
            .permute(2, 0, 1)
            .contiguous()
        )
        write_png(tensor, str(OUT_DIR / f"frame_{index:03d}.png"))
    except Exception as exc:  # noqa: BLE001 - saving is a nicety, never fatal
        print(f"  (could not save frame {index}: {exc})")


def _describe(observation: GameObservation) -> str:
    """One-line summary of the metrics the extractor read from a frame."""
    m = observation.metrics
    parts: list[str] = []
    parts.append(f"x_progress={m.x_progress:.3f}" if m.x_progress is not None else "x_progress=?")
    if m.score is not None:
        parts.append(f"score={m.score}")
    if m.lives is not None:
        parts.append(f"lives={m.lives}")
    if m.world is not None:
        parts.append(f"world={m.world}")
    return " ".join(parts)


def main() -> int:
    print("=" * 70)
    print("NeuroForge - BizHawk bridge smoke test")
    print("=" * 70)
    for label, path in (
        ("EmuHawk", EMUHAWK_PATH),
        ("ROM", ROM_PATH),
        ("Lua bridge", str(LUA_SCRIPT)),
    ):
        ok = Path(path).exists()
        print(f"  [{'OK ' if ok else 'MISSING'}] {label}: {path}")
        if not ok:
            print(f"\nERROR: {label} not found. Edit the CONFIG block in this file.")
            return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Screenshots will be saved to: {OUT_DIR}")
    print("\n  Starting... an EmuHawk window should open in a few seconds.")
    if MANUAL_MODE:
        print("  MANUAL MODE: click the EmuHawk window and PLAY with your keyboard.")
        print("  Walk Mario into a level and run right. We're recording frames + metrics.")
        print("  (Stop early with the red square when you've reached gameplay.)\n")
    else:
        print("  AUTO DEMO: taps Start, walks up to the level, enters it, runs right.")
        print("  Watch the x_progress readout climb once he's in a level.\n")

    client = BizHawkClient(
        BizHawkClientConfig(
            port=PORT,
            width=256,
            height=240,
            channels=3,
            connect_timeout_s=60.0,   # EmuHawk can take a moment to start
            step_timeout_s=45.0,      # tolerate brief pauses (e.g. focus loss)
            launch=True,
            transport="socket",
            passive_input=MANUAL_MODE,
        ),
        launcher=EmuHawkLauncher(
            emuhawk_path=EMUHAWK_PATH,
            lua_script=str(LUA_SCRIPT),
            rom_path=ROM_PATH,
        ),
    )
    # Vision-only metric extractor: x_progress works immediately; digit OCR
    # (score/lives/world) reads None until a glyph atlas is calibrated.
    extractor = SMB3HudExtractor(SMB3HudConfig())
    extractor.reset()
    print(f"  HUD digit OCR calibrated: {extractor.is_calibrated}")
    loop = VisionOnlyGameLoop(client=client, policy=DemoPolicy(), metric_extractor=extractor)

    max_steps = MANUAL_FRAMES if MANUAL_MODE else FRAMES
    received = 0
    saved = 0
    try:
        for transition in loop.run(max_steps=max_steps):
            received += 1
            if received in SAVE_AT or (MANUAL_MODE and received % MANUAL_SAVE_EVERY == 0):
                _save_frame_png(transition.after, received)
                saved += 1
            if received % 60 == 0:
                print(f"  ...{received} frames | {_describe(transition.after)}")
    except KeyboardInterrupt:
        print("\n  Interrupted by user.")
    except BizHawkConnectionError as exc:
        # The emulator stopped sending frames - usually EmuHawk pausing on focus
        # loss, or you closed/stopped it. Not fatal: keep what we captured.
        print(f"\n  Emulator stopped sending frames after {received} frames.")
        print(f"  ({exc})")
        print(
            "  Tip: BizHawk pauses when its window loses focus; "
            "keep it focused while playing."
        )
    finally:
        loop.close()

    print("\n" + "=" * 70)
    if received > 0:
        print(f"  SUCCESS: received {received} frames; saved {saved} screenshots.")
        print(f"  Open {OUT_DIR} - those PNGs are the brain's view (incl. in-level frames).")
    else:
        print("  No frames received. See docs/BIZHAWK_SETUP.md -> Troubleshooting.")
    print("=" * 70)
    return 0 if received > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
