"""See what the brain's perception learned: render A1's features to a PNG.

Loads the perception state saved in the policy checkpoint (the A1 STDP feature
detectors), renders each learned receptive field as a tile, and writes a PNG you
can open. Run after some training with the feature stack enabled.

  Run/▶ "NeuroForge: Visualize Features"  (or run from a terminal).
"""

from __future__ import annotations

import sys
from pathlib import Path

from neuroforge.perception.vision.encoding import (
    PerceptionStack,
    PerceptionStackConfig,
    RetinaEncoderConfig,
    render_feature_atlas,
)

_REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = _REPO / "artifacts" / "smb3_policy.pt"
OUT = _REPO / "artifacts" / "feature_atlas.png"


def main() -> int:
    import torch
    from torchvision.io import write_png

    if not CHECKPOINT.exists():
        print(f"No checkpoint at {CHECKPOINT} — train first.")
        return 1
    payload = torch.load(str(CHECKPOINT), map_location="cpu", weights_only=False)
    encoder_state = payload.get("encoder")
    if not encoder_state:
        print(
            "Checkpoint has no perception state. Train with the feature stack enabled "
            "(features=True in train_smb3.py) so A1 has something to show.",
        )
        return 1

    # Rebuild a stack matching the trained one and load the learned features.
    stack = PerceptionStack(
        PerceptionStackConfig(
            retina=RetinaEncoderConfig(out_h=28, out_w=32),
            features=True, objects=True, motion=True, learn=False,
        ),
    )
    stack.load_state_dict(encoder_state)
    layer = stack.feature_layer
    if layer is None:
        print("Stack has no feature layer.")
        return 1

    atlas = render_feature_atlas(layer, scale=12)  # [H, W] uint8
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_png(atlas.unsqueeze(0), str(OUT))  # [1, H, W]
    print(f"Wrote learned-feature atlas to {OUT}")
    print("Each tile is one A1 detector's receptive field (ON minus OFF contrast).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
