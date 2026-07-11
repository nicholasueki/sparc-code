#!/usr/bin/env python3
"""Enroll a face: stand in front of the camera, run this ON the vision Pi.

    ~/lucas_venv/bin/python scripts/enroll.py --name Nicholas

Collects N face embeddings from the live enrichment stream (lucas/vision/rich),
averages them, and stores ONE 512-number vector under the given name.
No images are captured or stored by enrollment — only the vector.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("common", "node_a"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

import numpy as np  # noqa: E402

from lucas_common import config  # noqa: E402
from lucas_common.bus import Bus  # noqa: E402
from lucas_common.types import DetectionFrame  # noqa: E402
from lucas_node_a.world_model import WorldModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    collected: list[np.ndarray] = []

    def on_rich(frame: DetectionFrame) -> None:
        for det in frame.detections:
            if det.face_embedding and det.conf >= 0.6:
                collected.append(np.asarray(det.face_embedding, dtype=np.float32))
                print(f"  sample {len(collected)}/{args.samples} (det conf {det.conf:.2f})")

    bus = Bus(client_id="enroll")
    bus.subscribe("lucas/vision/rich", DetectionFrame, on_rich)
    bus.start()
    print(f"Look at the camera, {args.name} — collecting {args.samples} samples...")
    deadline = time.time() + args.timeout
    while len(collected) < args.samples and time.time() < deadline:
        time.sleep(0.2)
    bus.stop()

    if len(collected) < 3:
        print(f"only {len(collected)} samples — stand closer/face the camera and retry")
        sys.exit(1)

    emb = np.mean(np.stack(collected), axis=0)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    # self-consistency check: every sample should match the mean strongly
    sims = [float(e @ emb) for e in collected]
    print(f"sample self-similarity: min {min(sims):.3f} (want > 0.6)")
    if min(sims) < 0.5:
        print("samples too inconsistent (multiple faces? movement?) — retry")
        sys.exit(1)

    world = WorldModel(config.get("node_a.db_path"))
    eid = world.enroll_face(args.name, [float(x) for x in emb])
    world.add_event("enrolled", f"{args.name} enrolled their face", [eid], 0.6)
    print(f"✓ enrolled {args.name} ({eid}) from {len(collected)} samples")


if __name__ == "__main__":
    main()
