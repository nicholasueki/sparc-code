"""M1b-lite — continuous face-identity enrichment on the Hailo-8 (v0.4).

While a person is in view (tracked via the tier0 bus), loop at ~1 Hz:
  frame (localhost frame server) -> SCRFD face detect + 5 landmarks ->
  similarity-align to 112x112 -> ArcFace -> L2-normalized 512-d embedding ->
  publish on lucas/vision/rich.

Identity RESOLUTION (cosine vs known_faces) happens in the orchestrator
(design LIM-M1-3) — this process only produces vectors. No frames or crops
are ever written to disk.

Run: python -m lucas_node_a.perception.face_enrich
"""
from __future__ import annotations

import logging
import threading
import time

import httpx
import numpy as np

from lucas_common import config
from lucas_common.bus import Bus
from lucas_common.types import Detection, DetectionFrame

log = logging.getLogger("lucas.face")

SCRFD_HEF = "/usr/share/hailo-models/scrfd_2.5g_h8l.hef"
ARCFACE_HEF = "/usr/share/hailo-models/arcface_mobilefacenet.hef"
IN_SIZE = 640

# stride -> (score, bbox, kps) output names, 2 anchors per cell
SCRFD_BRANCHES = {
    8: ("scrfd_2_5g/conv42", "scrfd_2_5g/conv43", "scrfd_2_5g/conv44"),
    16: ("scrfd_2_5g/conv49", "scrfd_2_5g/conv50", "scrfd_2_5g/conv51"),
    32: ("scrfd_2_5g/conv55", "scrfd_2_5g/conv56", "scrfd_2_5g/conv57"),
}

# ArcFace 112x112 canonical 5-point template (insightface standard)
ARCFACE_DST = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


class HailoModel:
    def __init__(self, vdevice, hef_path: str):
        from hailo_platform import FormatType

        self.model = vdevice.create_infer_model(hef_path)
        self.model.input().set_format_type(FormatType.UINT8)
        for out in self.model.outputs:
            self.model.output(out.name).set_format_type(FormatType.FLOAT32)
        self.cm = self.model.configure()
        self.out_shapes = {o.name: tuple(o.shape) for o in self.model.outputs}

    def run(self, img: np.ndarray) -> dict[str, np.ndarray]:
        bindings = self.cm.create_bindings()
        bindings.input().set_buffer(np.ascontiguousarray(img))
        # output buffers must be allocated and bound as views BEFORE run,
        # otherwise HailoRT raises "not configured as view" (status 6)
        outputs = {n: np.empty(s, dtype=np.float32) for n, s in self.out_shapes.items()}
        for name, buf in outputs.items():
            bindings.output(name).set_buffer(buf)
        self.cm.run([bindings], timeout=3000)
        return outputs


def _maybe_sigmoid(x: np.ndarray) -> np.ndarray:
    # zoo HEFs differ on whether scores are logits; detect and adapt
    if x.min() < 0.0 or x.max() > 1.0:
        return 1.0 / (1.0 + np.exp(-x))
    return x


def decode_scrfd(outs: dict, conf_t: float) -> tuple[np.ndarray, np.ndarray, float] | None:
    """-> (bbox_xyxy[4], landmarks[5,2], score) for the best face, in 640-space."""
    best = None
    for stride, (s_name, b_name, k_name) in SCRFD_BRANCHES.items():
        h = w = IN_SIZE // stride
        scores = _maybe_sigmoid(outs[s_name].reshape(-1))          # (h*w*2,)
        bbox = outs[b_name].reshape(-1, 4) * stride                # (h*w*2, 4)
        kps = outs[k_name].reshape(-1, 10) * stride                # (h*w*2, 10)
        ys, xs = np.mgrid[:h, :w]
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
        centers = np.repeat(centers, 2, axis=0).astype(np.float32)  # 2 anchors
        idx = int(np.argmax(scores))
        if best is None or scores[idx] > best[2]:
            cx, cy = centers[idx]
            x1, y1 = cx - bbox[idx, 0], cy - bbox[idx, 1]
            x2, y2 = cx + bbox[idx, 2], cy + bbox[idx, 3]
            lm = (centers[idx][None, :] + kps[idx].reshape(5, 2)).astype(np.float32)
            best = (np.array([x1, y1, x2, y2]), lm, float(scores[idx]))
    if best is None or best[2] < conf_t:
        return None
    return best


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    M, _ = cv2.estimateAffinePartial2D(landmarks / scale, ARCFACE_DST,
                                       method=cv2.LMEDS)
    if M is None:
        raise ValueError("alignment failed")
    return cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0)


class PresenceMirror:
    """Tracks whether anyone is in view by mirroring the tier0 topic."""

    def __init__(self, bus: Bus):
        self.tracks: set[str] = set()
        bus.subscribe("lucas/vision/tier0", DetectionFrame, self._on_frame)

    def _on_frame(self, f: DetectionFrame) -> None:
        for d in f.detections:
            if f.scene_delta == "new_track":
                self.tracks.add(d.track_id)
            elif f.scene_delta == "lost_track":
                self.tracks.discard(d.track_id)

    @property
    def anyone(self) -> bool:
        return bool(self.tracks)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    import cv2
    from hailo_platform import VDevice

    cfg = config.get("node_a.face", {})
    hz = float(cfg.get("enrich_hz", 1.0))
    conf_t = float(cfg.get("face_conf", 0.55))
    frame_url = f"http://127.0.0.1:{config.get('node_a.imx500.frame_port', 8600)}/frame.jpg"

    # round-robin scheduler lets both models be co-resident on one core and
    # auto-activate per inference (without it, running a 2-model vdevice raises
    # HAILO_INVALID_OPERATION because neither network group is active)
    from hailo_platform import HailoSchedulingAlgorithm

    vp = VDevice.create_params()
    vp.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
    vdevice = VDevice(vp)
    scrfd = HailoModel(vdevice, SCRFD_HEF)
    arcface = HailoModel(vdevice, ARCFACE_HEF)
    log.info("hailo-8 face pipeline resident (scrfd + arcface, round-robin)")

    bus = Bus(client_id="face-enrich")
    presence = PresenceMirror(bus)
    bus.start()
    http = httpx.Client(timeout=3)

    while True:
        if not presence.anyone:
            time.sleep(0.25)
            continue
        t0 = time.time()
        try:
            r = http.get(frame_url)
            if r.status_code != 200:
                time.sleep(1.0 / hz)
                continue
            frame = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            fh, fw = frame.shape[:2]
            scale = IN_SIZE / max(fh, fw)
            resized = cv2.resize(frame, (int(fw * scale), int(fh * scale)))
            canvas = np.zeros((IN_SIZE, IN_SIZE, 3), np.uint8)
            canvas[: resized.shape[0], : resized.shape[1]] = resized

            det = decode_scrfd(scrfd.run(canvas), conf_t)
            if det is None:
                time.sleep(1.0 / hz)
                continue
            box, lm, score = det
            aligned = align_face(frame, lm, scale)
            emb = arcface.run(aligned)["arcface_mobilefacenet/fc1"].reshape(-1)
            emb = emb / (np.linalg.norm(emb) + 1e-9)

            nb = (box / IN_SIZE).clip(0, 1)
            bus.publish("lucas/vision/rich", DetectionFrame(
                source="hailo8",
                scene_delta="periodic",
                detections=[Detection(
                    track_id=next(iter(presence.tracks), "unknown"),
                    cls="face", conf=score,
                    bbox=(float(nb[0]), float(nb[1]), float(nb[2]), float(nb[3])),
                    face_embedding=[float(x) for x in emb],
                )]))
            log.info("face embedded (det %.2f, %.0f ms)", score,
                     (time.time() - t0) * 1000)
        except Exception:
            log.exception("enrich tick failed")
        time.sleep(max(0.0, 1.0 / hz - (time.time() - t0)))


if __name__ == "__main__":
    main()
