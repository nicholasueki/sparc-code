"""M1b-lite — continuous face-identity enrichment on the Hailo-8 (v0.4).

While a person is in view (tracked via the tier0 bus), loop at ~1 Hz:
  frame (localhost frame server) -> SCRFD face detect + 5 landmarks ->
  similarity-align to 112x112 -> ArcFace -> L2-normalized 512-d embedding ->
  publish on sparc/vision/rich.

Identity RESOLUTION (cosine vs known_faces) happens in the orchestrator
(design LIM-M1-3) — this process only produces vectors.

Run: python -m sparc_node_a.perception.face_enrich
"""
from __future__ import annotations

import logging
import threading
import time

import httpx
import numpy as np

from sparc_common import config
from sparc_common.bus import Bus
from sparc_common.health import HealthReporter
from sparc_common.types import Detection, DetectionFrame

log = logging.getLogger("sparc.face")

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


def decode_scrfd(outs: dict, conf_t: float, max_faces: int = 4
                 ) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """-> [(bbox_xyxy[4], landmarks[5,2], score)] for ALL faces >= conf_t,
    NMS-deduplicated, best-first, in 640-canvas space."""
    cands: list[tuple[np.ndarray, np.ndarray, float]] = []
    for stride, (s_name, b_name, k_name) in SCRFD_BRANCHES.items():
        h = w = IN_SIZE // stride
        scores = _maybe_sigmoid(outs[s_name].reshape(-1))          # (h*w*2,)
        bbox = outs[b_name].reshape(-1, 4) * stride                # (h*w*2, 4)
        kps = outs[k_name].reshape(-1, 10) * stride                # (h*w*2, 10)
        ys, xs = np.mgrid[:h, :w]
        centers = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
        centers = np.repeat(centers, 2, axis=0).astype(np.float32)  # 2 anchors
        for idx in np.nonzero(scores >= conf_t)[0]:
            cx, cy = centers[idx]
            box = np.array([cx - bbox[idx, 0], cy - bbox[idx, 1],
                            cx + bbox[idx, 2], cy + bbox[idx, 3]])
            lm = (centers[idx][None, :] + kps[idx].reshape(5, 2)).astype(np.float32)
            cands.append((box, lm, float(scores[idx])))
    cands.sort(key=lambda c: -c[2])
    kept: list[tuple[np.ndarray, np.ndarray, float]] = []
    for box, lm, sc in cands:
        if all(_box_iou(box, k[0]) < 0.4 for k in kept):
            kept.append((box, lm, sc))
        if len(kept) >= max_faces:
            break
    return kept


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area > 0 else 0.0


def match_face_to_track(face_center_norm: tuple[float, float],
                        track_boxes: dict[str, tuple]) -> str | None:
    """Attribute a face to the person box containing its center; if several
    contain it (people overlapping), pick the smallest box (nearest person)."""
    cx, cy = face_center_norm
    containing = [
        (tid, (b[2] - b[0]) * (b[3] - b[1]))
        for tid, b in track_boxes.items()
        if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]
    ]
    if not containing:
        return None
    return min(containing, key=lambda t: t[1])[0]


def align_face(frame_bgr: np.ndarray, landmarks: np.ndarray, scale: float) -> np.ndarray:
    import cv2

    M, _ = cv2.estimateAffinePartial2D(landmarks / scale, ARCFACE_DST,
                                       method=cv2.LMEDS)
    if M is None:
        raise ValueError("alignment failed")
    return cv2.warpAffine(frame_bgr, M, (112, 112), borderValue=0)


class BoxMirror:
    """Mirrors tier0 to know WHO is in view and WHERE (track_id -> bbox)."""

    def __init__(self, bus: Bus):
        self.boxes: dict[str, tuple] = {}
        bus.subscribe("sparc/vision/tier0", DetectionFrame, self._on_frame)

    def _on_frame(self, f: DetectionFrame) -> None:
        for d in f.detections:
            if f.scene_delta == "lost_track":
                self.boxes.pop(d.track_id, None)
            else:  # new_track or periodic refresh
                self.boxes[d.track_id] = d.bbox

    @property
    def anyone(self) -> bool:
        return bool(self.boxes)


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
    mirror = BoxMirror(bus)
    bus.start()
    def health_probe() -> dict:
        details = {
            "mqtt_ready": bus.connected,
            "scrfd_loaded": scrfd is not None,
            "arcface_loaded": arcface is not None,
            "scrfd_model": SCRFD_HEF,
            "arcface_model": ARCFACE_HEF,
        }
        ready = bool(details["mqtt_ready"] and details["scrfd_loaded"] and
                     details["arcface_loaded"])
        missing = [name for name in ("mqtt_ready", "scrfd_loaded", "arcface_loaded")
                   if not details[name]]
        return {
            "ready": ready,
            "details": details,
            "failure_reason": None if ready else f"not ready: {', '.join(missing)}",
        }

    HealthReporter(bus, "enrich", health_probe).start()
    http = httpx.Client(timeout=3)

    while True:
        if not mirror.anyone:
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

            faces = decode_scrfd(scrfd.run(canvas), conf_t)
            track_boxes = dict(mirror.boxes)  # snapshot
            dets: list[Detection] = []
            for box, lm, score in faces:
                # face center in frame-normalized coords = person-box space
                fcx = (box[0] + box[2]) / 2 / scale / fw
                fcy = (box[1] + box[3]) / 2 / scale / fh
                tid = match_face_to_track((fcx, fcy), track_boxes)
                if tid is None and len(track_boxes) == 1:
                    tid = next(iter(track_boxes))  # single person: trivially theirs
                if tid is None:
                    continue  # face with no owner (box lag) — skip, next tick
                aligned = align_face(frame, lm, scale)
                emb = arcface.run(aligned)["arcface_mobilefacenet/fc1"].reshape(-1)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                dets.append(Detection(
                    track_id=tid, cls="face", conf=score,
                    bbox=(float(fcx), float(fcy), float(fcx), float(fcy)),
                    face_embedding=[float(x) for x in emb],
                ))
            if dets:
                bus.publish("sparc/vision/rich", DetectionFrame(
                    source="hailo8", scene_delta="periodic", detections=dets))
                log.info("embedded %d face(s) -> %s (%.0f ms)",
                         len(dets), [d.track_id[:6] for d in dets],
                         (time.time() - t0) * 1000)
        except Exception:
            log.exception("enrich tick failed")
        time.sleep(max(0.0, 1.0 / hz - (time.time() - t0)))


if __name__ == "__main__":
    main()
