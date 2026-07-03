"""M1a — IMX500 always-on tripwire (NanoDet-Plus, on-sensor).

Runs the postprocessed .rpk on the sensor NPU, tracks person boxes with a
tiny nearest-centroid tracker, and emits DetectionFrame(new_track/lost_track)
on the bus. ~0 host CPU: we only parse output tensors.

Runs standalone in its own process: python -m lucas_node_a.perception.imx500_tripwire
"""
from __future__ import annotations

import logging
import time

from lucas_common import config
from lucas_common.bus import Bus
from lucas_common.types import Detection, DetectionFrame, new_id

log = logging.getLogger("lucas.imx500")

PERSON_CLASS = 0  # COCO person in the RPi nanodet labels


class CentroidTracker:
    def __init__(self, lost_after_s: float):
        self.tracks: dict[str, dict] = {}  # id -> {cx, cy, last_seen}
        self.lost_after = lost_after_s

    def update(self, boxes: list[tuple[float, float, float, float]]
               ) -> tuple[list[tuple[str, tuple]], list[str], list[str]]:
        """-> (current [(track_id, box)], new_ids, lost_ids)"""
        now = time.time()
        assigned, new_ids, current = set(), [], []
        for box in boxes:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            best, best_d = None, 0.25  # max normalized jump between frames
            for tid, t in self.tracks.items():
                if tid in assigned:
                    continue
                d = ((t["cx"] - cx) ** 2 + (t["cy"] - cy) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = tid, d
            if best is None:
                best = new_id()
                new_ids.append(best)
            assigned.add(best)
            self.tracks[best] = {"cx": cx, "cy": cy, "last_seen": now}
            current.append((best, box))
        lost = [tid for tid, t in self.tracks.items()
                if now - t["last_seen"] > self.lost_after]
        for tid in lost:
            del self.tracks[tid]
        return current, new_ids, lost


def _parse_detections(outputs, imx500, conf_t: float
                      ) -> tuple[list[tuple[float, float, float, float]], list[float]]:
    """Person boxes+confs from IMX500 tensors. Handles both output layouts:
    - `_pp` rpk (postprocess on-sensor): outputs = [boxes, scores, classes]
    - raw rpk: host-side nanodet postprocess
    Boxes normalized to [0,1] xyxy regardless of source format.
    """
    boxes: list[tuple[float, float, float, float]] = []
    confs: list[float] = []
    if len(outputs) >= 3:  # on-sensor postprocessed
        b, scores, classes = outputs[0][0], outputs[1][0], outputs[2][0]
    else:  # raw output tensor -> host postprocess
        from picamera2.devices.imx500 import postprocess_nanodet_detection

        b, scores, classes = postprocess_nanodet_detection(
            outputs=outputs[0], conf=conf_t, iou_thres=0.6, max_out_dets=8
        )[0]
    in_w, in_h = imx500.get_input_size()
    for box, score, cls in zip(b, scores, classes):
        if int(cls) != PERSON_CLASS or float(score) < conf_t:
            continue
        v = [float(x) for x in box]
        if max(v) > 1.5:  # pixel coords -> normalize
            v = [v[0] / in_h, v[1] / in_w, v[2] / in_h, v[3] / in_w]
        y0, x0, y1, x1 = v  # sensor emits yxyx
        boxes.append((x0, y0, x1, y1))
        confs.append(float(score))
    return boxes, confs


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from picamera2 import Picamera2
    from picamera2.devices import IMX500

    cfg = config.get("node_a.imx500", {})
    imx500 = IMX500(cfg["model"])
    intrinsics = imx500.network_intrinsics
    picam2 = Picamera2(imx500.camera_num)
    conf_t = float(cfg.get("person_conf", 0.45))

    bus = Bus(client_id="imx500-tripwire")
    bus.start()
    tracker = CentroidTracker(float(cfg.get("person_lost_s", 5.0)))

    camera_config = picam2.create_preview_configuration(buffer_count=6)
    picam2.start(camera_config)
    log.info("IMX500 tripwire live (model=%s)", cfg["model"])

    while True:
        metadata = picam2.capture_metadata()
        outputs = imx500.get_outputs(metadata, add_batch=True)
        if outputs is None:
            continue
        try:
            boxes, confs = _parse_detections(outputs, imx500, conf_t)
        except Exception:
            log.exception("tensor parse failed")
            time.sleep(1)  # don't spam the log at 30 fps
            continue

        current, new_ids, lost_ids = tracker.update(boxes)
        if new_ids:
            dets = [
                Detection(track_id=tid, cls="person", conf=c, bbox=box)
                for (tid, box), c in zip(current, confs)
                if tid in new_ids
            ]
            bus.publish("lucas/vision/tier0", DetectionFrame(
                source="imx500", detections=dets, scene_delta="new_track"))
            log.info("new person track(s): %s", new_ids)
        for tid in lost_ids:
            bus.publish("lucas/vision/tier0", DetectionFrame(
                source="imx500",
                detections=[Detection(track_id=tid, cls="person", conf=0.0,
                                      bbox=(0, 0, 0, 0))],
                scene_delta="lost_track"))
            log.info("lost track: %s", tid)


if __name__ == "__main__":
    main()
