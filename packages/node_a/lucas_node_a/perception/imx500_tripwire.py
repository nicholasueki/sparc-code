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


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import postprocess_nanodet_detection

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
        boxes: list[tuple[float, float, float, float]] = []
        confs: list[float] = []
        try:
            # postprocessed rpk: outputs = boxes/scores/classes tensors
            b, scores, classes = postprocess_nanodet_detection(
                outputs=outputs[0], conf=conf_t, iou_thres=0.6, max_out_dets=8
            )[0]
            for box, score, cls in zip(b, scores, classes):
                if int(cls) == PERSON_CLASS and score >= conf_t:
                    x0, y0, x1, y1 = box
                    boxes.append((float(x0), float(y0), float(x1), float(y1)))
                    confs.append(float(score))
        except Exception:
            log.exception("tensor parse failed")
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
