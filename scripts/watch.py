#!/usr/bin/env python3
"""Live fleet watcher — one terminal window, everything Lucas sees/thinks/says.

Usage (from repo root on any machine on the LAN):
    .venv/bin/python scripts/watch.py            # or any python with paho-mqtt
Options:
    --host robot-vision.local   MQTT broker (default)
    --raw                       also dump unknown topics verbatim
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import paho.mqtt.client as mqtt

C = {
    "dim": "\033[2m", "reset": "\033[0m", "bold": "\033[1m",
    "cyan": "\033[36m", "green": "\033[32m", "yellow": "\033[33m",
    "magenta": "\033[35m", "blue": "\033[34m", "red": "\033[31m",
}


def ts() -> str:
    return time.strftime("%H:%M:%S")


def line(color: str, tag: str, text: str) -> None:
    print(f"{C['dim']}{ts()}{C['reset']} {C[color]}{C['bold']}{tag:<9}{C['reset']} {text}",
          flush=True)


def wrap(text: str, indent: int = 19, width: int = 100) -> str:
    words, out, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    out.append(cur)
    pad = "\n" + " " * indent
    return pad.join(out)


def on_message(client, userdata, msg):
    raw_mode = userdata["raw"]
    try:
        data = json.loads(msg.payload)
    except Exception:
        if raw_mode:
            line("red", "raw", f"{msg.topic}: {msg.payload[:200]!r}")
        return

    t = msg.topic
    if t == "lucas/vision/tier0":
        delta = data.get("scene_delta")
        n = len(data.get("detections", []))
        icon = {"new_track": "appeared", "lost_track": "left view"}.get(delta, delta)
        line("cyan", "VISION", f"person {icon} ({n} det, src={data.get('source')})")
    elif t == "lucas/audio/transcript":
        line("green", "HEARD", f"\"{data.get('text', '')}\"")
    elif t == "lucas/audio/sound":
        line("green", "SOUND", f"{data.get('cls')} (conf {data.get('conf', 0):.2f})")
    elif t == "lucas/tts/say":
        line("yellow", "LUCAS", f"{C['bold']}\"{data.get('text', '')}\"{C['reset']}")
    elif t == "lucas/debug/thought":
        cam = " 📷" if data.get("has_image") else ""
        line("magenta", "EVENT", data.get("event", "") + cam)
        if data.get("scene"):
            line("blue", "scene", C["dim"] + wrap(data["scene"]) + C["reset"])
        if data.get("memory"):
            line("blue", "memory", C["dim"] + wrap(data["memory"]) + C["reset"])
        if data.get("thinking"):
            line("magenta", "think", C["dim"] + wrap(data["thinking"]) + C["reset"])
        for o in data.get("options", []):
            marker = "→" if o["idx"] == data.get("choice") else " "
            argtxt = o["args"].get("text") or o["args"].get("statement") or ""
            picked = C["bold"] if o["idx"] == data.get("choice") else C["dim"]
            line("magenta", f"  opt {o['idx']}",
                 f"{picked}{marker} {o['action']}: {argtxt[:90]}{C['reset']}")
        line("magenta", "chose",
             f"#{data.get('choice')} because {data.get('why','')} "
             f"{C['dim']}({data.get('gen_ms')} ms, attempt {data.get('attempts')}){C['reset']}")
        print()
    elif raw_mode:
        line("red", "bus", f"{msg.topic}: {json.dumps(data)[:200]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="10.1.215.55")
    ap.add_argument("--raw", action="store_true")
    args = ap.parse_args()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                         client_id=f"watch-{int(time.time())}",
                         userdata={"raw": args.raw})
    client.on_message = on_message
    client.connect(args.host, 1883, keepalive=30)
    client.subscribe("lucas/#", qos=0)
    print(f"{C['bold']}— watching Lucas on {args.host} (Ctrl-C to quit) —{C['reset']}")
    try:
        client.loop_forever()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
