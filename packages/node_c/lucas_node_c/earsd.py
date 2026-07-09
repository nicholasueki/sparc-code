"""earsd — interim ears + voice on the tokenator Mac (v0.3-interim).

mic -> webrtcvad utterance gating -> genaid /stt (Whisper on the Hailo-10H,
the designed path) -> lucas/audio/transcript on the bus -> orchestrator thinks
-> lucas/tts/say -> spoken through the Mac with `say`.

Echo control is half-duplex: the mic is gated while Lucas speaks (+ a tail),
so he never transcribes himself. When the USB speakerphone arrives, this whole
file's job moves to Node B (Silero + openWakeWord + Piper per the design);
genaid /stt stays exactly as-is.

PRIVACY: audio is processed in memory and discarded; nothing is recorded to disk.

Run: python -m lucas_node_c.earsd  (needs macOS mic permission the first time)
"""
from __future__ import annotations

import base64
import collections
import logging
import subprocess
import threading
import time

import httpx

from lucas_common import config
from lucas_common.types import SpeakRequest, Transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("lucas.earsd")

CFG = config.get("node_c.ears", {})
GENAID = config.get("endpoints.genaid", "http://robot-genai.local:8700")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480

_speaking_until = 0.0  # half-duplex gate (epoch)


def _muted() -> bool:
    return time.time() < _speaking_until


# ------------------------------------------------------------------ voice out

def speaker_loop() -> None:
    """Subscribe to lucas/tts/say; speak via macOS `say`; gate the mic meanwhile."""
    import paho.mqtt.client as mqtt

    voice = CFG.get("tts_voice", "Samantha")
    rate = int(CFG.get("tts_rate_wpm", 190))

    def on_message(client, userdata, msg):
        global _speaking_until
        try:
            req = SpeakRequest.model_validate_json(msg.payload)
        except Exception:
            return
        est_s = max(1.0, len(req.text.split()) / (rate / 60.0)) + 1.0
        _speaking_until = time.time() + est_s + float(CFG.get("echo_tail_s", 0.7))
        log.info("SPEAKING: %s", req.text[:80])
        try:
            subprocess.run(["say", "-v", voice, "-r", str(rate), req.text],
                           timeout=60, check=False)
        finally:
            _speaking_until = time.time() + float(CFG.get("echo_tail_s", 0.7))

    def on_disconnect(client, *a, **k):
        log.warning("tts bus disconnected; reconnecting")
        while True:
            try:
                client.reconnect()
                return
            except Exception:
                time.sleep(3)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="earsd-voice")
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.connect(config.get("bus.host"), int(config.get("bus.port", 1883)), keepalive=30)
    client.subscribe("lucas/tts/say", qos=1)
    client.loop_forever(retry_first_connection=True)


# ------------------------------------------------------------------- ears in

def publish_transcript(text: str, conf: float) -> None:
    import json

    import paho.mqtt.publish as mqtt_publish

    tr = Transcript(text=text, conf=conf)
    mqtt_publish.single(
        "lucas/audio/transcript", tr.model_dump_json(), qos=1,
        hostname=config.get("bus.host"), port=int(config.get("bus.port", 1883)),
    )


_stt_backend = None  # resolved at first use: "genaid" (10H) or "mlx_local" (Mac)


def _resolve_stt_backend() -> str:
    global _stt_backend
    if _stt_backend:
        return _stt_backend
    forced = CFG.get("stt_backend", "auto")
    if forced != "auto":
        _stt_backend = forced
    else:
        try:  # designed home first: Whisper on the 10H
            h = httpx.get(f"{GENAID}/health", timeout=4).json()
            _stt_backend = "genaid" if h.get("stt_loaded") else "mlx_local"
        except Exception:
            _stt_backend = "mlx_local"
    log.info("STT backend: %s", _stt_backend)
    return _stt_backend


def transcribe(pcm: bytes) -> str:
    if _resolve_stt_backend() == "genaid":
        r = httpx.post(f"{GENAID}/stt", json={
            "audio_b64": base64.b64encode(pcm).decode(), "sample_rate": SAMPLE_RATE,
        }, timeout=20)
        r.raise_for_status()
        return r.json().get("text", "").strip()
    # interim local tier (v3 §U9): mlx-whisper on this Mac until HailoRT>=5.2
    import numpy as np

    import mlx_whisper

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    out = mlx_whisper.transcribe(
        audio, path_or_hf_repo=CFG.get("mlx_whisper_model",
                                       "mlx-community/whisper-base-mlx"),
        language="en", fp16=True)
    return (out.get("text") or "").strip()


def mic_loop() -> None:
    import sounddevice as sd
    import webrtcvad

    vad = webrtcvad.Vad(int(CFG.get("vad_aggressiveness", 2)))
    min_utt_frames = int(CFG.get("min_utt_ms", 400)) // FRAME_MS
    silence_end_frames = int(CFG.get("silence_end_ms", 700)) // FRAME_MS
    max_utt_frames = int(CFG.get("max_utt_s", 12)) * 1000 // FRAME_MS
    preroll = collections.deque(maxlen=8)  # ~240 ms before speech onset

    utterance: list[bytes] = []
    voiced_frames = 0
    silence_frames = 0
    in_speech = False

    def flush():
        nonlocal utterance, voiced_frames, silence_frames, in_speech
        frames, was_voiced = utterance, voiced_frames
        utterance, voiced_frames, silence_frames, in_speech = [], 0, 0, False
        if was_voiced < min_utt_frames:
            return  # too short: cough/click
        pcm = b"".join(frames)
        try:
            text = transcribe(pcm)
        except Exception as e:
            log.warning("stt failed: %s", e)
            return
        if text and len(text) > 1:
            log.info("HEARD: %r", text)
            publish_transcript(text, conf=0.9)

    log.info("ears live: mic -> VAD -> 10H whisper (device=%s)", sd.default.device)
    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                           blocksize=FRAME_SAMPLES) as stream:
        while True:
            frame, _ = stream.read(FRAME_SAMPLES)
            frame = bytes(frame)
            if _muted():
                if in_speech:
                    flush()
                preroll.clear()
                continue
            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            if in_speech:
                utterance.append(frame)
                if is_speech:
                    voiced_frames += 1
                    silence_frames = 0
                else:
                    silence_frames += 1
                if silence_frames >= silence_end_frames or len(utterance) >= max_utt_frames:
                    flush()
            elif is_speech:
                in_speech = True
                utterance = list(preroll) + [frame]
                voiced_frames, silence_frames = 1, 0
            else:
                preroll.append(frame)


def main() -> None:
    threading.Thread(target=speaker_loop, daemon=True, name="voice").start()
    while True:  # mic loop restarts on device errors (e.g., permission granted late)
        try:
            mic_loop()
        except Exception:
            log.exception("mic loop crashed; retrying in 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
