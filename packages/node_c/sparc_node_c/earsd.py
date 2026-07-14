"""earsd — interim ears + voice on the tokenator Mac (v0.3-interim).

mic -> webrtcvad utterance gating -> genaid /stt (Whisper on the Hailo-10H,
the designed path) -> sparc/audio/transcript on the bus -> orchestrator thinks
-> sparc/tts/say -> spoken through the Mac with `say`.

Echo control is half-duplex: the mic is gated while SPARC speaks (+ a tail),
so he never transcribes himself. When the USB speakerphone arrives, this whole
file's job moves to Node B (Silero + openWakeWord + Piper per the design);
genaid /stt stays exactly as-is.

Run: python -m sparc_node_c.earsd  (needs macOS mic permission the first time)
"""
from __future__ import annotations

import base64
import collections
import importlib.util
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import httpx

from sparc_common import config
from sparc_common.bus import Bus
from sparc_common.health import HealthReporter
from sparc_common.types import SpeakRequest, Transcript

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("sparc.earsd")

CFG = config.get("node_c.ears", {})
GENAID = config.get("endpoints.genaid", "http://robot-genai.local:8700")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480

_speaking_until = 0.0  # half-duplex gate (epoch)
_mic_open = False
_voice_mqtt_ready = False


def _muted() -> bool:
    return time.time() < _speaking_until


# ------------------------------------------------------------------ voice out

def speaker_loop() -> None:
    """Subscribe to sparc/tts/say; speak via macOS `say`; gate the mic meanwhile."""
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

    def on_connect(client, userdata, flags, reason_code, properties):
        global _voice_mqtt_ready
        if reason_code == 0:
            client.subscribe("sparc/tts/say", qos=1)
            _voice_mqtt_ready = True

    def on_disconnect(client, *a, **k):
        global _voice_mqtt_ready
        _voice_mqtt_ready = False
        log.warning("tts bus disconnected; reconnecting")
        while True:
            try:
                client.reconnect()
                return
            except Exception:
                time.sleep(3)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="earsd-voice")
    client.on_message = on_message
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(config.get("bus.host"), int(config.get("bus.port", 1883)), keepalive=30)
    client.loop_forever(retry_first_connection=True)


# ------------------------------------------------------------------- ears in

def publish_transcript(text: str, conf: float) -> None:
    import json

    import paho.mqtt.publish as mqtt_publish

    tr = Transcript(text=text, conf=conf)
    mqtt_publish.single(
        "sparc/audio/transcript", tr.model_dump_json(), qos=1,
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
    # hallucination gates (standard whisper thresholds): drop segments the model
    # itself flags as probably-not-speech / degenerate repetition
    kept = []
    for seg in out.get("segments", []):
        if seg.get("no_speech_prob", 0) > 0.5:
            continue
        if seg.get("avg_logprob", 0) < -1.0:
            continue
        if seg.get("compression_ratio", 1) > 2.4:
            continue
        kept.append(seg.get("text", ""))
    return " ".join(kept).strip()


def _looks_hallucinated(text: str) -> bool:
    """Final text-level guard: degenerate repetition ('paid paid paid...')."""
    words = text.lower().split()
    if len(words) >= 6:
        run = 1
        for a, b in zip(words, words[1:]):
            run = run + 1 if a == b else 1
            if run >= 4:
                return True
        if len(set(words)) <= max(2, len(words) // 6):
            return True
    return False


def mic_loop() -> None:
    global _mic_open
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
        if not text or len(text) <= 1:
            return
        if _looks_hallucinated(text):
            log.info("dropped (hallucination): %.60r", text)
            return
        # interim wake gate: without openWakeWord, only address-by-name reaches
        # SPARC — otherwise he answers the TV, music, and passing conversation
        if CFG.get("require_name", True) and "sparc" not in text.lower():
            log.info("dropped (not addressed to SPARC): %.60r", text)
            return
        log.info("HEARD: %r", text)
        publish_transcript(text, conf=0.9)

    log.info("ears live: mic -> VAD -> 10H whisper (device=%s)", sd.default.device)
    try:
        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                               blocksize=FRAME_SAMPLES) as stream:
            _mic_open = True
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
    finally:
        _mic_open = False


def health(bus: Bus) -> dict:
    backend = _resolve_stt_backend()
    stt_ready = False
    stt_reason = None
    if backend == "genaid":
        try:
            response = httpx.get(f"{GENAID}/health", timeout=4)
            response.raise_for_status()
            stt_ready = bool(response.json().get("stt_loaded"))
            if not stt_ready:
                stt_reason = "genaid selected but Whisper is not loaded"
        except Exception as exc:
            stt_reason = f"genaid health failed: {type(exc).__name__}: {exc}"
    elif backend == "mlx_local":
        mlx_model = CFG.get("mlx_whisper_model", "mlx-community/whisper-base-mlx")
        stt_ready = (importlib.util.find_spec("mlx_whisper") is not None and
                     _mlx_model_present(mlx_model))
        if not stt_ready:
            stt_reason = "mlx_local selected but its module or model setting is missing"
    else:
        stt_reason = f"unsupported STT backend: {backend}"

    details = {
        "mic_open": _mic_open,
        "tts_ready": shutil.which("say") is not None,
        "mqtt_ready": bus.connected and _voice_mqtt_ready,
        "stt_backend": backend,
        "stt_ready": stt_ready,
        "stt_model": (CFG.get("mlx_whisper_model", "mlx-community/whisper-base-mlx")
                      if backend == "mlx_local"
                      else config.get("node_b.stt_hef")),
    }
    ready = bool(details["mic_open"] and details["tts_ready"] and
                 details["mqtt_ready"] and details["stt_ready"])
    missing = [name for name in ("mic_open", "tts_ready", "mqtt_ready", "stt_ready")
               if not details[name]]
    reason = None if ready else f"not ready: {', '.join(missing)}"
    if stt_reason:
        reason = f"{reason}; {stt_reason}" if reason else stt_reason
    return {"ready": ready, "details": details, "failure_reason": reason}


def _mlx_model_present(model: str) -> bool:
    """Require provisioned local weights; a configured download name is not ready."""
    path = Path(model).expanduser()
    if path.exists():
        return True
    if "/" not in model:
        return False
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    cached = hf_home / "hub" / f"models--{model.replace('/', '--')}" / "snapshots"
    return cached.is_dir() and any(cached.iterdir())


def _forever(fn, name: str) -> None:
    while True:  # every loop survives transient failures (broker down, device busy)
        try:
            fn()
        except Exception:
            log.exception("%s crashed; retrying in 5s", name)
            time.sleep(5)


def main() -> None:
    health_bus = Bus(client_id="earsd-health")
    health_bus.start()
    HealthReporter(health_bus, "earsd", lambda: health(health_bus)).start()
    threading.Thread(target=_forever, args=(speaker_loop, "voice"),
                     daemon=True, name="voice").start()
    _forever(mic_loop, "ears")


if __name__ == "__main__":
    main()
