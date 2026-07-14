"""genaid — Node B model service on the native HailoRT GenAI API.

Single neural core => one lock serializes every model call (LIM-SYS-1).
Selector = single-letter greedy decode (v3 §U2): tiny prompt, max 3 tokens,
parse-clamp; salience = closed set {S,N,E}. STT wires in when the Whisper HEF
is downloaded (returns 503 until then -> orchestrator L2 fallbacks handle it).

Run: python -m sparc_node_b.genaid
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sparc_common import config
from sparc_common.health import runtime_version
from sparc_common.types import SalienceDecision, SalienceRequest, SelectorRequest, Selection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("sparc.genaid")

CFG = config.get("node_b", {})
GROUP_ID = "SHARED"  # hailo-apps SHARED_VDEVICE_GROUP_ID — do not change

app = FastAPI(title="sparc-genaid")
_core_lock = threading.Lock()  # the single neural core

llm = None
stt = None
vdevice = None
vdevice_stt = None


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


@app.on_event("startup")
def _startup() -> None:
    global llm, stt, vdevice, vdevice_stt
    from hailo_platform import VDevice

    def shared_vdevice():
        # one VDevice handle PER model, all in the SHARED group: this is how
        # co-residency works on the 10H (sharing one handle raises status 61)
        params = VDevice.create_params()
        params.group_id = GROUP_ID
        return VDevice(params)

    hef = Path(CFG.get("hef_dir", "")) / CFG.get("llm_hef", "")
    if hef.exists():
        from hailo_platform.genai import LLM

        t0 = time.time()
        vdevice = shared_vdevice()
        llm = LLM(vdevice, str(hef))
        log.info("10H LLM resident in %.1fs: %s", time.time() - t0, hef.name)
    else:
        log.warning("LLM HEF missing (%s) — /select and /salience will 503", hef)

    whisper_hef = Path(CFG.get("hef_dir", "")) / CFG.get("stt_hef", "Whisper-Base.hef")
    if whisper_hef.exists():
        from hailo_platform.genai import Speech2Text

        t0 = time.time()
        try:
            vdevice_stt = shared_vdevice()
            stt = Speech2Text(vdevice_stt, str(whisper_hef))
            log.info("10H Whisper resident in %.1fs (co-resident with LLM: %s)",
                     time.time() - t0, llm is not None)
        except Exception:
            log.exception("Whisper load failed — /stt will 503, LLM unaffected")
    else:
        log.warning("Whisper HEF missing (%s) — /stt will 503", whisper_hef)


def _require_llm() -> None:
    if llm is None:
        raise HTTPException(503, "LLM model not loaded on 10H")


def _gen(messages: list[dict], max_tokens: int, temperature: float = 0.0) -> str:
    with _core_lock:
        try:
            out = llm.generate_all(
                prompt=messages,
                temperature=max(temperature, 0.01),
                seed=7,
                max_generated_tokens=max_tokens,
            )
            return str(out)
        finally:
            try:
                llm.clear_context()
            except Exception:
                log.warning("clear_context failed", exc_info=True)


LETTERS = "ABCDEFGH"


@app.post("/select", response_model=Selection)
def select(req: SelectorRequest) -> Selection:
    _require_llm()
    n = len(req.options)
    if n == 0:
        raise HTTPException(400, "no options")
    menu = "\n".join(f"{LETTERS[i]}) {opt[:90]}" for i, opt in enumerate(req.options[:8]))
    messages = [
        _msg("system",
             f"You are SPARC's instinct. Persona: {req.persona_line}. Mood: {req.mood}. "
             "Pick the option that fits SPARC best. Answer with ONE letter only."),
        _msg("user", f"{menu}\nAnswer:"),
    ]
    t0 = time.time()
    raw = _gen(messages, max_tokens=int(CFG.get("selector", {}).get("max_tokens", 3)))
    m = re.search(r"[A-H]", raw.upper())
    idx = LETTERS.index(m.group(0)) if m else 0
    idx = idx if idx < n else 0
    log.info("select: %r -> %d (%.2fs)", raw[:20], idx, time.time() - t0)
    return Selection(choice=idx, backup=(idx + 1) % n, raw=raw[:40], source="h10_selector")


@app.post("/salience", response_model=SalienceDecision)
def salience(req: SalienceRequest) -> SalienceDecision:
    _require_llm()
    messages = [
        _msg("system",
             "Classify how much attention a home robot should pay to an event. "
             "Answer ONE letter: S = skip, N = note quietly, E = needs attention now."),
        _msg("user", f"Event: {req.event_line}\nContext: {req.context_line}\nAnswer:"),
    ]
    raw = _gen(messages, max_tokens=3)
    m = re.search(r"[SNE]", raw.upper())
    verdict = {"S": "SKIP", "N": "NOTE", "E": "ESCALATE"}.get(m.group(0) if m else "N", "NOTE")
    return SalienceDecision(verdict=verdict, raw=raw[:40])


class SttRequest(BaseModel):
    audio_b64: str          # 16 kHz mono WAV (int16) or raw int16 PCM, base64
    sample_rate: int = 16000
    language: str = "en"


@app.post("/stt")
def stt_endpoint(req: SttRequest) -> dict:
    if stt is None:
        raise HTTPException(503, "Whisper model not loaded on 10H")
    import base64
    import io
    import wave

    import numpy as np
    from hailo_platform.genai import Speech2TextTask

    raw = base64.b64decode(req.audio_b64)
    if raw[:4] == b"RIFF":  # WAV container
        with wave.open(io.BytesIO(raw), "rb") as w:
            if w.getframerate() != 16000 or w.getnchannels() != 1:
                raise HTTPException(400, "need 16 kHz mono WAV")
            raw = w.readframes(w.getnframes())
    audio = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0).astype("<f4")
    if audio.size < 1600:  # <0.1 s: silence guard, don't poke Whisper
        return {"text": "", "segments": 0, "ms": 0}
    t0 = time.time()
    with _core_lock:  # serial neural core (LIM-SYS-1)
        segments = stt.generate_all_segments(
            audio_data=audio,
            task=Speech2TextTask.TRANSCRIBE,
            language=req.language,
            timeout_ms=15000,
        )
    text = "".join(s.text for s in (segments or [])).strip()
    ms = int((time.time() - t0) * 1000)
    log.info("stt: %.1fs audio -> %r (%d ms)", audio.size / 16000, text[:60], ms)
    return {"text": text, "segments": len(segments or []), "ms": ms}


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "version": runtime_version(),
        "llm_loaded": llm is not None,
        "stt_loaded": stt is not None,
        "llm_hef": str(Path(CFG.get("hef_dir", "")) / CFG.get("llm_hef", "")),
        "stt_hef": str(Path(CFG.get("hef_dir", "")) / CFG.get("stt_hef", "")),
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(CFG.get("genaid_port", 8700)))


if __name__ == "__main__":
    main()
