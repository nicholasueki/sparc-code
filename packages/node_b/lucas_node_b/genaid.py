"""genaid — Node B model service on the native HailoRT GenAI API.

Single neural core => one lock serializes every model call (LIM-SYS-1).
Selector = single-letter greedy decode (v3 §U2): tiny prompt, max 3 tokens,
parse-clamp; salience = closed set {S,N,E}. STT wires in when the Whisper HEF
is downloaded (returns 503 until then -> orchestrator L2 fallbacks handle it).

Run: python -m lucas_node_b.genaid
"""
from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from lucas_common import config
from lucas_common.types import SalienceDecision, SalienceRequest, SelectorRequest, Selection

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("lucas.genaid")

CFG = config.get("node_b", {})
GROUP_ID = "SHARED"  # hailo-apps SHARED_VDEVICE_GROUP_ID — do not change

app = FastAPI(title="lucas-genaid")
_core_lock = threading.Lock()  # the single neural core

llm = None
vdevice = None


def _msg(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


@app.on_event("startup")
def _startup() -> None:
    global llm, vdevice
    hef = Path(CFG.get("hef_dir", "")) / CFG.get("llm_hef", "")
    if not hef.exists():
        log.warning("LLM HEF missing (%s) — /select and /salience will 503", hef)
        return
    from hailo_platform import VDevice
    from hailo_platform.genai import LLM

    params = VDevice.create_params()
    params.group_id = GROUP_ID
    vdevice = VDevice(params)
    t0 = time.time()
    llm = LLM(vdevice, str(hef))
    log.info("10H LLM resident in %.1fs: %s", time.time() - t0, hef.name)


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
             f"You are Lucas's instinct. Persona: {req.persona_line}. Mood: {req.mood}. "
             "Pick the option that fits Lucas best. Answer with ONE letter only."),
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


@app.post("/stt")
def stt() -> dict:
    raise HTTPException(503, "Whisper HEF not installed yet (P3)")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "llm_loaded": llm is not None,
            "hef": str(Path(CFG.get("hef_dir", "")) / CFG.get("llm_hef", ""))}


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(CFG.get("genaid_port", 8700)))


if __name__ == "__main__":
    main()
