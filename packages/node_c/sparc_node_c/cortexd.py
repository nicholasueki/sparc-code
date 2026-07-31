"""cortexd — Node C cognitive service.

FastAPI wrapper around one resident Ornith model + semantic memory.
One big model call in flight at a time (design LIM-M5-1): a single worker
thread serializes generation; the event loop stays free for /embed, /memory,
/health. Every response carries timing + fallback level for the trace.

Run: python -m sparc_node_c.cortexd
"""
from __future__ import annotations

import base64
import concurrent.futures
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from sparc_common import config
from sparc_common.types import (
    Action,
    MemoryProposal,
    RetrievedFact,
    RetrieveRequest,
    ThinkRequest,
    ThinkResponse,
    UpsertFactRequest,
    VlmAnswer,
    VlmQuery,
)

from . import prompts
from .backends import make_backend
from .memory import SemanticMemory

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("sparc.cortexd")

CFG = config.get("node_c", {})
PERSONA = config.get("personality.persona", "You are SPARC, a companion robot.")

app = FastAPI(title="sparc-cortexd")

# Single generation worker = the queue (depth enforced in the endpoint).
_gen_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="gen")
_inflight = 0

backend = None
memory: SemanticMemory | None = None
_debug_mqtt = None


@app.on_event("startup")
def _startup() -> None:
    global backend, memory
    memory = SemanticMemory(
        CFG["memory"]["db_path"], CFG["memory"]["embed_model"]
    )
    backend = make_backend(CFG)  # blocks ~60 s while the model loads
    log.info("cortexd ready: %s", backend.info())


def _queue_guard() -> None:
    if _inflight >= int(CFG.get("queue_depth", 3)):
        raise HTTPException(503, "cortexd queue saturated", headers={"Retry-After": "2"})


def _generate(system: str, user: str, image_b64=None, max_tokens=700, temperature=0.7) -> str:
    global _inflight
    _inflight += 1
    try:
        fut = _gen_pool.submit(
            backend.generate, system, user, image_b64, max_tokens, temperature
        )
        return fut.result(timeout=90)
    finally:
        _inflight -= 1


# --------------------------------------------------------------------- think

@app.post("/think", response_model=ThinkResponse)
def think(req: ThinkRequest) -> ThinkResponse:
    _queue_guard()
    t0 = time.time()
    user = prompts.build_think_user(req)
    motion_persona = config.get("personality.persona_motion", PERSONA)
    system = req.persona_override or (motion_persona if req.motion else PERSONA)
    base_temp = (req.temperature_override if req.temperature_override is not None
                 else float(CFG["mlx"].get("temperature", 0.7)))
    parsed = None
    raw = ""
    attempts = 0
    for attempt in range(2):  # L1: one retry
        attempts = attempt + 1
        raw = _generate(
            system, user, req.image_b64,
            max_tokens=int(CFG["mlx"].get("max_tokens_think", 700)),
            temperature=base_temp if attempt == 0 else 0.2,
        )
        parsed = prompts.parse_think(raw, req.max_options)
        if parsed:
            break
    gen_ms = int((time.time() - t0) * 1000)

    if not parsed:
        # L1 exhausted -> complete, safe partial (ask), never an error to the caller
        log.warning("think unparseable after %d attempts: %.200s", attempts, raw)
        from sparc_common.types import OptionMeta, OptionSet

        ask = OptionMeta(idx=0, action="ask_user",
                         args={"text": "I noticed something but I'm not sure what to do — any thoughts?"})
        wait = OptionMeta(idx=1, action="wait")
        return ThinkResponse(
            options=OptionSet(options=[ask, wait]), choice=1, backup=0,
            why="model output unparseable; deterministic fallback",
            action=prompts.option_to_action(wait, "fallback", fallback_level=2),
            timing_ms={"generate": gen_ms}, fallback_level=2,
        )

    options, choice, backup, why = parsed
    action = prompts.option_to_action(options.options[choice], why)
    import re as _re

    think_match = _re.search(r"<think>(.*?)</think>", raw, _re.DOTALL)
    return ThinkResponse(
        options=options, choice=choice, backup=backup, why=why, action=action,
        timing_ms={"generate": gen_ms, "attempts": attempts,
                   "prompt_chars": len(system) + len(user),
                   "raw_chars": len(raw)},
        fallback_level=0 if attempts == 1 else 1,
        thinking=(think_match.group(1).strip() if think_match else ""),
    )


# ----------------------------------------------------------------- vlm query

@app.post("/vlm_query", response_model=VlmAnswer)
def vlm_query(req: VlmQuery) -> VlmAnswer:
    _queue_guard()
    raw = _generate(prompts.VLM_SYSTEM, req.question, req.image_b64,
                    max_tokens=200, temperature=0.1)
    data = prompts.extract_json(raw) or {}
    return VlmAnswer(
        answer=str(data.get("answer", raw.strip()[:200])),
        confidence=float(data.get("confidence", 0.3)),
    )


# -------------------------------------------------------------------- memory

@app.post("/memory/upsert")
def memory_upsert(req: UpsertFactRequest) -> dict:
    memory.upsert(req.fact_id, req.statement)
    return {"ok": True, "count": memory.count()}


@app.post("/memory/retrieve", response_model=list[RetrievedFact])
def memory_retrieve(req: RetrieveRequest) -> list[RetrievedFact]:
    mcfg = CFG["memory"]
    hits = memory.retrieve(req.situation, req.k or mcfg["vector_k"], mcfg["min_score"])
    return [RetrievedFact(fact_id=f, statement=s, score=sc) for f, s, sc in hits]


class EmbedRequest(BaseModel):
    texts: list[str]


@app.post("/embed")
def embed(req: EmbedRequest) -> dict:
    return {"vectors": [memory.embed_one(t) for t in req.texts]}


class DistillRequest(BaseModel):
    episodes: list[str]


@app.post("/distill")
def distill(req: DistillRequest) -> dict:
    _queue_guard()
    user = "EVENT LOG:\n" + "\n".join(req.episodes[-100:])
    raw = _generate(prompts.DISTILL_SYSTEM, user, max_tokens=500, temperature=0.2)
    data = prompts.extract_json(raw) or {}
    proposals = []
    for p in data.get("proposals", []):
        try:
            proposals.append(MemoryProposal.model_validate(p).model_dump())
        except Exception:
            pass
    return {"proposals": proposals}


@app.get("/health")
def health() -> dict:
    return {
        "ok": backend is not None,
        "backend": backend.info() if backend else None,
        "inflight": _inflight,
        "facts": memory.count() if memory else 0,
    }


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(CFG.get("cortexd_port", 8800)))


if __name__ == "__main__":
    main()
