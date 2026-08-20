"""OpenAI-compatible client for the grader and researcher tiers.

The subject model (Ornith-35B on Node C) is never called through here — it runs
locally behind cortexd and is frozen. This client is only for the models that
*judge* and *reason about* the experiment.

Works against anything speaking /chat/completions: a local llama.cpp / vLLM /
LM Studio / Ollama server on the DGX, or a hosted gateway like OpenRouter.
The only differences are `base_url` and whether an API key is required, both of
which come from config.

Deliberately small: chat completions, defensive JSON extraction, bounded retries,
and a thread-pooled `map` because a slow local model makes the researcher tier
latency-bound rather than cost-bound.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

import httpx

log = logging.getLogger("evo.llm")

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}
T = TypeVar("T")
R = TypeVar("R")


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    calls: int = 0
    seconds: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost += other.cost
        self.calls += other.calls
        self.seconds += other.seconds


@dataclass
class LLMClient:
    """One model endpoint.

    base_url is the OpenAI-compatible root, with or without a trailing /v1:
      DGX (LM Studio / vLLM / llama.cpp)   http://10.0.0.9:1234/v1
      Ollama                               http://10.0.0.9:11434/v1
      OpenRouter                           https://openrouter.ai/api/v1
    """

    model: str
    base_url: str
    api_key: str = ""
    api_key_env: str = ""
    # Local 27B-class models on modest hardware can take minutes on a 20k-token
    # postmortem prompt, so the default ceiling is generous rather than snappy.
    timeout: float = 600.0
    max_retries: int = 4
    temperature: float = 0.2
    # Three independent capabilities, negotiated per server rather than assumed.
    # LM Studio, for one, accepts json_schema and REJECTS bare json_object
    # ("'response_format.type' must be 'json_schema' or 'text'"), so a fixed
    # schema->object->text ladder degrades in the wrong direction for it.
    json_mode: bool = True          # send response_format at all
    schema_mode: bool = True        # server accepts {"type": "json_schema"}
    json_object_mode: bool = True   # server accepts {"type": "json_object"}
    max_concurrency: int = 4
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        if not self.api_key and self.api_key_env:
            self.api_key = os.environ.get(self.api_key_env, "")
        if not self.base_url:
            raise LLMError("base_url is required (see evolution.researcher in sparc.yaml)")
        self.base_url = self.base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        if self.requires_key and not self.api_key:
            raise LLMError(
                f"{self.base_url} looks like a hosted gateway but no API key was found"
                + (f" in ${self.api_key_env}" if self.api_key_env else ""))
        self._client = httpx.Client(timeout=self.timeout)

    @property
    def requires_key(self) -> bool:
        """Local servers on a private address accept anything; hosted ones don't."""
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        private = (host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
                   or host.startswith(("10.", "192.168.", "100.64.", "100."))
                   or re.match(r"^172\.(1[6-9]|2\d|3[01])\.", host) is not None
                   or host.endswith(".local"))
        return not private

    # ------------------------------------------------------------------ core

    def complete(self, system: str, user: str, *, json_out: bool = False,
                 max_tokens: int = 2000, temperature: float | None = None,
                 schema: dict[str, Any] | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        def apply_format() -> None:
            """Pick the strongest output constraint this server still accepts.

            Grammar-constrained json_schema is strictly stronger than json_object:
            the model cannot emit a missing key or an invalid enum in the first
            place. That matters most for mutation, where a malformed gene costs a
            whole retry cycle.
            """
            payload.pop("response_format", None)
            if not (json_out and self.json_mode):
                return
            if schema and self.schema_mode:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "response", "strict": True,
                                    "schema": schema},
                }
            elif self.json_object_mode:
                payload["response_format"] = {"type": "json_object"}

        apply_format()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers.setdefault("HTTP-Referer", "https://github.com/sparc-rover")
            headers.setdefault("X-Title", "SPARC memory-prompt evolution")

        last_err: Exception | None = None
        attempt = 0
        renegotiations = 0
        while attempt < self.max_retries:
            t0 = time.time()
            try:
                r = self._client.post(f"{self.base_url}/chat/completions",
                                      json=payload, headers=headers)
                if r.status_code == 400 and "response_format" in r.text:
                    # Disable only the rejected mode, then re-pick. Capability
                    # negotiation is not a failed attempt — it must not consume the
                    # retry budget, or a client with max_retries=1 (the preflight)
                    # can never recover from the very first renegotiation.
                    rejected = (payload.get("response_format") or {}).get("type")
                    if rejected == "json_schema":
                        self.schema_mode = False
                    elif rejected == "json_object":
                        self.json_object_mode = False
                    before = payload.get("response_format")
                    apply_format()
                    if payload.get("response_format") == before:
                        self.json_mode = False       # nothing left to try
                        apply_format()
                    log.warning("%s rejected %s; using %s", self.model, rejected,
                                (payload.get("response_format") or {}).get("type", "text"))
                    renegotiations += 1
                    if renegotiations <= 3:
                        continue
                    attempt += 1
                    continue
                if r.status_code in RETRY_STATUS:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                data = r.json()
                if "error" in data and not data.get("choices"):
                    raise LLMError(str(data["error"])[:300])
                u = data.get("usage") or {}
                self.usage.add(Usage(int(u.get("prompt_tokens", 0) or 0),
                                     int(u.get("completion_tokens", 0) or 0),
                                     float(u.get("cost", 0.0) or 0.0),
                                     calls=1, seconds=time.time() - t0))
                msg = data["choices"][0].get("message", {})
                # Reasoning models may put the answer in reasoning_content when the
                # visible content is empty.
                return msg.get("content") or msg.get("reasoning_content") or ""
            except Exception as e:  # noqa: BLE001 — retry policy is status-agnostic
                last_err = e
                attempt += 1
                if attempt >= self.max_retries:
                    break
                sleep = min(30.0, 2.0 ** attempt) + random.uniform(0, 1.0)
                log.warning("llm attempt %d/%d failed (%s); retry in %.1fs",
                            attempt + 1, self.max_retries, e, sleep)
                time.sleep(sleep)
        raise LLMError(
            f"all {self.max_retries} attempts failed: "
            + (str(last_err) if last_err else
               "server rejected every response_format we offered "
               f"(schema={self.schema_mode}, json_object={self.json_object_mode})"))

    def complete_json(self, system: str, user: str, **kw: Any) -> dict:
        raw = self.complete(system, user, json_out=True, **kw)
        obj = extract_json(raw)
        if obj is None:
            raise LLMError(f"model did not return parseable JSON: {raw[:300]}")
        return obj

    def map(self, fn: Callable[[T], R], items: Iterable[T]) -> list[R]:
        """Run independent calls concurrently.

        The researcher tier is ~20 mutually independent postmortems per generation.
        Against a slow local model that is the difference between minutes and an hour
        added to every generation, so it is worth the small amount of machinery.
        Exceptions propagate per item rather than sinking the batch.
        """
        items = list(items)
        if len(items) <= 1 or self.max_concurrency <= 1:
            return [fn(i) for i in items]
        out: list[Any] = [None] * len(items)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(self.max_concurrency, len(items))) as pool:
            futs = {pool.submit(fn, item): idx for idx, item in enumerate(items)}
            for fut in concurrent.futures.as_completed(futs):
                idx = futs[fut]
                try:
                    out[idx] = fut.result()
                except Exception as e:  # noqa: BLE001
                    log.error("concurrent call %d failed: %s", idx, e)
                    out[idx] = e
        return out

    def health(self) -> dict:
        """Cheap reachability + model-availability probe for preflight.

        Short connect timeout on purpose: the generation timeout is minutes because
        a local 27B is slow, but an unreachable *address* should fail in seconds so
        preflight stays a fast check rather than a ten-minute block.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self._client.get(f"{self.base_url}/models", headers=headers,
                             timeout=httpx.Timeout(20.0, connect=5.0))
        r.raise_for_status()
        ids = [m.get("id") for m in (r.json().get("data") or [])]
        return {"reachable": True, "n_models": len(ids),
                "model_listed": self.model in ids if ids else "unknown",
                "sample": ids[:4]}

    def close(self) -> None:
        self._client.close()


# Back-compat alias: the harness used to be OpenRouter-only.
OpenRouterClient = LLMClient
OpenRouterError = LLMError


# --------------------------------------------------------------- json salvage

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """First balanced {...} after stripping think blocks and fences.

    Same defensive shape as sparc_node_c.prompts.extract_json. This matters more
    with a local model than with a hosted one: Qwen-class models emit <think>
    blocks and prose preambles often enough that strict json.loads on the whole
    body is not safe.
    """
    text = _THINK_RE.sub("", text or "")
    # An unterminated <think> (hit max_tokens mid-reasoning) would otherwise leave
    # the whole body looking like prose.
    if "<think>" in text and "</think>" not in text:
        text = text.split("<think>", 1)[0]
    text = text.replace("```json", "```")
    if "```" in text:
        for chunk in text.split("```"):
            obj = _balanced(chunk)
            if obj is not None:
                return obj
    return _balanced(text)


def _balanced(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None
