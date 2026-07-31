"""Model backends for cortexd.

MLXBackend: in-process mlx-vlm (primary — measured 58 tok/s decode on M1 Max).
OpenAICompatBackend: any OpenAI-compatible server (llama-server, LM Studio).
Both expose: generate(prompt, image_b64, max_tokens, temperature) -> str.
Serialization to one in-flight call is handled by cortexd, not here.
"""
from __future__ import annotations

import base64
import io
import logging
import tempfile
import time
from typing import Optional, Protocol

log = logging.getLogger("sparc.backend")


class ModelBackend(Protocol):
    def generate(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
        max_tokens: int = 700,
        temperature: float = 0.7,
    ) -> str: ...

    def info(self) -> dict: ...


class MLXBackend:
    """In-process mlx-vlm. Loads once at boot (~60 s), stays resident."""

    def __init__(self, model_path: str):
        t0 = time.time()
        from mlx_vlm import generate as _generate
        from mlx_vlm import load as _load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        self._generate = _generate
        self._apply_chat_template = apply_chat_template
        self.model, self.processor = _load(model_path)
        self.config = load_config(model_path)
        self.model_path = model_path
        self.load_s = round(time.time() - t0, 1)
        log.info("MLX model resident in %.1fs: %s", self.load_s, model_path)

    def generate(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
        max_tokens: int = 700,
        temperature: float = 0.7,
    ) -> str:
        images: list[str] = []
        tmp = None
        if image_b64:
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(base64.b64decode(image_b64))
            tmp.flush()
            images = [tmp.name]

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self._apply_chat_template(
            self.processor, self.config, messages, num_images=len(images)
        )
        result = self._generate(
            self.model,
            self.processor,
            prompt,
            image=images or None,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        text = getattr(result, "text", None)
        if text is None:  # older mlx-vlm returns str
            text = str(result)
        if tmp is not None:
            import os

            os.unlink(tmp.name)
        return text

    def info(self) -> dict:
        return {"backend": "mlx", "model": self.model_path, "load_s": self.load_s}


class OpenAICompatBackend:
    """llama-server / LM Studio / Ollama behind one URL."""

    def __init__(self, base_url: str, model: str):
        import httpx

        self._client = httpx.Client(base_url=base_url, timeout=120)
        self.model = model
        self.base_url = base_url

    def generate(
        self,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
        max_tokens: int = 700,
        temperature: float = 0.7,
    ) -> str:
        content: list | str
        if image_b64:
            content = [
                {"type": "text", "text": user},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ]
        else:
            content = user
        r = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def info(self) -> dict:
        return {"backend": "openai_compat", "model": self.model, "url": self.base_url}


def make_backend(cfg: dict) -> ModelBackend:
    kind = cfg.get("backend", "mlx")
    if kind == "mlx":
        return MLXBackend(cfg["mlx"]["model_path"])
    if kind == "openai_compat":
        oc = cfg["openai_compat"]
        return OpenAICompatBackend(oc["base_url"], oc["model"])
    raise ValueError(f"unknown backend: {kind}")
