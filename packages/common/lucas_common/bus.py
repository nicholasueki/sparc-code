"""Typed MQTT bus. The only file that knows the transport.

Topics carry exactly one Pydantic type each (TOPICS map). QoS 1 everywhere;
state-ish topics retained. A future ROS 2 / zenoh migration replaces this file.
"""
from __future__ import annotations

import json
import logging
import threading
from typing import Callable, Type, TypeVar

import paho.mqtt.client as mqtt
from pydantic import BaseModel

from . import config
from .types import (
    DetectionFrame,
    SoundEvent,
    SpeakRequest,
    SpeechEvent,
    Transcript,
)

log = logging.getLogger("lucas.bus")
T = TypeVar("T", bound=BaseModel)

TOPICS: dict[str, type[BaseModel]] = {
    "lucas/vision/tier0": DetectionFrame,
    "lucas/vision/rich": DetectionFrame,
    "lucas/audio/speech": SpeechEvent,
    "lucas/audio/sound": SoundEvent,
    "lucas/audio/transcript": Transcript,
    "lucas/tts/say": SpeakRequest,
}
RETAINED: set[str] = set()


class Bus:
    def __init__(self, client_id: str, host: str | None = None, port: int | None = None):
        self._host = host or config.get("bus.host", "127.0.0.1")
        self._port = port or int(config.get("bus.port", 1883))
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=client_id, clean_session=True
        )
        self._handlers: dict[str, list[tuple[type[BaseModel], Callable]]] = {}
        self._client.on_message = self._on_message
        self._client.on_connect = self._on_connect
        self._connected = threading.Event()

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._client.connect(self._host, self._port, keepalive=30)
        self._client.loop_start()
        if not self._connected.wait(timeout=10):
            raise ConnectionError(f"MQTT broker unreachable at {self._host}:{self._port}")

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self._connected.set()
        for topic in self._handlers:
            client.subscribe(topic, qos=1)
        log.info("bus connected to %s:%s", self._host, self._port)

    # -- pub/sub -----------------------------------------------------------
    def publish(self, topic: str, msg: BaseModel) -> None:
        expected = TOPICS.get(topic)
        if expected is not None and not isinstance(msg, expected):
            raise TypeError(f"{topic} expects {expected.__name__}, got {type(msg).__name__}")
        self._client.publish(
            topic, msg.model_dump_json(), qos=1, retain=topic in RETAINED
        )

    def publish_json(self, topic: str, payload: dict) -> None:
        """Untyped escape hatch for debug/telemetry topics only.

        Oversized string fields are trimmed so the JSON stays valid.
        """
        line = json.dumps(payload)
        if len(line) > 60000:
            payload = {
                k: (v[:8000] + "…[trimmed]") if isinstance(v, str) and len(v) > 8000 else v
                for k, v in payload.items()
            }
            line = json.dumps(payload)[:60000]
        self._client.publish(topic, line, qos=0, retain=False)

    def subscribe(self, topic: str, model: Type[T], handler: Callable[[T], None]) -> None:
        self._handlers.setdefault(topic, []).append((model, handler))
        if self._connected.is_set():
            self._client.subscribe(topic, qos=1)

    def _on_message(self, client, userdata, message) -> None:
        for model, handler in self._handlers.get(message.topic, []):
            try:
                obj = model.model_validate_json(message.payload)
            except Exception:
                log.exception("bad payload on %s: %.200s", message.topic, message.payload)
                continue
            try:
                handler(obj)
            except Exception:
                log.exception("handler error on %s", message.topic)


def parse(topic: str, payload: bytes) -> BaseModel:
    """Fixture/test helper: parse a payload per the TOPICS registry."""
    model = TOPICS[topic]
    return model.model_validate_json(payload)
