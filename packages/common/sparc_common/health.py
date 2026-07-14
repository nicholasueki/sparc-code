"""Periodic typed health publication for daemons without an HTTP health API."""
from __future__ import annotations

import importlib.metadata
import logging
import threading
from collections.abc import Callable
from typing import Any

from .bus import Bus
from .types import ServiceHealth

log = logging.getLogger("sparc.health")

HealthProbe = Callable[[], dict[str, Any]]


def runtime_version() -> str:
    try:
        return importlib.metadata.version("sparc-robot")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class HealthReporter:
    """Refresh a retained health record so a dead daemon becomes stale."""

    def __init__(
        self,
        bus: Bus,
        service: str,
        probe: HealthProbe,
        *,
        interval_s: float = 5.0,
    ) -> None:
        self.bus = bus
        self.service = service
        self.probe = probe
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _message(self) -> ServiceHealth:
        try:
            state = self.probe()
            details = dict(state.get("details", {}))
            ready = bool(state.get("ready", False))
            reason = state.get("failure_reason")
            if not ready and not reason:
                reason = "one or more functional readiness checks failed"
            return ServiceHealth(
                service=self.service,
                status=state.get("status", "ready" if ready else "degraded"),
                ready=ready,
                version=runtime_version(),
                details=details,
                failure_reason=reason,
            )
        except Exception as exc:
            log.exception("%s health probe failed", self.service)
            return ServiceHealth(
                service=self.service,
                status="failed",
                ready=False,
                version=runtime_version(),
                failure_reason=f"health probe failed: {type(exc).__name__}: {exc}",
            )

    def publish(self) -> ServiceHealth:
        message = self._message()
        self.bus.publish(f"sparc/health/{self.service}", message)
        return message

    def start(self) -> None:
        if self._thread is not None:
            return
        self.publish()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name=f"{self.service}-health",
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.publish()
            except Exception:
                log.exception("%s health publish failed", self.service)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s + 1)
