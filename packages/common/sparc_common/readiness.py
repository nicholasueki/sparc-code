"""Profile-aware fleet readiness verifier used by scripts/verify_capability.sh."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import paho.mqtt.client as mqtt

from . import config
from .types import ServiceHealth


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Evaluation:
    ok: bool
    checks: list[dict[str, Any]]


def evaluate_profile(
    profile: str,
    health: dict[str, ServiceHealth],
    *,
    broker_connected: bool,
    cortex: dict[str, Any] | None,
    genaid: dict[str, Any] | None,
    now: float | None = None,
    stale_s: float = 15.0,
) -> Evaluation:
    """Evaluate one sample. Pure and hardware-free so every gate is unit-testable."""
    if profile not in {"v0.3", "v0.4", "active"}:
        raise ValueError(f"unknown capability profile: {profile}")
    now = time.time() if now is None else now
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, observed: Any = None) -> None:
        checks.append({"name": name, "ok": bool(ok), "observed": observed})

    check("broker.reachable", broker_connected, broker_connected)

    def service(name: str, required_details: tuple[str, ...]) -> ServiceHealth | None:
        report = health.get(name)
        check(f"{name}.present", report is not None, report.model_dump() if report else None)
        if report is None:
            return None
        age = now - report.ts
        check(f"{name}.fresh", -5 <= age <= stale_s, round(age, 3))
        check(f"{name}.service", report.service == name, report.service)
        check(f"{name}.ready", report.ready and report.status == "ready",
              {"ready": report.ready, "status": report.status,
               "failure_reason": report.failure_reason})
        check(f"{name}.version", report.version not in {"", "unknown"}, report.version)
        for field in required_details:
            check(f"{name}.{field}", report.details.get(field) is True,
                  report.details.get(field))
        return report

    wants_v03 = profile in {"v0.3", "active"}
    wants_v04 = profile in {"v0.4", "active"}
    orchestrator_fields = ["db_ready", "mqtt_ready"]
    if wants_v03:
        orchestrator_fields.append("bounded_session_runtime")
    service("orchestrator", tuple(orchestrator_fields))

    cortex = cortex or {}
    check("cortexd.reachable", "_error" not in cortex and bool(cortex),
          cortex.get("_error"))
    check("cortexd.backend_ready", bool(cortex.get("ok") and cortex.get("backend")),
          cortex)
    check("cortexd.version", cortex.get("version") not in {None, "", "unknown"},
          cortex.get("version"))

    if wants_v03:
        ears = service("earsd", ("mic_open", "tts_ready", "mqtt_ready", "stt_ready"))
        if ears is not None:
            backend = ears.details.get("stt_backend")
            check("earsd.stt_backend", backend in {"mlx_local", "genaid"}, backend)
            if backend == "genaid":
                genaid = genaid or {}
                check("genaid.reachable", "_error" not in genaid and bool(genaid),
                      genaid.get("_error"))
                check("genaid.stt_loaded", genaid.get("stt_loaded") is True,
                      genaid.get("stt_loaded"))
                check("genaid.version", genaid.get("version") not in {None, "", "unknown"},
                      genaid.get("version"))

    if wants_v04:
        service("tripwire", ("mqtt_ready", "camera_ready", "frame_path_ready",
                             "model_present"))
        service("enrich", ("mqtt_ready", "scrfd_loaded", "arcface_loaded"))

    return Evaluation(ok=all(item["ok"] for item in checks), checks=checks)


class HealthCollector:
    def __init__(self, host: str, port: int) -> None:
        self._lock = threading.Lock()
        self._reports: dict[str, ServiceHealth] = {}
        self._connected = threading.Event()
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"sparc-readiness-{os.getpid()}",
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=5)
        self._client.connect_async(host, port, keepalive=15)

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code == 0:
            self._connected.set()
            client.subscribe("sparc/health/+", qos=1)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code,
                       properties) -> None:
        self._connected.clear()

    def _on_message(self, client, userdata, message) -> None:
        try:
            report = ServiceHealth.model_validate_json(message.payload)
        except Exception:
            return
        expected = message.topic.rsplit("/", 1)[-1]
        if report.service != expected:
            return
        with self._lock:
            self._reports[report.service] = report

    def start(self) -> None:
        self._client.loop_start()

    def stop(self) -> None:
        self._client.disconnect()
        self._client.loop_stop()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def snapshot(self) -> dict[str, ServiceHealth]:
        with self._lock:
            return dict(self._reports)


def _http_health(url: str) -> dict[str, Any]:
    try:
        response = httpx.get(f"{url.rstrip('/')}/health", timeout=4)
        response.raise_for_status()
        body = response.json()
        return body if isinstance(body, dict) else {"_error": "non-object response"}
    except Exception as exc:
        return {"_error": f"{type(exc).__name__}: {exc}"}


def _repo_root() -> Path:
    explicit = os.environ.get("SPARC_REPO")
    if explicit:
        return Path(explicit).resolve()
    for parent in Path.cwd(), *Path.cwd().parents:
        if (parent / "config" / "sparc.yaml").is_file():
            return parent
    raise RuntimeError("run the verifier from a SPARC checkout or set SPARC_REPO")


def _run_metadata(profile: str, repo: Path, config_path: Path) -> dict[str, Any]:
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("fleet evidence requires a clean integrated Git checkout")
    git_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cfg = config.load()
    return {
        "kind": "run",
        "profile": profile,
        "git_sha": git_sha,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "host_topology": {
            "node_a": {"broker": f"{cfg['bus']['host']}:{cfg['bus']['port']}",
                       "services": ["orchestrator", "tripwire", "enrich"]},
            "node_b": {"endpoint": cfg["endpoints"]["genaid"],
                       "services": ["genaid"]},
            "node_c": {"endpoint": cfg["endpoints"]["cortexd"],
                       "services": ["cortexd", "earsd"]},
        },
        "started_at": _utc_now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify SPARC capability readiness")
    parser.add_argument("profile", choices=("v0.3", "v0.4", "active"))
    args = parser.parse_args(argv)

    repo = _repo_root()
    config_path = Path(os.environ.get("SPARC_CONFIG", repo / "config" / "sparc.yaml"))
    cfg = config.load()
    deadline_s = float(os.environ.get("SPARC_VERIFY_DEADLINE_S", "180"))
    interval_s = float(os.environ.get("SPARC_VERIFY_INTERVAL_S", "5"))
    stale_s = float(os.environ.get("SPARC_HEALTH_STALE_S", "15"))
    samples_needed = int(os.environ.get("SPARC_VERIFY_SAMPLES", "3"))
    if deadline_s <= 0 or interval_s <= 0 or stale_s <= 0 or samples_needed <= 0:
        parser.error("deadline, interval, staleness, and sample count must be positive")

    evidence_dir = Path(os.environ.get("SPARC_EVIDENCE_DIR", repo / "evidence"))
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    evidence_path = evidence_dir / f"capability-{args.profile}-{stamp}.jsonl"
    try:
        metadata = _run_metadata(args.profile, repo, config_path)
    except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
        parser.error(str(exc))

    collector = HealthCollector(str(cfg["bus"]["host"]), int(cfg["bus"]["port"]))
    collector.start()
    started = time.monotonic()
    consecutive = 0
    last_counted_ids: dict[str, str] | None = None
    sample_index = 0
    result = "failed"
    try:
        with evidence_path.open("x", encoding="utf-8") as evidence:
            evidence.write(json.dumps(metadata, sort_keys=True) + "\n")
            while True:
                sample_index += 1
                reports = collector.snapshot()
                cortex = _http_health(str(cfg["endpoints"]["cortexd"]))
                ears = reports.get("earsd")
                genaid = None
                if ears is not None and ears.details.get("stt_backend") == "genaid":
                    genaid = _http_health(str(cfg["endpoints"]["genaid"]))
                evaluation = evaluate_profile(
                    args.profile,
                    reports,
                    broker_connected=collector.connected,
                    cortex=cortex,
                    genaid=genaid,
                    stale_s=stale_s,
                )
                required_services = {"orchestrator"}
                if args.profile in {"v0.3", "active"}:
                    required_services.add("earsd")
                if args.profile in {"v0.4", "active"}:
                    required_services.update(("tripwire", "enrich"))
                current_ids = {
                    name: reports[name].msg_id
                    for name in required_services
                    if name in reports
                }
                refreshed = bool(
                    evaluation.ok
                    and len(current_ids) == len(required_services)
                    and (last_counted_ids is None or all(
                        current_ids[name] != last_counted_ids.get(name)
                        for name in required_services
                    ))
                )
                if not evaluation.ok:
                    consecutive = 0
                    last_counted_ids = None
                elif refreshed:
                    consecutive += 1
                    last_counted_ids = current_ids
                record = {
                    "kind": "sample",
                    "index": sample_index,
                    "sampled_at": _utc_now(),
                    "elapsed_s": round(time.monotonic() - started, 3),
                    "healthy": evaluation.ok,
                    "counted_as_refreshed_sample": refreshed,
                    "consecutive_healthy": consecutive,
                    "checks": evaluation.checks,
                    "service_versions": {
                        name: {"version": report.version, "details": report.details}
                        for name, report in sorted(reports.items())
                    },
                    "cortexd": cortex,
                    "genaid": genaid,
                }
                evidence.write(json.dumps(record, sort_keys=True) + "\n")
                evidence.flush()
                failed = [item["name"] for item in evaluation.checks if not item["ok"]]
                disposition = ("healthy" if refreshed else
                               "healthy (awaiting refreshed retained health)"
                               if evaluation.ok else "unhealthy")
                print(f"sample {sample_index}: {disposition}"
                      f" ({consecutive}/{samples_needed})"
                      + (f"; failed: {', '.join(failed)}" if failed else ""))
                if consecutive >= samples_needed:
                    result = "passed"
                    break
                remaining = deadline_s - (time.monotonic() - started)
                if remaining <= 0:
                    break
                time.sleep(min(interval_s, remaining))

            summary = {
                "kind": "summary",
                "result": result,
                "ended_at": _utc_now(),
                "duration_s": round(time.monotonic() - started, 3),
                "samples": sample_index,
                "required_consecutive_healthy": samples_needed,
                "evidence_path": str(evidence_path),
            }
            evidence.write(json.dumps(summary, sort_keys=True) + "\n")
    finally:
        collector.stop()

    print(f"readiness {result}: {evidence_path}")
    return 0 if result == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
