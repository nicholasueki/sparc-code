"""Hardware-free tests for typed health and profile-specific release gates."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages" / "common"))

from sparc_common.bus import RETAINED, TOPICS, parse  # noqa: E402
from sparc_common.health import HealthReporter  # noqa: E402
from sparc_common.readiness import evaluate_profile  # noqa: E402
from sparc_common.types import ServiceHealth  # noqa: E402


NOW = 1_800_000_000.0
CORTEX = {"ok": True, "backend": {"model": "ornith"}, "version": "0.1.0"}


def report(service: str, details: dict, *, ts: float = NOW,
           ready: bool = True) -> ServiceHealth:
    return ServiceHealth(
        service=service,
        status="ready" if ready else "degraded",
        ready=ready,
        ts=ts,
        version="0.1.0",
        details=details,
        failure_reason=None if ready else "fixture failure",
    )


def healthy_reports(stt_backend: str = "mlx_local") -> dict[str, ServiceHealth]:
    return {
        "orchestrator": report("orchestrator", {
            "db_ready": True, "mqtt_ready": True, "bounded_session_runtime": True,
        }),
        "earsd": report("earsd", {
            "mic_open": True, "tts_ready": True, "mqtt_ready": True,
            "stt_ready": True, "stt_backend": stt_backend,
        }),
        "tripwire": report("tripwire", {
            "mqtt_ready": True, "camera_ready": True, "frame_path_ready": True,
            "model_present": True,
        }),
        "enrich": report("enrich", {
            "mqtt_ready": True, "scrfd_loaded": True, "arcface_loaded": True,
        }),
    }


@pytest.mark.parametrize("profile", ["v0.3", "v0.4", "active"])
def test_healthy_approved_topology_passes(profile):
    result = evaluate_profile(
        profile,
        healthy_reports(),
        broker_connected=True,
        cortex=CORTEX,
        genaid=None,
        now=NOW,
    )
    assert result.ok, [c for c in result.checks if not c["ok"]]


def test_v03_accepts_genaid_only_when_stt_is_loaded():
    reports = healthy_reports(stt_backend="genaid")
    failed = evaluate_profile(
        "v0.3", reports, broker_connected=True,
        cortex=CORTEX,
        genaid={"ok": True, "stt_loaded": False, "version": "0.1.0"}, now=NOW,
    )
    passed = evaluate_profile(
        "v0.3", reports, broker_connected=True,
        cortex=CORTEX,
        genaid={"ok": True, "stt_loaded": True, "version": "0.1.0"}, now=NOW,
    )
    assert not failed.ok
    assert passed.ok


def test_profiles_are_independent_not_union_by_accident():
    v03_reports = healthy_reports()
    del v03_reports["tripwire"], v03_reports["enrich"]
    v03 = evaluate_profile(
        "v0.3", v03_reports, broker_connected=True,
        cortex=CORTEX,
        genaid=None, now=NOW,
    )
    v04_reports = healthy_reports()
    del v04_reports["earsd"]
    v04_reports["orchestrator"].details["bounded_session_runtime"] = False
    v04 = evaluate_profile(
        "v0.4", v04_reports, broker_connected=True,
        cortex=CORTEX,
        genaid=None, now=NOW,
    )
    assert v03.ok and v04.ok


@pytest.mark.parametrize("service,field", [
    ("orchestrator", "bounded_session_runtime"),
    ("earsd", "mic_open"),
    ("earsd", "tts_ready"),
    ("earsd", "mqtt_ready"),
    ("earsd", "stt_ready"),
])
def test_v03_rejects_each_missing_defining_dependency(service, field):
    reports = healthy_reports()
    reports[service].details[field] = False
    result = evaluate_profile(
        "v0.3", reports, broker_connected=True,
        cortex=CORTEX,
        genaid=None, now=NOW,
    )
    assert not result.ok
    assert any(c["name"] == f"{service}.{field}" and not c["ok"]
               for c in result.checks)


@pytest.mark.parametrize("service,field", [
    ("tripwire", "camera_ready"),
    ("tripwire", "frame_path_ready"),
    ("tripwire", "model_present"),
    ("tripwire", "mqtt_ready"),
    ("enrich", "scrfd_loaded"),
    ("enrich", "arcface_loaded"),
    ("enrich", "mqtt_ready"),
])
def test_v04_rejects_each_missing_defining_dependency(service, field):
    reports = healthy_reports()
    reports[service].details[field] = False
    result = evaluate_profile(
        "v0.4", reports, broker_connected=True,
        cortex=CORTEX,
        genaid=None, now=NOW,
    )
    assert not result.ok
    assert any(c["name"] == f"{service}.{field}" and not c["ok"]
               for c in result.checks)


def test_unreachable_broker_unready_cortex_and_stale_health_fail():
    reports = healthy_reports()
    reports["orchestrator"].ts = NOW - 16
    result = evaluate_profile(
        "active", reports, broker_connected=False,
        cortex={"_error": "connection refused"}, genaid=None, now=NOW, stale_s=15,
    )
    failed = {c["name"] for c in result.checks if not c["ok"]}
    assert {"broker.reachable", "orchestrator.fresh", "cortexd.reachable",
            "cortexd.backend_ready"}.issubset(failed)


def test_health_topics_are_typed_and_retained():
    message = report("tripwire", {
        "mqtt_ready": True, "camera_ready": True, "frame_path_ready": True,
        "model_present": True,
    })
    topic = "sparc/health/tripwire"
    assert TOPICS[topic] is ServiceHealth
    assert topic in RETAINED
    assert parse(topic, message.model_dump_json().encode()) == message


def test_health_reporter_turns_probe_exception_into_typed_failure():
    class FakeBus:
        published = []

        def publish(self, topic, message):
            self.published.append((topic, message))

    def broken_probe():
        raise RuntimeError("camera vanished")

    bus = FakeBus()
    message = HealthReporter(bus, "tripwire", broken_probe).publish()
    assert bus.published[0][0] == "sparc/health/tripwire"
    assert message.ready is False and message.status == "failed"
    assert "camera vanished" in message.failure_reason


def test_installer_covers_all_services_without_process_pattern_kills():
    text = (ROOT / "scripts" / "install_services.sh").read_text()
    for service in ("sparc-orchestrator", "sparc-tripwire", "sparc-enrich",
                    "sparc-genaid", "com.sparc.cortexd", "com.sparc.earsd"):
        assert service in text
    assert "pkill" not in text
    assert "pgrep" not in text
    assert "|| echo" not in text


@pytest.mark.parametrize("script", ["install_services.sh", "verify_capability.sh"])
def test_service_scripts_reject_invalid_arguments_without_network(script):
    result = subprocess.run(
        [str(ROOT / "scripts" / script), "invalid"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "usage:" in result.stderr
