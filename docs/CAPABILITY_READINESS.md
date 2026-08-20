# Fleet installation and capability readiness

`deploy.sh`, `install_services.sh`, and `verify_capability.sh` have deliberately
separate jobs. Deployment synchronizes the checkout and locked Python environment;
installation copies supervisor definitions and restarts daemons; verification
proves a selected user-facing capability is functionally ready.

Run fleet operations only from one clean, integrated revision with exclusive
access to all nodes. Do not run concurrent `rsync --delete` deployments. Node C
LaunchAgents require the accepted GUI login/auto-login precondition.

## Install active services

After deployment, select nodes—not capabilities—with:

```bash
scripts/install_services.sh a
scripts/install_services.sh b
scripts/install_services.sh c
scripts/install_services.sh all
```

The installer always copies, enables, and restarts every active daemon on each
selected node:

| Node | Active daemons |
| --- | --- |
| A | `sparc-orchestrator`, `sparc-tripwire`, `sparc-enrich` |
| B | `sparc-genaid` |
| C | `com.sparc.cortexd`, `com.sparc.earsd` |

Lifecycle operations use exact systemd units and launchd labels. Any failed
copy, enable, restart, or loaded/running check exits nonzero.

## Verify a capability profile

From the same integrated checkout used for deployment:

```bash
scripts/verify_capability.sh v0.3
scripts/verify_capability.sh v0.4
scripts/verify_capability.sh active
```

The default cold-boot deadline is 180 seconds. A pass requires three healthy
samples five seconds apart, and every required daemon must refresh its retained
message between counted samples. Retained health older than 15 seconds is unhealthy.
For the 30-second supervisor-restart gate, set only the deadline:

```bash
SPARC_VERIFY_DEADLINE_S=30 scripts/verify_capability.sh active
```

`v0.3` accepts either local MLX Whisper on Node C or Node B genaid STT. When
earsd selects genaid, genaid `/health` must report `stt_loaded=true`. `v0.4`
does not require STT, and therefore remains independently verifiable. `active`
is the union of both profiles.

The verifier rejects a service that is merely running. Functional checks are:

| Profile | Defining checks |
| --- | --- |
| v0.3 | MQTT broker; orchestrator DB, MQTT, and bounded-session runtime; cortexd backend; earsd mic, TTS, MQTT, and usable approved STT backend |
| v0.4 | MQTT broker; orchestrator DB and MQTT; tripwire camera, model, and frame path; enrichment SCRFD, ArcFace, and MQTT; cortexd backend |

The non-HTTP daemons publish retained `ServiceHealth` records on
`sparc/health/{orchestrator,tripwire,enrich,earsd}`. Every record carries
`service`, `status`, `ready`, `ts`, `version`, `details`, and
`failure_reason`. cortexd and genaid expose equivalent model details through
their existing `/health` endpoints.

## Evidence and recovery runs

Each verifier invocation writes dated JSON Lines evidence under `evidence/`.
It refuses a dirty checkout so the recorded revision identifies all source.
The first record pins the Git SHA, SHA-256 configuration digest, profile, host
topology, and start time. Each sample records every probe plus service/model
versions. The summary records the end time, duration, result, and required
consecutive sample count. Generated evidence is intentionally ignored by Git;
attach the final integrated fleet evidence to the release record.

Use this dated run matrix for the exclusive fleet session:

| UTC date | Integrated SHA | Config SHA-256 | Profile | Scenario | Deadline | Result | Evidence file |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| YYYY-MM-DD | `<40-char SHA>` | `<64-char digest>` | v0.3 / v0.4 / active | cold boot / supervisor restart | 180s / 30s | pass / fail | `capability-<profile>-<UTC>.jsonl` |

For cold boot, start the verifier when all hosts are reachable and Node C has a
GUI login. For recovery, hard-kill each daemon, let its configured supervisor
restart it, then run the 30-second verifier. Preserve the full failed evidence
file on any miss; do not replace failure records with a later successful run.
