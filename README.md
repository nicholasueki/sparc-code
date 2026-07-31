# SPARC — MVP Implementation

Monorepo for the SPARC companion robot MVP. Design docs live one level up
(`SPARC_System_Design_v2.md`, `MVP_System_Design.md`, `Hardware_Audit_and_Design_v3.md`,
`Model_Selection_Plan.md`). This README records only what the *code* does and the
executive decisions taken during implementation.

## Layout

```
packages/common/sparc_common/   shared Pydantic types, config loader, MQTT bus, narrative time
packages/node_a/sparc_node_a/   orchestrator: world model, scheduler, perception, pipelines (vision Pi)
packages/node_b/sparc_node_b/   genaid: STT/select/salience on hailo_platform.genai + audio I/O (genai Pi)
packages/node_c/sparc_node_c/   cortexd: Ornith backend(s), /think, semantic memory (M1 Max)
config/sparc.yaml               single source of runtime config (per-node sections)
scripts/                        deploy + run helpers (rsync, systemd units)
tests/                          fixture-driven unit tests (no hardware needed)
```

## Executive decisions (deltas from the design docs)

1. **MQTT instead of ROS 2 for the MVP bus.** ROS 2 has no supported install on
   RPi OS Trixie (Debian 13). Mosquitto + paho is apt-installable, ~1 ms on LAN,
   QoS 1, retained messages for state topics. All messages remain typed Pydantic
   models serialized as JSON — the boundary types are the contract, so a future
   ROS 2/zenoh migration touches only `sparc_common/bus.py`.
2. **cortexd model backend = in-process mlx-vlm (`MLXBackend`), primary.**
   Measured on the M1 Max: 58 tok/s decode, 26 GB peak. LM Studio's bundled MLX
   engine cannot load Ornith yet (`qwen3_5_moe_vision` unsupported); pip mlx-vlm
   0.6.3 can. `OpenAICompatBackend` (llama-server / LM Studio GGUF+mmproj) kept as
   config-selectable fallback; it adds decoder-level JSON-schema grammars if we
   ever need them.
3. **Structured output enforcement = parse + one retry + Pydantic validation +
   deterministic fallback** (no decoder grammar in mlx-vlm). §16 fallback layers
   are the real safety net; measured schema-violation rate is logged to `trace`.
4. **Embeddings = fastembed (ONNX, all-MiniLM-L6-v2, 384-d) on Node C CPU.**
   Small, no torch, ~10 ms/text. Vector store = sqlite-vec on Node C. Canonical
   facts live on Node A (SQLite); Node C holds the embedding mirror (design §7.1).
5. **Audio is optional at boot.** genaid + orchestrator run without mic/speaker;
   `say` actions fall back to log lines (and MQTT `sparc/tts/say` for whenever a
   speaker exists). Piper/Silero wire in without code changes when hardware arrives.
6. **10H models:** v5.2.0 zoo HEFs verified working on the HailoRT 5.1.1 runtime
   (Qwen2.5-1.5B generates). Letter-selector protocol per v3 §U2.

## Runbook

- Node A: `python -m sparc_node_a.orchestrator` (needs mosquitto running locally)
- Node B: `python -m sparc_node_b.genaid`
- Node C: `python -m sparc_node_c.cortexd`
- Deploy: `scripts/deploy.sh {a|b|c|all}` (rsync to the node, restart systemd unit)
- Tests: `pytest tests/` from repo root (pure-Python, hardware mocked by fixtures)
