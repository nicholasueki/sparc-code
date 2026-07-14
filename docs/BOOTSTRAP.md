# Reproducible Python bootstrap

Lucas keeps readable direct dependency groups in `pyproject.toml` and installs
complete generated locks from `locks/`. Every package in a runtime lock is exact
and hash-checked; vendor libraries, OS packages, and model assets stay external.

| Target | Resolution platform | Lock |
| --- | --- | --- |
| Developer/test | macOS arm64 + x86_64, Python 3.12 | `locks/dev-macos-py312.txt` |
| Node A | Debian 13 arm64 (manylinux glibc 2.39+), Python 3.13 | `locks/node-a-debian13-arm64-py313.txt` |
| Node B | Debian 13 arm64 (manylinux glibc 2.39+), Python 3.13 | `locks/node-b-debian13-arm64-py313.txt` |
| Node C | macOS 14+ arm64, Python 3.12 (`~/mlx312`) | `locks/node-c-macos14-arm64-py312.txt` |

`scripts/bootstrap_python.sh` first installs the hash-locked packaging toolchain
from `locks/bootstrap.txt` (`pip==26.1.2`, `setuptools==83.0.0`,
`wheel==0.47.0`), then the selected runtime lock, then Lucas itself with no
dependency resolution and no isolated build environment.

## Developer/test bootstrap

From a clean checkout on macOS with Python 3.12:

```bash
python3.12 -m venv .venv
scripts/bootstrap_python.sh dev
.venv/bin/python -m pytest
.venv/bin/python -c 'import lucas_common, lucas_node_a, lucas_node_b, lucas_node_c'
```

The bootstrap fails if the host is not macOS or the interpreter is not Python
3.12. The repository-local `.venv` is ignored and never removed by deployment.

## Node A

Install the Debian 13/vendor layer first: Mosquitto, Picamera2/IMX500, OpenCV,
and the matching HailoRT `hailo_platform` package. These stay outside pip and are
visible through the system-site venv:

```bash
python3 -m venv --system-site-packages ~/lucas_venv
PYTHON="$HOME/lucas_venv/bin/python" scripts/bootstrap_python.sh node-a
~/lucas_venv/bin/python -c 'import lucas_node_a.orchestrator; import lucas_node_a.perception.imx500_tripwire; import lucas_node_a.perception.face_enrich'
```

## Node B

Install the vendor HailoRT release providing `hailo_platform` and
`hailo_platform.genai`, and provision the configured LLM/Whisper HEFs externally:

```bash
python3 -m venv --system-site-packages ~/lucas_venv
PYTHON="$HOME/lucas_venv/bin/python" scripts/bootstrap_python.sh node-b
~/lucas_venv/bin/python -c 'import lucas_node_b.genaid as g; assert g.health()["ok"]'
```

## Node C

Node C is supported only on Apple-silicon arm64 running macOS 14 or newer with
Python 3.12 at `~/mlx312`. This minimum is required by the locked MLX wheels.
Bootstrap checks the OS, architecture, OS major version, and Python version
before installing anything and emits a clear error if the contract is not met:

```bash
PYTHON="$HOME/mlx312/bin/python" scripts/bootstrap_python.sh node-c
~/mlx312/bin/python -c 'import lucas_node_c.cortexd as c; assert c.health()["ok"] is False'
~/mlx312/bin/python -c 'import lucas_node_c.earsd'
```

The Node C lock includes MLX VLM and local Whisper libraries, not model weights:

- Ornith/MLX VLM: `node_c.mlx.model_path` in `config/lucas.yaml`.
- Local Whisper fallback: `node_c.ears.mlx_whisper_model`, cached outside Git.
- Node B Hailo models: `node_b.hef_dir`, `node_b.llm_hef`, and `node_b.stt_hef`.
- Node A Hailo HEFs and IMX500 `.rpk`: OS/vendor paths in source/config.

## Deployment

`scripts/deploy.sh {a|b|c|all}` syncs the checkout and invokes the same bootstrap
script with the matching lock. It does not install OS packages, download models,
install services, or run fleet health checks. Use `scripts/install_services.sh`
only after external platform prerequisites exist, then follow
[`CAPABILITY_READINESS.md`](CAPABILITY_READINESS.md) for profile verification and
revision/config/version-pinned evidence.

## Reproducible lock regeneration

Locks are generated with `uv==0.9.28`, an upload cutoff of
`2026-07-14T18:17:36Z`, exact direct pins plus reviewed compatibility pins from
`constraints-*.txt`, binary-only target resolution, and SHA-256 hashes. The
script resolves Debian 13 against `aarch64-manylinux_2_39`, resolves Node C with
`MACOSX_DEPLOYMENT_TARGET=14.0`, and requires identical dev graphs for macOS
arm64 and x86_64.

Bootstrap the pinned resolver and regenerate:

```bash
python3.12 -m venv .lock-venv
.lock-venv/bin/python -m pip install --require-hashes --no-deps -r locks/bootstrap.txt
.lock-venv/bin/python -m pip install --require-hashes --no-deps -r locks/lock-tools.txt
UV_BIN=.lock-venv/bin/uv scripts/lock_dependencies.sh
git diff --exit-code -- locks/
```

The last command must be empty when regeneration is reproducible. Direct-version
changes begin in `pyproject.toml` and its matching `constraints-*.txt` resolution
input, followed by regeneration and clean verification of every supported lock.

Verify generated paths remain outside the index:

```bash
git ls-files | grep -E '(^|/)(\.venv|\.lock-venv|__pycache__|\.pytest_cache)(/|$)|\.py[co]$|\.(db|sqlite|sqlite3|log)$'
```

The command must produce no output.
