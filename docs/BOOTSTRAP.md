# Reproducible Python bootstrap

Lucas supports four committed dependency groups. Always install a group through
its `constraints-<group>.txt` file; do not copy package lists into deployment
commands. Every group file includes the shared root `constraints.txt` contract.

| Target | Python | Install group | External prerequisites |
| --- | --- | --- | --- |
| Developer/test | macOS 3.12 | `dev` | none for the hardware-free suite |
| Node A | Debian 13 3.13 | `node-a` | Mosquitto, HailoRT/Hailo Python API, Picamera2/IMX500, OpenCV |
| Node B | Debian 13 3.13 | `node-b` | HailoRT with the GenAI Python API and deployed HEFs |
| Node C | macOS 3.12 (`~/mlx312`) | `node-c` | PortAudio, MLX-capable Apple Silicon, local model weights |

The manifest constrains every direct Python dependency to a reviewed compatible
series. The group constraint files select exact reviewed direct versions.
Transitive dependencies are resolved by pip; after changing a direct dependency,
install and verify all supported groups before committing the new constraint.
Node C pins NumPy 2.3.5 because the reviewed `mlx-whisper`/Numba stack requires
NumPy below 2.4; other groups use NumPy 2.5.0.

## Developer/test bootstrap

From a clean checkout on macOS with Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -c constraints-dev.txt '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -c 'import lucas_common, lucas_node_a, lucas_node_b, lucas_node_c'
```

The repository-local `.venv` is ignored. Recreating it does not modify the Git
index, and deleting it is never part of deployment.

## Node A

Install the OS/vendor layer first. On the supported Debian 13 image this includes
Mosquitto, Picamera2/IMX500, OpenCV, and the matching HailoRT `hailo_platform`
package. Do not replace those packages with PyPI wheels. Then:

```bash
python3 -m venv --system-site-packages ~/lucas_venv
~/lucas_venv/bin/python -m pip install -c constraints-node-a.txt '.[node-a]'
~/lucas_venv/bin/python -c 'import lucas_node_a.orchestrator; import lucas_node_a.perception.imx500_tripwire; import lucas_node_a.perception.face_enrich'
```

Hardware-free import checks do not open a camera or Hailo device. A fleet smoke
must additionally verify `hailo_platform`, `picamera2`, and `cv2`, then start the
Node A services against the configured hardware.

## Node B

Install the vendor HailoRT release that provides both `hailo_platform` and
`hailo_platform.genai`, and provision the configured LLM/Whisper HEFs separately.
Then:

```bash
python3 -m venv --system-site-packages ~/lucas_venv
~/lucas_venv/bin/python -m pip install -c constraints-node-b.txt '.[node-b]'
~/lucas_venv/bin/python -c 'import lucas_node_b.genaid as g; assert g.health()["ok"]'
```

Importing `genaid` and calling its health function does not load Hailo or require
a HEF. On the actual node, start the service and confirm `/health`; absent model
files are reported as `llm_loaded=false` or `stt_loaded=false`, not downloaded.

## Node C

Node C deliberately keeps the established `~/mlx312` Python 3.12 environment:

```bash
~/.local/bin/uv pip install --python ~/mlx312/bin/python -c constraints-node-c.txt '.[node-c]'
~/mlx312/bin/python -c 'import lucas_node_c.cortexd as c; assert c.health()["ok"] is False'
~/mlx312/bin/python -c 'import lucas_node_c.earsd'
```

The `node-c` group includes the MLX VLM and local Whisper fallback libraries, but
pip does not fetch model weights. Keep these configured assets external:

- Ornith/MLX VLM: `node_c.mlx.model_path` in `config/lucas.yaml` (currently
  `/Users/tokenator/models/Ornith-N-24B-A3B-Thinking-MLX-4bit`).
- Local Whisper fallback: `node_c.ears.mlx_whisper_model` (currently the
  Hugging Face id `mlx-community/whisper-base-mlx`, cached outside this repo).
- Node B Hailo models: `node_b.hef_dir`, `node_b.llm_hef`, and `node_b.stt_hef`.
- Node A Hailo HEFs and the IMX500 `.rpk`: OS/vendor paths in source/config.

The source checkout must not contain model weights, HEFs, `.rpk` files, runtime
databases, logs, or generated evaluation output. Those paths are ignored by Git.

## Deployment and verification boundary

`scripts/deploy.sh {a|b|c|all}` syncs the checkout and installs the corresponding
named group through its group-specific constraints file. It does not install OS
packages, download models, install services, or run fleet health checks. Use
`scripts/install_services.sh` only after the platform prerequisites above exist.

Before committing dependency changes, run the developer bootstrap in a newly
created venv and verify the index is free of generated files:

```bash
git ls-files | grep -E '(^|/)(\.venv|__pycache__|\.pytest_cache)(/|$)|\.py[co]$|\.(db|sqlite|sqlite3|log)$'
```

The command must produce no output.
