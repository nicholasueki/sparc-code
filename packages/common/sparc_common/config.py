"""Config loader: one YAML, per-node sections, env overrides.

SPARC_CONFIG points at the YAML (default: <repo>/config/sparc.yaml).
SPARC_NODE names this node ("a" | "b" | "c") for node-scoped helpers.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "config" / "sparc.yaml").exists():
            return parent
    return Path.cwd()


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    path = os.environ.get("SPARC_CONFIG") or str(_repo_root() / "config" / "sparc.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get(dotted: str, default: Any = None) -> Any:
    """cfg.get('node_c.think_timeout_s', 6)"""
    cur: Any = load()
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def node() -> str:
    return os.environ.get("SPARC_NODE", "dev")
