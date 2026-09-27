"""Load YAML config and resolve environment secrets."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from tifbench.io import project_root


def load_config(path: Path | None = None) -> dict[str, Any]:
    load_dotenv(project_root() / ".env", override=False)
    cfg_path = path or (project_root() / "config" / "default.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = project_root()
    for key in ("raw", "curated", "derived", "benchmark"):
        if key in cfg.get("paths", {}) and cfg["paths"][key]:
            cfg["paths"][key] = str((root / cfg["paths"][key]).resolve())
    return cfg


def require_api_key(env_name: str) -> str:
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {env_name}")
    return value


def path_from_cfg(cfg: dict[str, Any], *parts: str) -> Path:
    base = Path(cfg["paths"][parts[0]])
    return base.joinpath(*parts[1:]) if len(parts) > 1 else base
