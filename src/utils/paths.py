from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root, assuming this file lives under src/utils."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the repository root unless already absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root() / p


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for a file path and return the resolved path."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return the resolved path."""
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file from an absolute path or repo-relative path."""
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
