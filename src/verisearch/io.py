"""Small atomic IO and provenance helpers shared by commands."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(value) if is_dataclass(value) and not isinstance(value, type) else value
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = handle.name
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        temporary = handle.name
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def provenance() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
