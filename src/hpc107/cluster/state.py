"""Atomic persistent run state used for reconnection and monitoring."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from hpc107.common.errors import HPC107Error
from hpc107.common.models import RunState, Settings


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_run_id(project_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(char if char.isalnum() or char in "-." else "-" for char in project_name)
    return f"{timestamp}-{safe.strip('-') or 'task'}"


def run_directory(root: Path, settings: Settings, run_id: str) -> Path:
    base = (root.resolve() / settings.paths.runs).resolve()
    if root.resolve() not in base.parents and base != root.resolve():
        raise HPC107Error("Run directory escapes the project")
    return base / run_id


def write_state(run_dir: Path, state: RunState) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    state.updated_at = utc_now()
    target = run_dir / "state.json"
    temporary = run_dir / f".state-{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(state.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def read_state(run_dir: Path) -> RunState:
    target = run_dir / "state.json"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return RunState(**data)
    except (OSError, ValueError, TypeError) as exc:
        raise HPC107Error(f"Could not read run state at {target}: {exc}") from exc


def find_run(root: Path, settings: Settings, run_id: str) -> Path:
    directory = run_directory(root, settings, run_id)
    if not (directory / "state.json").is_file():
        raise HPC107Error(f"Unknown run ID: {run_id}")
    return directory
