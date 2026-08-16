"""Deterministic artifact inventory and safe transfer plans."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from hpc107.common.models import Settings
from hpc107.common.transfer import validate_pan_remote


def artifact_inventory(root: Path, settings: Settings) -> list[dict[str, int | str]]:
    result: list[dict[str, int | str]] = []
    for directory in (settings.paths.logs, settings.paths.outputs, settings.paths.checkpoints):
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and not path.is_symlink():
                result.append(
                    {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
                )
    return result


def write_inventory(run_dir: Path, root: Path, settings: Settings) -> Path:
    target = run_dir / "artifacts.json"
    target.write_text(
        json.dumps(artifact_inventory(root, settings), indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def pan_upload_plan(root: Path, settings: Settings) -> list[str]:
    remote = settings.datasets.pan_remote
    if not remote:
        return []
    validate_pan_remote(remote)
    result: list[str] = []
    for directory in (settings.paths.outputs, settings.paths.checkpoints, settings.paths.logs):
        if (root / directory).exists():
            result.append(
                shlex.join(["rclone", "copy", f"./{directory}", f"{remote}/{directory}", "-P"])
            )
    return result
