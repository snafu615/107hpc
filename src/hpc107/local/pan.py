"""Generate, but never execute, safe USTC Pan copy commands."""

from __future__ import annotations

import shlex
from pathlib import Path

from hpc107.common.transfer import validate_pan_remote


def upload_commands(root: Path, remote: str) -> list[list[str]]:
    validate_pan_remote(remote)
    commands: list[list[str]] = []
    for name in ("data", "datasets"):
        if (root / name).exists():
            commands.append(["rclone", "copy", f"./{name}", f"{remote}/{name}", "-P"])
    return commands


def result_upload_commands(root: Path, remote: str) -> list[list[str]]:
    validate_pan_remote(remote)
    commands: list[list[str]] = []
    for name in ("outputs", "checkpoints", "logs"):
        if (root / name).exists():
            commands.append(["rclone", "copy", f"./{name}", f"{remote}/{name}", "-P"])
    return commands


def result_download_commands(remote: str) -> list[list[str]]:
    validate_pan_remote(remote)
    return [
        ["rclone", "copy", f"{remote}/{name}", f"./{name}", "-P"]
        for name in ("outputs", "checkpoints", "logs")
    ]


def reverse_for_cluster(commands: list[list[str]]) -> list[list[str]]:
    result: list[list[str]] = []
    for command in commands:
        if len(command) == 5 and command[:2] == ["rclone", "copy"]:
            result.append(["rclone", "copy", command[3], command[2], "-P"])
    return result


def display(command: list[str]) -> str:
    return shlex.join(command)
