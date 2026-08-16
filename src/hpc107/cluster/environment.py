"""Deterministic project-environment checks and preparation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hpc107.common.errors import HPC107Error
from hpc107.common.models import Settings


def check_environment(root: Path, settings: Settings) -> list[str]:
    manager = settings.environment.manager
    errors: list[str] = []
    if manager == "uv":
        if shutil.which("uv") is None and not (Path.home() / ".local/bin/uv").is_file():
            errors.append("uv is not available on PATH or at ~/.local/bin/uv")
        if not (root / "pyproject.toml").is_file() and not (root / "requirements.txt").is_file():
            errors.append("uv requires pyproject.toml or requirements.txt")
    elif manager == "conda" and shutil.which("conda") is None:
        errors.append("conda is not available on PATH")
    elif manager == "system" and shutil.which("python") is None:
        errors.append("python is not available on PATH")
    return errors


def _uv_executable() -> str:
    value = shutil.which("uv")
    if value:
        return value
    fallback = Path.home() / ".local/bin/uv"
    if fallback.is_file():
        return str(fallback)
    raise HPC107Error("uv is not available")


def prepare_environment(root: Path, settings: Settings) -> None:
    errors = check_environment(root, settings)
    if errors:
        raise HPC107Error("Environment is not ready:\n- " + "\n- ".join(errors))
    if settings.environment.manager != "uv" or (root / ".venv").is_dir():
        return
    uv = _uv_executable()
    if (root / "pyproject.toml").is_file():
        command = [uv, "sync"]
        result = subprocess.run(command, cwd=root, check=False)
    else:
        created = subprocess.run([uv, "venv"], cwd=root, check=False)
        if created.returncode != 0:
            raise HPC107Error("uv venv failed")
        result = subprocess.run(
            [uv, "pip", "install", "-r", "requirements.txt"],
            cwd=root,
            check=False,
        )
    if result.returncode != 0 or not (root / ".venv").is_dir():
        raise HPC107Error("Python environment preparation failed")
