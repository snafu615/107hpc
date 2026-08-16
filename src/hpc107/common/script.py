"""Owned deterministic Slurm template and semantic validation."""

from __future__ import annotations

import hashlib
import re
import shlex
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from .errors import ProjectValidationError
from .models import Settings

_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={{ job_name }}
#SBATCH --partition={{ partition }}
#SBATCH --qos={{ qos }}
#SBATCH --cpus-per-task={{ cpus }}
#SBATCH --mem={{ memory }}
{% if gpus > 0 -%}
#SBATCH --gres=gpu:{{ gpus }}
{% endif -%}
#SBATCH --time={{ walltime }}
#SBATCH --output={{ logs }}/%x_%j.out
#SBATCH --error={{ logs }}/%x_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

exec {{ command }}
"""


def _command(settings: Settings) -> list[str]:
    entry = settings.project.entry
    manager = settings.environment.manager
    if manager == "uv":
        return [".venv/bin/python", entry]
    if manager == "conda":
        # Kept as a Bash expression because activation changes the current shell.
        return [
            "bash",
            "-lc",
            (
                f"source activate {shlex.quote(settings.environment.conda_env)} && "
                f"exec python {shlex.quote(entry)}"
            ),
        ]
    return ["python", entry]


def render_script(settings: Settings) -> str:
    command = shlex.join(_command(settings))
    return (
        Environment(undefined=StrictUndefined, autoescape=False)
        .from_string(_SBATCH_TEMPLATE)
        .render(
            job_name=settings.project.job_name,
            partition=settings.resources.partition,
            qos=settings.resources.qos,
            cpus=settings.resources.cpus,
            memory=settings.resources.memory,
            gpus=settings.resources.gpus,
            walltime=settings.resources.walltime,
            logs=settings.paths.logs,
            command=command,
        )
    )


def script_path(project_root: Path, settings: Settings) -> Path:
    root = project_root.resolve()
    target = (root / settings.paths.script).resolve()
    if target != root and root not in target.parents:
        raise ProjectValidationError("Configured script path escapes the project")
    return target


def write_script(project_root: Path, settings: Settings, *, force: bool = True) -> Path:
    target = script_path(project_root, settings)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise ProjectValidationError(f"Script already exists: {target}")
    if target.exists():
        index = 0
        while True:
            suffix = ".bak" if index == 0 else f".bak{index}"
            backup = target.with_name(target.name + suffix)
            if not backup.exists():
                backup.write_bytes(target.read_bytes())
                break
            index += 1
    target.write_text(render_script(settings), encoding="utf-8", newline="\n")
    target.chmod(0o755)
    return target


def validate_script(path: Path, settings: Settings) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"Missing Slurm script: {path}"]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"Could not read Slurm script: {exc}"]
    required = {
        "shebang": r"\A#!/bin/bash",
        "partition": rf"^#SBATCH --partition={re.escape(settings.resources.partition)}$",
        "qos": rf"^#SBATCH --qos={re.escape(settings.resources.qos)}$",
        "cpus": rf"^#SBATCH --cpus-per-task={settings.resources.cpus}$",
        "memory": rf"^#SBATCH --mem={re.escape(str(settings.resources.memory))}$",
        "walltime": rf"^#SBATCH --time={re.escape(str(settings.resources.walltime))}$",
        "stdout": rf"^#SBATCH --output={re.escape(settings.paths.logs)}/%x_%j\.out$",
        "stderr": rf"^#SBATCH --error={re.escape(settings.paths.logs)}/%x_%j\.err$",
        "strict mode": r"^set -euo pipefail$",
        "submit directory": r'^cd "\$SLURM_SUBMIT_DIR"$',
    }
    if settings.resources.gpus > 0:
        required["gpu"] = rf"^#SBATCH --gres=gpu:{settings.resources.gpus}$"
    elif re.search(r"^#SBATCH --gres=gpu", text, re.MULTILINE):
        errors.append("GPU directive is present while resources.gpus is zero")
    for label, pattern in required.items():
        if not re.search(pattern, text, re.MULTILINE):
            errors.append(f"Missing or inconsistent {label}")
    if settings.project.entry not in text:
        errors.append("Configured project entry is not executed by the script")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
