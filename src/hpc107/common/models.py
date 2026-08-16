"""Typed configuration, inspection, and run-state models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ProjectSettings:
    entry: str = "src/train.py"
    job_name: str = "train"


@dataclass(slots=True)
class EnvironmentSettings:
    manager: str = "uv"
    conda_env: str = ""


@dataclass(slots=True)
class ResourceSettings:
    partition: str = "Students"
    qos: str = "qos_stu_default"
    cpus: int = 4
    memory: str = "16G"
    gpus: int = 0
    walltime: str = "1:00:00"


@dataclass(slots=True)
class PathSettings:
    script: str = "scripts/train.sbatch"
    logs: str = "logs"
    outputs: str = "outputs"
    checkpoints: str = "checkpoints"
    runs: str = ".hpc107/runs"


@dataclass(slots=True)
class DatasetSettings:
    pan_remote: str = ""


@dataclass(slots=True)
class Settings:
    version: int = 1
    project: ProjectSettings = field(default_factory=ProjectSettings)
    environment: EnvironmentSettings = field(default_factory=EnvironmentSettings)
    resources: ResourceSettings = field(default_factory=ResourceSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    datasets: DatasetSettings = field(default_factory=DatasetSettings)

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProjectReport:
    root: str
    valid: bool
    entry: str | None
    dependencies: tuple[str, ...]
    inferred_environment: str
    inferred_resources: dict[str, Any]
    missing: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RunState:
    run_id: str
    state: str
    project_root: str
    script: str
    script_sha256: str
    job_id: str = ""
    submitted_at: str = ""
    updated_at: str = ""
    exit_code: str = ""
    elapsed: str = ""
    node_list: str = ""
    alloc_tres: str = ""
    diagnosis: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
