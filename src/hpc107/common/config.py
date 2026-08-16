"""Native configuration plus one-way submit107 compatibility mapping."""

from __future__ import annotations

import os
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError
from .models import (
    DatasetSettings,
    EnvironmentSettings,
    PathSettings,
    ProjectSettings,
    ResourceSettings,
    Settings,
)

NATIVE_CONFIG = "hpc107.yaml"
LEGACY_CONFIG = ".submit107.yaml"
_MEMORY_RE = re.compile(r"^[1-9][0-9]*(?:\.[0-9]+)?[KMGTP]?$", re.IGNORECASE)
_WALLTIME_RE = re.compile(r"^(?:[0-9]+-)?[0-9]{1,2}:[0-9]{2}:[0-9]{2}$")
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SAFE_PATH_PART_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigurationError(f"{location} must be a YAML mapping")
    return value


def _known_values(cls: type, data: dict[str, Any], location: str) -> dict[str, Any]:
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigurationError(f"Unknown {location} fields: {', '.join(unknown)}")
    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))
    except OSError as exc:
        raise ConfigurationError(f"Could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc


def _legacy_to_native(data: dict[str, Any]) -> dict[str, Any]:
    """Map source-backed submit107 fields into the native schema."""
    resources: dict[str, Any] = {}
    for old, new in {
        "partition": "partition",
        "qos": "qos",
        "cpus": "cpus",
        "mem": "memory",
        "gpu": "gpus",
        "time": "walltime",
    }.items():
        if old in data:
            resources[new] = data[old]
    paths: dict[str, Any] = {}
    if "sbatch_dir" in data:
        paths["script"] = f"{str(data['sbatch_dir']).rstrip('/')}/train.sbatch"
    for old, new in {"log_dir": "logs"}.items():
        if old in data:
            paths[new] = data[old]
    environment: dict[str, Any] = {}
    if "env" in data:
        environment["manager"] = data["env"]
    if "conda_env" in data:
        environment["conda_env"] = data["conda_env"]
    datasets: dict[str, Any] = {}
    if "pan_remote" in data:
        datasets["pan_remote"] = data["pan_remote"]
    return {
        "project": {},
        "environment": environment,
        "resources": resources,
        "paths": paths,
        "datasets": datasets,
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _environment_overrides(prefix: str) -> dict[str, Any]:
    names = {
        "PARTITION": ("resources", "partition", str),
        "QOS": ("resources", "qos", str),
        "CPUS": ("resources", "cpus", int),
        "MEM": ("resources", "memory", str),
        "MEMORY": ("resources", "memory", str),
        "GPU": ("resources", "gpus", int),
        "GPUS": ("resources", "gpus", int),
        "TIME": ("resources", "walltime", str),
        "WALLTIME": ("resources", "walltime", str),
        "ENV": ("environment", "manager", str),
        "ENTRY": ("project", "entry", str),
    }
    result: dict[str, Any] = {}
    for suffix, (section, key, converter) in names.items():
        raw = os.environ.get(prefix + suffix)
        if raw is None or raw == "":
            continue
        try:
            value = converter(raw)
        except ValueError as exc:
            raise ConfigurationError(f"{prefix}{suffix} has an invalid value") from exc
        result.setdefault(section, {})[key] = value
    return result


def load_settings(project_root: Path) -> Settings:
    """Load legacy, native, and environment configuration.

    Precedence is built-ins < .submit107.yaml < hpc107.yaml < SUBMIT107_*
    < HPC107_*.
    """
    root = project_root.resolve()
    data: dict[str, Any] = Settings().to_dict()
    legacy = root / LEGACY_CONFIG
    if legacy.exists():
        data = _deep_merge(data, _legacy_to_native(_read_yaml(legacy)))
    native = root / NATIVE_CONFIG
    if native.exists():
        raw = _read_yaml(native)
        version = raw.pop("version", 1)
        if version != 1:
            raise ConfigurationError(f"Unsupported hpc107.yaml version: {version!r}")
        data = _deep_merge(data, raw)
    data = _deep_merge(data, _environment_overrides("SUBMIT107_"))
    data = _deep_merge(data, _environment_overrides("HPC107_"))

    project = ProjectSettings(
        **_known_values(ProjectSettings, _mapping(data.get("project"), "project"), "project")
    )
    environment = EnvironmentSettings(
        **_known_values(
            EnvironmentSettings,
            _mapping(data.get("environment"), "environment"),
            "environment",
        )
    )
    resources = ResourceSettings(
        **_known_values(ResourceSettings, _mapping(data.get("resources"), "resources"), "resources")
    )
    paths = PathSettings(
        **_known_values(PathSettings, _mapping(data.get("paths"), "paths"), "paths")
    )
    datasets = DatasetSettings(
        **_known_values(DatasetSettings, _mapping(data.get("datasets"), "datasets"), "datasets")
    )
    settings = Settings(
        version=1,
        project=project,
        environment=environment,
        resources=resources,
        paths=paths,
        datasets=datasets,
    )
    validate_settings(settings)
    return settings


def validate_settings(settings: Settings) -> None:
    r = settings.resources
    if not _SAFE_NAME_RE.fullmatch(settings.project.job_name):
        raise ConfigurationError("project.job_name contains unsupported characters")
    if not _SAFE_NAME_RE.fullmatch(r.partition):
        raise ConfigurationError("resources.partition contains unsupported characters")
    if r.qos and not _SAFE_NAME_RE.fullmatch(r.qos):
        raise ConfigurationError("resources.qos contains unsupported characters")
    if not 1 <= r.cpus <= 1024:
        raise ConfigurationError("resources.cpus must be between 1 and 1024")
    if not 0 <= r.gpus <= 64:
        raise ConfigurationError("resources.gpus must be between 0 and 64")
    if not _MEMORY_RE.fullmatch(str(r.memory)):
        raise ConfigurationError("resources.memory must look like 16G")
    if not _WALLTIME_RE.fullmatch(str(r.walltime)):
        raise ConfigurationError("resources.walltime must look like H:MM:SS or D-HH:MM:SS")
    if settings.environment.manager not in {"uv", "conda", "system"}:
        raise ConfigurationError("environment.manager must be uv, conda, or system")
    if settings.environment.manager == "conda" and not settings.environment.conda_env:
        raise ConfigurationError("environment.conda_env is required for conda")
    for name, value in {
        "project.entry": settings.project.entry,
        "paths.script": settings.paths.script,
        "paths.logs": settings.paths.logs,
        "paths.outputs": settings.paths.outputs,
        "paths.checkpoints": settings.paths.checkpoints,
        "paths.runs": settings.paths.runs,
    }.items():
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ConfigurationError(f"{name} must be a contained relative path")
        if not path.parts or any(not _SAFE_PATH_PART_RE.fullmatch(part) for part in path.parts):
            raise ConfigurationError(
                f"{name} may contain only letters, numbers, dots, underscores, hyphens, and /"
            )


def write_default_config(project_root: Path, settings: Settings, *, force: bool = False) -> Path:
    path = project_root.resolve() / NATIVE_CONFIG
    if path.exists() and not force:
        return path
    path.write_text(
        yaml.safe_dump(settings.to_dict(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path
