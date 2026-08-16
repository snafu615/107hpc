"""Deterministic inspection for the submit107-style computing-task layout."""

from __future__ import annotations

import re
from pathlib import Path

from .errors import ProjectValidationError
from .models import ProjectReport

GPU_PACKAGES = frozenset(
    {
        "torch",
        "pytorch",
        "torchvision",
        "torchaudio",
        "tensorflow",
        "tf-keras",
        "keras",
        "jax",
        "jaxlib",
        "flax",
    }
)


def _package_name(value: str) -> str | None:
    match = re.match(r"^\s*([A-Za-z0-9_.-]+)", value)
    return match.group(1).lower().replace("_", "-") if match else None


def detect_dependencies(root: Path) -> set[str]:
    dependencies: set[str] = set()
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            import tomllib

            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            project = data.get("project", {})
            values = list(project.get("dependencies", []))
            for group in project.get("optional-dependencies", {}).values():
                values.extend(group)
            for value in values:
                name = _package_name(str(value))
                if name:
                    dependencies.add(name)
        except (OSError, ValueError, TypeError):
            pass
    requirements = root / "requirements.txt"
    if requirements.exists():
        for line in requirements.read_text(encoding="utf-8").splitlines():
            content = line.split("#", 1)[0].strip()
            if not content or content.startswith(("-", "http://", "https://")):
                continue
            name = _package_name(content)
            if name:
                dependencies.add(name)
    return dependencies


def detect_entry(root: Path) -> Path | None:
    """Prefer the documented template layout, then submit107 root fallbacks."""
    preferred = [
        Path("src/train.py"),
        Path("train.py"),
        Path("main.py"),
    ]
    for candidate in preferred:
        if (root / candidate).is_file():
            return candidate
    root_python = sorted(path.relative_to(root) for path in root.glob("*.py"))
    if root_python:
        return root_python[0]
    notebooks = sorted(path.relative_to(root) for path in root.glob("*.ipynb"))
    return notebooks[0] if notebooks else None


def infer_defaults(root: Path) -> dict[str, object]:
    dependencies = detect_dependencies(root)
    if (root / "pyproject.toml").exists() or (root / "requirements.txt").exists():
        environment = "uv"
    elif (root / "environment.yml").exists():
        environment = "conda"
    else:
        environment = "system"
    if dependencies & GPU_PACKAGES:
        return {
            "environment": environment,
            "gpus": 1,
            "cpus": 4,
            "memory": "16G",
            "walltime": "2:00:00",
        }
    return {
        "environment": environment,
        "gpus": 0,
        "cpus": 2,
        "memory": "4G",
        "walltime": "1:00:00",
    }


def inspect_project(project_root: Path) -> ProjectReport:
    root = project_root.resolve()
    missing: list[str] = []
    warnings: list[str] = []
    if not root.is_dir():
        missing.append("project directory")
    if not (root / "src").is_dir():
        missing.append("src/")
    entry = detect_entry(root) if root.is_dir() else None
    if entry is None:
        missing.append("src/train.py (or a compatible root entry point)")
    elif entry != Path("src/train.py"):
        warnings.append(f"Compatibility entry layout in use: {entry.as_posix()}")
    if not any(
        (root / name).is_file()
        for name in ("pyproject.toml", "requirements.txt", "environment.yml")
    ):
        missing.append("pyproject.toml, requirements.txt, or environment.yml")
    inferred = infer_defaults(root) if root.is_dir() else {}
    dependencies = tuple(sorted(detect_dependencies(root))) if root.is_dir() else ()
    return ProjectReport(
        root=str(root),
        valid=not missing,
        entry=entry.as_posix() if entry else None,
        dependencies=dependencies,
        inferred_environment=str(inferred.get("environment", "system")),
        inferred_resources={key: value for key, value in inferred.items() if key != "environment"},
        missing=tuple(missing),
        warnings=tuple(warnings),
    )


def require_valid_project(project_root: Path) -> ProjectReport:
    report = inspect_project(project_root)
    if not report.valid:
        raise ProjectValidationError(
            "Project does not satisfy the hpc107 task layout: " + "; ".join(report.missing)
        )
    return report
