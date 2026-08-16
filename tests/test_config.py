from pathlib import Path

import pytest

from hpc107.common.config import load_settings
from hpc107.common.errors import ConfigurationError


def test_defaults_match_submit107_platform_conventions(tmp_path: Path) -> None:
    settings = load_settings(tmp_path)
    assert settings.resources.partition == "Students"
    assert settings.resources.qos == "qos_stu_default"
    assert settings.resources.cpus == 4
    assert settings.resources.memory == "16G"
    assert settings.resources.gpus == 0
    assert settings.resources.walltime == "1:00:00"
    assert settings.project.entry == "src/train.py"


def test_legacy_submit107_config_maps_one_way(tmp_path: Path) -> None:
    (tmp_path / ".submit107.yaml").write_text(
        "partition: Students\nqos: qos_stu_default\ncpus: 8\nmem: 32G\n"
        "gpu: 2\ntime: '3:00:00'\nenv: conda\nconda_env: task\n"
        "sbatch_dir: jobs\nlog_dir: run-logs\npan_remote: pan:task\n",
        encoding="utf-8",
    )
    settings = load_settings(tmp_path)
    assert settings.resources.cpus == 8
    assert settings.resources.memory == "32G"
    assert settings.resources.gpus == 2
    assert settings.resources.walltime == "3:00:00"
    assert settings.environment.manager == "conda"
    assert settings.environment.conda_env == "task"
    assert settings.paths.script == "jobs/train.sbatch"
    assert settings.paths.logs == "run-logs"
    assert settings.datasets.pan_remote == "pan:task"


def test_native_overrides_legacy(tmp_path: Path) -> None:
    (tmp_path / ".submit107.yaml").write_text("cpus: 8\n", encoding="utf-8")
    (tmp_path / "hpc107.yaml").write_text("version: 1\nresources:\n  cpus: 12\n", encoding="utf-8")
    assert load_settings(tmp_path).resources.cpus == 12


def test_hpc107_environment_overrides_submit107(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUBMIT107_CPUS", "8")
    monkeypatch.setenv("HPC107_CPUS", "16")
    assert load_settings(tmp_path).resources.cpus == 16


def test_rejects_escaping_paths(tmp_path: Path) -> None:
    (tmp_path / "hpc107.yaml").write_text(
        "version: 1\npaths:\n  script: ../escape.sbatch\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="contained relative"):
        load_settings(tmp_path)


def test_rejects_unknown_native_fields(tmp_path: Path) -> None:
    (tmp_path / "hpc107.yaml").write_text(
        "version: 1\nresources:\n  imaginary: true\n", encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="Unknown resources"):
        load_settings(tmp_path)


def test_rejects_sbatch_field_injection(tmp_path: Path) -> None:
    (tmp_path / "hpc107.yaml").write_text(
        'version: 1\nproject:\n  job_name: "ok\\n#SBATCH --exclusive"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="job_name"):
        load_settings(tmp_path)


def test_rejects_unsafe_log_path(tmp_path: Path) -> None:
    (tmp_path / "hpc107.yaml").write_text(
        'version: 1\npaths:\n  logs: "logs folder"\n', encoding="utf-8"
    )
    with pytest.raises(ConfigurationError, match="paths.logs"):
        load_settings(tmp_path)
