from pathlib import Path

import pytest

from hpc107.cluster.artifacts import artifact_inventory, pan_upload_plan
from hpc107.common.errors import ConfigurationError
from hpc107.common.models import Settings
from hpc107.local.pan import result_download_commands, upload_commands


def test_dataset_plan_covers_both_conventional_directories(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "datasets").mkdir()
    commands = upload_commands(tmp_path, "pan:task")
    assert [command[2] for command in commands] == ["./data", "./datasets"]


def test_pan_remote_validation_is_shared_by_cluster_side(tmp_path: Path) -> None:
    settings = Settings()
    settings.datasets.pan_remote = "pan:task;unsafe"
    with pytest.raises(ConfigurationError, match="metacharacters"):
        pan_upload_plan(tmp_path, settings)


def test_result_download_plan_is_copy_only() -> None:
    commands = result_download_commands("pan:task")
    assert len(commands) == 3
    assert all(command[:2] == ["rclone", "copy"] for command in commands)


def test_artifact_inventory_is_scoped_to_declared_outputs(tmp_path: Path) -> None:
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "result.txt").write_text("done", encoding="utf-8")
    (tmp_path / "source.txt").write_text("not an artifact", encoding="utf-8")
    inventory = artifact_inventory(tmp_path, Settings())
    assert inventory == [{"path": "outputs/result.txt", "bytes": 4}]
