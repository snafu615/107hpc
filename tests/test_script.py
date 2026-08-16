from pathlib import Path

from hpc107.common.models import Settings
from hpc107.common.script import render_script, validate_script, write_script


def test_cpu_script_matches_proven_directives() -> None:
    settings = Settings()
    script = render_script(settings)
    assert "#SBATCH --partition=Students" in script
    assert "#SBATCH --qos=qos_stu_default" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=1:00:00" in script
    assert "#SBATCH --output=logs/%x_%j.out" in script
    assert "#SBATCH --error=logs/%x_%j.err" in script
    assert "--gres=gpu" not in script
    assert 'cd "$SLURM_SUBMIT_DIR"' in script
    assert "set -euo pipefail" in script
    assert "exec .venv/bin/python src/train.py" in script


def test_gpu_script_uses_generic_gres() -> None:
    settings = Settings()
    settings.resources.gpus = 2
    assert "#SBATCH --gres=gpu:2" in render_script(settings)


def test_written_script_validates(tmp_path: Path) -> None:
    settings = Settings()
    path = write_script(tmp_path, settings, force=False)
    assert validate_script(path, settings) == []


def test_existing_script_is_backed_up(tmp_path: Path) -> None:
    settings = Settings()
    path = write_script(tmp_path, settings, force=False)
    original = path.read_text(encoding="utf-8")
    settings.resources.cpus = 8
    write_script(tmp_path, settings, force=True)
    assert path.with_name("train.sbatch.bak").read_text(encoding="utf-8") == original


def test_validator_detects_resource_mismatch(tmp_path: Path) -> None:
    settings = Settings()
    path = write_script(tmp_path, settings, force=False)
    path.write_text(path.read_text().replace("--cpus-per-task=4", "--cpus-per-task=8"))
    assert any("cpus" in error for error in validate_script(path, settings))
