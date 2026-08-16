from pathlib import Path

from hpc107.common.project import detect_entry, infer_defaults, inspect_project


def _project(root: Path, *, dependency: str = "numpy") -> None:
    (root / "src").mkdir()
    (root / "src" / "train.py").write_text("print('ok')\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "task"\ndependencies = ["{dependency}"]\n', encoding="utf-8"
    )


def test_canonical_submit107_template_layout_is_valid(tmp_path: Path) -> None:
    _project(tmp_path)
    report = inspect_project(tmp_path)
    assert report.valid
    assert report.entry == "src/train.py"


def test_canonical_entry_precedes_root_compatibility_entry(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "train.py").write_text("print('root')\n", encoding="utf-8")
    assert detect_entry(tmp_path) == Path("src/train.py")


def test_root_train_is_accepted_with_warning(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "train.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")
    report = inspect_project(tmp_path)
    assert report.valid
    assert report.warnings


def test_gpu_dependency_uses_submit107_defaults(tmp_path: Path) -> None:
    _project(tmp_path, dependency="torch>=2")
    defaults = infer_defaults(tmp_path)
    assert defaults == {
        "environment": "uv",
        "gpus": 1,
        "cpus": 4,
        "memory": "16G",
        "walltime": "2:00:00",
    }


def test_cpu_dependency_uses_submit107_defaults(tmp_path: Path) -> None:
    _project(tmp_path, dependency="scikit-learn")
    defaults = infer_defaults(tmp_path)
    assert defaults["gpus"] == 0
    assert defaults["cpus"] == 2
    assert defaults["memory"] == "4G"


def test_missing_dependency_manifest_is_invalid(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").touch()
    report = inspect_project(tmp_path)
    assert not report.valid
    assert any("pyproject" in item for item in report.missing)
