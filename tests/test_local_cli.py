from pathlib import Path

from hpc107.local.cli import main


def test_template_creates_canonical_task(tmp_path: Path) -> None:
    assert main(["template", "demo", "--path", str(tmp_path)]) == 0
    root = tmp_path / "demo"
    assert (root / "src" / "train.py").is_file()
    assert (root / "scripts" / "train.sbatch").is_file()
    assert (root / "hpc107.yaml").is_file()


def test_inspect_json_is_nonmutating(tmp_path: Path, capsys) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").touch()
    (tmp_path / "requirements.txt").write_text("numpy\n")
    assert main(["inspect", str(tmp_path), "--json"]) == 0
    assert '"entry": "src/train.py"' in capsys.readouterr().out
    assert not (tmp_path / "hpc107.yaml").exists()


def test_prepare_derives_gpu_defaults_and_writes_script(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "train.py").touch()
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname="task"\ndependencies=["torch"]\n', encoding="utf-8"
    )
    assert main(["prepare", str(tmp_path)]) == 0
    text = (tmp_path / "scripts" / "train.sbatch").read_text(encoding="utf-8")
    assert "--gres=gpu:1" in text


def test_pan_plan_never_executes_and_prints_copy(tmp_path: Path, capsys) -> None:
    (tmp_path / "data").mkdir()
    assert main(["pan-plan", str(tmp_path), "--remote", "pan:project"]) == 0
    output = capsys.readouterr().out
    assert "rclone copy ./data pan:project/data -P" in output
    assert "nothing was uploaded" in output


def test_fetch_plan_prints_result_copy_commands(tmp_path: Path, capsys) -> None:
    assert main(["fetch-plan", str(tmp_path), "--remote", "pan:project"]) == 0
    output = capsys.readouterr().out
    assert "rclone copy pan:project/outputs ./outputs -P" in output
    assert "nothing was downloaded" in output


def test_template_train_imports_sibling_modules(tmp_path: Path) -> None:
    assert main(["template", "demo", "--path", str(tmp_path)]) == 0
    text = (tmp_path / "demo" / "src" / "train.py").read_text(encoding="utf-8")
    assert "from data import load_data" in text
