"""Create a submit107-style deterministic computing-task project."""

from __future__ import annotations

from pathlib import Path

from .config import write_default_config
from .errors import ProjectValidationError
from .models import Settings
from .script import write_script

_FILES = {
    "src/__init__.py": "",
    "src/data.py": '''"""Data loading for the computing task."""\n\n\ndef load_data():\n    raise NotImplementedError("Implement src/data.py")\n''',
    "src/model.py": '''"""Model construction for the computing task."""\n\n\ndef build_model():\n    raise NotImplementedError("Implement src/model.py")\n''',
    "src/train.py": '''"""Training entry point."""\n\nfrom data import load_data\nfrom model import build_model\n\n\ndef main() -> None:\n    data = load_data()\n    model = build_model()\n    print(f"Data: {data}")\n    print(f"Model: {model}")\n\n\nif __name__ == "__main__":\n    main()\n''',
    "pyproject.toml": """[project]\nname = "computing-task"\nversion = "0.1.0"\nrequires-python = ">=3.10"\ndependencies = [\n    "torch>=2.0",\n    "torchvision>=0.15",\n    "tqdm>=4.65",\n]\n""",
    ".gitignore": """.venv/\n__pycache__/\n*.pyc\ndata/\nlogs/\noutputs/\ncheckpoints/\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n.hpc107/runs/\n""",
    "README.md": "# Computing task\n\nPrepare locally with `hpc107-local prepare .`, then run on 107 with `hpc107 submit .`.\n",
}


def create_scaffold(target: Path) -> Path:
    target = target.resolve()
    if target.exists():
        raise ProjectValidationError(f"Target already exists: {target}")
    target.mkdir(parents=True)
    settings = Settings()
    settings.resources.gpus = 1
    settings.resources.walltime = "2:00:00"
    for relative, content in _FILES.items():
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    for directory in ("scripts", "logs", "outputs", "checkpoints"):
        (target / directory).mkdir(exist_ok=True)
    write_default_config(target, settings)
    write_script(target, settings, force=False)
    return target
