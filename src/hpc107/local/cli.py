"""Local CLI: prepare a conventional task, push code, and plan data transfer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hpc107 import __version__
from hpc107.common.config import LEGACY_CONFIG, NATIVE_CONFIG, load_settings, write_default_config
from hpc107.common.errors import HPC107Error
from hpc107.common.project import infer_defaults, inspect_project, require_valid_project
from hpc107.common.scaffold import create_scaffold
from hpc107.common.script import validate_script, write_script

from .git import push_project, repository_info
from .pan import display, result_download_commands, reverse_for_cluster, upload_commands


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpc107-local",
        description="Prepare and hand off deterministic computing tasks to USTC 107",
    )
    parser.add_argument("--version", action="version", version=f"hpc107-local {__version__}")
    commands = parser.add_subparsers(dest="command")

    template = commands.add_parser("template", help="Create a conventional computing task")
    template.add_argument("name")
    template.add_argument("--path", default=".")

    inspect = commands.add_parser("inspect", help="Inspect the project without modifying it")
    inspect.add_argument("project", nargs="?", default=".")
    inspect.add_argument("--json", action="store_true")

    prepare = commands.add_parser("prepare", help="Create config and deterministic Slurm script")
    prepare.add_argument("project", nargs="?", default=".")
    prepare.add_argument("--force-script", action="store_true")

    push = commands.add_parser("push", help="Commit and push the prepared project")
    push.add_argument("project", nargs="?", default=".")
    push.add_argument("--remote")
    push.add_argument("--commit", action="store_true")
    push.add_argument("--message", default="prepare: deterministic USTC 107 task")

    pan = commands.add_parser("pan-plan", help="Print dataset upload commands")
    pan.add_argument("project", nargs="?", default=".")
    pan.add_argument("--remote")

    fetch = commands.add_parser("fetch-plan", help="Print result download commands")
    fetch.add_argument("project", nargs="?", default=".")
    fetch.add_argument("--remote")

    handoff = commands.add_parser("handoff", help="Print exact cluster-side follow-up")
    handoff.add_argument("project", nargs="?", default=".")
    return parser


def _prepare(root: Path, force_script: bool) -> Path:
    report = require_valid_project(root)
    native = root / NATIVE_CONFIG
    if not native.exists():
        settings = load_settings(root)
        inferred = infer_defaults(root)
        settings.project.entry = report.entry or settings.project.entry
        settings.project.job_name = Path(settings.project.entry).stem
        if not (root / LEGACY_CONFIG).exists():
            settings.environment.manager = str(inferred["environment"])
            settings.resources.gpus = int(inferred["gpus"])
            settings.resources.cpus = int(inferred["cpus"])
            settings.resources.memory = str(inferred["memory"])
            settings.resources.walltime = str(inferred["walltime"])
        write_default_config(root, settings)
    settings = load_settings(root)
    for name in (settings.paths.logs, settings.paths.outputs, settings.paths.checkpoints):
        (root / name).mkdir(parents=True, exist_ok=True)
    target = root / settings.paths.script
    if target.exists() and not force_script:
        errors = validate_script(target, settings)
        if errors:
            raise HPC107Error(
                "Existing Slurm script is incompatible; pass --force-script to replace it:\n- "
                + "\n- ".join(errors)
            )
        return target
    return write_script(root, settings, force=force_script)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        _parser().print_help()
        return 0
    try:
        if args.command == "template":
            target = create_scaffold(Path(args.path) / args.name)
            print(f"Created deterministic computing task: {target}")
            return 0

        root = Path(args.project).resolve()
        if args.command == "inspect":
            report = inspect_project(root)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(f"Project: {report.root}")
                print(f"Valid: {report.valid}")
                print(f"Entry: {report.entry or 'not found'}")
                print(f"Dependencies: {', '.join(report.dependencies) or 'none'}")
                print(f"Inferred environment: {report.inferred_environment}")
                print(f"Inferred resources: {report.inferred_resources}")
                for item in report.missing:
                    print(f"Missing: {item}")
                for item in report.warnings:
                    print(f"Warning: {item}")
            return 0 if report.valid else 1
        if args.command == "prepare":
            target = _prepare(root, args.force_script)
            print(f"Prepared project with script: {target}")
            print("Next: hpc107-local push . --remote <git-url> --commit")
            return 0
        if args.command == "push":
            _prepare(root, False)
            result = push_project(
                root, remote=args.remote, message=args.message, commit=args.commit
            )
            print(f"Pushed {result['branch']} to {result['remote']} ({result['url']})")
            return 0
        if args.command == "pan-plan":
            settings = load_settings(root)
            remote = args.remote or settings.datasets.pan_remote
            commands = upload_commands(root, remote)
            if not commands:
                print("No data/ directory found; no Pan command is needed.")
            for command in commands:
                print(display(command))
            print("Commands were generated only; nothing was uploaded.")
            return 0
        if args.command == "fetch-plan":
            settings = load_settings(root)
            remote = args.remote or settings.datasets.pan_remote
            for command in result_download_commands(remote):
                print(display(command))
            print("Commands were generated only; nothing was downloaded.")
            return 0
        if args.command == "handoff":
            settings = load_settings(root)
            info = repository_info(root)
            if not info or not info["url"]:
                raise HPC107Error("A pushed Git remote is required before handoff")
            project_name = root.name
            print("On the 107 login node:")
            print(display(["git", "clone", info["url"], project_name]))
            print(f"cd {project_name}")
            if settings.datasets.pan_remote:
                for command in reverse_for_cluster(
                    upload_commands(root, settings.datasets.pan_remote)
                ):
                    print(display(command))
            print("hpc107 check .")
            print("hpc107 submit . --yes --watch")
            return 0
    except HPC107Error as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
