"""Cluster CLI for deterministic validation, Slurm execution, and monitoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hpc107 import __version__
from hpc107.common.config import load_settings
from hpc107.common.errors import HPC107Error
from hpc107.common.models import RunState, Settings
from hpc107.common.project import require_valid_project
from hpc107.common.script import (
    script_path,
    sha256_file,
    validate_script,
    write_script,
)

from .artifacts import pan_upload_plan, write_inventory
from .diagnostics import diagnose
from .environment import check_environment, prepare_environment
from .slurm import JobStatus, SlurmClient
from .state import find_run, new_run_id, read_state, run_directory, utc_now, write_state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hpc107",
        description="Run and monitor deterministic computing tasks on USTC 107",
    )
    parser.add_argument("--version", action="version", version=f"hpc107 {__version__}")
    commands = parser.add_subparsers(dest="command")

    render = commands.add_parser("render", help="Render the deterministic Slurm script")
    render.add_argument("project", nargs="?", default=".")
    render.add_argument("--force", action="store_true")

    check = commands.add_parser(
        "check", help="Validate project, environment, script, and partition"
    )
    check.add_argument("project", nargs="?", default=".")
    check.add_argument("--skip-live", action="store_true")

    submit = commands.add_parser("submit", help="Submit the validated deterministic script")
    submit.add_argument("project", nargs="?", default=".")
    submit.add_argument("--yes", action="store_true")
    submit.add_argument("--auto", action="store_true", help="Compatibility alias for --yes")
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--skip-check", action="store_true")
    submit.add_argument("--watch", action="store_true")
    submit.add_argument("--poll-seconds", type=int, default=10)

    status = commands.add_parser("status", help="Refresh and print one recorded run")
    status.add_argument("run_id")
    status.add_argument("--project", default=".")

    watch = commands.add_parser("watch", help="Resume monitoring a recorded run")
    watch.add_argument("run_id")
    watch.add_argument("--project", default=".")
    watch.add_argument("--poll-seconds", type=int, default=10)

    artifacts = commands.add_parser("artifacts", help="Inventory results and print Pan copy plans")
    artifacts.add_argument("run_id")
    artifacts.add_argument("--project", default=".")

    diagnostic = commands.add_parser("diagnose", help="Classify a recorded run deterministically")
    diagnostic.add_argument("run_id")
    diagnostic.add_argument("--project", default=".")
    return parser


def _validate_local(root: Path, settings: Settings) -> Path:
    report = require_valid_project(root)
    configured_entry = root / settings.project.entry
    if not configured_entry.is_file():
        raise HPC107Error(
            f"Configured entry does not exist: {settings.project.entry}; detected {report.entry}"
        )
    target = script_path(root, settings)
    errors = validate_script(target, settings)
    errors.extend(check_environment(root, settings))
    if errors:
        raise HPC107Error("Validation failed:\n- " + "\n- ".join(errors))
    return target


def _apply_status(
    state: RunState,
    current: JobStatus,
    run_dir: Path,
    settings: Settings,
) -> None:
    state.state = current.state
    state.exit_code = current.exit_code
    state.elapsed = current.elapsed
    state.node_list = current.node_list
    state.alloc_tres = current.alloc_tres
    state.diagnosis = diagnose(state, _log_tail(Path(state.project_root), state, settings))
    write_state(run_dir, state)


def _log_tail(root: Path, state: RunState, settings: Settings, limit: int = 80_000) -> str:
    log_root = root / settings.paths.logs
    candidates = [
        log_root / f"{settings.project.job_name}_{state.job_id}.out",
        log_root / f"{settings.project.job_name}_{state.job_id}.err",
    ]
    blocks: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            blocks.append(stream.read().decode("utf-8", errors="replace"))
    return "\n".join(blocks)


def _print_state(state: RunState) -> None:
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command is None:
        _parser().print_help()
        return 0
    try:
        if args.command in {"render", "check", "submit"}:
            root = Path(args.project).resolve()
            settings = load_settings(root)
        else:
            root = Path(args.project).resolve()
            settings = load_settings(root)

        if args.command == "render":
            require_valid_project(root)
            target = write_script(root, settings, force=args.force)
            print(f"Rendered: {target}")
            return 0

        if args.command == "check":
            target = _validate_local(root, settings)
            if not args.skip_live:
                client = SlurmClient()
                client.require_commands("sinfo", "sbatch", "bash")
                snapshot = client.preflight(settings.resources)
                client.validate_script(root, target)
                print(snapshot)
            print(f"Valid project and Slurm script: {target}")
            return 0

        if args.command == "submit":
            target = _validate_local(root, settings)
            digest = sha256_file(target)
            print("SUBMISSION SUMMARY")
            print(f"  Project: {root}")
            print(f"  Entry: {settings.project.entry}")
            print(f"  Script: {target}")
            print(f"  Script SHA-256: {digest}")
            print(f"  Partition/QOS: {settings.resources.partition}/{settings.resources.qos}")
            print(
                f"  Resources: CPU={settings.resources.cpus} MEM={settings.resources.memory} "
                f"GPU={settings.resources.gpus} TIME={settings.resources.walltime}"
            )
            if args.dry_run:
                print("Dry run: no environment changes or Slurm submission were performed.")
                return 0
            if (
                not (args.yes or args.auto)
                and input("Type SUBMIT to continue: ").strip() != "SUBMIT"
            ):
                print("Not submitted.")
                return 0

            prepare_environment(root, settings)
            for directory in (
                settings.paths.logs,
                settings.paths.outputs,
                settings.paths.checkpoints,
            ):
                (root / directory).mkdir(parents=True, exist_ok=True)
            client = SlurmClient()
            required_commands = ["sbatch", "bash"]
            if not args.skip_check:
                required_commands.append("sinfo")
            if args.watch:
                required_commands.extend(("squeue", "sacct"))
            client.require_commands(*required_commands)
            run_id = new_run_id(root.name)
            run_dir = run_directory(root, settings, run_id)
            state = RunState(
                run_id=run_id,
                state="VALIDATED",
                project_root=str(root),
                script=str(target.relative_to(root)),
                script_sha256=digest,
            )
            write_state(run_dir, state)
            if not args.skip_check:
                client.preflight(settings.resources)
            client.validate_script(root, target)
            state.state = "PREFLIGHTED"
            write_state(run_dir, state)
            if sha256_file(target) != digest:
                raise HPC107Error("The Slurm script changed after validation")
            job_id = client.submit(root, target)
            state.job_id = job_id
            state.state = "SUBMITTED"
            state.submitted_at = utc_now()
            write_state(run_dir, state)
            print(f"Run ID: {run_id}")
            print(f"Submitted batch job {job_id}")
            print(f"State: {run_dir / 'state.json'}")
            if args.watch:
                current = client.watch(job_id, poll_seconds=args.poll_seconds)
                _apply_status(state, current, run_dir, settings)
                write_inventory(run_dir, root, settings)
                _print_state(state)
            return 0

        run_dir = find_run(root, settings, args.run_id)
        state = read_state(run_dir)
        if args.command == "status":
            if state.job_id:
                client = SlurmClient()
                client.require_commands("squeue", "sacct")
                _apply_status(state, client.status(state.job_id), run_dir, settings)
            _print_state(state)
            return 0
        if args.command == "watch":
            if not state.job_id:
                raise HPC107Error("The run has no recorded Slurm job ID")
            client = SlurmClient()
            client.require_commands("squeue", "sacct")
            current = client.watch(state.job_id, poll_seconds=args.poll_seconds)
            _apply_status(state, current, run_dir, settings)
            write_inventory(run_dir, root, settings)
            _print_state(state)
            return 0
        if args.command == "artifacts":
            target = write_inventory(run_dir, root, settings)
            print(f"Artifact inventory: {target}")
            plans = pan_upload_plan(root, settings)
            if plans:
                print("Review and run these commands manually:")
                for command in plans:
                    print(command)
            else:
                print("No Pan remote configured; use rsync or configure datasets.pan_remote.")
            return 0
        if args.command == "diagnose":
            state.diagnosis = diagnose(state, _log_tail(root, state, settings))
            write_state(run_dir, state)
            print(state.diagnosis)
            return 0
    except (HPC107Error, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
