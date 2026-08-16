"""Slurm subprocess adapter with parsable results and resumable monitoring."""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hpc107.common.errors import SlurmError
from hpc107.common.models import ResourceSettings

TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}


@dataclass(frozen=True, slots=True)
class JobStatus:
    job_id: str
    state: str
    exit_code: str = ""
    elapsed: str = ""
    node_list: str = ""
    alloc_tres: str = ""


Runner = Callable[..., subprocess.CompletedProcess[str]]


class SlurmClient:
    def __init__(self, runner: Runner = subprocess.run) -> None:
        self.runner = runner

    def _run(self, args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
        return self.runner(args, cwd=cwd, capture_output=True, text=True, check=False)

    def require_commands(self, *names: str) -> None:
        required = names or ("sinfo", "sbatch", "squeue", "sacct", "bash")
        missing = [name for name in required if shutil.which(name) is None]
        if missing:
            raise SlurmError("Missing Slurm commands: " + ", ".join(missing))

    def preflight(self, resources: ResourceSettings) -> str:
        result = self._run(["sinfo", "-p", resources.partition, "-h"])
        if result.returncode != 0 or not result.stdout.strip():
            detail = result.stderr.strip() or "partition returned no rows"
            raise SlurmError(f"Partition {resources.partition!r} is unavailable: {detail}")
        return result.stdout.strip()

    def validate_script(self, root: Path, script: Path) -> None:
        syntax = self._run(["bash", "-n", str(script)], cwd=root)
        if syntax.returncode != 0:
            raise SlurmError(syntax.stderr.strip() or "bash syntax validation failed")
        help_result = self._run(["sbatch", "--help"])
        if "--test-only" in help_result.stdout:
            tested = self._run(["sbatch", "--test-only", str(script)], cwd=root)
            if tested.returncode != 0:
                raise SlurmError(tested.stderr.strip() or "sbatch --test-only rejected the script")

    def submit(self, root: Path, script: Path) -> str:
        result = self._run(["sbatch", "--parsable", str(script)], cwd=root)
        if result.returncode != 0:
            raise SlurmError(result.stderr.strip() or "sbatch failed")
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise SlurmError(f"Could not parse sbatch job ID from {result.stdout!r}")
        return job_id

    def status(self, job_id: str) -> JobStatus:
        if not job_id.isdigit():
            raise SlurmError("Job ID must be numeric")
        queue = self._run(["squeue", "-h", "-j", job_id, "-o", "%i|%T|%M|%L|%R|%N"])
        if queue.returncode != 0:
            raise SlurmError(queue.stderr.strip() or "squeue failed")
        line = queue.stdout.strip().splitlines()
        if line:
            values = line[0].split("|")
            return JobStatus(
                job_id=job_id,
                state=values[1].upper() if len(values) > 1 else "UNKNOWN",
                elapsed=values[2] if len(values) > 2 else "",
                node_list=values[5] if len(values) > 5 else "",
            )
        accounting = self._run(
            [
                "sacct",
                "-X",
                "-j",
                job_id,
                "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,AllocTRES",
                "--parsable2",
                "--noheader",
            ]
        )
        if accounting.returncode != 0:
            raise SlurmError(accounting.stderr.strip() or "sacct failed")
        for row in accounting.stdout.splitlines():
            values = row.split("|")
            if len(values) >= 6 and values[0].strip() == job_id:
                return JobStatus(
                    job_id=job_id,
                    state=values[1].strip().split()[0].rstrip("+").upper(),
                    exit_code=values[2].strip(),
                    elapsed=values[3].strip(),
                    node_list=values[4].strip(),
                    alloc_tres="|".join(values[5:]).strip("|"),
                )
        return JobStatus(job_id=job_id, state="UNKNOWN")

    def watch(self, job_id: str, *, poll_seconds: int = 10) -> JobStatus:
        last = ""
        while True:
            current = self.status(job_id)
            if current.state != last:
                print(f"{job_id}: {current.state}")
                last = current.state
            if current.state in TERMINAL_STATES:
                return current
            time.sleep(max(2, poll_seconds))
