import subprocess
from pathlib import Path

import pytest

from hpc107.cluster.slurm import SlurmClient
from hpc107.common.errors import SlurmError
from hpc107.common.models import ResourceSettings


def result(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_preflight_uses_submit107_sinfo_contract() -> None:
    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return result(args, stdout="Students up gpu:1\n")

    output = SlurmClient(runner).preflight(ResourceSettings())
    assert output.startswith("Students")
    assert calls == [["sinfo", "-p", "Students", "-h"]]


def test_invalid_partition_blocks() -> None:
    def runner(args, **kwargs):
        return result(args, returncode=1, stderr="invalid partition")

    with pytest.raises(SlurmError, match="unavailable"):
        SlurmClient(runner).preflight(ResourceSettings())


def test_submit_uses_parsable_job_id(tmp_path: Path) -> None:
    def runner(args, **kwargs):
        return result(args, stdout="12345;cluster\n")

    assert SlurmClient(runner).submit(tmp_path, Path("train.sbatch")) == "12345"


def test_queue_status_is_parsed() -> None:
    def runner(args, **kwargs):
        return result(args, stdout="12345|RUNNING|00:01|00:10|node|node1\n")

    status = SlurmClient(runner).status("12345")
    assert status.state == "RUNNING"
    assert status.node_list == "node1"


def test_terminal_accounting_is_parsed() -> None:
    def runner(args, **kwargs):
        if args[0] == "squeue":
            return result(args, stdout="")
        return result(args, stdout="12345|COMPLETED|0:0|00:05|node1|cpu=4,mem=16G|\n")

    status = SlurmClient(runner).status("12345")
    assert status.state == "COMPLETED"
    assert status.exit_code == "0:0"
