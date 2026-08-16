from pathlib import Path

from hpc107.cluster.diagnostics import diagnose
from hpc107.cluster.state import read_state, write_state
from hpc107.common.models import RunState


def _state(state: str = "SUBMITTED") -> RunState:
    return RunState(
        run_id="run-1",
        state=state,
        project_root="/project",
        script="scripts/train.sbatch",
        script_sha256="abc",
        job_id="12345",
    )


def test_state_round_trip_is_atomic(tmp_path: Path) -> None:
    state = _state()
    write_state(tmp_path, state)
    loaded = read_state(tmp_path)
    assert loaded.job_id == "12345"
    assert not list(tmp_path.glob("*.tmp"))


def test_diagnoses_cuda_oom() -> None:
    assert "memory exhaustion" in diagnose(_state("FAILED"), "CUDA out of memory")


def test_diagnoses_timeout() -> None:
    assert "wall-time" in diagnose(_state("TIMEOUT"))
