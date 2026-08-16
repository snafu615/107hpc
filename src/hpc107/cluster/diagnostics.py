"""Deterministic classifications for common Slurm and application failures."""

from __future__ import annotations

from hpc107.common.models import RunState


def diagnose(state: RunState, log_tail: str = "") -> str:
    slurm_state = state.state.upper()
    text = log_tail.lower()
    if slurm_state in {"OUT_OF_MEMORY", "OOM"} or "cuda out of memory" in text:
        return "memory exhaustion: reduce batch size or request appropriate memory"
    if slurm_state == "TIMEOUT":
        return "wall-time exhausted: measure a bounded run and request a justified limit"
    if slurm_state in {"NODE_FAIL", "BOOT_FAIL"}:
        return "cluster infrastructure failure: inspect platform status before retrying"
    if "modulenotfounderror" in text or "no module named" in text:
        return "missing Python dependency: correct the environment manifest and prepare again"
    if "no space left on device" in text:
        return "storage exhausted: inspect quota and move large datasets to approved storage"
    if slurm_state == "COMPLETED" and state.exit_code in {"", "0:0", "0"}:
        return "completed successfully"
    if slurm_state in {"FAILED", "CANCELLED", "PREEMPTED"}:
        return f"Slurm ended in {slurm_state}; inspect the recorded stdout/stderr logs"
    return "unclassified state: inspect Slurm accounting and bounded log tails"
