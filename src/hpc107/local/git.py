"""Small, argument-list-based Git adapter for the local CLI."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from hpc107.common.errors import HPC107Error


def _run(root: Path, args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=root, capture_output=True, text=True, check=check)


def ensure_repository(root: Path) -> None:
    if shutil.which("git") is None:
        raise HPC107Error("git is not installed or is not on PATH")
    result = _run(root, ["git", "rev-parse", "--show-toplevel"])
    if result.returncode != 0:
        result = _run(root, ["git", "init"])
        if result.returncode != 0:
            raise HPC107Error(result.stderr.strip() or "git init failed")


def ensure_remote(root: Path, remote: str | None) -> tuple[str, str]:
    if remote:
        current = _run(root, ["git", "remote", "get-url", "origin"])
        command = (
            ["git", "remote", "set-url", "origin", remote]
            if current.returncode == 0
            else ["git", "remote", "add", "origin", remote]
        )
        result = _run(root, command)
        if result.returncode != 0:
            raise HPC107Error(result.stderr.strip() or "Could not configure origin")
        return "origin", remote
    remotes = _run(root, ["git", "remote"])
    names = [line.strip() for line in remotes.stdout.splitlines() if line.strip()]
    if not names:
        raise HPC107Error("No Git remote is configured; provide --remote")
    name = names[0]
    url = _run(root, ["git", "remote", "get-url", name])
    if url.returncode != 0:
        raise HPC107Error(url.stderr.strip() or f"Could not read remote {name}")
    return name, url.stdout.strip()


def current_branch(root: Path) -> str:
    result = _run(root, ["git", "branch", "--show-current"])
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise HPC107Error("No current Git branch; detached HEAD is not supported")
    return branch


def push_project(
    root: Path,
    *,
    remote: str | None,
    message: str,
    commit: bool,
) -> dict[str, str]:
    ensure_repository(root)
    name, url = ensure_remote(root, remote)
    status = _run(root, ["git", "status", "--porcelain"])
    if status.returncode != 0:
        raise HPC107Error(status.stderr.strip() or "git status failed")
    if status.stdout.strip():
        if not commit:
            raise HPC107Error("The project has uncommitted changes; pass --commit to include them")
        added = _run(root, ["git", "add", "-A"])
        if added.returncode != 0:
            raise HPC107Error(added.stderr.strip() or "git add failed")
        committed = _run(root, ["git", "commit", "-m", message])
        if committed.returncode != 0:
            raise HPC107Error(committed.stderr.strip() or committed.stdout.strip())
    branch = current_branch(root)
    pushed = _run(root, ["git", "push", "-u", name, branch])
    if pushed.returncode != 0:
        raise HPC107Error(pushed.stderr.strip() or pushed.stdout.strip())
    return {"remote": name, "url": url, "branch": branch}


def repository_info(root: Path) -> dict[str, str] | None:
    if (
        shutil.which("git") is None
        or _run(root, ["git", "rev-parse", "--is-inside-work-tree"]).returncode
    ):
        return None
    branch = current_branch(root)
    remotes = _run(root, ["git", "remote"])
    names = [line.strip() for line in remotes.stdout.splitlines() if line.strip()]
    if not names:
        return {"remote": "", "url": "", "branch": branch}
    name = names[0]
    url = _run(root, ["git", "remote", "get-url", name]).stdout.strip()
    return {"remote": name, "url": url, "branch": branch}
