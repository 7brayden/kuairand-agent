"""Subprocess execution and workspace git plumbing.

Generated pipeline code runs via ``subprocess`` with a hard timeout — never
``exec()``. We need process isolation, reliable kill-on-timeout (the pipeline may
spawn children, so we kill the whole process group), and clean stdout/stderr
capture, because the recovery path (critic / debug action) consumes them.

``pipeline/workspace/`` is its own git repo (gitignored from the outer repo):

  accepted iteration  -> commit; diff deliverable = ``git format-patch -1 HEAD --stdout``
  rejected iteration  -> ``git checkout . && git clean -fd``
  policy backtrack    -> branch from the earlier commit

so the workspace commit graph IS the search tree, mirrored by parent_id links in
the journal.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Cap on captured stream text. Stderr is fed back to the LLM on the debug path, so
#: an unbounded traceback would blow context and cost tokens (15% of the score).
MAX_CAPTURE_CHARS = 8000


def _truncate(text: str, limit: int = MAX_CAPTURE_CHARS) -> str:
    """Keep the head and tail — the head has the command echo, the tail has the error."""
    if len(text) <= limit:
        return text
    head, tail = text[: limit // 2], text[-limit // 2:]
    return f"{head}\n... [{len(text) - limit} chars elided] ...\n{tail}"


@dataclass
class ExecutionResult:
    """Outcome of one subprocess run of the generated pipeline."""

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    wall_seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


def run_pipeline(cmd: list[str], cwd: Path, timeout_seconds: float,
                 env: Optional[dict[str, str]] = None) -> ExecutionResult:
    """Run ``cmd`` in ``cwd`` with a hard timeout. Never raises on pipeline failure.

    On timeout the whole process group is killed (SIGKILL after SIGTERM), so a
    runaway trainer cannot outlive its iteration and stall the run.
    """
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", start_new_session=True,
        env={**os.environ, **(env or {})},
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_group(proc)
        stdout, stderr = proc.communicate()
        stderr = (stderr or "") + f"\n[harness] killed after {timeout_seconds}s hard timeout"
    return ExecutionResult(
        stdout=_truncate(stdout or ""),
        stderr=_truncate(stderr or ""),
        returncode=proc.returncode if proc.returncode is not None else -9,
        timed_out=timed_out,
        wall_seconds=time.time() - t0,
    )


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGTERM the process group, then SIGKILL anything still alive."""
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


class GitError(RuntimeError):
    """A git command against the workspace failed."""


@dataclass
class Workspace:
    """Git operations on the inner ``pipeline/workspace/`` repo.

    The commit graph is the search tree, so these operations are not bookkeeping —
    they ARE the agent's memory of what it tried and where it can go back to.
    """

    path: Path

    def _git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", "-C", str(self.path), *args],
                           capture_output=True, text=True)
        if check and r.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def init_if_needed(self, template_dir: Path) -> bool:
        """Seed the workspace from the template and ``git init`` on first run.

        Returns True if it initialised, False if the workspace already existed.
        Idempotent: resuming a run must not wipe the agent's history.
        """
        if (self.path / ".git").exists():
            return False
        self.path.mkdir(parents=True, exist_ok=True)
        for src in template_dir.iterdir():
            if src.name in {".git", "__pycache__"}:
                continue
            dst = self.path / src.name
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        self._git("init", "-q")
        self._git("config", "user.email", "agent@kuairand.local")
        self._git("config", "user.name", "kuairand-agent")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "seed: pipeline template (I/O contract only)")
        return True

    def head_sha(self) -> str:
        return self._git("rev-parse", "HEAD")

    def is_dirty(self) -> bool:
        return bool(self._git("status", "--porcelain"))

    def commit(self, message: str) -> str:
        """Commit all changes (accepted iteration). Returns the commit sha."""
        self._git("add", "-A")
        self._git("commit", "-q", "-m", message)
        return self.head_sha()

    def revert_all(self) -> None:
        """Rejected iteration: discard every working-tree change."""
        self._git("checkout", "--", ".")
        self._git("clean", "-fdq")

    def branch_from(self, commit_sha: str, branch_name: str) -> None:
        """Backtrack: create and check out a branch at an earlier accepted node."""
        self.revert_all()
        self._git("checkout", "-q", "-B", branch_name, commit_sha)

    def export_patch(self, out_path: Path) -> Optional[Path]:
        """Write ``git format-patch -1 HEAD --stdout`` to ``out_path``.

        This is the per-iteration "code diff applied" judge deliverable. Returns
        ``None`` when HEAD is the seed commit and there is nothing to export.
        """
        patch = self._git("format-patch", "-1", "HEAD", "--stdout")
        if not patch.strip():
            return None
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(patch + "\n", encoding="utf-8")
        return out_path

    def read_file(self, rel: str) -> str:
        return (self.path / rel).read_text(encoding="utf-8")

    def write_file(self, rel: str, content: str) -> None:
        p = self.path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
