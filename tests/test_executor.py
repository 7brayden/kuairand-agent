"""Timeout kill, capture truncation, and the workspace git plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

from agent.executor import MAX_CAPTURE_CHARS, Workspace, _truncate, run_pipeline


def test_timeout_kills_and_reports(tmp_path: Path) -> None:
    (tmp_path / "spin.py").write_text("import time\nwhile True: time.sleep(0.1)\n")
    res = run_pipeline([sys.executable, "spin.py"], tmp_path, timeout_seconds=2)
    assert res.timed_out and not res.ok
    assert "hard timeout" in res.stderr


def test_nonzero_exit_is_captured_not_raised(tmp_path: Path) -> None:
    (tmp_path / "boom.py").write_text("raise SystemExit(3)\n")
    res = run_pipeline([sys.executable, "boom.py"], tmp_path, timeout_seconds=30)
    assert res.returncode == 3 and not res.ok and not res.timed_out


def test_success_path(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text("print('hello')\n")
    res = run_pipeline([sys.executable, "ok.py"], tmp_path, timeout_seconds=30)
    assert res.ok and "hello" in res.stdout


def test_capture_is_truncated_head_and_tail() -> None:
    out = _truncate("A" * 100 + "B" * 100, limit=40)
    assert out.startswith("A") and out.endswith("B") and "elided" in out
    assert len(_truncate("x" * (MAX_CAPTURE_CHARS * 3))) < MAX_CAPTURE_CHARS * 2


def _seed(tmp_path: Path) -> Workspace:
    template = tmp_path / "template"
    template.mkdir()
    (template / "main.py").write_text("v = 1\n")
    ws = Workspace(tmp_path / "ws")
    assert ws.init_if_needed(template) is True
    return ws


def test_init_is_idempotent(tmp_path: Path) -> None:
    ws = _seed(tmp_path)
    assert ws.init_if_needed(tmp_path / "template") is False  # resume must not wipe history


def test_accept_commits_and_exports_patch(tmp_path: Path) -> None:
    ws = _seed(tmp_path)
    ws.write_file("main.py", "v = 2\n")
    sha = ws.commit("model: bump")
    patch = ws.export_patch(tmp_path / "diffs" / "n1.patch")
    assert sha and patch is not None and "v = 2" in patch.read_text()


def test_reject_reverts_to_clean_tree(tmp_path: Path) -> None:
    ws = _seed(tmp_path)
    ws.write_file("main.py", "v = 999\n")
    ws.write_file("stray.txt", "junk")
    ws.revert_all()
    assert ws.read_file("main.py") == "v = 1\n"
    assert not (ws.path / "stray.txt").exists()
    assert not ws.is_dirty()


def test_backtracking_branches_from_earlier_commit(tmp_path: Path) -> None:
    """The commit graph is the search tree: an old node must stay reachable."""
    ws = _seed(tmp_path)
    ws.write_file("main.py", "v = 2\n")
    first = ws.commit("a")
    ws.write_file("main.py", "v = 3\n")
    ws.commit("b")
    ws.branch_from(first, "backtrack-1")
    assert ws.read_file("main.py") == "v = 2\n"
