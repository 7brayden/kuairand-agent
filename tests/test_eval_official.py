"""The vendored organiser code is the scoring authority — guard it and the adapter."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from eval import scorer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/raw/KuaiRand-Pure/data"
needs_data = pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not downloaded")


def test_vendored_files_are_unmodified() -> None:
    """Any edit here silently changes the scoring rules — fail loudly instead."""
    r = subprocess.run(["shasum", "-a", "256", "-c", "CHECKSUMS.txt"],
                       cwd=ROOT / "eval/official", capture_output=True, text=True)
    assert r.returncode == 0, f"vendored organiser code was modified:\n{r.stdout}{r.stderr}"


def test_loop_cannot_score_test_split() -> None:
    """The honour-system guard: only scripts/final_eval.py may pass allow_test."""
    with pytest.raises(scorer.TestSetAccessError):
        scorer.score_submission("ignored.csv", str(DATA_DIR), split="test")


@needs_data
def test_random_self_check_matches_published() -> None:
    """The starter kit's mandated harness check: random -> valid primary ~ 0.4834."""
    res = scorer.self_check(str(DATA_DIR))
    assert res["ok"], f"expected ~{res['expected']}, got {res['got']:.4f}"


@needs_data
def test_split_sizes_match_official_windows() -> None:
    sizes = {k: len(v) for k, v in scorer.load_splits(str(DATA_DIR)).items()}
    assert sizes == {"train": 1_141_112, "valid": 124_909, "test": 170_588}


@needs_data
def test_pipeline_row_order_matches_the_official_loader_exactly() -> None:
    """The submission contract: predictions must be in the official row order.

    The template joins item features onto the logs (a left merge, which preserves left
    order) so `author_id` is reachable — the agent crashed twice reaching for a column
    the baseline uses but the raw log does not contain. Any change to loading must keep
    this exact, so it is pinned rather than checked by hand.
    """
    import sys
    sys.path.insert(0, str(ROOT / "pipeline/template"))
    import main as template

    df = template.load_logs(str(DATA_DIR))
    for split in ("train", "valid", "test"):
        mine = template.split_of(df, split)
        official = scorer.split_rows(str(DATA_DIR), split)
        assert len(mine) == len(official), split
        assert list(mine["user_id"]) == [r[1] for r in official], f"{split}: user order"
        assert list(mine["video_id"]) == [r[2] for r in official], f"{split}: video order"
        assert [1 if v else 0 for v in mine["long_view"]] == \
               [r[6] for r in official], f"{split}: labels"


@needs_data
def test_columns_the_prompt_promises_actually_exist() -> None:
    """The agent is told an exact column list; a stale list costs it an iteration."""
    import re
    import sys
    sys.path.insert(0, str(ROOT / "pipeline/template"))
    import main as template

    brief = (ROOT / "agent/prompts/_task_brief_v1.md").read_text()
    section = brief.split("The exact columns of")[1].split("`dur_bucket` is not a column")[0]
    promised = set(re.findall(r"`([a-z_]+)`", section)) - {
        "train", "valid", "target", "video_features_basic_pure.csv", "duration_ms_"}
    actual = set(template.load_logs(str(DATA_DIR)).columns)
    missing = {c for c in promised if c not in actual and "_" in c}
    assert not missing, f"prompt promises columns that do not exist: {sorted(missing)}"
