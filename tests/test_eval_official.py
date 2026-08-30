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
