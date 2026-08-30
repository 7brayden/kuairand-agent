"""Thin adapter over the vendored official evaluator. The ONLY scoring door.

Design note — there is deliberately no fast approximation. The brief anticipated an
``eval/fast.py`` asserted equal to the official script, but the official
``evaluate.py`` turned out to be pure-stdlib and cheap: a full load + score of the
valid split measures ~2.7s end to end. A second implementation would add a real
correctness risk (silently diverging metrics) for no measurable payoff, so the loop
calls the organisers' code directly. If profiling ever justifies a fast path, it
must ship with a parity assertion against this module.

Test-set integrity
------------------
The test split's labels are present locally (KuaiRand is public), so "hidden test"
is an honour-system constraint that this module enforces mechanically:
``score_submission`` refuses ``split="test"`` unless ``allow_test=True``, which only
``scripts/final_eval.py`` passes, exactly once, at convergence. The loop has no path
to a test score. See CLAUDE.md > Test-set integrity.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

_OFFICIAL = Path(__file__).resolve().parent / "official"
if str(_OFFICIAL) not in sys.path:
    # The vendored modules import each other by bare name (``from data import load``),
    # so their directory must be importable. Appended, never prepended, so we cannot
    # shadow a stdlib module for the rest of the process.
    sys.path.append(str(_OFFICIAL))

import data as official_data          # noqa: E402  (vendored, unmodified)
import evaluate as official_evaluate  # noqa: E402  (vendored, unmodified — the authority)
import submit as official_submit      # noqa: E402  (vendored, unmodified)

#: Row tuple layout produced by the official loader: (date, user, video, author, tab, dur, label)
USER_IDX, LABEL_IDX = 1, 6

#: Splits the agent loop is allowed to score. Test is excluded by design.
LOOP_SPLITS = ("valid",)


class TestSetAccessError(RuntimeError):
    """Raised when anything but the one-shot final evaluation asks for a test score."""


@lru_cache(maxsize=4)
def load_splits(data_dir: str) -> dict[str, list[tuple]]:
    """Load and cache the official splits (~2.7s cold, free thereafter).

    Cached per-process: the loop scores once per iteration and would otherwise
    re-parse 100MB of CSV every time.
    """
    return official_data.load(data_dir)


def split_rows(data_dir: str, split: str) -> list[tuple]:
    """Rows of one official split, in the canonical order submissions must match."""
    return load_splits(data_dir)[split]


def score_arrays(user_ids: Sequence, labels: Sequence, scores: Sequence) -> dict[str, float]:
    """Score raw arrays with the official evaluator. Returns GAUC / nDCG@5 / primary."""
    return official_evaluate.evaluate(list(user_ids), list(labels), list(scores))


def score_submission(path: str | Path, data_dir: str, split: str = "valid",
                     *, allow_test: bool = False) -> dict[str, float]:
    """Validate a submission CSV against the split, then score it.

    Alignment is checked by the organisers' own ``submit.read_submission`` — header,
    row count, contiguous ``row_id``, matching ``user_id``/``video_id``, and no
    NaN/Inf. Running that on EVERY iteration means the submission format is exercised
    continuously instead of being discovered broken at submission time.

    Raises :class:`TestSetAccessError` if asked for the test split without the
    explicit one-shot opt-in.
    """
    if split == "test" and not allow_test:
        raise TestSetAccessError(
            "the agent loop must never score the test split; test is evaluated exactly "
            "once by scripts/final_eval.py from the validation-best checkpoint"
        )
    rows = split_rows(data_dir, split)
    scores = official_submit.read_submission(str(path), rows)
    return score_arrays([r[USER_IDX] for r in rows], [r[LABEL_IDX] for r in rows], scores)


def self_check(data_dir: str, seed: int = 0) -> dict[str, Any]:
    """The starter kit's mandated harness self-check.

    Scoring uniform-random predictions must land at valid primary ~= 0.4834
    (published; +/-0.001). A miss means the evaluation path is broken and nothing
    downstream can be trusted — fix it before running the agent.
    """
    import numpy as np

    rows = split_rows(data_dir, "valid")
    rng = np.random.default_rng(seed)
    got = score_arrays([r[USER_IDX] for r in rows], [r[LABEL_IDX] for r in rows],
                       rng.random(len(rows)))
    expected = 0.4834
    ok = abs(got["primary"] - expected) <= 0.001
    return {"ok": ok, "expected": expected, "got": got["primary"], "detail": got}
