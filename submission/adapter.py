"""The ONLY file that knows the organiser submission schema.

Schema (published in the starter kit, `eval/official/submit.py`):

    row_id,user_id,video_id,score
    0,0,7531,-3.34176

* ``row_id`` — 0-based, contiguous, in the canonical split order (both standard log
  files read in order, original row order kept, then date-filtered).
* ``user_id`` / ``video_id`` — redundant, used only to verify alignment.
* ``score`` — any finite real; only relative order within a user matters.

``(user_id, video_id)`` is NOT a key: 3.06% of test rows are duplicate pairs, up to
12 repeats. That is exactly why ``row_id`` exists, and why nothing here may sort,
group, or deduplicate.

Because the pipeline already emits this schema on every iteration (and the official
validator runs on every iteration's output), the format is exercised continuously
instead of being discovered broken on submission day. This adapter is therefore
thin by design: validate with the organisers' own code, then place the file.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from eval import scorer


def validate(predictions_csv: str | Path, data_dir: str, split: str) -> int:
    """Validate alignment using the organisers' own reader. Returns the row count.

    Raises ValueError (from the official reader) on a bad header, wrong row count,
    non-contiguous row_id, misaligned user/video ids, or NaN/Inf scores.
    """
    rows = scorer.split_rows(data_dir, split)
    scores = scorer.official_submit.read_submission(str(predictions_csv), rows)
    return len(scores)


def export_submission(predictions_csv: str | Path, out_path: str | Path,
                      data_dir: str, split: str = "test") -> Path:
    """Validate and place the final submission file.

    The pipeline's own output is already in the organiser schema, so this copies
    rather than converts — but it never copies unvalidated: a submission that fails
    the official reader must never reach the organisers.
    """
    n = validate(predictions_csv, data_dir, split)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(predictions_csv, out_path)
    print(f"submission validated and written: {out_path} ({n:,d} rows, split={split})")
    return out_path
