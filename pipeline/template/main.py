"""KuaiRand-Pure ranking pipeline — the agent owns everything below the CONTRACT line.

Run by the harness as a subprocess with a hard timeout:

    python3 main.py --data-dir <raw csv dir> --split valid --out-dir <dir>

Writes:
  <out-dir>/predictions.csv   official schema: row_id,user_id,video_id,score
  <out-dir>/checkpoint/       optional; whatever is needed to reproduce inference

TASK CONTRACT (fixed by the organisers — see eval/official/evaluate.py):
  * within-user ranking over logged impressions; no full-catalogue retrieval
  * label is `long_view` (0/1); metrics are GAUC and nDCG@5, primary = their mean
  * ONLY relative order within a user matters. Any term constant within a user
    (e.g. a pure user-side bias) cannot change the score at all.

ROW ORDER IS LOAD-BEARING: predictions must be in the canonical order — the two
standard log files read in the order listed in LOG_FILES, original row order kept
within each file, then filtered by date. The official validator rejects any other
order, so do not sort, group, or shuffle the target frame.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

# The organisers' published baseline ships alongside this file, in official/. The rules
# permit any public solution, so the agent may import and adapt it rather than
# reimplementing a factorization machine from memory. Its modules import each other by
# bare name, so their directory goes on the path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "official"))

# ----------------------------- CONTRACT (do not change) -----------------------------

LABEL = "long_view"
SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}
LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")
VIDEO_FEATURES = "video_features_basic_pure.csv"
RANDOM_LOG = "log_random_4_22_to_5_08_pure.csv"

#: The random-exposure log spans 04-22..05-08, which covers the TEST window. Only its
#: validation-window portion is ever exposed, so no future label can reach the model —
#: the restriction is structural rather than a rule the agent is asked to respect.
RANDOM_LOG_WINDOW = SPLITS["valid"]


def load_logs(data_dir: str) -> pd.DataFrame:
    """Load both standard logs in canonical order, with item attributes joined on.

    ids stay `str` so the submission's user_id/video_id match the official validator
    byte for byte (it compares raw CSV text, and any float coercion would render 0 as
    "0.0" and fail alignment).

    The item-side join mirrors the official loader, which pulls `author_id` from
    `video_features_basic_pure.csv` — that column is NOT in the log files, and the
    baseline's five fields include it. A left merge preserves the left frame's row
    order, which is the submission contract.
    """
    frames = [pd.read_csv(os.path.join(data_dir, f), dtype={"user_id": str, "video_id": str})
              for f in LOG_FILES]
    logs = pd.concat(frames, ignore_index=True)
    vf = pd.read_csv(os.path.join(data_dir, VIDEO_FEATURES), dtype={"video_id": str})
    return logs.merge(vf, on="video_id", how="left")


def load_unbiased(data_dir: str) -> pd.DataFrame:
    """Random-exposure log, restricted to the validation window.

    These impressions were shown uniformly at random rather than chosen by the
    recommender, so per-video statistics computed here measure what people actually
    watch. Popularity measured on the standard log correlates only 0.375 with this
    (see reports/data_bias_analysis.md), because the standard log largely records what
    the recommender promoted.

    Use it to build better ITEM-side features. Do not train the main objective on it and
    do not reweight toward it: scoring ranks within the recommender's own impressions, so
    biased traffic is the target distribution.
    """
    lo, hi = RANDOM_LOG_WINDOW
    df = pd.read_csv(os.path.join(data_dir, RANDOM_LOG), dtype={"user_id": str, "video_id": str})
    df = df[(df["date"] >= lo) & (df["date"] <= hi)].reset_index(drop=True)
    # Join the same item attributes as load_logs. Without this the frame silently lacks
    # author_id, tag, video_duration and friends — and item-side features are the entire
    # reason this frame exists, so the first thing anyone reaches for would raise.
    vf = pd.read_csv(os.path.join(data_dir, VIDEO_FEATURES), dtype={"video_id": str})
    return df.merge(vf, on="video_id", how="left")


def split_of(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Rows of one official split, canonical order preserved."""
    lo, hi = SPLITS[name]
    return df[(df["date"] >= lo) & (df["date"] <= hi)].reset_index(drop=True)


def write_predictions(out_dir: str, target: pd.DataFrame, scores: np.ndarray) -> str:
    """Write predictions.csv in the official submission schema."""
    scores = np.asarray(scores, dtype=float).ravel()
    if len(scores) != len(target):
        raise ValueError(f"got {len(scores)} scores for {len(target)} rows")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN/Inf, which the official validator rejects")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame({
        "row_id": np.arange(len(target), dtype=int),
        "user_id": target["user_id"].values,
        "video_id": target["video_id"].values,
        "score": [f"{s:.6g}" for s in scores],
    }).to_csv(path, index=False)
    return path


# ------------------------------- AGENT-OWNED ZONE -----------------------------------
# Everything below is what the agent rewrites. The seed implementation deliberately
# contains no ML: it scores every row 0.0, which is a valid-but-worthless submission.
# Milestone 0 (harness proof) runs exactly this.

def fit_predict(train: pd.DataFrame, valid: pd.DataFrame, target: pd.DataFrame,
                checkpoint_dir: str, unbiased: pd.DataFrame) -> np.ndarray:
    """Fit on `train`, return one score per row of `target` (canonical order).

    Args:
        train: the training split (20220408-0421), all raw log columns.
        valid: the validation split (20220422-0428). **Use this for early stopping and
            model selection** — the official FM baseline evaluates on it every epoch and
            stops after 4 rounds without improvement. Never early-stop on `target`.
        target: the split to score; return exactly len(target) scores.
        checkpoint_dir: write anything needed to reproduce inference here.
        unbiased: random-exposure impressions from the validation window (~500k rows,
            same columns as `train`). Shown uniformly at random, so per-video rates here
            measure genuine appeal rather than what the recommender promoted — the two
            correlate only 0.375. Useful for item-side features; not for training the
            main objective, and not for reweighting.

    Note during development `target` IS `valid` (same rows) — that is the organisers'
    own methodology, matching how the published baseline is tuned. At final evaluation
    `target` becomes the held-out test split while `valid` stays the validation split,
    so code that selects on `valid` transfers correctly and code that peeks at `target`'s
    labels does not.

    Returns:
        np.ndarray of shape (len(target),), finite floats, relative order only.
    """
    return np.zeros(len(target), dtype=float)


# --------------------------- END AGENT-OWNED ZONE -----------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="KuaiRand ranking pipeline (agent-owned).")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    df = load_logs(args.data_dir)
    train = split_of(df, "train")
    valid = split_of(df, "valid")
    target = split_of(df, args.split)
    print(f"train={len(train):,d} | valid={len(valid):,d} | "
          f"{args.split}(target)={len(target):,d} rows", flush=True)

    unbiased = load_unbiased(args.data_dir)
    print(f"unbiased(random exposure, valid window)={len(unbiased):,d} rows", flush=True)

    checkpoint_dir = os.path.join(args.out_dir, "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    scores = fit_predict(train, valid, target, checkpoint_dir, unbiased)

    path = write_predictions(args.out_dir, target, scores)
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
