"""Does matching training group size to evaluation group size help a listwise objective?

Motivation (reports/data_bias_analysis.md, finding 5): the evaluator groups by user_id,
giving median 4-5 impressions per group, while the train split has median 31 per user. A
listwise/pairwise ranker trained on whole users therefore learns to order ~31-43 item
lists and is scored on ~5-7 item lists. Every listwise attempt the agent made
(lambdarank, BPR, softmax) failed to beat a pointwise FM, and this is a candidate
explanation.

Controlled comparison: identical model, features, hyperparameters and seeds; the ONLY
variable is how training rows are chunked into groups.

  whole_user  group = all of a user's train impressions   (median 31)
  chunk_5     each user's impressions split into blocks of 5, matching evaluation
  chunk_8     blocks of 8, a midpoint

Uses train + valid only. Test is not touched.

Run: python3 scripts/experiment_group_size.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline/template"))
sys.path.insert(0, str(ROOT / "eval/official"))

import lightgbm as lgb  # noqa: E402
from evaluate import evaluate  # noqa: E402  (the scoring authority)

import main as template  # noqa: E402

DATA = str(ROOT / "data/raw/KuaiRand-Pure/data")
FIELDS = ["user_id", "video_id", "author_id", "tab", "dur_bucket"]
SEEDS = [0, 1, 2]
ROUNDS = 300


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = template.load_logs(DATA)
    tr, va = template.split_of(df, "train"), template.split_of(df, "valid")
    edges = np.quantile(tr["duration_ms"].to_numpy(float), np.linspace(0, 1, 11)[1:-1])
    for d in (tr, va):
        d["dur_bucket"] = np.searchsorted(edges, d["duration_ms"].to_numpy(float))
    # ids -> contiguous ints, fit on train; unseen values share a bucket
    for f in FIELDS:
        cats = pd.Index(tr[f].astype(str).unique())
        for d in (tr, va):
            d[f + "_e"] = cats.get_indexer(d[f].astype(str))  # -1 when unseen
    return tr, va


def groups_of(users: np.ndarray, chunk: int | None) -> np.ndarray:
    """Group sizes for LightGBM. `chunk=None` keeps whole users."""
    _, counts = np.unique(users, return_counts=True)
    if chunk is None:
        return counts
    out = []
    for c in counts:
        full, rem = divmod(int(c), chunk)
        out += [chunk] * full + ([rem] if rem else [])
    return np.asarray(out)


def run(tr: pd.DataFrame, va: pd.DataFrame, chunk: int | None, seed: int) -> float:
    tr = tr.sort_values("user_id", kind="stable")          # LightGBM needs grouped rows
    cols = [f + "_e" for f in FIELDS]
    model = lgb.LGBMRanker(
        objective="lambdarank", n_estimators=ROUNDS, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, random_state=seed,
        label_gain=[0, 1], verbose=-1, eval_at=[5],
    )
    model.fit(tr[cols], tr["long_view"].to_numpy(int),
              group=groups_of(tr["user_id"].to_numpy(), chunk),
              categorical_feature=cols)
    scores = model.predict(va[cols])
    r = evaluate(list(va["user_id"]), [int(v) for v in va["long_view"]], list(scores))
    return r["primary"]


def main() -> None:
    print("building features ...", flush=True)
    tr, va = build()
    g = tr.groupby("user_id").size()
    print(f"train {len(tr):,} rows | valid {len(va):,} rows | "
          f"train group median {g.median():.0f}, eval median "
          f"{va.groupby('user_id').size().median():.0f}\n")

    conditions = [("whole_user", None), ("chunk_8", 8), ("chunk_5", 5)]
    results: dict[str, list[float]] = {}
    for name, chunk in conditions:
        n_groups = len(groups_of(tr["user_id"].to_numpy(), chunk))
        vals = []
        for seed in SEEDS:
            t0 = time.time()
            p = run(tr, va, chunk, seed)
            vals.append(p)
            print(f"  {name:11s} seed={seed} primary={p:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
        results[name] = vals
        print(f"  {name:11s} groups={n_groups:,} mean={np.mean(vals):.4f} "
              f"+/-{np.std(vals):.4f}\n", flush=True)

    base = np.mean(results["whole_user"])
    print("=" * 62)
    print(f"{'condition':12s} {'mean':>8s} {'std':>8s} {'vs whole_user':>15s}")
    for name, _ in conditions:
        v = results[name]
        print(f"{name:12s} {np.mean(v):8.4f} {np.std(v):8.4f} "
              f"{np.mean(v) - base:+15.4f}")
    print("\nFM pointwise reference (published): 0.6016")
    print("seed std of the published FM: 0.0008 — a difference under ~0.002 is noise")


if __name__ == "__main__":
    main()
