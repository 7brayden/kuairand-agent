"""Reproduce reports/data_bias_analysis.md.

Reads train, valid and the random-exposure log. Deliberately does not read test labels —
only test's row structure, which the pipeline must predict on anyway.

Run: python3 scripts/analyse_bias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "data/raw/KuaiRand-Pure/data"
SPLITS = {"train": (20220408, 20220421), "valid": (20220422, 20220428),
          "test": (20220429, 20220508)}
IDS = {"user_id": str, "video_id": str}


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def main() -> None:
    if not D.exists():
        sys.exit(f"dataset not found at {D}; run data/download.sh")
    std = pd.concat([pd.read_csv(D / f, dtype=IDS) for f in
                     ("log_standard_4_08_to_4_21_pure.csv",
                      "log_standard_4_22_to_5_08_pure.csv")], ignore_index=True)
    rnd = pd.read_csv(D / "log_random_4_22_to_5_08_pure.csv", dtype=IDS)
    sp = {k: std[(std.date >= a) & (std.date <= b)] for k, (a, b) in SPLITS.items()}
    tr, va, te = sp["train"], sp["valid"], sp["test"]

    print("1. cold pairs (the model has never seen this user with this video)")
    pairs = set(zip(tr.user_id, tr.video_id))
    for name, d in (("valid", va), ("test", te)):
        unseen = np.mean([(u, v) not in pairs for u, v in zip(d.user_id, d.video_id)])
        print(f"   {name}: pair {unseen:6.1%} | video "
              f"{(~d.video_id.isin(set(tr.video_id))).mean():5.1%} | user "
              f"{(~d.user_id.isin(set(tr.user_id))).mean():5.1%}")

    print("\n2. training volume concentration")
    top = tr.groupby("date").size().sort_values(ascending=False)
    print(f"   top 3 days = {top.head(3).sum() / len(tr):.0%} of all training rows "
          f"({list(top.head(3).index)})")
    print(f"   rows/day: train {len(tr)/14:,.0f} | valid {len(va)/7:,.0f} | test {len(te)/10:,.0f}")

    print("\n3. label rate")
    for k, v in sp.items():
        print(f"   {k:6s} {v.long_view.mean():.4f}")

    print("\n4. exposure bias, same window")
    std2 = std[(std.date >= 20220422) & (std.date <= 20220508)]
    print(f"   standard {len(std2):>9,} rows  long_view {std2.long_view.mean():.4f}")
    print(f"   random   {len(rnd):>9,} rows  long_view {rnd.long_view.mean():.4f}")
    print(f"   lift {std2.long_view.mean() / rnd.long_view.mean():.2f}x | "
          f"impression gini train {gini(tr.video_id.value_counts().values):.3f}")

    print("\n5. is biased item quality the real thing?")
    a = tr.groupby("video_id").long_view.agg(["mean", "size"]).query("size >= 50")
    b = rnd.groupby("video_id").long_view.agg(["mean", "size"]).query("size >= 50")
    j = a.join(b, lsuffix="_biased", rsuffix="_unbiased", how="inner")
    print(f"   {len(j):,} comparable videos | correlation "
          f"{j['mean_biased'].corr(j['mean_unbiased']):.3f} | "
          f"means {j['mean_biased'].mean():.4f} biased vs {j['mean_unbiased'].mean():.4f} unbiased")


if __name__ == "__main__":
    main()
