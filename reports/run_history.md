# Run history

> GENERATED from every journal in `runs/` plus the active one by `scripts/run_summary.py` — do not edit by hand.


The submitted result is a single run, but many were performed while the harness was being debugged. Selecting the best of several runs is worth more than any one run is expected to score, so the full distribution is published here rather than only the number we chose.


| run | iters | accepts | edits | errors | interv. | tokens | best valid primary | vs baseline |
|---|---|---|---|---|---|---|---|---|
| 20260830T064003Z | 4 | 1 | 0 | 2 | 0 | 0 | 0.5807 | -0.0209 |
| 20260830T080905Z | 3 | 1 | 0 | 2 | 1 | 67,850 | 0.5981 | -0.0035 |
| 20260830T173952Z | 4 | 1 | 0 | 0 | 0 | 81,070 | 0.5977 | -0.0039 |
| 20260830T174733Z | 5 | 2 | 0 | 0 | 0 | 87,659 | 0.5977 | -0.0039 |
| 20260830T175437Z | 4 | 1 | 0 | 0 | 0 | 85,946 | 0.5970 | -0.0046 |
| 20260831T072047Z | 4 | 1 | 0 | 1 | 0 | 97,682 | 0.6023 | +0.0007 |
| 20260831T074356Z | 4 | 1 | 0 | 0 | 0 | 107,834 | 0.6020 | +0.0004 |
| 20260831T082338Z | 6 | 4 | 4 | 0 | 0 | 114,681 | 0.5973 | -0.0043 |
| 20260831T084036Z | 5 | 1 | 3 | 1 | 0 | 113,263 | 0.5763 | -0.0253 |
| 20260831T084710Z | 6 | 2 | 2 | 2 | 0 | 107,926 | 0.6029 | +0.0013 |
| 20260831T090922Z | 5 | 2 | 2 | 1 | 0 | 99,108 | 0.6015 | -0.0001 |
| **(submitted)** | 4 | 2 | 2 | 1 | 0 | 106,320 | 0.6033 | +0.0017 |

## Distribution

- runs: **12**
- best: **0.6033** (+0.0017 vs baseline)
- median: 0.5979
- mean: 0.5964 ± 0.0084
- runs at or above baseline (0.6016): **4 of 12**

Most of the spread comes from harness bugs fixed between runs (truncated code generation, a missing validation split, guessed API signatures, a column the pipeline never joined) rather than from the agent's reasoning, which was consistent throughout. Runs before a given fix are not comparable with runs after it.

