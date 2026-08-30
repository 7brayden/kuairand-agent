# Vendored organiser code — DO NOT EDIT

Copied verbatim from the official KuaiRand-Pure starter kit on 2026-08-28.
Source: `kuairand-starter-kit/` as delivered by the organisers.

| file | role |
|---|---|
| `evaluate.py` | **The scoring authority.** GAUC + nDCG@5 + primary. All conventions are pinned in its header comment. Never edit. |
| `data.py` | Official loader, split boundaries, and feature encoding. |
| `baseline.py` | `pop` / `fm` / `random` baselines. FM is the one to beat. |
| `submit.py` | Submission writer/validator (`row_id,user_id,video_id,score`). |
| `baseline_scores.json` | Published scores, seed variance, convergence rule (ε=0.002, N=3). |
| `ablation_features.py` | Reproduces the "more features don't help" result. |

Integrity: any change here changes the scoring rules. If a file must be adapted,
adapt it in `eval/scorer.py` instead and leave these bytes alone.

sha256 at vendoring time is recorded in `CHECKSUMS.txt`; `tests/test_eval_official.py`
verifies the files still match, so accidental edits fail the test suite.
