# Results

> GENERATED from `logs/journal.jsonl` by `scripts/generate_reports.py` — do not edit by hand.


## Validation-best checkpoint

Node `38bddbc4` (iteration 0, action `model`)


| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM baseline (valid) | 0.6674 | 0.5357 | 0.6016 |
| **agent (valid-best)** | **0.6682** | **0.5365** | **0.6023** |
| **absolute delta** | **+0.0008** | **+0.0008** | **+0.0007** |
| oracle ceiling | 1.0000 | — | 0.8484 |

Headroom used: +0.3% of the 0.2468 available above baseline.


## Every iteration

| it | node | action | GAUC | nDCG@5 | primary | accepted | errors |
|---|---|---|---|---|---|---|---|
| 0 | `38bddbc4` | model | 0.6682 | 0.5365 | 0.6023 | yes | — |
| 1 | `dc55a44f` | model | — | — | — | no | code_error |
| 2 | `cd6bd8ff` | debug | 0.6650 | 0.5349 | 0.6000 | no | — |
| 3 | `0cc5becb` | model | 0.6665 | 0.5360 | 0.6013 | no | — |
