# Results

> GENERATED from `logs/journal.jsonl` by `scripts/generate_reports.py` — do not edit by hand.


## Validation-best checkpoint

Node `6a261631` (iteration 3, action `feature`)


| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM baseline (valid) | 0.6674 | 0.5357 | 0.6016 |
| **agent (valid-best)** | **0.6694** | **0.5371** | **0.6033** |
| **absolute delta** | **+0.0020** | **+0.0014** | **+0.0017** |
| oracle ceiling | 1.0000 | — | 0.8484 |

Headroom used: +0.7% of the 0.2468 available above baseline.


## Every iteration

| it | node | action | mode | GAUC | nDCG@5 | primary | accepted | errors |
|---|---|---|---|---|---|---|---|---|
| 0 | `384a0a76` | model | rewrite | 0.6675 | 0.5358 | 0.6016 | yes | — |
| 1 | `1e6ee9d6` | model | rewrite | — | — | — | no | code_error |
| 2 | `73f751cc` | debug | edit×1 | 0.6473 | 0.5283 | 0.5878 | no | — |
| 3 | `6a261631` | feature | edit×3 | 0.6694 | 0.5371 | 0.6033 | yes | — |

## Hidden test (scored once)

See `final_result.json`.

