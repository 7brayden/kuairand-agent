# Results

> GENERATED from `logs/journal.jsonl` by `scripts/generate_reports.py` — do not edit by hand.


## Validation-best checkpoint

Node `d0df631a` (iteration 2, action `model`)


| | GAUC | nDCG@5 | primary |
|---|---|---|---|
| FM baseline (valid) | 0.6674 | 0.5357 | 0.6016 |
| **agent (valid-best)** | **0.6626** | **0.5335** | **0.5981** |
| **absolute delta** | **-0.0048** | **-0.0022** | **-0.0035** |
| oracle ceiling | 1.0000 | — | 0.8484 |

Headroom used: -1.4% of the 0.2468 available above baseline.


## Every iteration

| it | node | action | GAUC | nDCG@5 | primary | accepted | errors |
|---|---|---|---|---|---|---|---|
| 0 | `94d0365d` | model | — | — | — | no | bad_llm_output |
| 1 | `dde1d206` | model | — | — | — | no | bad_llm_output |
| 2 | `d0df631a` | model | 0.6626 | 0.5335 | 0.5981 | yes | — |
| 3 | `c5d675b0` | intervention | — | — | — | no | — |
