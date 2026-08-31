# official/ — the organisers' published baseline, verbatim

Copied unmodified from `eval/official/` (which is itself the starter kit, checksum-guarded).
The competition rules permit any public solution, and this is the organisers' own.

The agent may import and adapt anything here. It is a **starting point, not a target**:
`baseline.py`'s FM scores 0.6016 valid / 0.5946 test, which is the number to beat.

| file | what it gives you |
|---|---|
| `baseline.py` | `FM` (numpy factorization machine: `.step(X, y)`, `.predict(X)`), `run_fm`, `run_pop` |
| `data.py` | `load(data_dir)` -> split dict; `encode(splits)` -> `({split: (X, y, users)}, dim)`; `FIELDS` |
| `evaluate.py` | the scoring function (identical to the one the harness uses) |

Do not edit these files — copy what you need into your own code and change that instead.
