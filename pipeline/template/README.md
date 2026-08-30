# pipeline/template/ — the seed for the agent's workspace

`pipeline/workspace/` is initialised by copying this directory and running `git init`
inside it (`agent/executor.py: Workspace.init_if_needed`). From then on the AGENT owns
the workspace: accepted iteration = commit, rejection = revert, backtrack = branch —
the commit graph is the search tree. The workspace is gitignored from the outer repo.

The template contains **no ML logic** — `fit_predict` returns zeros. What it does pin
is the I/O contract, because two things are load-bearing and expensive to rediscover:

1. **Canonical row order.** Predictions must be in the exact order the official loader
   produces, or `eval/official/submit.py` rejects the file. `load_logs` + `split_of`
   produce that order; the agent must not sort or shuffle the target frame.
2. **Id string fidelity.** `user_id`/`video_id` stay `str` so they match the official
   validator's raw-text comparison.

Both are contract, not modelling, so they are given rather than discovered.

## Contract

    python3 main.py --data-dir <raw csv dir> --split valid --out-dir <dir>

writes `<out-dir>/predictions.csv` (`row_id,user_id,video_id,score`) and optionally
`<out-dir>/checkpoint/`. Exit 0 on success; any non-zero exit or timeout becomes an
ErrorEvent and routes to the debug action.

The harness passes `--split valid` during the run. `test` is scored exactly once at
the end by `scripts/final_eval.py`; generated code that reaches for it is rejected.
