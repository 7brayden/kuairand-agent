# kuairand-agent — Autonomous ML Research Agent (hackathon submission)

## The task

Build an LLM-driven agent that, given a dataset and metrics, writes its own code to explore
data, engineer features, train, evaluate, reflect, and iterate. **The submitted artifact is the
AGENT (harness + loop), not a model.** The ML pipeline is what the agent produces at runtime;
none of it is written in advance by a human. Reference systems judges have in mind: MLE-Bench
(OpenAI), AIDE (Weco AI), AI-Scientist-v2 (Sakana) — ML engineering as agentic tree search.

## ⚠️ The starter kit overrides the original brief

The starter kit (`../kuairand-starter-kit/`, released after the brief) is AUTHORITATIVE.
It contradicts the pre-release brief on nearly every fixed parameter. Do not re-derive from
the old brief — these are the real contracts:

| | Pre-release brief said | **Starter kit (authoritative)** |
|---|---|---|
| Label | `click` | **`long_view`** (native 0/1 column) |
| Metrics | NDCG@10, Recall@50 | **GAUC, nDCG@5** |
| AUC | "explicitly NOT scored" | **GAUC is half the primary score** |
| Primary | mean of two deltas | **mean(GAUC, nDCG@5)**, then delta vs FM baseline |
| Task | candidate set TBD | **within-user ranking over logged impressions** — no retrieval |
| Splits | val/test = 50/50 of one window | **train 20220408–0421 / valid 20220422–0428 / test 20220429–0508** |
| ε, N | TBD | **ε = 0.002, N = 3** (on validation primary) |
| Baseline | TBD | **FM: valid primary 0.6016 / test primary 0.5946** |
| Submission | TBD | **CSV `row_id,user_id,video_id,score`** |
| Deps | LightGBM/pandas/polars | starter kit is **numpy-only**; ours may use more |

Nothing is TBD any more. `OrganiserTBDError` remains in `state.py` as a guard but should
never fire now.

## Benchmark contracts (from eval/official/, do not reinterpret)

- **Task**: within-user ranking. Each user is ranked only over their own impressions in the
  eval split. There is no full-catalogue retrieval. **Consequence: any term that is constant
  within a user cannot change the score** — pure user-side first-order features contribute
  exactly 0. User features only matter through crosses with item-side features.
- **Label**: `long_view` (0/1).
- **Metrics**: GAUC (impression-weighted per-user AUC, counting only users with
  `0 < positives < impressions`, weighted by positive count) and nDCG@5 (zero-positive users
  score 0.0 and ARE counted in the mean). `primary = (GAUC + nDCG@5) / 2`.
- **Convergence**: ε = 0.002, N = 3 — three consecutive iterations without validation primary
  improving by more than 0.002. ε ≈ 2.5σ (FM's seed std is 0.0008), so anything under 0.002
  is noise, not progress.
- **Scoring**: absolute delta over FM. `mean(Δ GAUC, Δ nDCG@5) == Δ primary` exactly, so
  primary is both the selection scalar and the score.

### The real headroom (test set)

|  | random | **FM (to beat)** | oracle ceiling |
|---|---|---|---|
| GAUC | 0.4996 | **0.6610** | 1.0000 |
| nDCG@5 | 0.4511 | **0.5282** | 0.7289 |
| **primary** | 0.4753 | **0.5946** | **0.8645** |

nDCG@5 cannot exceed 0.729: 27.1% of test users are all-negative (nDCG ≡ 0, unfixable) and
9.2% all-positive (nDCG ≡ 1). FM has already taken 30.7% of the available range. **Measure
progress against the oracle ceiling (0.8645), not against 1.0.** Remaining headroom is 0.27.

### Organiser-measured dead ends — do NOT spend iterations here

- **More static features**: all 13 CWM feature fields → 0.5940 vs 5 fields' 0.5950. No gain.
- **More capacity**: embedding k = 8/16/32 → 0.5895/0.5902/0.5887. Flat.
- Reason: `user_id × video_id` crosses already absorb most learnable signal, and 1.14M rows
  won't support more capacity. **The bottleneck is neither features nor capacity.**

### Unexplored directions (organisers' own ranking, untested by them)

1. **Loss function** — currently pointwise logloss while the metrics are ranking metrics.
   Pairwise (BPR) or listwise (softmax over the user's impressions) aligns objective with
   metric. Organisers rate this most likely to work.
2. **User behaviour sequences** — completely unused. Hundreds to thousands of interactions per
   user in train; DIN/SIM-style interest modelling is untouched.
3. **Multi-task** — `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`,
   `play_time_ms` as auxiliary tasks for the `long_view` main task.
4. **Watch-time modelling** — CWM-style censored regression (watch time is truncated when the
   video ends, so a one-sided loss beats squared error).
5. **Model swap** — DeepFM/DCN/xDeepFM. Deprioritised: capacity is measurably not the bottleneck.
6. **Time features / distribution drift** — `hourmin`, `date`, train→test drift.
7. **Unbiased validation** — `log_random_4_22_to_5_08_pure.csv` (1.18M rows of random exposure)
   as a bias check.

This list is given to the agent as measured prior evidence, not as a plan: the policy must
choose and justify its own hypothesis (Innovation is 20% and judged on reasoning).

## Scoring — this drives the architecture

- **Technical Execution 35%** = primary delta + Robustness. Robustness is NOT failure count —
  it's recovery: retry/reroute/revert around code errors, timeouts, bad inputs, never crashing,
  stalling, or diverging.
- **Innovation & Problem Insight 20%** = what the agent chose to try and WHY (hypotheses and
  reasoning; explicitly not implementation quality).
- **Impact & Relevance 20%** = Autonomy, measured primarily by COUNT OF MANUAL INTERVENTIONS.
- **Feasibility & Practicality 15%** = total LLM tokens (in + out) and total GPU-hours.
- **Presentation 10%** (final event only).

Built-in tension: 35% rewards a better score, 20% rewards touching it less, 15% rewards
spending less.

## Test-set integrity — read this before touching eval

The test split's labels are in the local data (it's a public dataset). "Hidden test" is
therefore an **honour-system constraint we enforce in the harness**:

- The loop only ever scores `valid`. `eval/scorer.py` exposes no test path.
- Test is scored exactly once, at the end, by `scripts/final_eval.py`, from the
  validation-best checkpoint. That script is never imported by the loop.
- Generated pipeline code is linted for references to the test split; a hit is journaled as an
  error event and the iteration is rejected.

Violating this would invalidate the submission, and it is invisible to the organisers, so the
guard exists to protect us from ourselves.

## Core design principle

`logs/journal.jsonl` is the SPINE, not an afterthought. It is simultaneously (a) the judge
deliverable, (b) the agent's own memory for its reflect step, (c) the source from which the
results table and resource report are generated. One append-only journal, everything else
derived. Never a second source of truth. Schema in `agent/journal.py` — change it there first,
bump `SCHEMA_VERSION`, keep readers backward-compatible.

## Stack decisions — deliberate, do not relitigate

- **Hand-roll the loop.** No LangGraph / CrewAI / AutoGen. Everything scored outside the raw
  delta lives in the plumbing, and frameworks hide the plumbing.
- **ONE LLM call site: `agent/llm.py`.** Every call returns usage; every call is logged,
  including failures and retries. Token accounting (15%) depends on this being the only door.
- **Model: `claude-opus-5`.** Two API constraints are load-bearing and easy to reintroduce
  by habit: **never pass `temperature`/`top_p`/`top_k`** (sampling params are removed on
  Opus 5 and Sonnet 5 — HTTP 400 on every call), and **never pass a `thinking` block**
  (thinking is adaptive by default on Opus 5; `budget_tokens` is a 400). Reasoning depth is
  controlled by `llm.effort` in the config instead. `claude-sonnet-5` is a drop-in swap at
  ~40% of the cost — measure hypothesis quality before downgrading, since Innovation is 20%.
- **Prompt caching is not worth it here.** It would cut dollar cost, but cached reads still
  count as input tokens in `usage`, and the judged Feasibility metric is total tokens — so
  caching saves money without moving the score.
- **LightGBM lambdarank** is the default inner model — and note it directly attacks unexplored
  direction #1: lambdarank IS a listwise objective, with `group` = user, which is exactly the
  within-user ranking structure. CPU-first; GPU-hours cost 15% of the score.
- **pandas** in agent-facing generated code (LLMs write it fluently). polars was dropped: the
  harness no longer does data prep — the vendored `eval/official/data.py` owns loading.
- **Execution via subprocess with hard timeout, never `exec()`.**
- **`pipeline/workspace/` is its own git repo**, gitignored from the outer repo. Accepted
  iteration = commit. Diff = `git format-patch -1 HEAD --stdout`. Rejection =
  `git checkout . && git clean -fd`. Backtracking = branch from an earlier commit, so the
  commit graph IS the search tree.
- **Vendored organiser code lives untouched in `eval/official/`** (`evaluate.py`, `data.py`,
  `submit.py`). `eval/scorer.py` is a thin adapter over it. There is deliberately NO fast
  approximation: `evaluate.py` is pure-stdlib and fast enough, so a second implementation
  would be pure risk with no payoff. (This replaces the brief's `eval/fast.py` plan.)
- **No W&B / MLflow.** Static committed HTML report generated from the journal.
- **uv** for dependency management. NOTE: uv is installed at
  `~/Library/Python/3.12/bin/uv` and is NOT on PATH; `run.sh` resolves it.
- Checkpoints under `checkpoints/<node_id>/` (gitignored), referenced from the journal — never
  committed into the workspace repo.

## Build order — harness before intelligence

1. **DONE**: loop runs end to end with `RandomPolicy` (`--policy random`, no provider),
   proving journal, diffs, token accounting, timeout recovery, and convergence on garbage
   models. Its popularity variant scores 0.5807 = the published `item_popularity` valid
   primary, confirming loader/order/writer/scorer all agree with the organisers'.
2. **DONE**: eval self-check — random gives valid primary 0.4827 vs published 0.4834.
   Wired into loop startup; a failure aborts before any iteration runs.
3. **DONE**: `LLMPolicy` + `LLMCodeGenerator` (milestone 1). Prompts share
   `_task_brief_v1.md`. Both are non-raising: failures journal an error event and fall
   back. Covered by 24 tests with a scripted fake client (no network).
4. **NEXT**: a live run (`ANTHROPIC_API_KEY` required), then tune prompts against what the
   journal actually shows. Nothing about the harness should need to change for that.

## Repo conventions

- Prompts are versioned `.md` files in `agent/prompts/` — never inline strings in Python.
- `reports/*.md` are GENERATED from the journal; never hand-edit.
- Data: `data/raw/KuaiRand-Pure/data/` (gitignored), fetched by `data/download.sh`.
