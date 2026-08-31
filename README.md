# kuairand-agent

An autonomous ML research agent for the **KuaiRand-Pure** short-video ranking benchmark.
Given the dataset and the metrics, the agent writes its own code to explore the data,
engineer features, train, evaluate, reflect, and iterate.

**The submitted artifact is the agent — the harness and loop in this repo — not a model.**
The ML pipeline is produced by the agent at runtime inside `pipeline/workspace/`; none of
it is written in advance by a human.

## Task (organiser-fixed)

| | |
|---|---|
| Task | within-user ranking over logged impressions (no full-catalogue retrieval) |
| Label | `long_view` (0/1) |
| Metrics | GAUC, nDCG@5 — **primary = their mean** |
| Splits | train `20220408–0421` / valid `20220422–0428` / test `20220429–0508` |
| Baseline to beat | FM: valid primary **0.6016**, test primary **0.5946** |
| Oracle ceiling | test primary **0.8645** (nDCG@5 caps at 0.729 — 27% of users are all-negative) |
| Convergence | ε = 0.002, N = 3 on validation primary |
| Score | absolute delta over the FM baseline |

Full contracts, measured dead ends, and the unexplored directions are in `CLAUDE.md`.

## How it works

```
policy ──proposes action + hypothesis──▶ codegen ──▶ lint (contract + test-leak guard)
                                                        │
   journal.jsonl ◀── critic ◀── official eval ◀── subprocess (hard timeout, never exec)
        │                          (valid only)
        ├─▶ agent memory (reflect)      accept → git commit + format-patch + checkpoint
        ├─▶ reports/ (all of them)      reject → git checkout . && git clean -fd
        └─▶ convergence check           backtrack → branch from an earlier commit
```

`logs/journal.jsonl` is the spine: one append-only entry per attempted iteration, serving
simultaneously as the judge deliverable, the agent's own memory, and the only source for
every report. The workspace commit graph **is** the search tree.

## Status

| component | state |
|---|---|
| journal schema, run state, convergence | implemented + tested |
| executor (subprocess timeout, git workspace) | implemented + tested |
| eval adapter over vendored official scorer | implemented + verified against published numbers |
| pipeline I/O contract + canonical row order | implemented + verified against official loader |
| submission adapter, one-shot final eval | implemented |
| report generation (md + static HTML) | implemented |
| **milestone 0 — harness proof** | **passing** (see below) |
| **milestone 1 — LLM policy + code generation** | **implemented + tested** |
| **live runs** | **16 runs, 73 iterations, 1 logged intervention** |
| **hidden test** | **scored once: primary 0.59494, delta +0.00034** |
| Devpost writeup | `reports/devpost.html` |

### Result

Hidden test, scored once from validation-best node `6a261631`:

| | agent | FM baseline | delta |
|---|---|---|---|
| GAUC | 0.66091 | 0.66100 | −0.00009 |
| nDCG@5 | 0.52897 | 0.52820 | +0.00077 |
| **primary** | **0.59494** | 0.59460 | **+0.00034** |

A tie with the baseline — +0.00034 sits inside FM's own seed std of 0.0008. Validation-best
was 0.6033 (+0.0017); roughly 20% of that transferred, the rest being ordinary validation
overfitting to the seven-day window.

What the harness did do: reached baseline quality autonomously from an empty file, recovered
from 13 of 13 failures, ran 16 times with zero in-run interventions and zero GPU-hours, and
produced one genuine finding about the benchmark (see below). Full distribution across all
runs is in `reports/run_history.md` — best 0.6033, median 0.6018, mean 0.5980 ± 0.0077,
8 of 16 at or above baseline.

### Finding: match the metric's k, not the group size

The evaluator groups by user; train has median 31 impressions per user, evaluation 4–5. Every
listwise attempt across 15 runs lost to a pointwise FM because of it. A controlled experiment
(`scripts/experiment_group_size.py`, identical model/features/seeds, 8 seeds) found chunking
training groups to **exactly 5** — the metric's cutoff, nDCG@**5** — worth +0.0024, with a
sharp peak rather than a broad optimum:

| whole user (31) | 8 | 6 | **5** | 4 | 2 |
|---|---|---|---|---|---|
| 0.5989 | 0.5998 | 0.5999 | **0.6013** | 0.6000 | 0.5957 |

### Milestone 0: harness before intelligence

The loop runs end to end with a deliberately dumb policy (`policy.kind: random`) that
cycles through known-good, known-broken, and known-slow pipeline variants, so a single
short run proves every plumbing path on garbage models:

```
eval self-check: PASS (random valid primary 0.4827, expected ~0.4834)
  [00] tune     ACCEPT | primary 0.5807 | 1.3s
  [01] tune     reject | no score | 0.0s | 1 error(s): code_error
  [02] eda      reject | primary 0.4827 | 1.0s
  [03] model    reject | no score | 45.1s | 1 error(s): timeout
CONVERGED: validation primary plateaued: no gain > 0.002 over the last 3 iterations
validation-best: node 3089190e primary 0.5807 (baseline 0.6016, delta -0.0209)
tokens: 0 | gpu_s: 0.0 | wall_s: 47.5 | interventions: 0
```

That run demonstrates: journal written per iteration, diff exported and checkpoint kept for
the accepted node only, rejections reverted to a clean tree, a syntax error and a hard
timeout both recovered without crashing, token accounting at zero (no LLM involved), and
convergence firing on the organisers' own ε/N.

Its accepted iteration scored **0.5807**, which matches the organisers' published
`item_popularity` valid primary exactly — independent confirmation that our loader,
canonical row order, submission writer, and scorer agree with theirs.

### Milestone 1: the agent decides

`LLMPolicy` reflects over the journal and returns a JSON proposal (action, falsifiable
hypothesis, optional `backtrack_to`); `LLMCodeGenerator` turns that into a new
agent-owned zone, with one built-in repair round when the guards reject its first
attempt. Prompts are versioned files sharing one `_task_brief_v1.md` so the contracts,
the measured dead ends, and the noise floor have a single source of truth.

Neither can crash a run. A provider outage or an unparseable response is journaled as
an error event and the policy returns a labelled `FALLBACK` proposal — `debug` if the
previous iteration failed, otherwise the cheapest action. All of that is covered by 20
tests driven by a scripted fake client, plus four integration tests that exercise the
real loop, real data, and the real official scorer with only the provider faked.

## Setup

```bash
bash data/download.sh          # KuaiRand-Pure from Zenodo into data/raw/ (~200MB)
uv sync                        # numpy, pandas, lightgbm, pyyaml, anthropic
python3 -m pytest tests/ -q    # 64 tests
export ANTHROPIC_API_KEY=...   # or `ant auth login` — only for a live run
```

## Reproduction

```bash
bash run.sh                                              # full agent run to convergence
bash run.sh --policy random --max-iterations 6 --timeout 45   # harness proof, no provider
python3 scripts/generate_reports.py                      # regenerate reports from the journal
python3 scripts/final_eval.py --yes                      # ONE-SHOT hidden-test evaluation
```

`--policy random` is the milestone-0 mode and needs no credentials. The default is the
real agent, which fails fast at startup if no credential source is found rather than
burning iterations on fallbacks. The SDK resolves `ANTHROPIC_API_KEY`, then
`ANTHROPIC_AUTH_TOKEN`, then an `ant auth login` profile — the preflight accepts any.

Cost: adaptive thinking spends *output* tokens, so a real iteration runs ~4k input and
~10-17k output. On `claude-sonnet-5` (the default) that is roughly **$2-3 for a
20-iteration run**; `claude-opus-5` is about 2.5x that. Lower `llm.effort` to trade
reasoning depth for tokens — thinking counts toward the scored total, not just the bill.

`scripts/final_eval.py` refuses to run twice (it checks for `reports/final_result.json`)
and is the only code path allowed to touch the test split.

## Layout

- `agent/` — the harness: `loop.py`, `state.py`, `policy.py`, `critic.py`, `codegen.py`,
  `executor.py`, `journal.py`, `memory.py` (journal→prompt compression), and `llm.py`
  (the single provider call site); `agent/prompts/` (versioned `.md`, one shared brief)
- `eval/official/` — organiser code, **vendored untouched**, checksum-guarded by a test;
  `eval/scorer.py` — the thin adapter and the test-split guard
- `pipeline/template/` → `pipeline/workspace/` — the agent's own git repo (gitignored)
- `logs/journal.jsonl`, `logs/diffs/` — the append-only run record (deliverable)
- `reports/` — generated; `submission/adapter.py` — the only file that knows the schema

## Limitations (honest reflection)

- **The critic is greedy.** Accept-iff-better-than-best is hill-climbing; it never keeps a
  neutral change that might enable a later win. Tree search mitigates this (the policy may
  branch from any earlier accepted commit) but there is no explicit exploration budget yet.
- **Journal-as-memory compresses crudely.** `agent/memory.py` renders the last 8 iterations
  in full and collapses older ones to a row, preserving the "already rejected" and "failures"
  sections. That holds for tens of iterations; at hundreds it will need real summarisation,
  and because the journal *is* the agent's memory, any lossy scheme changes behaviour.
- **`gpu_seconds` is currently an honour-system field.** Token accounting is airtight (one
  call site); GPU attribution inside the subprocess is not yet measured. It reads 0 because
  the default path is CPU-only, but a torch escalation needs a real meter first.
- **Failed iterations spend convergence patience.** A crash streak counts as
  no-improvement and can end a run at N=3. Defensible (a stalled agent should stop burning
  budget) but it is a judgement call, not an organiser rule.
- **Checkpoint retention is unbounded** — every accepted node keeps one.
- **The hidden test is honour-system.** Its labels are in the public data; we enforce
  single-use in the harness rather than relying on discipline.

## Contributions

Written for the hackathon track by the repository author, with the agent harness
implemented in collaboration with Claude Code. The organisers' starter kit
(`eval/official/`) is vendored unmodified and is not our work.
