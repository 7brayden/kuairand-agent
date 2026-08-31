## Inspiration

MLE-Bench, AIDE and AI-Scientist-v2 all ask the same question: can a language model do the
*job* of an ML engineer — not write a model when told exactly what to build, but decide what
to try, measure it, work out why it failed, and try again?

We wanted to answer that honestly, which meant building the thing that could be wrong. So we
submitted **the agent, not the model**. Every line of the ranking pipeline is written by the
agent at runtime; we never wrote a feature, a loss function, or a hyperparameter.

The scoring rubric shaped the architecture more than the dataset did. It rewards a better
score (35%), *touching it less* (20%, measured by counting manual interventions), and
*spending less* (15%, tokens and GPU-hours). Those pull against each other, and the tension
is the actual question being asked. You cannot fake your way through a metric that counts how
often you intervened.

## What it does

Given KuaiRand-Pure — 1.4M short-video impressions, 27K users, 7.6K items — and two metrics,
the agent runs a loop with nobody watching:

1. **Reflect** over its own journal and propose an action with a falsifiable hypothesis
2. **Write code** — a surgical `SEARCH/REPLACE` edit, or a full rewrite if the approach changes
3. **Guard** it: syntax, I/O contract, and test-split leakage, all checked *before* execution
4. **Execute** in a subprocess with a hard timeout — never `exec()`
5. **Score** with the organisers' own evaluator, vendored byte-for-byte
6. **Judge**: accept → git commit + patch + checkpoint; reject → revert
7. **Append** to the journal, then check convergence

The task is within-user ranking over logged impressions. Label is `long_view`; the score is

$$\text{primary} = \tfrac{1}{2}\left(\text{GAUC} + \text{nDCG@5}\right), \qquad
\Delta = \text{primary}_{\text{agent}} - \text{primary}_{\text{baseline}}$$

## How we built it

**`logs/journal.jsonl` is the spine.** One append-only entry per attempted iteration, serving
three roles at once: the judge deliverable, the agent's own memory for its reflect step, and
the sole source for every report. If a number isn't derivable from the journal, it isn't
reported. There is never a second source of truth.

**The workspace is its own git repository, so the commit graph *is* the search tree.** Not a
metaphor — an accepted iteration is a commit, a rejection is `git checkout . && git clean -fd`,
and backtracking is a branch from an earlier commit. The per-iteration diff deliverable is
just `git format-patch -1 HEAD --stdout`.

**We hand-rolled the loop.** No LangGraph, no CrewAI, no AutoGen. Everything scored outside
the raw delta — token accounting, intervention counting, failure recovery — lives in exactly
the plumbing those frameworks hide. There is **one LLM call site**, so every token, including
failed calls and retries inside rejected iterations, is auditable.

**Two guards protect us from ourselves.** The test split's labels sit in the same local CSVs as
validation, so "scored once" is enforced structurally: the loop has no code path to a test
score, generated code is linted for test-split references before it runs, and the one-shot
evaluation refuses to run twice. Separately, the unbiased random-exposure log spans the test
window, so the harness slices it to the validation window *before* the agent ever sees it.

CPU-only throughout — GPU-hours are 15% of the score, so LightGBM and a numpy factorization
machine are a scoring advantage, not a compromise.

## Challenges we ran into

**Almost every agent failure turned out to be our failure.** This was the most uncomfortable
and most useful lesson of the project. Each of these cost real iterations, and under the
convergence rule the agent only gets three after its best result:

- **Truncated code generation.** We set `max_output_tokens: 8192`, forgetting that adaptive
  thinking spends *output* tokens. The model burned the budget reasoning and got cut off
  mid-function. Two iterations lost to what looked like the model writing garbage.
- **No validation split.** We handed the pipeline only `train` and `target`, so the agent
  carved its own holdout out of train. But train and validation are separated in time, so an
  internal split is inflated by user-video memorisation — it was early-stopping on a number
  reading 0.95 while reality was 0.60. *The agent diagnosed this itself* once we gave it
  visibility into its own training logs.
- **Interfaces documented by name but not by shape.** We told it the organisers' baseline
  existed without giving `evaluate()`'s signature. It wrote a wrapper that guessed argument
  order until one stopped throwing, silently picked a wrong one, and spent four iterations
  debugging its own guesswork.
- **A column we never joined.** We told it the baseline uses `author_id`; that column lives in
  the video-features file, not the log. Two crashes.
- **A discarded winning checkpoint.** One epsilon was doing two unrelated jobs. The critic
  accepted an iteration only if it beat the best by more than the *convergence* epsilon, so a
  run scoring 0.5984 against a tip of 0.5970 was rejected for gaining "only" +0.0014 — and its
  checkpoint deleted — while still being named the run's validation best. The submission
  script would have refused at the very end. Acceptance and convergence are now separate.
- **A 44-minute stall.** The SDK defaults to a 600s request timeout and we layered our own
  retry loop on top. Nothing crashed, which is exactly why it was bad — robustness is judged
  on never stalling, and a healthy-looking 44-minute iteration is a stall.

**Blind agents can't distinguish a bad idea from a bad implementation.** Early on the agent
scored 0.4983 reimplementing a factorization machine, noted correctly that random scores 0.4834,
concluded the *approach* was wrong, and abandoned it. Its SGD simply hadn't converged. With one
number per iteration and no training curve, those two are indistinguishable — so it discarded a
sound idea. Feeding pipeline stdout back into its memory fixed this, and its reasoning changed
immediately.

## Accomplishments we're proud of

**It reasons about its own evidence.** These are unedited journal entries:

> "train_loss keeps falling every epoch while valid_primary peaks at epoch 4 and then degrades
> for 4 more — classic overfitting of the user_id × video_id embeddings, not underfitting."

> "hourmin varies row-to-row even within a single user's impression list, so unlike user_id it
> isn't nulled out by within-user ranking."

That second one is the agent independently deriving the structural rule that makes this
benchmark unusual — anything constant within a user contributes *exactly zero* — and then
choosing a feature that survives it.

**A real finding about the benchmark: match the metric's $k$, not the group size.**

The evaluator groups by user. Train has a median of 31 impressions per user; the evaluation
splits have 4–5. So any listwise ranker trained on whole users learns to order ~31-item lists
and is scored on ~5-item ones. Across 15 runs, *every* listwise attempt — lambdarank, BPR,
softmax — lost to a pointwise factorization machine. A controlled experiment (identical model,
features and seeds; only the grouping varied, 8 seeds across two independent sets):

| training group | validation primary | vs whole-user |
|---|---|---|
| whole user (median 31) | 0.5989 | — |
| chunks of 8 | 0.5998 | +0.0009 |
| chunks of 6 | 0.5999 | +0.0010 |
| **chunks of 5** | **0.6013** | **+0.0024** |
| chunks of 4 | 0.6000 | +0.0011 |
| chunks of 2 | 0.5957 | −0.0032 |

The peak is *sharp* at exactly five, and the mechanism is the metric itself: it is
**nDCG@5**. A training list of exactly 5 sits entirely inside the cutoff, so every position
contributes gradient through the discount $1/\log_2(i+1)$ for $i < 5$. At 6+ some positions
fall outside; at 4 the list is shorter than the cutoff and carries less signal per group. The
rule isn't "match the average group size" — it's **match the metric's $k$**, which predicts a
sharp peak rather than a broad optimum. The neighbours behaving differently is evidence *for*
the mechanism, not noise against it.

## Results, stated honestly

**Hidden test, evaluated exactly once** from the validation-best checkpoint:

| | agent | FM baseline | delta |
|---|---|---|---|
| GAUC | 0.66091 | 0.66100 | −0.00009 |
| nDCG@5 | 0.52897 | 0.52820 | +0.00077 |
| **primary** | **0.59494** | 0.59460 | **+0.00034** |

**That is a tie, not a win.** The baseline's own seed-to-seed standard deviation is 0.0008 —
more than twice our margin. The defensible claim is that the agent reached *parity* with a
hand-built baseline, autonomously, from an empty file.

| | |
|---|---|
| Manual interventions | **1** across 73 iterations (zero inside any run; the one logged was environment setup) |
| Failures recovered | **13 / 13** — every crash, timeout and malformed generation journaled with its recovery |
| GPU-hours | **0.00** |
| Tokens | 1,584,277 across all 16 runs |
| Wall-clock | 2.6 hours |
| Tests | 121 |

We ran 16 times while debugging the harness and submitted the best, so we publish the *whole
distribution* rather than the number we chose: best 0.6033, median 0.6018, mean 0.5980 ±
0.0077, with 8 of 16 at or above baseline. Selecting the maximum of several runs is worth more
than any single run is expected to score, and hiding that would misrepresent the result.

## What we learned

**Validation-best selection offers no protection against validation overfitting.** Our
validation lead of +0.0017 became +0.0003 on test — roughly 20% transfer. The winning iteration
added hour-of-day and recency features that helped on the 7-day validation window and didn't
survive the 10-day test window after it. The thing you select on is the thing you overfit, and
the competition's own rule selects on validation.

**Observability is the difference between a researcher and a random search.** Every capability
we added that mattered was about letting the agent *see*: its training curves, the exact
signatures of code it could call, the columns actually present in its dataframes, a measured
account of what we'd already learned about the data. Its reasoning was consistently good; what
limited it was what we let it know.

**Guards beat instructions.** We asked the agent to prefer surgical edits over rewrites and it
rewrote anyway, four times out of four. Making it a mechanical guard — reject a rewrite for a
refinement action, before execution — changed the behaviour immediately. Anything that matters
should be enforced, not requested.

**Know when to stop.** At ~20% validation-to-test transfer, pushing validation to 0.605 would
buy roughly +0.0007 on test. Two independent methods — the agent's own search and an offline
blend of a matched-group lambdarank with a pointwise model — converged on the same ~0.603
ceiling. We stopped because the evidence said further optimisation wasn't worth it, not because
we ran out of ideas.

## What's next

- **An exploration budget for the critic.** It's greedy hill-climbing: accept-if-best never
  keeps a neutral change that might unlock something two moves later.
- **Real journal summarisation.** The last eight iterations render in full and older ones
  collapse to a row. That holds for tens of iterations, not hundreds — and because the journal
  *is* the agent's memory, any lossy scheme changes its behaviour.
- **Measured `gpu_seconds`.** Token accounting is airtight; GPU attribution inside the
  subprocess isn't. It reads zero because the path is genuinely CPU-only, but a torch
  escalation would need a real meter first.
- **Design for the temporal shift, not the validation score** — the one change most likely to
  turn a validation lead into a test lead.
