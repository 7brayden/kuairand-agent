<!-- Shared contract block, injected into every prompt as the task-brief variable.
     Single source of truth: edit here, not in the individual prompts. -->

## The benchmark

**KuaiRand-Pure**, short-video feed logs. You improve a ranking pipeline.

- **Within-user ranking** over logged impressions. Each user is ranked only against
  their own impressions in the eval split. There is no retrieval over the catalogue.
- **Label**: `long_view` (0/1). The other signals (`is_click`, `is_like`, `is_follow`,
  `is_comment`, `is_forward`, `play_time_ms`) may be used as **features or auxiliary
  tasks** — never as the target.
- **Metrics**: GAUC and nDCG@5; **primary = their mean**. You are scored on the
  absolute delta over the FM baseline.
- **Structural fact that kills whole families of ideas**: because ranking happens
  *within* a user, any term constant across that user's rows cannot change the score.
  Pure user-side first-order features contribute **exactly zero**. User features only
  matter through crosses with item-side features.

## Measured facts about this data — established, do not re-derive

These come from `reports/data_bias_analysis.md`, measured on train, valid and the
random-exposure log. Treat them as given; spending an iteration rediscovering any of
them is an iteration not spent improving.

**1. Almost every evaluated pairing is new.** 98.4% of valid rows and 98.7% of test rows
are a `(user_id, video_id)` pair that never appears in train. Every evaluated video and
~97% of users DO appear — it is the *pairing* that is novel.

> Consequence: an explicit `user_id × video_id` cross, or any per-pair lookup, fires on
> about 1% of evaluation rows and is dead weight. Only **factorised** interactions — a
> learned user vector dotted with a learned item vector, as in FM — generalise to unseen
> pairs. A LightGBM categorical `uv_cross` has already been tried and fails for exactly
> this reason.

**2. Training volume is lopsided in time.** 59% of the 1.14M training rows fall on three
days (04-10 to 04-12). An average training day carries 81.5k rows; an average evaluation
day carries 17.8k. A uniformly weighted fit is dominated by one short, high-traffic
early-April regime and then asked to rank a thinner, later one. Recency weighting or
inverse-day-frequency weighting of training rows is an untested, well-motivated lever.

**3. Label prevalence shifts once, then holds.** long_view rate is 0.3366 in train,
0.3133 in valid, 0.3135 in test. The drift is train→evaluation, not valid→test, so valid
is a well-calibrated proxy for test on this axis.

**4. Exposure is heavily biased, and that bias is the target.** Over the same window the
recommender's own impressions convert at 0.3134 against 0.0850 for uniformly random
exposure — a 3.69x lift, impression gini 0.72.

> Your `fit_predict` receives an `unbiased` frame — ~288k of those random-exposure
> impressions, restricted to the validation window so no test-period label can reach you.
> Use it to measure item quality cleanly, not to reweight training.
>
> **Do not attempt IPS or propensity debiasing.** You are scored on ranking *within the
> recommender's own impressions*. Biased traffic is the distribution you are being tested
> on, not a nuisance to correct away; debiasing optimises for an exam you are not sitting.

**5. Training groups are 6-8x larger than evaluation groups — and training group
construction is yours to choose.** The evaluator groups strictly by `user_id`, giving
median 4 impressions per group in valid and 5 in test. The train split has median **31**
per user. So a ranker trained with `group = user_id` learns to order ~31-43 item lists
and is scored on ~5-7 item lists; nDCG@5 is nearly a whole evaluation list but the top
~12% of a training one, and both softmax and lambdarank gradients depend on list length.

> This is a strong candidate explanation for why lambdarank, BPR and listwise softmax
> have each failed to beat a pointwise FM here: aligned in form, mismatched in scale.
> You cannot change how you are evaluated, but nothing requires training groups to be
> whole users — chunking each user's training impressions into blocks of ~5-7, or
> grouping by `(user_id, date)` (median 3), matches the two. **Untested.**
>
> Related: 36-42% of evaluation users are all-positive or all-negative, so they add a
> constant to nDCG and are dropped from GAUC entirely, against only 7.3% in train.

**6. Popularity measured on biased traffic is a weak proxy for quality.** Per-video
long_view rate on train correlates only **0.375** with the same rate measured on
unbiased random exposure (3,398 comparable videos; means 0.304 vs 0.108). An item-
popularity feature built from the standard log is substantially measuring what the
recommender promoted rather than what people actually watch.

## Scoreboard (validation primary)

| | |
|---|---|
| random (lower bound) | 0.4834 |
| item popularity | 0.5807 |
| **FM baseline — beat this** | **0.6016** |
| oracle ceiling | 0.8484 |

Headroom above baseline is **0.247**, not 0.40. Improvements below **0.002** are seed
noise (FM's std over 5 seeds is 0.0008) and will be rejected by the critic.

## Measured dead ends — do not re-test these

- **More static features**: all 13 CWM feature fields → 0.5940 vs 5 fields' 0.5950.
- **More capacity**: embedding k = 8/16/32 → 0.5895/0.5902/0.5887.
- `user_id × video_id` crosses already absorb most learnable signal, and 1.14M rows
  will not support more capacity. The bottleneck is neither features nor capacity.

## Untested directions (ranked by the organisers, who did not try them)

1. **Loss function** — the pipeline optimises pointwise logloss while the metrics are
   ranking metrics. Pairwise (BPR) or listwise (softmax over each user's impressions,
   or LightGBM `lambdarank` with `group` = user) aligns objective with evaluation.
2. **User behaviour sequences** — entirely unused; hundreds to thousands of
   interactions per user exist in train.
3. **Multi-task** — auxiliary heads on the other feedback signals.
4. **Watch-time modelling** — watch time is censored when the video ends, so a
   one-sided loss beats squared error.
5. **Model family** — DeepFM/DCN/xDeepFM (deprioritised: capacity is not the limit).
6. **Time features and train→test drift** (`hourmin`, `date`).
7. **Unbiased validation** against the random-exposure log.

Treat these as evidence, not orders.

## You have the organisers' baseline — use it

`official/` sits next to your code and is importable. It is the organisers' published
baseline, verbatim, and the rules permit any public solution.

**These are the exact signatures. Do not guess them, and never write a shim that tries
several argument orders until one stops throwing — that silently picks a wrong one and
reports nonsense metrics.** If something still surprises you, `print()` it and read the
value in the next iteration's log.

```python
from evaluate import evaluate
evaluate(user_ids, labels, scores, k=5)
#   -> {'GAUC': float, 'nDCG@5': float, 'primary': float, 'users': int, 'rows': int}
#   three EQUAL-LENGTH sequences, one entry per row: the row's user id, its 0/1
#   long_view label, and your score. 'primary' is the number you are judged on.

from baseline import FM, run_fm, run_pop
FM(dim, k=16, lr=0.001, l2=1e-06, seed=0)   # dim = total vocab size, from encode()
FM.step(X, y)      # one minibatch; X int32 (B, n_fields), y float32 (B,) -> loss
FM.predict(X, bs=200000)                                        # -> scores (N,)
run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True)
run_pop(splits, prior=20.0)

from data import load, encode, FIELDS
FIELDS   # ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']
load(data_dir)     # -> {split: [(date, user_id, video_id, author_id, tab,
                   #              duration_ms, label), ...]}
encode(splits)     # -> ({split: (X, y, users)}, dim); X int32 (N, 5) of offset ids
```

**Important shape mismatch.** `run_fm`, `run_pop`, `load` and `encode` all operate on
that dict-of-row-tuples format, NOT on the pandas DataFrames your `fit_predict` is
handed. Passing a DataFrame into them raises. Either build the tuple format from your
frames, or use `FM` directly with your own id encoding — `FM` only needs integer id
arrays and does not care where they came from.

**Do not reimplement FM from memory.** That has been tried repeatedly and the
from-scratch version lands anywhere between 0.5579 and 0.6023 depending on details that
have nothing to do with your hypothesis — burning the run's limited iterations on
reconstruction instead of improvement. Import it, confirm it reproduces ~0.6016, and
spend your iterations on what comes *after* the baseline.

**0.6016 is a known-achievable number, not an aspiration.** If your pipeline scores
meaningfully below it while using the same architecture and hyperparameters, your
implementation has a bug — that is not evidence about your hypothesis. Fix the
implementation with a `tune` or `debug` edit; do not abandon a sound approach because a
buggy version of it scored badly.

## What "trained enough" means here

The published FM baseline is not a quick fit: `k=16, lr=0.001, batch=8192,
max_epochs=40, patience=4` — 40 epochs with early stopping, ~40s on one CPU core for
1.14M training rows. Any model you write from scratch has to actually converge to be
worth judging.

Your pipeline is handed `train`, `valid`, and `target` separately. Select on `valid`
exactly as the baseline does — an internal holdout carved from `train` will look far
better than it is, because `train` and `valid` are separated in time and user-video
pairs memorised in `train` go cold afterwards.

A from-scratch SGD/embedding model that runs a handful of epochs will score far below
its potential for reasons that have **nothing to do with your hypothesis**. If you
implement FM, BPR, or anything else trained by gradient descent: give it a comparable
epoch budget, use early stopping on validation, and print the loss curve so the next
iteration can tell convergence failure from a wrong idea. A score near 0.50 (random is
0.4834) almost always means the model did not train, not that the idea was bad.

## Environment

- CPU only. No GPU budget. `numpy`, `pandas`, and `lightgbm` are available; torch is not.
- **The exact columns of `train` / `valid` / `target`.** These are all of them; anything
  else raises a KeyError, so do not reach for a column that is not on this list:

  From the interaction log —
  `user_id`, `video_id`, `date`, `hourmin`, `time_ms`, `tab`, `duration_ms`,
  `play_time_ms`, `profile_stay_time`, `comment_stay_time`, `long_view` (the label),
  `is_click`, `is_like`, `is_follow`, `is_comment`, `is_forward`, `is_hate`,
  `is_profile_enter`, `is_rand`.

  Joined from `video_features_basic_pure.csv` (item-side, one row per video) —
  `author_id`, `video_type`, `upload_dt`, `upload_type`, `visible_status`,
  `video_duration`, `server_width`, `server_height`, `music_id`, `music_type`, `tag`.

  `user_id` and `video_id` are **strings**, not ints. `dur_bucket` is not a column: the
  baseline derives it by quantile-bucketing `duration_ms` over the train split.
- Runtime limit per iteration is a hard timeout — a run that does not finish is a
  wasted iteration and is rejected.
