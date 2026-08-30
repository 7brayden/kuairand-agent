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
- Available raw log columns include: `user_id`, `video_id`, `date`, `hourmin`, `tab`,
  `duration_ms`, `play_time_ms`, `is_click`, `is_like`, `is_follow`, `is_comment`,
  `is_forward`, `long_view`, plus `is_hate`, `is_profile_enter`, `is_rand`.
- Runtime limit per iteration is a hard timeout — a run that does not finish is a
  wasted iteration and is rejected.
