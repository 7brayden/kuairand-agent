# Data bias analysis

Measured on **train, valid, and the random-exposure log only**. The test split's labels
were not inspected; only its row structure, which the pipeline must predict on anyway.

Reproduce with `scripts/analyse_bias.py`.

## 1. Almost every evaluation row is a pair the model has never seen

| split | unseen (user, video) pair | unseen video | unseen user |
|---|---|---|---|
| valid | **98.4%** | 0.0% | 1.6% |
| test | **98.7%** | 0.0% | 3.6% |

Every evaluated video appears in train, and nearly every user does — but the *pairing*
is new essentially always. Any explicit `user_id × video_id` cross is therefore a dead
feature at evaluation time: it fires on ~1% of rows. Only **factorised** interactions
(FM's latent dot product) generalise here, which is why FM holds up and a LightGBM
categorical `uv_cross` did not.

## 2. Training data is dominated by three days

59% of all 1.14M training rows fall on 2022-04-10, 04-11 and 04-12. An average training
day carries 81.5k rows; an average evaluation day carries 17.8k — about 22%. A uniformly
weighted model is therefore fit mostly to one short, high-traffic early-April regime and
then asked to rank a much thinner, later one.

## 3. The label rate shifts from train to evaluation, then holds

| | long_view rate |
|---|---|
| train | 0.3366 |
| valid | 0.3133 |
| test | 0.3135 |

The drift is train → evaluation, **not** valid → test. Valid is a well-calibrated proxy
for test on this axis (0.3133 vs 0.3135), so a valid/test generalisation gap is not
explained by label prevalence.

## 4. Exposure bias is severe

Same 04-22..05-08 window, biased vs uniformly random exposure:

| log | rows | long_view |
|---|---|---|
| standard (recommender's choices) | 295,497 | 0.3134 |
| random (uniform exposure) | 1,186,059 | **0.0850** |

The deployed recommender lifts long_view **3.69x** over showing videos at random, and
top-decile videos receive ~62x the relative exposure of bottom-decile ones (gini 0.72).

## 5. Item quality learned from biased traffic is a weak signal

Per-video long_view rate, biased train vs unbiased random log, over the 3,398 videos with
>=50 impressions in both: **correlation 0.375**; mean rate 0.304 biased vs 0.108 unbiased.

An item-popularity feature built from the standard log is therefore measuring "what the
recommender promoted" at least as much as "what people actually watch".

## What this does and does not license

**Actionable.** Recency or inverse-frequency weighting of training rows (finding 2);
preferring factorised interactions over explicit crosses (finding 1); using the random
log as an unbiased *diagnostic* of item quality (finding 5).

**A trap.** Full exposure debiasing (IPS and friends). The evaluation ranks *within the
recommender's own impressions* — biased traffic is the target distribution, not a
nuisance to correct away. Debiasing optimises for a distribution we are not scored on.

**Off limits as training data.** The random log spans 04-22..05-08, which covers the test
window. Using its labels to train would import future information the temporal split
exists to withhold. The organisers propose it as a validation aid; restricting any use to
its 04-22..04-28 portion keeps the test period untouched.
