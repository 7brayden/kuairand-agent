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

## 6. Training groups are 6-8x larger than evaluation groups

The evaluator groups strictly by `user_id` (`eval/official/evaluate.py`), so a "group" is
one user's impressions within the split:

| split | groups | median size | mean | users with mixed labels |
|---|---|---|---|---|
| train | 26,210 | **31** | 43.5 | 92.7% |
| valid | 22,377 | **4** | 5.6 | 57.8% |
| test | 23,875 | **5** | 7.1 | 63.7% |

Any pairwise or listwise objective trained with `group = user_id` on the train split
therefore learns to order ~31-43 item lists, and is then scored on ~5-7 item lists.
nDCG@5 covers nearly an entire evaluation list but only the top ~12% of a training one,
and softmax/lambdarank gradients both depend on list length. The objective is aligned in
form and mismatched in scale — a plausible reason lambdarank, BPR and listwise softmax
have each failed to beat a pointwise FM here.

**Training group construction is a free parameter.** The evaluator's grouping is fixed,
but nothing requires training groups to be whole users: chunking each user's training
impressions into blocks of ~5-7, or grouping by `(user_id, date)` (median 3 in train),
would match training list length to evaluation list length. Untested.

Note also that 36-42% of evaluation users have all-positive or all-negative labels, so
they contribute a constant to nDCG and are excluded from GAUC entirely — against 7.3% in
train. Training over-represents the discriminative case.

## What this does and does not license

**Actionable.** Matching training group size to evaluation group size for any
pairwise/listwise objective (finding 6 — the most concrete untested lever, and it
subsumes several failed attempts); recency or inverse-frequency weighting of training
rows (finding 2); preferring factorised interactions over explicit crosses (finding 1);
using the random log as an unbiased *diagnostic* of item quality (finding 5).

**A trap.** Full exposure debiasing (IPS and friends). The evaluation ranks *within the
recommender's own impressions* — biased traffic is the target distribution, not a
nuisance to correct away. Debiasing optimises for a distribution we are not scored on.

**Off limits as training data.** The random log spans 04-22..05-08, which covers the test
window. Using its labels to train would import future information the temporal split
exists to withhold. The organisers propose it as a validation aid; restricting any use to
its 04-22..04-28 portion keeps the test period untouched.
