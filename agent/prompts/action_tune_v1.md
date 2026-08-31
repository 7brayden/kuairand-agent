Change **hyperparameters only** — learning rate, L2/regularisation, embedding size,
tree/leaf counts, smoothing priors, epochs, early-stopping patience, and the
**construction of training groups** (how rows are chunked into lists for a
pairwise/listwise objective). Do not restructure the model or the features.

Every gain this project has actually banked came from this action: raising FM's L2 from
1e-6 to ~1e-4 in response to a loss curve that kept falling while validation degraded.
Regularisation and learning rate are the highest-yield knobs measured so far.

Training group size is also yours: the evaluator groups by `user_id` (median 4-5
impressions), while a user has median 31 in train. Matching them is untested.

This is the cheapest action. It is worth taking only when the metric history suggests
the current structure is under-tuned rather than wrong. Remember a gain under 0.002 is
noise and will be rejected.

This is a refinement of working code: answer with search/replace blocks, not a rewrite.
