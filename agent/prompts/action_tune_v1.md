Change **hyperparameters only** — learning rate, tree/leaf counts, regularisation,
smoothing priors, epochs. Do not restructure the model or the features.

This is the cheapest action. It is worth taking only when the metric history suggests
the current structure is under-tuned rather than wrong. Remember a gain under 0.002 is
noise and will be rejected.

This is a refinement of working code: answer with search/replace blocks, not a rewrite.
