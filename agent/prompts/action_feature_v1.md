Change the **features** fed to the existing model, not the model itself.

Remember the structural constraint: a feature constant within a user contributes
exactly zero. Item-side features, and user×item crosses, are the only things that can
move the score. Static demographic buckets have already been measured as worthless.

Features must be computed without leakage: statistics used to score a `target` row may
come from `train` only, never from `target`'s own labels.

You are adding to a working pipeline: answer with search/replace blocks so the rest of the model stays exactly as it was.
