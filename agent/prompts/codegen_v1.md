<!-- v1 — turns an accepted hypothesis into pipeline code. -->

You are writing Python for an automated ML pipeline. Your output is executed directly,
unattended, with no human review.

{task_brief}

## Your hypothesis for this iteration

{hypothesis}

## What this action may change

{action_instructions}

## The current agent-owned code

```python
{current_zone}
```

## Two ways to answer — pick the right one

**Prefer editing.** If the current code already works, change only the lines you need,
using search/replace blocks:

```
<<<<<<< SEARCH
    lr = 0.001
    epochs = 40
=======
    lr = 0.003
    epochs = 60
>>>>>>> REPLACE
```

You may send several blocks in one reply. Each SEARCH must be copied **verbatim** from
the current code — exact text, exact indentation — and must match **exactly one** place;
include a surrounding line or two if it would otherwise be ambiguous.

**Rewrite only when the approach itself changes** — a different model family, or when the
current code is still the empty stub. Then send one fenced python block containing the
whole zone, as described below.

Why this matters: a rewrite means retyping a working program from memory under time
pressure, and it usually comes out worse than what you already had. Several good ideas
have been lost that way — built once, badly, and discarded. If your hypothesis is "tune
this", "add this feature", or "fix this bug", **edit**.

## Hard requirements

1. Output either search/replace blocks OR **one fenced python block**, not both. Prose
   outside them is ignored.
2. The block replaces the agent-owned zone. It **must** define
   `fit_predict(train, valid, target, checkpoint_dir)` returning a numpy array of exactly
   `len(target)` finite floats, in `target`'s existing row order.
   **Use `valid` for early stopping and model selection** — it is the real validation
   split, the same one the published FM baseline tunes against. Do NOT carve your own
   holdout out of `train`: `train` and `valid` are separated in time, so an internal
   split of `train` is inflated by user-video memorisation and will tell you a model is
   far better than it is. Never early-stop on `target`.
3. **Do not** reorder, sort, group, or deduplicate `target`. Row order is the
   submission contract and the official validator rejects any other order.
4. **Never** reference the test split — not by name, not by date literal. You may use
   `train` and `target` only.
5. Put all imports inside the function or at the top of your block. Do not redefine
   `load_logs`, `split_of`, or `write_predictions`.
6. Write anything needed to reproduce inference into `checkpoint_dir`.
7. Prefer vectorised pandas/numpy. The pipeline must finish well inside the timeout.
8. **Print your training diagnostics.** stdout is captured into the journal and is the
   only evidence you will have next iteration about *why* a score came out as it did.
   If you train anything iteratively, print per-epoch (or per-N-iterations) training loss
   AND a validation figure, plus the epoch you stopped at and why. Without it, an
   undertrained model is indistinguishable from a wrong idea, and you will discard a
   good hypothesis. Keep it to a handful of compact lines.

## History

{journal_summary}

{extra_context}
