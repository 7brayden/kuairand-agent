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

## Hard requirements

1. Output **one fenced python block** and nothing else that matters. Prose outside the
   fence is ignored.
2. The block replaces the agent-owned zone. It **must** define
   `fit_predict(train, target, checkpoint_dir)` returning a numpy array of exactly
   `len(target)` finite floats, in `target`'s existing row order.
3. **Do not** reorder, sort, group, or deduplicate `target`. Row order is the
   submission contract and the official validator rejects any other order.
4. **Never** reference the test split — not by name, not by date literal. You may use
   `train` and `target` only.
5. Put all imports inside the function or at the top of your block. Do not redefine
   `load_logs`, `split_of`, or `write_predictions`.
6. Write anything needed to reproduce inference into `checkpoint_dir`.
7. Prefer vectorised pandas/numpy. The pipeline must finish well inside the timeout.

## History

{journal_summary}

{extra_context}
