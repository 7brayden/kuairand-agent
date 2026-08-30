"""Accept/reject decision and reflection after each iteration.

The critic decides whether the workspace change is committed (accepted) or reverted
(rejected), and produces the reflection text the policy reads back from the journal
on later iterations.

Acceptance rule
---------------
Accept iff the run produced a valid submission AND its validation primary is the best
seen so far — by ANY margin, not by epsilon.

Epsilon belongs to convergence, not to acceptance, and conflating them was a real bug:
a run scored 0.5984 against a tip of 0.5970, the +0.0014 gain fell under epsilon, so the
iteration was rejected and its workspace reverted — while ``RunState.best_node()``, which
scans every scored node, still named it the validation best. The submission pipeline then
had a validation-best node with no commit and no checkpoint, and ``final_eval.py`` would
have refused to produce a submission at all.

So: whatever scores best MUST be committed and checkpointed, because that is the artifact
the hidden test is run on. Epsilon still governs convergence (``agent/state.py``), which
reads the score sequence and is unaffected by what the critic accepts.

The cost is accepting improvements smaller than seed noise. That is the right trade: an
over-retained checkpoint costs disk, while a discarded best checkpoint costs the score.

Known limitation (deliberate, documented): this is still greedy hill-climbing — it never
keeps a neutral change that might enable a later win. Tree search mitigates it, since the
policy can branch from any earlier accepted node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from agent.executor import ExecutionResult
from agent.state import RunState


@dataclass
class Verdict:
    accepted: bool
    reason: str                        # journaled; part of the reasoning deliverable
    reflection: Optional[str] = None   # guidance for future iterations (agent memory)


def judge(state: RunState, execution: Optional[ExecutionResult],
          metrics: Optional[dict], epsilon: float = 0.002) -> Verdict:
    """Decide accept/reject for one iteration.

    ``epsilon`` is used only to describe whether a gain clears the noise floor in the
    journalled reason; it does NOT gate acceptance. See the module docstring.
    """
    if execution is not None and execution.timed_out:
        return Verdict(False, "timed out — no score produced",
                       "The change did not terminate inside the budget. Prefer a cheaper "
                       "formulation or cap the training loop explicitly.")
    if execution is not None and not execution.ok:
        return Verdict(False, f"pipeline exited {execution.returncode}",
                       "The change did not run. Fix the error before trying a new idea.")
    if metrics is None:
        return Verdict(False, "no valid metrics produced",
                       "The run finished but produced no scoreable submission.")

    primary = metrics["primary"]
    best = state.best_node()
    best_primary = best.selection_score if best else None

    if best_primary is None:
        return Verdict(True, f"first scored iteration (primary {primary:.4f})",
                       "Baseline established; subsequent changes are measured against it.")
    delta = primary - best_primary
    if delta > 0:
        noise = ("above the noise floor" if delta > epsilon
                 else f"within seed noise (eps {epsilon}), retained anyway because the "
                      f"best-scoring checkpoint is what the hidden test is run on")
        return Verdict(True, f"primary {primary:.4f} beats best {best_primary:.4f} "
                             f"(+{delta:.4f}, {noise})",
                       f"Improvement of {delta:.4f}"
                       + ("" if delta > epsilon else
                          " — under epsilon, so it does not reset convergence patience; "
                          "a bigger effect is needed to keep the run alive."))
    return Verdict(False, f"primary {primary:.4f} vs best {best_primary:.4f} "
                          f"({delta:+.4f}, no improvement)",
                   f"No improvement ({delta:+.4f}). This direction is exhausted or the "
                   f"effect is smaller than seed variance; try a different lever.")
