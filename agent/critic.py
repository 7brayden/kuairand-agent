"""Accept/reject decision and reflection after each iteration.

The critic decides whether the workspace change is committed (accepted) or reverted
(rejected), and produces the reflection text the policy reads back from the journal
on later iterations.

Acceptance rule
---------------
Accept iff the run produced a valid submission AND its validation primary beats the
current best by more than the organisers' epsilon (0.002 ~ 2.5 sigma of seed noise).
Comparing against the RUN BEST rather than the parent means a lucky-looking sideways
move cannot ratchet the tree sideways forever.

Known limitation (deliberate, documented): this is greedy hill-climbing. It never
keeps a neutral change that might enable a later win. Tree search mitigates it —
the policy can branch from any earlier accepted node — but a real exploration budget
is the obvious next upgrade.
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
    """Decide accept/reject for one iteration."""
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
    if delta > epsilon:
        return Verdict(True, f"primary {primary:.4f} beats best {best_primary:.4f} "
                             f"(+{delta:.4f} > eps {epsilon})",
                       f"Improvement of {delta:.4f} confirmed above noise.")
    return Verdict(False, f"primary {primary:.4f} vs best {best_primary:.4f} "
                          f"({delta:+.4f}, not > eps {epsilon})",
                   f"No gain above noise ({delta:+.4f}). This direction is exhausted or "
                   f"the effect is smaller than seed variance; try a different lever.")
