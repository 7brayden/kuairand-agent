"""The invariant that matters at submission time: the validation-best must be reproducible.

At convergence the organisers evaluate the VALIDATION-BEST checkpoint once on hidden
test. If the best-scoring node was rejected — its workspace reverted, its checkpoint
discarded — there is nothing to submit, and the failure only surfaces at the very end.
That happened: a 0.5984 iteration was rejected for gaining only +0.0014 (under epsilon)
while still being the run's best score.

These tests pin the fix: acceptance follows "is this the best score", convergence follows
epsilon, and the two must never be conflated again.
"""

from __future__ import annotations

import pytest

from agent.critic import judge
from agent.executor import ExecutionResult
from agent.state import Node, RunState, is_converged

OK = ExecutionResult(stdout="", stderr="", returncode=0, timed_out=False, wall_seconds=1.0)


def metrics(primary: float) -> dict:
    return {"GAUC": primary, "nDCG@5": primary, "primary": primary}


def state_with_best(primary: float) -> RunState:
    s = RunState()
    s.record_iteration(Node("best", None, "sha", primary, primary, True),
                       tokens_in=0, tokens_out=0, gpu_seconds=0, wall_seconds=0)
    return s


def test_sub_epsilon_improvement_is_still_accepted() -> None:
    """The exact case that broke: +0.0014 over the tip, under epsilon 0.002."""
    v = judge(state_with_best(0.5970), OK, metrics(0.5984), epsilon=0.002)
    assert v.accepted, "the best-scoring node must be committed — it is the submission"
    assert "0.0014" in v.reason


def test_a_worse_result_is_still_rejected() -> None:
    assert not judge(state_with_best(0.5984), OK, metrics(0.5900), epsilon=0.002).accepted


def test_an_equal_result_is_rejected() -> None:
    """Ties break toward the earlier node; re-accepting churns the tree for nothing."""
    assert not judge(state_with_best(0.5984), OK, metrics(0.5984), epsilon=0.002).accepted


def test_acceptance_does_not_change_convergence() -> None:
    """Epsilon still governs convergence: accepting a tiny gain must not keep a
    plateaued run alive forever."""
    scores = [0.5970, 0.5984, 0.5986, 0.5988]      # every step accepted, all under eps
    assert is_converged(scores, epsilon=0.002, patience=3)


def test_a_real_gain_still_resets_patience() -> None:
    assert not is_converged([0.5970, 0.5984, 0.5986, 0.6100], epsilon=0.002, patience=3)


@pytest.mark.parametrize("primary", [0.4900, 0.5984, 0.7000])
def test_the_best_node_is_always_an_accepted_node(primary: float) -> None:
    """The invariant, stated directly: whatever best_node() returns must be something
    the critic accepted, because only accepted nodes keep a commit and a checkpoint."""
    s = RunState()
    for i, p in enumerate([0.5970, primary]):
        verdict = judge(s, OK, metrics(p))
        s.record_iteration(
            Node(f"n{i}", None, "sha" if verdict.accepted else None, p, p, verdict.accepted),
            tokens_in=0, tokens_out=0, gpu_seconds=0, wall_seconds=0)
    best = s.best_node()
    assert best.accepted and best.commit_sha is not None
