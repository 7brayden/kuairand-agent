"""Tests for convergence, budget, and validation-best selection."""

from __future__ import annotations

import pytest

from agent.state import (
    Budget,
    Node,
    OrganiserTBDError,
    RunState,
    check_convergence,
    is_converged,
)


def node(node_id: str, gauc: float | None, ndcg5: float | None,
         accepted: bool = True) -> Node:
    return Node(node_id=node_id, parent_id=None, commit_sha=None,
                val_gauc=gauc, val_ndcg5=ndcg5, accepted=accepted)


class TestIsConverged:
    def test_not_converged_while_improving(self) -> None:
        assert not is_converged([0.1, 0.2, 0.3, 0.4], epsilon=0.01, patience=2)

    def test_converged_when_plateaued_within_epsilon(self) -> None:
        assert is_converged([0.1, 0.3, 0.305, 0.301], epsilon=0.01, patience=2)

    def test_improvement_larger_than_epsilon_resets(self) -> None:
        assert not is_converged([0.1, 0.3, 0.305, 0.35], epsilon=0.01, patience=2)

    def test_needs_more_than_patience_iterations(self) -> None:
        assert not is_converged([0.5, 0.5], epsilon=0.01, patience=2)
        assert not is_converged([0.5], epsilon=0.01, patience=2)

    def test_failed_iterations_count_as_no_improvement(self) -> None:
        assert is_converged([0.3, None, None], epsilon=0.01, patience=2)

    def test_all_failed_never_diverges(self) -> None:
        assert is_converged([None, None, None], epsilon=0.01, patience=2)

    def test_invalid_patience_raises(self) -> None:
        with pytest.raises(ValueError):
            is_converged([0.1], epsilon=0.01, patience=0)


class TestSelection:
    def test_selection_score_is_official_primary(self) -> None:
        """primary = mean(GAUC, nDCG@5) — reproduce the published FM valid row."""
        assert node("a", 0.6674, 0.5357).selection_score == pytest.approx(0.6016, abs=5e-5)

    def test_selection_score_none_when_metric_missing(self) -> None:
        assert node("a", None, 0.4).selection_score is None

    def test_best_node_ties_break_earlier(self) -> None:
        state = RunState()
        for n in [node("a", 0.2, 0.4), node("b", 0.3, 0.3), node("c", None, None)]:
            state.record_iteration(n, tokens_in=10, tokens_out=5,
                                   gpu_seconds=0.0, wall_seconds=1.0)
        best = state.best_node()
        assert best is not None and best.node_id == "a"  # equal scores -> earlier wins

    def test_resource_totals_accumulate(self) -> None:
        state = RunState()
        state.record_iteration(node("a", 0.1, 0.1), tokens_in=100, tokens_out=50,
                               gpu_seconds=2.0, wall_seconds=10.0)
        state.record_iteration(node("b", 0.2, 0.2), tokens_in=200, tokens_out=25,
                               gpu_seconds=0.0, wall_seconds=5.0)
        assert state.tokens_total == 375
        assert state.gpu_seconds == pytest.approx(2.0)


class TestCheckConvergence:
    def test_tbd_epsilon_raises_instead_of_guessing(self) -> None:
        state = RunState()
        with pytest.raises(OrganiserTBDError):
            check_convergence(state, {"epsilon": None, "patience": None}, Budget())

    def test_budget_exhaustion_converges_even_with_tbd_epsilon(self) -> None:
        state = RunState()
        state.record_iteration(node("a", 0.1, 0.1), tokens_in=900, tokens_out=200,
                               gpu_seconds=0.0, wall_seconds=1.0)
        budget = Budget(max_tokens=1000)
        assert check_convergence(state, {"epsilon": None, "patience": None}, budget)[0]

    def test_unset_budget_limits_never_trip(self) -> None:
        assert not Budget().exhausted(10**9, 10**9, 10**9)


class TestInterventionsDoNotSpendPatience:
    """A logged intervention must not count as a no-improvement iteration — otherwise
    honest bookkeeping would end a run early, penalising the honesty."""

    def test_intervention_does_not_append_a_score_slot(self) -> None:
        state = RunState()
        state.record_iteration(node("a", 0.66, 0.53), tokens_in=0, tokens_out=0,
                               gpu_seconds=0, wall_seconds=0)
        state.record_iteration(node("i", None, None, False), tokens_in=0, tokens_out=0,
                               gpu_seconds=0, wall_seconds=0,
                               intervention=True, is_iteration=False)
        assert state.iteration_scores == [pytest.approx(0.595)]
        assert state.interventions == 1
        assert state.iteration == 1        # the intervention is not an attempt

    def test_journal_rebuild_excludes_interventions_from_patience(self, tmp_path) -> None:
        from agent.journal import INTERVENTION, JournalEntry, append_entry
        j = tmp_path / "j.jsonl"

        def write(i, action, gauc, ndcg5, interv=False):
            append_entry(j, JournalEntry(
                node_id=f"n{i}", parent_id=None, iteration=i, timestamp="t",
                action_type=action, hypothesis="h", config={}, diff_path=None,
                commit_sha=None, checkpoint_path=None, val_gauc=gauc, val_ndcg5=ndcg5,
                wall_seconds=0.0, gpu_seconds=0.0, tokens_in=0, tokens_out=0,
                error_events=[], accepted=False, intervention=interv))

        write(0, "model", 0.66, 0.53)
        write(1, INTERVENTION, None, None, interv=True)
        write(2, "tune", 0.60, 0.50)
        rebuilt = RunState.from_journal(j)
        assert len(rebuilt.iteration_scores) == 2      # not 3
        assert rebuilt.interventions == 1
