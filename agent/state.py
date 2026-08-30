"""Run state: search-tree nodes, cumulative resource accounting, and convergence.

Convergence rule (organiser-fixed, from ``eval/official/baseline_scores.json``):
a run is converged when the VALIDATION primary has not improved by more than
``epsilon = 0.002`` over the last ``N = 3`` consecutive iterations, OR the compute
budget is exhausted — whichever comes first.

Selection scalar
----------------
``primary = mean(GAUC, nDCG@5)`` — this is the organisers' own primary metric, not
a choice we made. The judged score is the absolute delta over the published FM
baseline, and because the baseline is a constant, mean-of-per-metric-deltas equals
delta-of-primary exactly. So one number drives convergence, checkpoint selection,
and the final score alike, and no baseline value is needed to rank checkpoints.

Epsilon is calibrated, not arbitrary: FM's std over 5 seeds is 0.0008, so
0.002 ~= 2.5 sigma. A "gain" below epsilon is seed noise.

Working assumption (organiser-unspecified): every ATTEMPTED iteration counts toward
the N-window, and a failed iteration (no score) counts as no improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from agent.journal import JournalEntry, load_journal


class OrganiserTBDError(RuntimeError):
    """Raised when a computation needs an organiser-fixed value that is missing.

    Every such value is now published in the starter kit, so this should never
    fire. It remains as a guard: if a config key goes missing, fail loudly rather
    than silently substituting an invented number.
    """


@dataclass
class Node:
    """One node of the search tree = one attempted iteration.

    Thin, derived view over the journal (the journal stays the source of truth);
    ``RunState`` holds these for cheap in-memory traversal. ``parent_id`` links
    mirror the workspace git commit graph.
    """

    node_id: str
    parent_id: Optional[str]
    commit_sha: Optional[str]
    val_gauc: Optional[float]
    val_ndcg5: Optional[float]
    accepted: bool

    @property
    def selection_score(self) -> Optional[float]:
        """Validation primary = mean(GAUC, nDCG@5); ``None`` if either is missing."""
        if self.val_gauc is None or self.val_ndcg5 is None:
            return None
        return (self.val_gauc + self.val_ndcg5) / 2.0


@dataclass
class Budget:
    """Compute ceilings. ``None`` means "no limit of this kind"."""

    max_tokens: Optional[int] = None
    max_gpu_seconds: Optional[float] = None
    max_wall_seconds: Optional[float] = None

    def exhausted(self, tokens_total: int, gpu_seconds: float, wall_seconds: float) -> bool:
        """True if any SET limit is met or exceeded. Unset limits never trip.

        Note ``max_gpu_seconds = 0.0`` is a real limit (CPU-only run), not "unset":
        it trips the moment any GPU time is recorded.
        """
        if self.max_tokens is not None and tokens_total >= self.max_tokens:
            return True
        if self.max_gpu_seconds is not None and gpu_seconds > self.max_gpu_seconds:
            return True
        if self.max_wall_seconds is not None and wall_seconds >= self.max_wall_seconds:
            return True
        return False

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Budget":
        """Build from the ``budget:`` block of configs/agent.yaml (hours -> seconds)."""
        gpu_h = cfg.get("max_gpu_hours")
        wall_h = cfg.get("max_wall_hours")
        return cls(
            max_tokens=cfg.get("max_tokens"),
            max_gpu_seconds=None if gpu_h is None else float(gpu_h) * 3600.0,
            max_wall_seconds=None if wall_h is None else float(wall_h) * 3600.0,
        )


@dataclass
class RunState:
    """In-memory state of one agent run, rebuildable from the journal at any time.

    ``iteration_scores`` has one slot per ATTEMPTED iteration, in order;
    ``None`` marks an iteration that failed before producing a validation score.
    """

    nodes: dict[str, Node] = field(default_factory=dict)
    iteration_scores: list[Optional[float]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    gpu_seconds: float = 0.0
    wall_seconds: float = 0.0
    interventions: int = 0

    @property
    def tokens_total(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def iteration(self) -> int:
        """Number of attempted iterations so far = the next iteration's index."""
        return len(self.iteration_scores)

    def record_iteration(self, node: Node, *, tokens_in: int, tokens_out: int,
                         gpu_seconds: float, wall_seconds: float,
                         intervention: bool = False, is_iteration: bool = True) -> None:
        """Fold one finished journal entry into the state (called after journal append).

        ``is_iteration=False`` marks an entry that is recorded but is NOT an attempt —
        a human intervention. Those must not append to ``iteration_scores``, or a logged
        intervention would count as a no-improvement round and spend convergence patience,
        letting honest bookkeeping end the run early.
        """
        self.nodes[node.node_id] = node
        if is_iteration:
            self.iteration_scores.append(node.selection_score)
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.gpu_seconds += gpu_seconds
        self.wall_seconds += wall_seconds
        if intervention:
            self.interventions += 1

    def best_node(self) -> Optional[Node]:
        """The validation-best node — the one whose checkpoint goes to hidden test.

        Ties break toward the EARLIER node (cheaper, and no evidence the later one
        is better). Returns ``None`` if no iteration has produced a score yet.
        """
        best: Optional[Node] = None
        for node in self.nodes.values():  # insertion order == iteration order
            s = node.selection_score
            if s is None:
                continue
            if best is None or s > best.selection_score:  # type: ignore[operator]
                best = node
        return best

    def tip(self) -> Optional[Node]:
        """Most recent ACCEPTED node — where the next iteration builds from by default."""
        accepted = [n for n in self.nodes.values() if n.accepted]
        return accepted[-1] if accepted else None

    @classmethod
    def from_journal(cls, journal_path: str | Path) -> "RunState":
        """Rebuild state purely from the journal — used on resume after a crash.

        The journal is the single source of truth; this reproduces exactly the
        state the loop would have had in memory. Missing file = fresh run.
        """
        state = cls()
        if not Path(journal_path).exists():
            return state
        from agent.journal import INTERVENTION
        for e in load_journal(journal_path):
            state.record_iteration(
                Node(node_id=e.node_id, parent_id=e.parent_id, commit_sha=e.commit_sha,
                     val_gauc=e.val_gauc, val_ndcg5=e.val_ndcg5, accepted=e.accepted),
                tokens_in=e.tokens_in, tokens_out=e.tokens_out,
                gpu_seconds=e.gpu_seconds, wall_seconds=e.wall_seconds,
                intervention=e.intervention,
                is_iteration=e.action_type != INTERVENTION,
            )
        return state


def is_converged(scores: list[Optional[float]], epsilon: float, patience: int) -> bool:
    """Score-based half of the convergence rule.

    ``scores``: one entry per attempted iteration in order; ``None`` = failed
    iteration, treated as no improvement. Converged when the best of the last
    ``patience`` iterations does not exceed the best of everything before them by
    more than ``epsilon``. Needs strictly more than ``patience`` iterations before
    it can fire (there must be a "before" to compare against).
    """
    if patience <= 0:
        raise ValueError(f"patience must be positive, got {patience}")
    if len(scores) <= patience:
        return False
    vals = [s if s is not None else float("-inf") for s in scores]
    best_before = max(vals[:-patience])
    recent_best = max(vals[-patience:])
    return recent_best <= best_before + epsilon


def check_convergence(state: RunState, convergence_cfg: dict[str, Any],
                      budget: Budget) -> tuple[bool, str]:
    """Full organiser rule: score plateau OR budget exhausted, whichever first.

    Returns ``(converged, reason)`` — the reason string is journaled and shown in
    the run report, so a run always says why it stopped.
    """
    if budget.exhausted(state.tokens_total, state.gpu_seconds, state.wall_seconds):
        return True, (f"budget exhausted (tokens={state.tokens_total}, "
                      f"gpu_s={state.gpu_seconds:.1f}, wall_s={state.wall_seconds:.1f})")
    epsilon = convergence_cfg.get("epsilon")
    patience = convergence_cfg.get("patience")
    if epsilon is None or patience is None:
        raise OrganiserTBDError(
            "convergence epsilon/patience missing from config; they are organiser-fixed "
            "(epsilon=0.002, patience=3) and must not be invented at the call site"
        )
    if is_converged(state.iteration_scores, float(epsilon), int(patience)):
        return True, (f"validation primary plateaued: no gain > {epsilon} over the last "
                      f"{patience} iterations")
    return False, ""
