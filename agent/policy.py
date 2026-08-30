"""Action selection: what to try next, and why.

The hypothesis attached to each proposal is a judge deliverable (Innovation &
Problem Insight, 20% — judged on reasoning, not implementation quality). It goes
verbatim into the journal, so no policy may leave it empty.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from agent.journal import POLICY_ACTIONS, ErrorEvent, JournalEntry
from agent.llm import LLMError, extract_json_block
from agent.state import RunState


@dataclass
class ActionProposal:
    """What the policy wants to do next."""

    action_type: str            # one of journal.ACTION_TYPES
    hypothesis: str             # why — journaled verbatim; never empty
    parent_id: Optional[str] = None   # node to branch from (None = current tip)
    context: dict[str, Any] = field(default_factory=dict)  # action-specific inputs


class Policy(ABC):
    """Interface: read state + journal history, propose the next action."""

    @abstractmethod
    def propose(self, state: RunState, history: list[JournalEntry]) -> ActionProposal:
        ...


class RandomPolicy(Policy):
    """Milestone 0: pick a random action from a fixed list, seeded for reproducibility.

    Deliberately dumb. Its job is to prove the harness — journal, diffs, token
    accounting, timeout recovery, convergence — on a run that produces garbage
    models, before any intelligence is added. It cycles deterministically through
    the stub variants (including the deliberately broken and deliberately slow ones)
    so a single short run exercises both recovery paths.
    """

    def __init__(self, actions: list[str], seed: int = 0,
                 variants: Optional[list[str]] = None) -> None:
        self.actions = actions
        self.rng = random.Random(seed)
        # Fixed order, not sampled: a harness proof that only *probably* hits the
        # timeout path is not a proof.
        self.variants = variants or ["popularity", "broken_syntax", "random_scores",
                                     "infinite_loop", "popularity"]

    def propose(self, state: RunState, history: list[JournalEntry]) -> ActionProposal:
        i = state.iteration
        variant = self.variants[i % len(self.variants)]
        return ActionProposal(
            action_type=self.rng.choice(self.actions),
            hypothesis=(f"random baseline action (harness proof, iteration {i}); "
                        f"stub variant '{variant}' — no reasoning involved, this policy "
                        f"exists only to exercise the loop"),
            parent_id=None,
            context={"stub_variant": variant},
        )


class LLMPolicy(Policy):
    """Reflect over the journal, choose an action, articulate a falsifiable hypothesis,
    and decide whether to extend the current tip or backtrack to an earlier accepted
    node (tree search — cf. AIDE / AI-Scientist-v2).

    **This method never raises.** A provider outage or an unparseable response must not
    end a run: the failure is recorded in :attr:`pending_errors` for the loop to journal,
    and a safe fallback proposal is returned instead. Robustness is judged on what
    happens after a failure, and "the policy threw" would be the worst possible answer.
    """

    def __init__(self, client, actions: list[str], prompt_name: str = "policy_v1",
                 effort: str = "high") -> None:
        self.client = client
        self.actions = actions
        self.prompt_name = prompt_name
        self.effort = effort
        #: Errors from the most recent propose(); the loop drains these into the journal.
        self.pending_errors: list[ErrorEvent] = []

    def propose(self, state: RunState, history: list[JournalEntry]) -> ActionProposal:
        from agent import memory  # local import: memory imports state, avoid a cycle

        self.pending_errors = []
        variables = {
            "task_brief": self.client.load_prompt("_task_brief_v1"),
            "journal_summary": memory.summarize_journal(history),
            "resource_summary": memory.summarize_resources(state),
            "current_zone": self.current_zone,
            "actions": ", ".join(self.actions),
        }
        try:
            # Reasoning quality here IS the Innovation score (20%) — worth the tokens.
            response = self.client.complete(self.prompt_name, variables,
                                            effort=self.effort)
            return self._parse(response.text, state)
        except LLMError as exc:
            kind = ("llm_api_error" if "provider failed" in str(exc)
                    else "bad_llm_output")
            self.pending_errors.append(
                ErrorEvent(kind, str(exc)[:2000], "reroute", True))
        except Exception as exc:
            self.pending_errors.append(
                ErrorEvent("bad_llm_output", f"{type(exc).__name__}: {exc}"[:2000],
                           "reroute", True))
        return self._fallback(history)

    #: Set by the loop before each call so the policy can see the code it is changing.
    current_zone: str = "(pipeline source unavailable)"

    def _parse(self, text: str, state: RunState) -> ActionProposal:
        data = extract_json_block(text)
        action = str(data.get("action", "")).strip()
        hypothesis = str(data.get("hypothesis", "")).strip()
        if action not in self.actions or action not in POLICY_ACTIONS:
            raise LLMError(f"action {action!r} is not one of {self.actions}")
        if not hypothesis:
            raise LLMError("empty hypothesis — it is a scored deliverable, not optional")

        backtrack = data.get("backtrack_to")
        parent_id = None
        if backtrack:
            node = state.nodes.get(str(backtrack))
            # Only accepted nodes have a commit to branch from; ignore bad ids rather
            # than failing the iteration over them.
            if node is not None and node.accepted:
                parent_id = node.node_id
        return ActionProposal(action_type=action, hypothesis=hypothesis,
                              parent_id=parent_id, context={})

    def _fallback(self, history: list[JournalEntry]) -> ActionProposal:
        """Safe proposal when the model could not be reached or understood.

        Debug if the last iteration failed (there is a concrete thing to fix), else
        tune — the cheapest action, so a degraded policy burns as little as possible.
        """
        last_failed = bool(history and history[-1].error_events)
        action = "debug" if last_failed and "debug" in self.actions else (
            "tune" if "tune" in self.actions else self.actions[0])
        return ActionProposal(
            action_type=action,
            hypothesis=("FALLBACK: the policy model could not be reached or parsed, so "
                        f"the harness selected `{action}` "
                        + ("to repair the previous failure" if last_failed
                           else "as the cheapest safe action")
                        + ". This hypothesis was written by the harness, not the agent."),
            parent_id=None, context={})


def build(policy_cfg: dict[str, Any], client=None) -> Policy:
    """Construct the policy named in configs/agent.yaml."""
    kind = policy_cfg.get("kind", "random")
    actions = policy_cfg.get("actions", ["model"])
    if kind == "random":
        return RandomPolicy(actions, seed=policy_cfg.get("seed", 0))
    if kind == "llm":
        return LLMPolicy(client, actions)
    raise ValueError(f"unknown policy kind {kind!r}")
