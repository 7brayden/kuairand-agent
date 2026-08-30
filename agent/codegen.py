"""Turning a policy proposal into new pipeline source, plus the safety guards.

Two generators share one interface:

* :class:`StubCodeGenerator` — deterministic, no LLM, no tokens. It exists for
  milestone 0 (harness before intelligence): it emits known-good, known-broken, and
  known-slow variants so the journal, diff, timeout, and recovery paths can all be
  proven on a run that produces garbage models. Fault injection is the point, not a
  side effect.
* :class:`LLMCodeGenerator` — the real one, via ``agent/llm.py`` (the single door).

Both write only inside the AGENT-OWNED ZONE of ``main.py``; the I/O contract above
that marker (canonical row order, submission schema) is harness territory and must
survive every iteration.
"""

from __future__ import annotations

import ast
import random
import re
from dataclasses import dataclass
from typing import Optional, Protocol

from agent import actions as actions_registry
from agent.llm import LLMClient, LLMError, extract_code_block  # noqa: F401 (re-export)
from agent.policy import ActionProposal

ZONE_START = "# ------------------------------- AGENT-OWNED ZONE"
ZONE_END = "# --------------------------- END AGENT-OWNED ZONE"


class CodeGenError(RuntimeError):
    """Generated code violated a contract before it was ever executed."""


# --------------------------------- guards -------------------------------------------

#: Patterns that indicate generated code is reaching for the held-out test split.
#: The test labels sit in the same local CSVs as valid, so this is the mechanical
#: half of the honour-system guard described in CLAUDE.md > Test-set integrity.
_TEST_ACCESS_PATTERNS = (
    re.compile(r"""["']test["']"""),
    re.compile(r"\b2022(?:04(?:29|3\d)|050\d)\b"),   # test-window dates as literals
)


def lint_generated_code(source: str) -> list[str]:
    """Return a list of contract violations. Empty list = clean.

    Checks, in order of how expensive the mistake would be:
      1. it parses at all (a syntax error costs a whole iteration otherwise);
      2. the I/O contract zone markers survive;
      3. ``fit_predict`` still exists with the expected signature;
      4. nothing references the test split.
    """
    problems: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax error at line {exc.lineno}: {exc.msg}"]

    if ZONE_START not in source:
        problems.append("AGENT-OWNED ZONE start marker was removed")

    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "fit_predict" not in fns:
        problems.append("fit_predict() is missing — the harness calls it by name")
    else:
        args = [a.arg for a in fns["fit_predict"].args.args]
        if args[:3] != ["train", "target", "checkpoint_dir"]:
            problems.append(f"fit_predict signature changed: {args}")
    for guard in ("load_logs", "split_of", "write_predictions"):
        if guard not in fns:
            problems.append(f"contract function {guard}() was removed")

    zone = source.split(ZONE_START, 1)[-1].split(ZONE_END, 1)[0]
    for pat in _TEST_ACCESS_PATTERNS:
        if pat.search(zone):
            problems.append(
                f"generated code references the held-out test split ({pat.pattern}); "
                "the loop may only ever touch train and valid")
            break
    return problems


def extract_agent_zone(source: str) -> str:
    """Return just the agent-owned region — what the model is asked to rewrite.

    Sending only this instead of the whole file cuts input tokens (15% of the score)
    and removes any chance of the model "helpfully" rewriting the I/O contract.
    """
    if ZONE_START not in source or ZONE_END not in source:
        raise CodeGenError("main.py is missing its AGENT-OWNED ZONE markers")
    zone = source.split(ZONE_START, 1)[1].split(ZONE_END, 1)[0]
    return zone.split("\n", 1)[1] if "\n" in zone else zone


def replace_agent_zone(current_source: str, new_zone_body: str) -> str:
    """Swap the agent-owned region of main.py, leaving the I/O contract untouched."""
    if ZONE_START not in current_source:
        raise CodeGenError("current main.py has no AGENT-OWNED ZONE marker")
    if ZONE_END not in current_source:
        raise CodeGenError("current main.py has no END AGENT-OWNED ZONE marker")
    head = current_source.split(ZONE_START, 1)[0]
    tail = ZONE_END + current_source.split(ZONE_END, 1)[1]
    return (f"{head}{ZONE_START} -----------------------------------\n"
            f"{new_zone_body.strip()}\n\n\n{tail}")


# ------------------------------- generators -----------------------------------------

@dataclass
class GeneratedCode:
    """New pipeline source plus what the agent claims it does."""

    source: str
    summary: str
    config: dict


class CodeGenerator(Protocol):
    def generate(self, proposal: ActionProposal, current_source: str,
                 context: dict) -> GeneratedCode: ...


#: Milestone-0 variants. Two work, two fail on purpose — the failing ones are how we
#: prove the recovery path without waiting for a real LLM to make a real mistake.
STUB_VARIANTS = ("popularity", "random_scores", "broken_syntax", "infinite_loop")

_STUB_BODIES = {
    "popularity": '''
def fit_predict(train, target, checkpoint_dir):
    """Smoothed item popularity: P(long_view | video), Bayesian-shrunk to the mean."""
    import os
    g = train.groupby("video_id")["long_view"]
    stats = g.agg(["sum", "count"])
    prior, gmean = 20.0, float(train["long_view"].mean())
    rate = (stats["sum"] + prior * gmean) / (stats["count"] + prior)
    stats.assign(rate=rate).to_csv(os.path.join(checkpoint_dir, "item_rates.csv"))
    return target["video_id"].map(rate).fillna(gmean).to_numpy(dtype=float)
''',
    "random_scores": '''
def fit_predict(train, target, checkpoint_dir):
    """Uniform random scores — a deliberate lower bound (primary ~ 0.483)."""
    import numpy as np
    return np.random.default_rng(0).random(len(target))
''',
    "broken_syntax": '''
def fit_predict(train, target, checkpoint_dir):
    """Deliberately malformed: exercises the code_error -> debug recovery path."""
    scores = target["video_id"].map(  # unbalanced paren, fails at import time
    return scores
''',
    "infinite_loop": '''
def fit_predict(train, target, checkpoint_dir):
    """Deliberately non-terminating: exercises the hard-timeout kill path."""
    import time
    while True:
        time.sleep(1)
''',
}


@dataclass
class StubCodeGenerator:
    """Deterministic generator for the harness proof. Consumes zero tokens."""

    seed: int = 0
    _rng: Optional[random.Random] = None

    def generate(self, proposal: ActionProposal, current_source: str,
                 context: dict) -> GeneratedCode:
        if self._rng is None:
            self._rng = random.Random(self.seed)
        variant = proposal.context.get("stub_variant") or self._rng.choice(STUB_VARIANTS)
        body = _STUB_BODIES[variant]
        return GeneratedCode(
            source=replace_agent_zone(current_source, body),
            summary=f"stub variant: {variant}",
            config={"generator": "stub", "variant": variant, "action": proposal.action_type},
        )


@dataclass
class LLMCodeGenerator:
    """The real generator: a shared skeleton prompt plus per-action instructions.

    One repair attempt is built in. A lint failure caught here costs one extra LLM call;
    letting it through costs a whole iteration (subprocess start, data load, run, score),
    so repairing in place is the cheaper trade even though tokens are scored.
    """

    client: LLMClient
    prompt_name: str = "codegen_v1"
    max_repairs: int = 1
    #: Writing the code is mechanical once the hypothesis is fixed; deep reasoning here
    #: buys little and thinking tokens count toward the scored total. Keep it lower than
    #: the policy's.
    effort: str = "medium"

    def generate(self, proposal: ActionProposal, current_source: str,
                 context: dict) -> GeneratedCode:
        variables = {
            "task_brief": self.client.load_prompt("_task_brief_v1"),
            "hypothesis": proposal.hypothesis,
            "action_instructions": self.client.load_prompt(
                actions_registry.prompt_name(proposal.action_type)),
            "current_zone": extract_agent_zone(current_source),
            "journal_summary": context.get("journal_summary", "(no history)"),
            "extra_context": context.get("extra_context", ""),
        }

        problems: list[str] = []
        for attempt in range(self.max_repairs + 1):
            if problems:  # repair round: tell it exactly what the guard rejected
                variables["extra_context"] = (
                    f"{context.get('extra_context', '')}\n\n"
                    f"## Your previous attempt was REJECTED before execution\n\n"
                    + "\n".join(f"- {p}" for p in problems)
                    + "\n\nReturn a corrected block that fixes exactly these problems.")
            response = self.client.complete(self.prompt_name, variables,
                                            effort=self.effort)
            try:
                zone = extract_code_block(response.text)
            except LLMError as exc:
                problems = [str(exc)]
                continue
            candidate = replace_agent_zone(current_source, zone)
            problems = lint_generated_code(candidate)
            if not problems:
                return GeneratedCode(
                    source=candidate,
                    summary=_summarise(proposal.hypothesis),
                    config={"generator": "llm", "action": proposal.action_type,
                            "model": self.client.model, "prompt": self.prompt_name,
                            "repairs": attempt},
                )
        raise CodeGenError("generated code failed the guards after "
                           f"{self.max_repairs + 1} attempts: {'; '.join(problems)}")


def _summarise(hypothesis: str, limit: int = 72) -> str:
    """One-line commit subject from the hypothesis."""
    line = " ".join(hypothesis.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"



