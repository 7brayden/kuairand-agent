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


class NoEditBlocks(Exception):
    """The response contained no SEARCH/REPLACE blocks — treat it as a full rewrite."""


#: Actions that are refinements of working code by definition. For these, a full rewrite
#: is rejected before execution rather than merely discouraged: the first live run with
#: editing available chose `rewrite` for all four iterations, including a `feature` step
#: whose prompt asked for edits. Asking politely did not work; the guard does.
EDIT_REQUIRED_ACTIONS = frozenset({"tune", "feature", "debug"})

#: Marker for the untouched template body — there is nothing to edit yet.
_STUB_MARKER = "return np.zeros(len(target)"


def is_stub_zone(zone: str) -> bool:
    """True while the agent-owned zone is still the empty template implementation."""
    return _STUB_MARKER in zone


#: A surgical edit, in the search/replace form LLMs handle most reliably.
#: Unified diffs need exact line numbers and hunk headers, which models get wrong often
#: enough to cost iterations; an exact-string match either applies or fails loudly.
EDIT_BLOCK_RE = re.compile(
    r"<{5,}\s*SEARCH\s*\n(.*?)\n?={5,}\s*\n(.*?)\n?>{5,}\s*REPLACE",
    re.DOTALL)


def apply_edit_blocks(zone: str, text: str) -> tuple[str, int]:
    """Apply SEARCH/REPLACE blocks to the agent-owned zone.

    Why this exists: for four runs the agent could only replace its entire program, so
    every refinement was a rushed from-scratch rewrite of working code. Good ideas
    (listwise loss, multi-task heads) never got tuned — they got rebuilt badly once and
    discarded. Editing lets an idea survive long enough to be worth judging.

    Each SEARCH must match EXACTLY ONCE in the zone. Ambiguity is rejected rather than
    guessed at: silently editing the wrong one of two identical blocks produces code that
    runs and is subtly wrong, which is the worst possible failure here.

    Returns (new_zone, number_of_edits). Raises NoEditBlocks if this is a rewrite.
    """
    blocks = EDIT_BLOCK_RE.findall(text)
    if not blocks:
        raise NoEditBlocks()

    out = zone
    for i, (search, replace) in enumerate(blocks, 1):
        if not search.strip():
            raise CodeGenError(f"edit {i}: empty SEARCH block")
        found = out.count(search)
        if found == 0:
            preview = " / ".join(search.strip().splitlines()[:2])[:160]
            raise CodeGenError(
                f"edit {i}: SEARCH block not found in the current code — it must be "
                f"copied verbatim, including indentation. Looked for: {preview!r}")
        if found > 1:
            raise CodeGenError(
                f"edit {i}: SEARCH block matches {found} places; it must be unique. "
                f"Include surrounding lines to disambiguate.")
        out = out.replace(search, replace, 1)
    return out, len(blocks)


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
        if args[:5] != ["train", "valid", "target", "checkpoint_dir", "unbiased"]:
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
def fit_predict(train, valid, target, checkpoint_dir, unbiased):
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
def fit_predict(train, valid, target, checkpoint_dir, unbiased):
    """Uniform random scores — a deliberate lower bound (primary ~ 0.483)."""
    import numpy as np
    return np.random.default_rng(0).random(len(target))
''',
    "broken_syntax": '''
def fit_predict(train, valid, target, checkpoint_dir, unbiased):
    """Deliberately malformed: exercises the code_error -> debug recovery path."""
    scores = target["video_id"].map(  # unbalanced paren, fails at import time
    return scores
''',
    "infinite_loop": '''
def fit_predict(train, valid, target, checkpoint_dir, unbiased):
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
    max_repairs: int = 2   # 3 attempts. Two consecutive syntax errors ended an
                           # iteration that a third attempt would likely have saved.
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

        current_zone = variables["current_zone"]
        problems: list[str] = []
        for attempt in range(self.max_repairs + 1):
            if problems:  # repair round: tell it exactly what the guard rejected
                variables["extra_context"] = (
                    f"{context.get('extra_context', '')}\n\n"
                    f"## Your previous attempt was REJECTED before execution\n\n"
                    + "\n".join(f"- {p}" for p in problems)
                    + "\n\nReturn a corrected response that fixes exactly these problems.")
            response = self.client.complete(self.prompt_name, variables,
                                            effort=self.effort)

            # Prefer a surgical edit; fall back to a full rewrite. Which one arrived is
            # the model's choice, and is journaled so we can see whether editing helps.
            mode, edits = "edit", 0
            try:
                zone, edits = apply_edit_blocks(current_zone, response.text)
            except NoEditBlocks:
                mode = "rewrite"
                try:
                    zone = extract_code_block(response.text)
                except LLMError as exc:
                    problems = [f"{exc} (and no SEARCH/REPLACE blocks either)"]
                    continue
            except CodeGenError as exc:
                problems = [str(exc)]
                continue

            # Refinement actions should edit. Push back once, but do not spend the whole
            # iteration on the argument: on the last attempt a rewrite is accepted, since
            # a worse-shaped change still beats no change at all.
            if (mode == "rewrite"
                    and proposal.action_type in EDIT_REQUIRED_ACTIONS
                    and not is_stub_zone(current_zone)
                    and attempt < self.max_repairs):
                problems = [
                    f"`{proposal.action_type}` is a refinement of code that already works, "
                    f"so it must be expressed as SEARCH/REPLACE edits, not a full rewrite. "
                    f"Rewriting retypes a working program from memory and reliably comes "
                    f"out worse. Send only the lines you are changing."]
                continue

            candidate = replace_agent_zone(current_source, zone)
            problems = lint_generated_code(candidate)
            if not problems:
                return GeneratedCode(
                    source=candidate,
                    summary=_summarise(proposal.hypothesis),
                    config={"generator": "llm", "action": proposal.action_type,
                            "model": self.client.model, "prompt": self.prompt_name,
                            "repairs": attempt, "mode": mode, "edits": edits},
                )
        raise CodeGenError("generated code failed the guards after "
                           f"{self.max_repairs + 1} attempts: {'; '.join(problems)}")


def _summarise(hypothesis: str, limit: int = 72) -> str:
    """One-line commit subject from the hypothesis."""
    line = " ".join(hypothesis.split())
    return line if len(line) <= limit else line[: limit - 1] + "…"



