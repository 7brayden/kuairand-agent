"""Action registry — one module per entry in ``journal.POLICY_ACTIONS``.

Each module carries its action's identity (``ACTION``, ``PROMPT_NAME``, ``DESCRIPTION``);
the prompt body lives in ``agent/prompts/action_<name>_v1.md``. Context assembly and
validation are deliberately NOT per-action: they are shared in ``agent/codegen.py`` so
that no action can be accidentally exempted from the guards (contract preservation and
the test-split lint).

Generated pipeline code is pandas-based — LLMs write pandas fluently, and the
organisers' own loader is plain csv+numpy, so there is no house dataframe library to
conform to.
"""

from __future__ import annotations

from agent.actions import debug, eda, feature, model, tune

MODULES = {m.ACTION: m for m in (eda, feature, model, tune, debug)}


def prompt_name(action: str) -> str:
    """Prompt file (without extension) for an action. Raises on an unknown action."""
    if action not in MODULES:
        raise KeyError(f"unknown action {action!r}; known: {sorted(MODULES)}")
    return MODULES[action].PROMPT_NAME


def describe(action: str) -> str:
    return MODULES[action].DESCRIPTION


__all__ = ["MODULES", "prompt_name", "describe", "eda", "feature", "model", "tune", "debug"]
