"""Investigate the data and encode what is learned as a scoring change.

Findings are printed by the pipeline and land in the journal, so an EDA iteration adds knowledge even when its score is flat.

The prompt text for this action lives in ``agent/prompts/action_eda_v1.md`` and is
loaded by :class:`agent.codegen.LLMCodeGenerator`. This module holds only the action's
identity — there is deliberately no ``build_context``/``validate_output`` here: context
assembly is shared across all actions in ``agent/codegen.py``, and validation is the
single guard in ``agent.codegen.lint_generated_code`` so that no action can be
accidentally exempted from it.
"""

from __future__ import annotations

ACTION = "eda"
PROMPT_NAME = "action_eda_v1"
DESCRIPTION = "Investigate the data and encode what is learned as a scoring change."
