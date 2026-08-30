"""Repair the previous failure, and nothing else.

The recovery path: scheduled by the policy (or by its fallback) after an iteration produced error events.

The prompt text for this action lives in ``agent/prompts/action_debug_v1.md`` and is
loaded by :class:`agent.codegen.LLMCodeGenerator`. This module holds only the action's
identity — there is deliberately no ``build_context``/``validate_output`` here: context
assembly is shared across all actions in ``agent/codegen.py``, and validation is the
single guard in ``agent.codegen.lint_generated_code`` so that no action can be
accidentally exempted from it.
"""

from __future__ import annotations

ACTION = "debug"
PROMPT_NAME = "action_debug_v1"
DESCRIPTION = "Repair the previous failure, and nothing else."
