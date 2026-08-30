"""Change hyperparameters only.

The cheapest action. Worth taking only when the history suggests the structure is under-tuned rather than wrong.

The prompt text for this action lives in ``agent/prompts/action_tune_v1.md`` and is
loaded by :class:`agent.codegen.LLMCodeGenerator`. This module holds only the action's
identity — there is deliberately no ``build_context``/``validate_output`` here: context
assembly is shared across all actions in ``agent/codegen.py``, and validation is the
single guard in ``agent.codegen.lint_generated_code`` so that no action can be
accidentally exempted from it.
"""

from __future__ import annotations

ACTION = "tune"
PROMPT_NAME = "action_tune_v1"
DESCRIPTION = "Change hyperparameters only."
