"""Change the model or its training objective.

The organisers' highest-rated untested lever: the metrics are ranking metrics while the baseline optimises pointwise logloss.

The prompt text for this action lives in ``agent/prompts/action_model_v1.md`` and is
loaded by :class:`agent.codegen.LLMCodeGenerator`. This module holds only the action's
identity — there is deliberately no ``build_context``/``validate_output`` here: context
assembly is shared across all actions in ``agent/codegen.py``, and validation is the
single guard in ``agent.codegen.lint_generated_code`` so that no action can be
accidentally exempted from it.
"""

from __future__ import annotations

ACTION = "model"
PROMPT_NAME = "action_model_v1"
DESCRIPTION = "Change the model or its training objective."
