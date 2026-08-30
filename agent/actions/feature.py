"""Change the features fed to the existing model.

Constrained by the within-user structure: a feature constant within a user contributes exactly zero, so only item-side and cross terms move the score.

The prompt text for this action lives in ``agent/prompts/action_feature_v1.md`` and is
loaded by :class:`agent.codegen.LLMCodeGenerator`. This module holds only the action's
identity — there is deliberately no ``build_context``/``validate_output`` here: context
assembly is shared across all actions in ``agent/codegen.py``, and validation is the
single guard in ``agent.codegen.lint_generated_code`` so that no action can be
accidentally exempted from it.
"""

from __future__ import annotations

ACTION = "feature"
PROMPT_NAME = "action_feature_v1"
DESCRIPTION = "Change the features fed to the existing model."
