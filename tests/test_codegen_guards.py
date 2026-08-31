"""Guards that run before any generated code is executed."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.codegen import (STUB_VARIANTS, CodeGenError, _STUB_BODIES, extract_agent_zone,
                           extract_code_block, lint_generated_code, replace_agent_zone)
from agent.llm import LLMError

TEMPLATE = (Path(__file__).resolve().parent.parent / "pipeline/template/main.py").read_text()


def test_template_itself_is_clean() -> None:
    assert lint_generated_code(TEMPLATE) == []


@pytest.mark.parametrize("variant", ["popularity", "random_scores", "infinite_loop"])
def test_valid_stub_variants_pass_lint(variant: str) -> None:
    assert lint_generated_code(replace_agent_zone(TEMPLATE, _STUB_BODIES[variant])) == []


def test_syntax_error_is_caught_before_execution() -> None:
    problems = lint_generated_code(replace_agent_zone(TEMPLATE, _STUB_BODIES["broken_syntax"]))
    assert any("syntax error" in p for p in problems)


def test_test_split_access_is_rejected() -> None:
    leak = replace_agent_zone(TEMPLATE, '''
def fit_predict(train, valid, target, checkpoint_dir):
    extra = split_of(load_logs("d"), "test")
    return extra
''')
    assert any("test split" in p for p in lint_generated_code(leak))


def test_removing_contract_function_is_rejected() -> None:
    broken = TEMPLATE.replace("def write_predictions(", "def _disabled_write_predictions(")
    assert any("write_predictions" in p for p in lint_generated_code(broken))


def test_changed_fit_predict_signature_is_rejected() -> None:
    bad = replace_agent_zone(TEMPLATE, "def fit_predict(df):\n    return []\n")
    assert any("signature" in p for p in lint_generated_code(bad))


def test_dropping_the_valid_arg_is_rejected() -> None:
    """`valid` is how the pipeline does honest model selection; losing it silently
    sends the agent back to an internal split of train, which is memorisation-inflated."""
    old = replace_agent_zone(
        TEMPLATE, "def fit_predict(train, target, checkpoint_dir):\n    return []\n")
    assert any("signature" in p for p in lint_generated_code(old))


def test_zone_replacement_preserves_io_contract() -> None:
    out = replace_agent_zone(TEMPLATE, _STUB_BODIES["popularity"])
    for contract in ("def load_logs(", "def split_of(", "def write_predictions(", "row_id"):
        assert contract in out


def test_missing_marker_raises() -> None:
    with pytest.raises(CodeGenError):
        replace_agent_zone("print('no markers here')", "def fit_predict(a,b,c): pass")


def test_extract_code_block() -> None:
    assert extract_code_block("blah\n```python\nx = 1\n```\ntrailing").strip() == "x = 1"
    with pytest.raises(LLMError):
        extract_code_block("no fence at all")


def test_extract_agent_zone_excludes_the_io_contract() -> None:
    """Only the agent-owned region is sent to the model — cheaper, and it cannot
    rewrite the submission contract if it never sees it."""
    zone = extract_agent_zone(TEMPLATE)
    assert "def fit_predict(" in zone
    assert "def write_predictions(" not in zone and "row_id" not in zone


# --------------------------- surgical edits (edit vs rewrite) -------------------------
# For four runs the agent could only replace its whole program, so every refinement was
# a rushed rewrite of working code and good ideas were rebuilt badly, then discarded.

from agent.codegen import NoEditBlocks, apply_edit_blocks  # noqa: E402

ZONE = "def fit_predict(train, valid, target, checkpoint_dir):\n    k = 16\n    lr = 0.001\n    return k, lr\n"


def test_single_edit_applies() -> None:
    new, n = apply_edit_blocks(
        ZONE, "reasoning\n<<<<<<< SEARCH\n    k = 16\n=======\n    k = 32\n>>>>>>> REPLACE\n")
    assert n == 1 and "k = 32" in new and "lr = 0.001" in new   # untouched line survives


def test_several_edits_apply_in_order() -> None:
    new, n = apply_edit_blocks(ZONE, (
        "<<<<<<< SEARCH\n    k = 16\n=======\n    k = 64\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\n    lr = 0.001\n=======\n    lr = 0.01\n>>>>>>> REPLACE\n"))
    assert n == 2 and "k = 64" in new and "lr = 0.01" in new


def test_a_plain_rewrite_is_signalled_not_misparsed() -> None:
    with pytest.raises(NoEditBlocks):
        apply_edit_blocks(ZONE, "```python\ndef fit_predict(a,b,c,d): pass\n```")


def test_unmatched_search_is_rejected_with_a_usable_message() -> None:
    with pytest.raises(CodeGenError, match="not found"):
        apply_edit_blocks(ZONE, "<<<<<<< SEARCH\n    k = 99\n=======\n    k = 1\n>>>>>>> REPLACE")


def test_ambiguous_search_is_rejected_rather_than_guessed() -> None:
    """Editing the wrong one of two identical blocks yields code that runs and is
    subtly wrong — the worst failure mode available here."""
    dup = "def f():\n    x = 1\n    y = 2\n    x = 1\n"
    with pytest.raises(CodeGenError, match="matches 2 places"):
        apply_edit_blocks(dup, "<<<<<<< SEARCH\n    x = 1\n=======\n    x = 5\n>>>>>>> REPLACE")


def test_empty_search_is_rejected() -> None:
    with pytest.raises(CodeGenError, match="empty SEARCH"):
        apply_edit_blocks(ZONE, "<<<<<<< SEARCH\n\n=======\n    k = 5\n>>>>>>> REPLACE")
