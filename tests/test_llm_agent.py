"""Milestone 1: the LLM policy and code generator, driven by a fake client.

No network and no API key: a scripted client returns canned responses, so the parsing,
validation, repair, and — most importantly — the failure/fallback paths are all
deterministic and run in CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.codegen import LLMCodeGenerator, CodeGenError
from agent.journal import ErrorEvent, JournalEntry
from agent.llm import LLMClient, LLMError, extract_json_block
from agent.policy import ActionProposal, LLMPolicy
from agent.state import Node, RunState

PROMPTS = Path(__file__).resolve().parent.parent / "agent/prompts"
TEMPLATE = (Path(__file__).resolve().parent.parent / "pipeline/template/main.py").read_text()


class FakeClient(LLMClient):
    """LLMClient with the provider replaced by a scripted list of responses."""

    def __init__(self, responses: list, **kw):
        super().__init__(prompts_dir=PROMPTS, **kw)
        self.responses = list(responses)
        self.prompts_seen: list[str] = []
        self.efforts_seen: list = []

    def complete(self, prompt_name, variables, system=None, effort=None):
        self.efforts_seen.append(effort)
        self.prompts_seen.append(self.render(prompt_name, variables))
        if not self.responses:
            raise LLMError("script exhausted")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        self.tokens_in += 100
        self.tokens_out += 20
        from agent.llm import LLMResponse
        return LLMResponse(nxt, 100, 20, prompt_name, self.model)


def entry(iteration: int, *, accepted=False, gauc=None, ndcg5=None, errors=None):
    return JournalEntry(
        node_id=f"n{iteration}", parent_id=None, iteration=iteration,
        timestamp="2026-08-30T00:00:00+00:00", action_type="model",
        hypothesis=f"hypothesis {iteration}", config={}, diff_path=None,
        commit_sha="c" * 8 if accepted else None, checkpoint_path=None,
        val_gauc=gauc, val_ndcg5=ndcg5, wall_seconds=1.0, gpu_seconds=0.0,
        tokens_in=0, tokens_out=0, error_events=errors or [], accepted=accepted)


# ------------------------------- policy ---------------------------------------------

def test_policy_parses_a_well_formed_proposal() -> None:
    client = FakeClient(['```json\n{"action":"model","hypothesis":"lambdarank aligns '
                         'the objective with nDCG","backtrack_to":null}\n```'])
    p = LLMPolicy(client, ["model", "tune", "debug"])
    prop = p.propose(RunState(), [])
    assert prop.action_type == "model" and "lambdarank" in prop.hypothesis
    assert p.pending_errors == []


def test_policy_accepts_unfenced_json() -> None:
    """Models drop the fence sometimes; recovering beats wasting an iteration."""
    client = FakeClient(['Sure! {"action":"tune","hypothesis":"lower the lr"}'])
    prop = LLMPolicy(client, ["model", "tune"]).propose(RunState(), [])
    assert prop.action_type == "tune"


def test_provider_outage_falls_back_and_never_raises() -> None:
    client = FakeClient([LLMError("provider failed after 3 attempts: 529")])
    p = LLMPolicy(client, ["model", "tune", "debug"])
    prop = p.propose(RunState(), [])
    assert prop.action_type == "tune"                    # cheapest safe action
    assert "FALLBACK" in prop.hypothesis
    assert p.pending_errors[0].error_type == "llm_api_error"
    assert p.pending_errors[0].recovery == "reroute" and p.pending_errors[0].recovered


def test_fallback_prefers_debug_after_a_failure() -> None:
    client = FakeClient([LLMError("provider failed after 3 attempts")])
    history = [entry(0, errors=[ErrorEvent("timeout", "killed", "revert", True)])]
    prop = LLMPolicy(client, ["model", "tune", "debug"]).propose(RunState(), history)
    assert prop.action_type == "debug"


@pytest.mark.parametrize("bad", [
    "no json here at all",
    '```json\n{"action":"delete_everything","hypothesis":"x"}\n```',   # not an allowed action
    '```json\n{"action":"model","hypothesis":"   "}\n```',             # empty hypothesis
    '```json\n{"action":"model",,,}\n```',                             # malformed
])
def test_bad_policy_output_is_journaled_not_raised(bad: str) -> None:
    p = LLMPolicy(FakeClient([bad]), ["model", "tune", "debug"])
    prop = p.propose(RunState(), [])
    assert "FALLBACK" in prop.hypothesis
    assert p.pending_errors[0].error_type == "bad_llm_output"


def test_backtrack_only_targets_accepted_nodes() -> None:
    state = RunState()
    state.record_iteration(Node("good", None, "sha1", 0.66, 0.53, True),
                           tokens_in=0, tokens_out=0, gpu_seconds=0, wall_seconds=0)
    state.record_iteration(Node("bad", None, None, 0.60, 0.50, False),
                           tokens_in=0, tokens_out=0, gpu_seconds=0, wall_seconds=0)

    def propose_backtrack(target):
        client = FakeClient([f'```json\n{{"action":"model","hypothesis":"branch",'
                             f'"backtrack_to":"{target}"}}\n```'])
        return LLMPolicy(client, ["model"]).propose(state, [])

    assert propose_backtrack("good").parent_id == "good"
    assert propose_backtrack("bad").parent_id is None       # never committed
    assert propose_backtrack("nonexistent").parent_id is None


def test_policy_prompt_carries_the_contracts_and_history() -> None:
    client = FakeClient(['```json\n{"action":"tune","hypothesis":"h"}\n```'])
    p = LLMPolicy(client, ["model", "tune"])
    p.current_zone = "def fit_predict(train, valid, target, checkpoint_dir): ..."
    p.propose(RunState(), [entry(0, accepted=True, gauc=0.66, ndcg5=0.53)])
    prompt = client.prompts_seen[0]
    for must in ["Within-user", "long_view", "0.6016", "0.002", "hypothesis 0",
                 "fit_predict"]:
        assert must in prompt, f"policy prompt is missing {must!r}"


# ------------------------------ code generation --------------------------------------

GOOD_ZONE = '''```python
def fit_predict(train, valid, target, checkpoint_dir):
    """Item popularity."""
    import numpy as np
    rate = train.groupby("video_id")["long_view"].mean()
    return target["video_id"].map(rate).fillna(0.0).to_numpy(dtype=float)
```'''


def test_generator_produces_lintable_source() -> None:
    client = FakeClient([GOOD_ZONE])
    gen = LLMCodeGenerator(client)
    out = gen.generate(ActionProposal("model", "popularity floor"), TEMPLATE, {})
    assert "groupby" in out.source
    assert "def write_predictions(" in out.source      # I/O contract preserved
    assert out.config["repairs"] == 0


def test_generator_repairs_a_rejected_first_attempt() -> None:
    broken = '```python\ndef fit_predict(train, valid, target, checkpoint_dir):\n    x = (\n```'
    client = FakeClient([broken, GOOD_ZONE])
    out = LLMCodeGenerator(client).generate(
        ActionProposal("model", "h"), TEMPLATE, {})
    assert out.config["repairs"] == 1
    assert "syntax error" in client.prompts_seen[1]     # told exactly what was wrong


def test_generator_gives_up_after_max_repairs() -> None:
    broken = '```python\ndef fit_predict(train, valid, target, checkpoint_dir):\n    x = (\n```'
    with pytest.raises(CodeGenError):
        LLMCodeGenerator(FakeClient([broken, broken])).generate(
            ActionProposal("model", "h"), TEMPLATE, {})


def test_generator_rejects_test_split_access_before_execution() -> None:
    leak = ('```python\ndef fit_predict(train, valid, target, checkpoint_dir):\n'
            '    t = split_of(train, "test")\n    return t\n```')
    with pytest.raises(CodeGenError):
        LLMCodeGenerator(FakeClient([leak, leak])).generate(
            ActionProposal("model", "h"), TEMPLATE, {})


def test_generator_prompt_includes_action_instructions() -> None:
    client = FakeClient([GOOD_ZONE])
    LLMCodeGenerator(client).generate(
        ActionProposal("debug", "fix it"), TEMPLATE,
        {"extra_context": "TRACEBACK: boom"})
    prompt = client.prompts_seen[0]
    assert "Fix the failure below" in prompt        # action_debug_v1.md
    assert "TRACEBACK: boom" in prompt
    assert "row order" in prompt.lower()


def test_every_action_has_a_prompt_file() -> None:
    from agent.journal import POLICY_ACTIONS
    for action in POLICY_ACTIONS:
        assert (PROMPTS / f"action_{action}_v1.md").exists(), action


# --------------------------------- parsing -------------------------------------------

def test_extract_json_block_variants() -> None:
    assert extract_json_block('```json\n{"a":1}\n```') == {"a": 1}
    assert extract_json_block('text {"a":2} more') == {"a": 2}
    with pytest.raises(LLMError):
        extract_json_block("[1,2,3]")          # array, not an object
    with pytest.raises(LLMError):
        extract_json_block("nothing here")


def test_effort_is_higher_for_reasoning_than_for_codegen() -> None:
    """Hypothesis quality is the 20% Innovation score; writing the code is mechanical.
    Thinking tokens count toward the scored total, so the split is deliberate."""
    pc = FakeClient(['```json\n{"action":"model","hypothesis":"h"}\n```'])
    LLMPolicy(pc, ["model"]).propose(RunState(), [])
    gc = FakeClient([GOOD_ZONE])
    LLMCodeGenerator(gc).generate(ActionProposal("model", "h"), TEMPLATE, {})
    assert pc.efforts_seen == ["high"]
    assert gc.efforts_seen == ["medium"]


# ------------------------- stop_reason handling (regression) --------------------------
# A truncated response and a refusal both return HTTP 200. Before this check, truncation
# surfaced downstream as "no fenced python block", which sent the debug action hunting a
# code bug that did not exist — it cost a real iteration on the first live run.

class _FakeMessage:
    def __init__(self, stop_reason, text="ok", stop_details=None):
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.content = [type("B", (), {"type": "text", "text": text})()]
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 16000})()


class _FakeProvider:
    def __init__(self, message):
        self.messages = type("M", (), {"create": lambda _self, **kw: message})()


def _client_returning(message) -> LLMClient:
    c = LLMClient(prompts_dir=PROMPTS, max_retries=1)
    c._client = _FakeProvider(message)
    return c


def test_truncation_at_max_tokens_is_named_precisely() -> None:
    c = _client_returning(_FakeMessage("max_tokens", text="def fit_predict(  # cut off"))
    with pytest.raises(LLMError, match="truncated at max_tokens"):
        c.complete("policy_v1", {"task_brief": "", "journal_summary": "",
                                 "resource_summary": "", "current_zone": "", "actions": "model"})


def test_refusal_is_reported_as_a_refusal() -> None:
    details = type("D", (), {"category": "cyber"})()
    c = _client_returning(_FakeMessage("refusal", stop_details=details))
    with pytest.raises(LLMError, match="declined"):
        c.complete("policy_v1", {"task_brief": "", "journal_summary": "",
                                 "resource_summary": "", "current_zone": "", "actions": "model"})


def test_usage_is_counted_even_on_a_truncated_call() -> None:
    """Truncated calls still cost tokens; the scored total must include them."""
    c = _client_returning(_FakeMessage("max_tokens"))
    with pytest.raises(LLMError):
        c.complete("policy_v1", {"task_brief": "", "journal_summary": "",
                                 "resource_summary": "", "current_zone": "", "actions": "model"})
    assert c.take_iteration_usage() == (100, 16000)


# --------------------- training-log feedback (the 0.4983 lesson) ----------------------

def test_recent_pipeline_output_is_replayed_to_the_policy() -> None:
    """Without the training log the agent cannot tell a wrong hypothesis from an
    undertrained model — in the first live run that cost it two sound ideas."""
    from agent import memory
    history = [entry(i, gauc=0.6, ndcg5=0.5) for i in range(5)]
    for i, e in enumerate(history):
        e.stdout_tail = f"epoch {i} loss 0.4{i}\nstopped early"
    text = memory.summarize_journal(history)
    assert "Pipeline output from recent iterations" in text
    assert "epoch 4 loss 0.44" in text          # most recent replayed
    assert "epoch 0 loss 0.40" not in text      # older elided to control tokens
    assert "undertrained" in text               # the framing the agent needs


def test_no_output_section_when_nothing_was_printed() -> None:
    from agent import memory
    assert "Pipeline output" not in memory.summarize_journal(
        [entry(0, gauc=0.6, ndcg5=0.5)])


def test_codegen_prompt_demands_training_diagnostics() -> None:
    client = FakeClient([GOOD_ZONE])
    LLMCodeGenerator(client).generate(ActionProposal("model", "h"), TEMPLATE, {})
    prompt = client.prompts_seen[0]
    assert "Print your training diagnostics" in prompt
    assert "max_epochs=40" in prompt      # the measured training budget from eval/official/


# ------------------- generator chooses edit vs rewrite (end to end) -------------------

EDIT_RESPONSE = """I'll tune the learning rate only.

<<<<<<< SEARCH
    rate = train.groupby("video_id")["long_view"].mean()
=======
    rate = train.groupby("video_id")["long_view"].mean() * 1.0
>>>>>>> REPLACE
"""

WORKING_ZONE = '''def fit_predict(train, valid, target, checkpoint_dir):
    """Item popularity."""
    import numpy as np
    rate = train.groupby("video_id")["long_view"].mean()
    return target["video_id"].map(rate).fillna(0.0).to_numpy(dtype=float)
'''


def _source_with(zone: str) -> str:
    from agent.codegen import replace_agent_zone
    return replace_agent_zone(TEMPLATE, zone)


def test_generator_applies_an_edit_without_touching_the_rest() -> None:
    client = FakeClient([EDIT_RESPONSE])
    out = LLMCodeGenerator(client).generate(
        ActionProposal("tune", "raise the smoothing"), _source_with(WORKING_ZONE), {})
    assert out.config["mode"] == "edit" and out.config["edits"] == 1
    assert '* 1.0' in out.source
    assert '"""Item popularity."""' in out.source     # surrounding code preserved verbatim
    assert "def write_predictions(" in out.source     # I/O contract still intact


def test_generator_still_accepts_a_full_rewrite() -> None:
    out = LLMCodeGenerator(FakeClient([GOOD_ZONE])).generate(
        ActionProposal("model", "new approach"), TEMPLATE, {})
    assert out.config["mode"] == "rewrite" and out.config["edits"] == 0


def test_a_bad_edit_is_repaired_with_the_specific_reason() -> None:
    bad = "<<<<<<< SEARCH\n    nonexistent = 1\n=======\n    x = 2\n>>>>>>> REPLACE"
    client = FakeClient([bad, EDIT_RESPONSE])
    out = LLMCodeGenerator(client).generate(
        ActionProposal("tune", "h"), _source_with(WORKING_ZONE), {})
    assert out.config["repairs"] == 1
    assert "SEARCH block not found" in client.prompts_seen[1]


def test_codegen_prompt_teaches_when_to_edit() -> None:
    client = FakeClient([EDIT_RESPONSE])
    LLMCodeGenerator(client).generate(
        ActionProposal("tune", "h"), _source_with(WORKING_ZONE), {})
    prompt = client.prompts_seen[0]
    assert "Prefer editing" in prompt
    assert "SEARCH" in prompt and "REPLACE" in prompt
    assert "refinement of working code" in prompt      # from action_tune_v1.md
