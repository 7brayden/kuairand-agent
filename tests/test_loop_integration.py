"""End-to-end: the real loop, real data, real official scorer — only the provider faked.

The unit tests cover parsing and fallbacks in isolation. This one runs
``AgentRun.run_iteration`` for real: workspace git, subprocess execution with a hard
timeout, the official evaluator, the critic, and the journal append. The only fake is
the LLM response, so an integration bug between those pieces cannot hide.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent.codegen import LLMCodeGenerator
from agent.journal import load_journal
from agent.loop import AgentRun
from agent.policy import LLMPolicy
from agent.state import RunState

from tests.test_llm_agent import FakeClient

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data/raw/KuaiRand-Pure/data"
pytestmark = pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not downloaded")

POP_ZONE = '''```python
def fit_predict(train, valid, target, checkpoint_dir):
    """Smoothed item popularity."""
    import numpy as np
    g = train.groupby("video_id")["long_view"].agg(["sum", "count"])
    gmean = float(train["long_view"].mean())
    rate = (g["sum"] + 20.0 * gmean) / (g["count"] + 20.0)
    return target["video_id"].map(rate).fillna(gmean).to_numpy(dtype=float)
```'''

CONSTANT_ZONE = '''```python
def fit_predict(train, valid, target, checkpoint_dir):
    """Constant score — within-user ranking makes this exactly as good as random."""
    import numpy as np
    return np.zeros(len(target), dtype=float)
```'''


def _run(tmp_path: Path, responses: list) -> AgentRun:
    """An AgentRun rooted in tmp_path, wired to a scripted client."""
    agent_cfg = yaml.safe_load((ROOT / "configs/agent.yaml").read_text())
    data_cfg = yaml.safe_load((ROOT / "configs/data.yaml").read_text())
    agent_cfg["policy"]["kind"] = "llm"

    for sub in ("logs/diffs", "checkpoints", "pipeline"):
        (tmp_path / sub).mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs/journal.jsonl").touch()
    # configs point at repo-relative paths; template and data stay in the real repo
    agent_cfg["paths"]["template"] = str(ROOT / "pipeline/template")
    data_cfg["paths"]["raw"] = str(DATA_DIR)

    run = AgentRun(agent_cfg, data_cfg, tmp_path, timeout_override=120)
    client = FakeClient(responses)
    run.client = client
    run.policy = LLMPolicy(client, agent_cfg["policy"]["actions"])
    run.generator = LLMCodeGenerator(client)
    run.workspace.init_if_needed(Path(agent_cfg["paths"]["template"]))
    return run


def test_accepted_iteration_produces_every_deliverable(tmp_path: Path) -> None:
    run = _run(tmp_path, [
        '```json\n{"action":"model","hypothesis":"smoothed item popularity gives a '
        'non-trivial floor above random","backtrack_to":null}\n```',
        POP_ZONE,
    ])
    state = RunState()
    entry = run.run_iteration(state, [])

    assert entry.accepted, entry.config.get("verdict")
    assert entry.error_events == []
    # matches the organisers' published item_popularity valid primary
    assert entry.val_primary == pytest.approx(0.5807, abs=0.002)
    assert entry.tokens_in > 0 and entry.tokens_out > 0      # usage actually accounted
    assert entry.commit_sha and entry.diff_path and entry.checkpoint_path
    assert (tmp_path / entry.diff_path).exists()
    assert (tmp_path / entry.checkpoint_path / "predictions.csv").exists()
    assert "popularity" in entry.hypothesis
    assert entry.stdout_tail and "rows" in entry.stdout_tail   # pipeline output captured
    assert not run.workspace.is_dirty()


def test_rejected_iteration_leaves_no_trace_but_is_journaled(tmp_path: Path) -> None:
    run = _run(tmp_path, [
        '```json\n{"action":"model","hypothesis":"popularity floor"}\n```', POP_ZONE,
        '```json\n{"action":"model","hypothesis":"constant scores as a control"}\n```',
        CONSTANT_ZONE,
    ])
    state = RunState()
    history = []
    for _ in range(2):
        e = run.run_iteration(state, history)
        history.append(e)
        state.record_iteration(
            __import__("agent.state", fromlist=["Node"]).Node(
                e.node_id, e.parent_id, e.commit_sha, e.val_gauc, e.val_ndcg5, e.accepted),
            tokens_in=e.tokens_in, tokens_out=e.tokens_out,
            gpu_seconds=e.gpu_seconds, wall_seconds=e.wall_seconds)

    good, ctrl = history
    assert good.accepted and not ctrl.accepted
    assert ctrl.val_primary is not None and ctrl.val_primary < good.val_primary
    assert ctrl.commit_sha is None and ctrl.checkpoint_path is None
    assert not run.workspace.is_dirty()          # reverted cleanly
    assert state.best_node().node_id == good.node_id


def test_provider_outage_mid_run_is_survived(tmp_path: Path) -> None:
    """The whole point of Robustness: an outage produces a journaled fallback, not a crash."""
    from agent.llm import LLMError
    run = _run(tmp_path, [LLMError("provider failed after 3 attempts: 529 overloaded")])
    entry = run.run_iteration(RunState(), [])

    kinds = {e.error_type for e in entry.error_events}
    assert "llm_api_error" in kinds
    assert all(e.recovered for e in entry.error_events)
    assert not entry.accepted
    assert "FALLBACK" in entry.hypothesis        # honest about who wrote it
    assert not run.workspace.is_dirty()


def test_journal_round_trips_through_disk(tmp_path: Path) -> None:
    from agent.journal import append_entry
    run = _run(tmp_path, [
        '```json\n{"action":"model","hypothesis":"popularity floor"}\n```', POP_ZONE])
    entry = run.run_iteration(RunState(), [])
    append_entry(run.journal_path, entry)

    (loaded,) = load_journal(run.journal_path)
    assert loaded == entry
    rebuilt = RunState.from_journal(run.journal_path)
    assert rebuilt.best_node().node_id == entry.node_id
    assert rebuilt.tokens_total == entry.tokens_in + entry.tokens_out
    json.loads((run.journal_path).read_text().strip())   # one valid JSON object per line
