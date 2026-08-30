"""Tests for the journal schema — the interface everything else depends on."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent.journal import (
    ACTION_TYPES,
    SCHEMA_VERSION,
    ErrorEvent,
    JournalEntry,
    append_entry,
    load_journal,
)


def make_entry(**overrides) -> JournalEntry:
    base = dict(
        node_id="a1b2c3",
        parent_id=None,
        iteration=0,
        timestamp="2026-08-26T00:00:00+00:00",
        action_type="model",
        hypothesis="lambdarank baseline on raw interaction counts",
        config={"model": "lightgbm", "objective": "lambdarank"},
        diff_path="logs/diffs/a1b2c3.patch",
        commit_sha="deadbeef",
        checkpoint_path="checkpoints/a1b2c3",
        val_gauc=0.6674,
        val_ndcg5=0.5357,
        wall_seconds=12.5,
        gpu_seconds=0.0,
        tokens_in=1000,
        tokens_out=200,
        error_events=[
            ErrorEvent(
                error_type="timeout",
                message="killed after 1800s",
                recovery="debug_action",
                recovered=True,
            )
        ],
        accepted=True,
        intervention=False,
        intervention_note=None,
    )
    base.update(overrides)
    return JournalEntry(**base)


def test_roundtrip_through_jsonl(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    first = make_entry()
    second = make_entry(node_id="d4e5f6", parent_id="a1b2c3", iteration=1,
                        accepted=False, commit_sha=None, error_events=[])
    append_entry(journal, first)
    append_entry(journal, second)

    loaded = load_journal(journal)
    assert loaded == [first, second]
    assert loaded[0].error_events[0].recovery == "debug_action"


def test_entries_are_one_json_object_per_line(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    append_entry(journal, make_entry())
    lines = journal.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["schema_version"] == SCHEMA_VERSION
    assert parsed["node_id"] == "a1b2c3"


def test_invalid_action_type_rejected() -> None:
    with pytest.raises(ValueError):
        make_entry(action_type="refactor")


def test_failed_iteration_may_have_no_metrics(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    failed = make_entry(val_gauc=None, val_ndcg5=None, accepted=False,
                        commit_sha=None, checkpoint_path=None)
    append_entry(journal, failed)
    (loaded,) = load_journal(journal)
    assert loaded.val_gauc is None and loaded.val_primary is None and not loaded.accepted


def test_primary_is_derived_not_stored(tmp_path: Path) -> None:
    """primary must never be persisted: a stored copy could disagree with the metrics."""
    entry = make_entry()
    assert entry.val_primary == pytest.approx((0.6674 + 0.5357) / 2)
    assert "val_primary" not in entry.to_dict()


# ----------------------------- interventions (Autonomy, 20%) --------------------------

def test_intervention_is_a_valid_entry_type_but_not_a_policy_action() -> None:
    from agent.journal import INTERVENTION, POLICY_ACTIONS
    assert INTERVENTION in ACTION_TYPES          # the journal accepts it
    assert INTERVENTION not in POLICY_ACTIONS    # no policy can ever propose it


def test_intervention_entry_round_trips(tmp_path: Path) -> None:
    from agent.journal import INTERVENTION
    journal = tmp_path / "journal.jsonl"
    e = make_entry(action_type=INTERVENTION, intervention=True,
                   intervention_note="installed lightgbm by hand",
                   val_gauc=None, val_ndcg5=None, error_events=[])
    append_entry(journal, e)
    (loaded,) = load_journal(journal)
    assert loaded.intervention and loaded.intervention_note == "installed lightgbm by hand"


# ------------------------- stdout capture (schema v2) ---------------------------------

def test_stdout_tail_round_trips(tmp_path: Path) -> None:
    journal = tmp_path / "journal.jsonl"
    e = make_entry(stdout_tail="epoch 1 loss 0.51\nepoch 2 loss 0.48\nearly stop at 12")
    append_entry(journal, e)
    (loaded,) = load_journal(journal)
    assert "early stop at 12" in loaded.stdout_tail


def test_v1_entries_without_stdout_still_load(tmp_path: Path) -> None:
    """Schema v2 added stdout_tail; v1 journals must keep loading without migration."""
    journal = tmp_path / "journal.jsonl"
    d = make_entry().to_dict()
    del d["stdout_tail"]
    d["schema_version"] = 1
    journal.write_text(json.dumps(d) + "\n")
    (loaded,) = load_journal(journal)
    assert loaded.stdout_tail is None and loaded.schema_version == 1
