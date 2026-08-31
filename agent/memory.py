"""Journal -> prompt context. The agent's memory, and where it gets compressed.

The reflect step reads the journal to decide what to try next, so this module is
literally the agent's memory. Two forces pull against each other:

* Every past hypothesis and outcome is signal — dropping the wrong one makes the
  agent repeat an experiment it already ran.
* Total LLM tokens are 15% of the score, and the journal grows without bound.

So recent iterations are rendered in full and older ones are compressed to a single
line each, with an explicit "ruled out" section that survives compression — because
*what failed and why* is the part the policy most needs and the part a naive
truncation would lose first.

Everything here is derived from ``logs/journal.jsonl``. Nothing else stores it.
"""

from __future__ import annotations

from typing import Optional

from agent.journal import JournalEntry
from agent.state import RunState

#: Iterations rendered with their full hypothesis text. Older ones collapse to a row.
FULL_DETAIL_WINDOW = 8

#: Hypotheses are the agent's own prose and can run long; cap them in the recap.
HYPOTHESIS_CHARS = 220

#: Iterations whose pipeline stdout is replayed in full. Training logs are the most
#: useful thing in the journal for diagnosing a bad score, and also the bulkiest, so
#: only the most recent few are shown — tokens are 15% of the score.
STDOUT_WINDOW = 3


def _clip(text: str, limit: int = HYPOTHESIS_CHARS) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _outcome(entry: JournalEntry) -> str:
    if entry.error_events:
        return "/".join(e.error_type for e in entry.error_events)
    return "accept" if entry.accepted else "reject"


def summarize_journal(entries: list[JournalEntry],
                      full_detail_window: int = FULL_DETAIL_WINDOW) -> str:
    """Render the run history for the policy prompt."""
    if not entries:
        return ("No iterations yet. This is the first one: establish a working pipeline "
                "before attempting anything clever.")

    scored = [e for e in entries if e.val_primary is not None]
    best = max(scored, key=lambda e: e.val_primary) if scored else None
    head = [f"{len(entries)} iterations attempted."]
    if best:
        head.append(f"Best so far: primary {best.val_primary:.4f} at node `{best.node_id}` "
                    f"(iteration {best.iteration}, action `{best.action_type}`).")
    else:
        head.append("Nothing has produced a valid score yet.")

    lines = ["", "| it | action | primary | outcome | hypothesis |",
             "|---|---|---|---|---|"]
    cutoff = max(0, len(entries) - full_detail_window)
    for e in entries:
        primary = f"{e.val_primary:.4f}" if e.val_primary is not None else "—"
        hyp = _clip(e.hypothesis) if e.iteration >= cutoff else "(older — see above)"
        lines.append(f"| {e.iteration} | {e.action_type} | {primary} | "
                     f"{_outcome(e)} | {hyp} |")

    ruled_out = [e for e in entries if not e.accepted and e.val_primary is not None]
    if ruled_out:
        lines += ["", "## Already tried and rejected (do not repeat)", ""]
        for e in ruled_out:
            delta = (f"{e.val_primary - best.val_primary:+.4f} vs best"
                     if best else f"{e.val_primary:.4f}")
            lines.append(f"- `{e.action_type}` ({delta}): {_clip(e.hypothesis, 160)}")

    stuck = detect_stall(entries)
    if stuck:
        lines += ["", "## ⚠ You are stuck — change something structural", "", stuck, ""]

    recent_logs = [e for e in entries[-STDOUT_WINDOW:] if e.stdout_tail]
    if recent_logs:
        lines += ["", "## Pipeline output from recent iterations", "",
                  "Use this to tell a wrong *hypothesis* from a broken *implementation*: "
                  "a model whose loss was still falling when training stopped was "
                  "undertrained, and its score says nothing about the idea.", ""]
        for e in recent_logs:
            lines += [f"### iteration {e.iteration} (`{e.action_type}`, "
                      f"primary {e.val_primary:.4f})" if e.val_primary is not None
                      else f"### iteration {e.iteration} (`{e.action_type}`, no score)",
                      "```", e.stdout_tail.strip(), "```"]

    failures = [e for e in entries if e.error_events]
    if failures:
        lines += ["", "## Failures and how they were handled", ""]
        for e in failures:
            for ev in e.error_events:
                lines.append(f"- it{e.iteration} `{ev.error_type}` → recovery "
                             f"`{ev.recovery}`: {_clip(ev.message, 200)}")

    return "\n".join(head + lines)


#: How many identical-scoring or same-action iterations count as a stall.
STALL_WINDOW = 3


def detect_stall(entries: list[JournalEntry]) -> Optional[str]:
    """Describe a stall, if the recent history shows one.

    Two shapes, both observed live. A run once spent four consecutive `debug`
    iterations producing byte-identical scores of 0.5763 while the agent re-diagnosed
    the same problem each time — the loop was healthy, the search was not. Robustness is
    judged on never stalling, so the condition is named for the policy rather than left
    for it to notice.
    """
    scored = [e for e in entries if e.val_primary is not None]
    if len(scored) >= STALL_WINDOW:
        recent = [round(e.val_primary, 4) for e in scored[-STALL_WINDOW:]]
        if len(set(recent)) == 1:
            return (f"The last {STALL_WINDOW} scored iterations all produced primary "
                    f"{recent[0]:.4f} — byte-identical. Your changes are not reaching the "
                    f"scoring path at all. Stop refining this line; either verify what you "
                    f"changed is actually executed (print it), or switch approach.")

    if len(entries) >= STALL_WINDOW:
        actions = [e.action_type for e in entries[-STALL_WINDOW:]]
        if len(set(actions)) == 1:
            return (f"The last {STALL_WINDOW} iterations were all `{actions[0]}` and none "
                    f"was accepted. Repeating the same action is not working — pick a "
                    f"different one.")
    return None


def summarize_resources(state: RunState, budget_cfg: Optional[dict] = None) -> str:
    """Render the spend so far. The policy is told its own cost — cheaper hypotheses
    are worth something (Feasibility is 15% of the score)."""
    parts = [f"tokens used: {state.tokens_total:,d}",
             f"GPU-hours: {state.gpu_seconds / 3600:.3f}",
             f"wall-hours: {state.wall_seconds / 3600:.2f}",
             f"iterations: {state.iteration}"]
    if budget_cfg:
        if budget_cfg.get("max_tokens"):
            pct = 100.0 * state.tokens_total / float(budget_cfg["max_tokens"])
            parts.append(f"token budget used: {pct:.1f}%")
        if budget_cfg.get("max_gpu_hours") == 0.0:
            parts.append("GPU budget is ZERO — CPU-only; a torch model would exceed budget")
    return " | ".join(parts)


def error_context(entries: list[JournalEntry], limit: int = 1) -> str:
    """The most recent failures, formatted for the debug action's prompt."""
    failed = [e for e in entries if e.error_events][-limit:]
    if not failed:
        return "No recent failure."
    out = []
    for e in failed:
        for ev in e.error_events:
            out.append(f"iteration {e.iteration} — {ev.error_type} (recovery: {ev.recovery})\n"
                       f"hypothesis was: {_clip(e.hypothesis)}\n"
                       f"error:\n{ev.message}")
    return "\n\n".join(out)
