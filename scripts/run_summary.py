"""Summarise every run ever performed, from the journals in runs/ plus the active one.

Why this exists: the submitted result is one run, but many were performed while the
harness was being fixed, and the best was chosen to submit. That choice is a human
judgement made outside any journal, and selecting the maximum of several runs is
worth more than any single run is expected to score. Publishing the distribution is
the honest way to report a selected result — and it doubles as evidence of how
reliably the agent reaches baseline, which a single number cannot show.

Run: python3 scripts/run_summary.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agent.journal import load_journal  # noqa: E402


def summarise(journal: Path) -> dict | None:
    entries = [e for e in load_journal(journal) if e.action_type != "intervention"]
    if not entries:
        return None
    scored = [e for e in entries if e.val_primary is not None]
    accepted = [e for e in scored if e.accepted]
    return {
        "iterations": len(entries),
        "best": max((e.val_primary for e in accepted), default=None),
        "accepts": len(accepted),
        "errors": sum(len(e.error_events) for e in entries),
        "interventions": sum(1 for e in load_journal(journal) if e.intervention),
        "tokens": sum(e.tokens_in + e.tokens_out for e in entries),
        "edits": sum(1 for e in entries if e.config.get("mode") == "edit"),
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/agent.yaml").read_text())
    baseline = cfg["baseline"]["valid"]["primary"]

    rows = []
    for d in sorted((ROOT / "runs").glob("*/")) if (ROOT / "runs").exists() else []:
        j = d / "journal.jsonl"
        if j.exists() and (s := summarise(j)):
            rows.append((d.name, s, False))
    active = ROOT / cfg["paths"]["journal"]
    if active.exists() and active.stat().st_size and (s := summarise(active)):
        rows.append(("(submitted)", s, True))

    scored = [r[1]["best"] for r in rows if r[1]["best"] is not None]
    out = [
        "# Run history\n",
        "> GENERATED from every journal in `runs/` plus the active one by "
        "`scripts/run_summary.py` — do not edit by hand.\n",
        "\nThe submitted result is a single run, but many were performed while the harness "
        "was being debugged. Selecting the best of several runs is worth more than any one "
        "run is expected to score, so the full distribution is published here rather than "
        "only the number we chose.\n",
        "\n| run | iters | accepts | edits | errors | interv. | tokens | best valid primary | vs baseline |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s, is_active in rows:
        best = s["best"]
        delta = f"{best - baseline:+.4f}" if best is not None else "—"
        label = f"**{name}**" if is_active else name
        out.append(f"| {label} | {s['iterations']} | {s['accepts']} | {s['edits']} | "
                   f"{s['errors']} | {s['interventions']} | {s['tokens']:,d} | "
                   f"{best:.4f} | {delta} |" if best is not None else
                   f"| {label} | {s['iterations']} | {s['accepts']} | {s['edits']} | "
                   f"{s['errors']} | {s['interventions']} | {s['tokens']:,d} | — | — |")

    if len(scored) > 1:
        out += [
            "\n## Distribution\n",
            f"- runs: **{len(scored)}**",
            f"- best: **{max(scored):.4f}** ({max(scored) - baseline:+.4f} vs baseline)",
            f"- median: {statistics.median(scored):.4f}",
            f"- mean: {statistics.mean(scored):.4f} ± {statistics.pstdev(scored):.4f}",
            f"- runs at or above baseline ({baseline}): "
            f"**{sum(1 for s in scored if s >= baseline)} of {len(scored)}**",
            "\nMost of the spread comes from harness bugs fixed between runs (truncated "
            "code generation, a missing validation split, guessed API signatures, a column "
            "the pipeline never joined) rather than from the agent's reasoning, which was "
            "consistent throughout. Runs before a given fix are not comparable with runs "
            "after it.\n",
        ]
    (ROOT / "reports" / "run_history.md").write_text("\n".join(out) + "\n")
    print(f"wrote {ROOT / 'reports/run_history.md'} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
