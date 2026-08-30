"""Generate every human-facing report from logs/journal.jsonl — never by hand.

The journal is the single source of truth; everything in reports/ is a derived view:

  reports/results_table.md   validation-best scores + absolute delta vs the FM baseline
  reports/resource_usage.md  total tokens in/out and GPU-hours to convergence
  reports/interventions.md   every intervention-flagged entry + the summary count
  reports/run_report.html    static self-contained run report (replaces W&B/MLflow)

Run: python3 scripts/generate_reports.py
"""

from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agent.journal import JournalEntry, load_journal  # noqa: E402

GENERATED = ("> GENERATED from `logs/journal.jsonl` by `scripts/generate_reports.py` "
             "— do not edit by hand.\n")


def _fmt(v, nd=4):
    return "—" if v is None else f"{v:.{nd}f}"


def results_table(entries: list[JournalEntry], cfg: dict) -> str:
    scored = [e for e in entries if e.val_primary is not None]
    base = cfg["baseline"]["valid"]
    out = ["# Results\n", GENERATED]
    if not scored:
        out.append("\nNo scored iteration yet.\n")
        return "\n".join(out)

    best = max(scored, key=lambda e: e.val_primary)
    ceil = cfg["baseline"]["oracle_ceiling"]["valid"]["primary"]
    out += [
        "\n## Validation-best checkpoint\n",
        f"Node `{best.node_id}` (iteration {best.iteration}, action `{best.action_type}`)\n",
        "\n| | GAUC | nDCG@5 | primary |",
        "|---|---|---|---|",
        f"| FM baseline (valid) | {base['GAUC']} | {base['nDCG@5']} | {base['primary']} |",
        f"| **agent (valid-best)** | **{_fmt(best.val_gauc)}** | "
        f"**{_fmt(best.val_ndcg5)}** | **{_fmt(best.val_primary)}** |",
        f"| **absolute delta** | **{best.val_gauc - base['GAUC']:+.4f}** | "
        f"**{best.val_ndcg5 - base['nDCG@5']:+.4f}** | "
        f"**{best.val_primary - base['primary']:+.4f}** |",
        f"| oracle ceiling | 1.0000 | — | {ceil} |",
        "",
        f"Headroom used: {(best.val_primary - base['primary']) / (ceil - base['primary']) * 100:+.1f}%"
        f" of the {ceil - base['primary']:.4f} available above baseline.\n",
        "\n## Every iteration\n",
        "| it | node | action | GAUC | nDCG@5 | primary | accepted | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        errs = ", ".join(x.error_type for x in e.error_events) or "—"
        out.append(f"| {e.iteration} | `{e.node_id}` | {e.action_type} | "
                   f"{_fmt(e.val_gauc)} | {_fmt(e.val_ndcg5)} | {_fmt(e.val_primary)} | "
                   f"{'yes' if e.accepted else 'no'} | {errs} |")
    final = ROOT / "reports" / "final_result.json"
    if final.exists():
        out.append(f"\n## Hidden test (scored once)\n\nSee `{final.name}`.\n")
    return "\n".join(out) + "\n"


def resource_usage(entries: list[JournalEntry]) -> str:
    ti = sum(e.tokens_in for e in entries)
    to = sum(e.tokens_out for e in entries)
    gpu = sum(e.gpu_seconds for e in entries)
    wall = sum(e.wall_seconds for e in entries)
    return "\n".join([
        "# Resource usage\n", GENERATED,
        "\n| | |", "|---|---|",
        f"| iterations attempted | {len(entries)} |",
        f"| LLM tokens in | {ti:,d} |",
        f"| LLM tokens out | {to:,d} |",
        f"| **LLM tokens total** | **{ti + to:,d}** |",
        f"| **GPU-hours** | **{gpu / 3600:.4f}** |",
        f"| wall-clock hours | {wall / 3600:.4f} |",
        "\nToken counts come from `agent/llm.py`, the single provider call site, and "
        "include failed calls, retries, and calls made inside rejected iterations.\n",
        "GPU-hours are 0 by design: the default inner model is LightGBM lambdarank on CPU.\n",
    ]) + "\n"


def interventions(entries: list[JournalEntry]) -> str:
    flagged = [e for e in entries if e.intervention]
    out = ["# Manual interventions\n", GENERATED,
           f"\n## Total: {len(flagged)}\n"]
    if not flagged:
        out.append("\nNo manual interventions were recorded across "
                   f"{len(entries)} attempted iterations.\n")
    else:
        out += ["\n| it | node | note |", "|---|---|---|"]
        out += [f"| {e.iteration} | `{e.node_id}` | {e.intervention_note or '—'} |"
                for e in flagged]
    out.append("\nCounted conservatively: any human action that altered a run — editing "
               "code, restarting, changing config mid-run — is logged as an intervention.\n")
    return "\n".join(out) + "\n"


def html_report(entries: list[JournalEntry], cfg: dict) -> str:
    base = cfg["baseline"]["valid"]["primary"]
    rows = []
    for e in entries:
        errs = "<br>".join(f"<code>{html.escape(x.error_type)}</code> &rarr; "
                           f"{html.escape(x.recovery)}" for x in e.error_events) or "—"
        cls = "acc" if e.accepted else ("err" if e.error_events else "rej")
        rows.append(
            f"<tr class='{cls}'><td>{e.iteration}</td><td><code>{e.node_id}</code></td>"
            f"<td>{html.escape(e.action_type)}</td>"
            f"<td class='hyp'>{html.escape(e.hypothesis)}</td>"
            f"<td>{_fmt(e.val_gauc)}</td><td>{_fmt(e.val_ndcg5)}</td>"
            f"<td><b>{_fmt(e.val_primary)}</b></td>"
            f"<td>{'✓' if e.accepted else '✗'}</td><td>{errs}</td>"
            f"<td>{e.wall_seconds:.1f}s</td>"
            f"<td>{e.tokens_in + e.tokens_out:,d}</td></tr>")
    scored = [e for e in entries if e.val_primary is not None]
    best = max(scored, key=lambda e: e.val_primary) if scored else None
    summary = (f"validation-best <code>{best.node_id}</code> primary "
               f"<b>{best.val_primary:.4f}</b> (baseline {base}, "
               f"delta {best.val_primary - base:+.4f})" if best else "no scored iteration")
    return f"""<!doctype html>
<meta charset="utf-8"><title>kuairand-agent run report</title>
<style>
 body{{font:14px/1.5 -apple-system,system-ui,sans-serif;margin:2rem auto;max-width:1200px;padding:0 1rem}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top}}
 th{{background:#f4f4f6}} tr.acc{{background:#eefbf0}} tr.rej{{background:#fafafa}}
 tr.err{{background:#fdf0ee}} .hyp{{max-width:420px;font-size:12px;color:#333}}
 code{{font:12px ui-monospace,Menlo,monospace}}
 .sum{{padding:10px 14px;background:#f4f4f6;border-left:4px solid #888;margin:1rem 0}}
</style>
<h1>kuairand-agent — run report</h1>
<div class="sum">{summary}<br>
 {len(entries)} attempted iterations &middot;
 {sum(e.tokens_in + e.tokens_out for e in entries):,d} tokens &middot;
 {sum(e.gpu_seconds for e in entries) / 3600:.4f} GPU-hours &middot;
 {sum(1 for e in entries if e.intervention)} manual interventions</div>
<p>Generated from <code>logs/journal.jsonl</code>. Every row is one attempted iteration:
its hypothesis, the metrics it produced, and any failure with the recovery taken.</p>
<table><thead><tr><th>it</th><th>node</th><th>action</th><th>hypothesis</th>
<th>GAUC</th><th>nDCG@5</th><th>primary</th><th>acc</th><th>errors &rarr; recovery</th>
<th>wall</th><th>tokens</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", type=Path, default=ROOT / "logs/journal.jsonl")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--agent-config", type=Path, default=ROOT / "configs/agent.yaml")
    args = ap.parse_args()

    entries = load_journal(args.journal) if args.journal.exists() else []
    cfg = yaml.safe_load(args.agent_config.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name, text in (("results_table.md", results_table(entries, cfg)),
                       ("resource_usage.md", resource_usage(entries)),
                       ("interventions.md", interventions(entries)),
                       ("run_report.html", html_report(entries, cfg))):
        (args.out_dir / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.out_dir / name}")


if __name__ == "__main__":
    main()
