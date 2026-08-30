"""Record a manual intervention in the journal. Run this EVERY time you touch a run.

Autonomy is 20% of the score and is measured primarily by the count of manual
interventions. That count is generated from the journal, so an intervention that is not
logged is a number the submission gets wrong in its own favour — which is exactly the
kind of thing this whole harness exists to make impossible.

Log it whenever a human changes what a run does: editing generated code by hand, killing
and restarting a run, changing a config mid-run, fixing the environment, unsticking the
agent. When in doubt, log it — over-counting costs a little Autonomy score, under-counting
misrepresents the system.

An intervention entry is a first-class journal entry, but it is NOT an attempted
iteration: it does not consume convergence patience and it holds no metrics.

Usage:
    python3 scripts/log_intervention.py "installed lightgbm; agent's code needed it"
    python3 scripts/log_intervention.py --tokens-in 0 "restarted after a laptop reboot"
"""

from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agent.journal import INTERVENTION, JournalEntry, append_entry, load_journal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("note", help="what you did and why — this is read by the judges")
    ap.add_argument("--agent-config", type=Path, default=ROOT / "configs/agent.yaml")
    ap.add_argument("--tokens-in", type=int, default=0,
                    help="tokens spent outside the harness while intervening, if any")
    ap.add_argument("--tokens-out", type=int, default=0)
    args = ap.parse_args()

    if not args.note.strip():
        sys.exit("an intervention note is required — the count alone is not the deliverable")

    cfg = yaml.safe_load(args.agent_config.read_text())
    journal = ROOT / cfg["paths"]["journal"]
    prior = load_journal(journal) if journal.exists() and journal.stat().st_size else []

    entry = JournalEntry(
        node_id=uuid.uuid4().hex[:8],
        parent_id=prior[-1].node_id if prior else None,
        iteration=len(prior),
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        action_type=INTERVENTION,
        hypothesis=f"MANUAL INTERVENTION (not the agent's reasoning): {args.note.strip()}",
        config={"logged_by": "scripts/log_intervention.py"},
        diff_path=None, commit_sha=None, checkpoint_path=None,
        val_gauc=None, val_ndcg5=None,
        wall_seconds=0.0, gpu_seconds=0.0,
        tokens_in=args.tokens_in, tokens_out=args.tokens_out,
        error_events=[], accepted=False,
        intervention=True, intervention_note=args.note.strip(),
    )
    append_entry(journal, entry)

    total = sum(1 for e in load_journal(journal) if e.intervention)
    print(f"logged intervention {entry.node_id}: {args.note.strip()}")
    print(f"this run now has {total} manual intervention(s)")


if __name__ == "__main__":
    main()
