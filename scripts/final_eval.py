"""One-shot hidden-test evaluation — run EXACTLY ONCE, at convergence.

This is deliberately a separate entry point that the agent loop never imports. The
test labels sit in the same local CSVs as valid, so "scored once" is an honour-system
constraint; isolating it here (and refusing to run when the journal says a test score
already exists) is how we keep ourselves honest.

What it does:
  1. read the journal, pick the VALIDATION-BEST node (not the last, not the peak-looking);
  2. check out that node's commit in the workspace — the commit graph is the search tree,
     so the winning pipeline is recoverable exactly;
  3. run it once with --split test;
  4. validate + score with the vendored official evaluator;
  5. write submission/submission.csv and reports/final_result.json.

Usage:  python3 scripts/final_eval.py [--yes]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agent.executor import Workspace, run_pipeline  # noqa: E402
from agent.state import RunState  # noqa: E402
from eval import scorer  # noqa: E402
from submission.adapter import export_submission  # noqa: E402

RESULT_PATH = ROOT / "reports" / "final_result.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yes", action="store_true",
                    help="confirm the one-shot test evaluation")
    ap.add_argument("--agent-config", type=Path, default=ROOT / "configs/agent.yaml")
    ap.add_argument("--data-config", type=Path, default=ROOT / "configs/data.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.agent_config.read_text())
    data_cfg = yaml.safe_load(args.data_config.read_text())
    data_dir = str(ROOT / data_cfg["paths"]["raw"])

    if RESULT_PATH.exists():
        sys.exit(f"REFUSING: {RESULT_PATH} already exists — test was already scored once. "
                 f"Delete it deliberately if you truly mean to re-score.")

    state = RunState.from_journal(ROOT / cfg["paths"]["journal"])
    best = state.best_node()
    if best is None:
        sys.exit("no scored iteration in the journal; nothing to submit")
    if best.commit_sha is None:
        sys.exit(f"validation-best node {best.node_id} has no commit — cannot reproduce it")

    print(f"validation-best: node {best.node_id} primary {best.selection_score:.4f} "
          f"(commit {best.commit_sha[:8]})")
    if not args.yes:
        sys.exit("re-run with --yes to spend the single hidden-test evaluation")

    ws = Workspace(ROOT / cfg["paths"]["workspace"])
    ws.branch_from(best.commit_sha, f"final-{best.node_id}")

    out_dir = ROOT / "logs" / "_final" / best.node_id
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_pipeline([cfg["executor"].get("python", "python3"), "main.py",
                        "--data-dir", data_dir, "--split", "test",
                        "--out-dir", str(out_dir)],
                       cwd=ws.path,
                       timeout_seconds=float(cfg["executor"]["timeout_seconds"]))
    if not res.ok:
        sys.exit(f"final pipeline run failed (rc={res.returncode}, "
                 f"timed_out={res.timed_out}):\n{res.stderr[-2000:]}")

    preds = out_dir / "predictions.csv"
    # allow_test=True is passed here and NOWHERE else in the codebase.
    metrics = scorer.score_submission(preds, data_dir, split="test", allow_test=True)
    export_submission(preds, ROOT / "submission" / "submission.csv", data_dir, "test")

    base = cfg["baseline"]["test"]
    result = {
        "node_id": best.node_id,
        "commit_sha": best.commit_sha,
        "validation_primary": best.selection_score,
        "test": {k: metrics[k] for k in ("GAUC", "nDCG@5", "primary")},
        "baseline_test": {"GAUC": base["GAUC"], "nDCG@5": base["nDCG@5"],
                          "primary": base["primary"]},
        "delta": {
            "GAUC": metrics["GAUC"] - base["GAUC"],
            "nDCG@5": metrics["nDCG@5"] - base["nDCG@5"],
            "primary": metrics["primary"] - base["primary"],
        },
        "oracle_ceiling_primary": cfg["baseline"]["oracle_ceiling"]["test"]["primary"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["delta"], indent=2))
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
