"""The outer agent loop — the artifact being judged.

One iteration, end to end:

  1. ``policy.propose(state, history)``   -> ActionProposal (action_type + hypothesis)
  2. ``codegen.generate(...)``            -> new main.py source (stub or LLM)
  3. ``codegen.lint_generated_code(...)`` -> contract + test-leak guard, pre-execution
  4. ``executor.run_pipeline(...)``       -> subprocess, hard timeout, never exec()
  5. ``eval.scorer.score_submission(...)``-> official GAUC / nDCG@5 / primary on VALID
  6. ``critic.judge(...)``                -> accept / reject + reflection
  7. accept: workspace commit + ``git format-patch`` into ``logs/diffs/``, checkpoint kept;
     reject: ``git checkout . && git clean -fd``
  8. ``journal.append_entry(...)``        -> the spine; state folds it in
  9. ``state.check_convergence(...)``     -> stop or continue

Robustness is the point of steps 3-5, not an afterthought: every failure becomes an
ErrorEvent carrying the recovery taken, and the loop is written so that no failure
mode — bad codegen, syntax error, timeout, misaligned submission, provider outage —
can crash, stall, or diverge the run. The only way out of the loop is convergence.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import yaml

from agent import codegen, critic, memory, policy as policy_mod
from agent.executor import ExecutionResult, Workspace, run_pipeline
from agent.journal import ErrorEvent, JournalEntry, append_entry, load_journal
from agent.llm import LLMClient, LLMError
from agent.state import Budget, Node, RunState, check_convergence
from eval import scorer


#: Pipeline stdout is fed back into LLM context, so it is capped. Head keeps setup
#: (row counts, feature shapes), tail keeps the final epochs — the two ends that say
#: whether training actually converged.
STDOUT_HEAD, STDOUT_TAIL = 700, 1500


def _tail(text: str) -> Optional[str]:
    """Trim pipeline stdout for the journal, keeping both ends."""
    text = (text or "").strip()
    if not text:
        return None
    if len(text) <= STDOUT_HEAD + STDOUT_TAIL:
        return text
    return f"{text[:STDOUT_HEAD]}\n...[elided]...\n{text[-STDOUT_TAIL:]}"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _has_credentials() -> bool:
    """True if the Anthropic SDK will find credentials.

    An unset ANTHROPIC_API_KEY does NOT mean there are none: the SDK resolves
    ANTHROPIC_API_KEY -> ANTHROPIC_AUTH_TOKEN -> an `ant auth login` profile on disk.
    Checking only the env var would refuse to start for a user authenticated via the CLI.
    """
    import os
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config/anthropic").exists()


def archive_run(root: Path, cfg: dict[str, Any]) -> Optional[Path]:
    """Move the current run's artifacts into ``runs/<timestamp>/`` and start clean.

    A run is defined by its journal: resuming appends to it, so a *new* run needs a new
    journal. Archiving rather than deleting keeps every prior run reproducible — the
    journal is a judge deliverable, and a harness proof is still evidence the harness works.
    Returns the archive directory, or None if there was nothing to archive.
    """
    journal = root / cfg["paths"]["journal"]
    workspace = root / cfg["paths"]["workspace"]
    if not (journal.exists() and journal.stat().st_size) and not (workspace / ".git").exists():
        return None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = root / "runs" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for rel in (cfg["paths"]["journal"], cfg["paths"]["diffs"], cfg["paths"]["checkpoints"],
                cfg["paths"]["workspace"]):
        src = root / rel
        if src.exists():
            shutil.move(str(src), str(dest / Path(rel).name))

    (root / cfg["paths"]["diffs"]).mkdir(parents=True, exist_ok=True)
    (root / cfg["paths"]["workspace"]).mkdir(parents=True, exist_ok=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.touch()
    return dest


class AgentRun:
    """One full run to convergence. Holds paths and collaborators; owns no metrics
    state of its own — that all lives in the journal and the RunState derived from it."""

    def __init__(self, agent_cfg: dict[str, Any], data_cfg: dict[str, Any],
                 root: Path, max_iterations: Optional[int] = None,
                 timeout_override: Optional[float] = None,
                 policy_override: Optional[str] = None) -> None:
        if policy_override:
            agent_cfg = {**agent_cfg,
                         "policy": {**agent_cfg["policy"], "kind": policy_override}}
        self.cfg, self.data_cfg, self.root = agent_cfg, data_cfg, Path(root)
        self.paths = agent_cfg["paths"]
        self.journal_path = self.root / self.paths["journal"]
        self.diffs_dir = self.root / self.paths["diffs"]
        self.checkpoints_dir = self.root / self.paths["checkpoints"]
        self.workspace = Workspace(self.root / self.paths["workspace"])
        self.template_dir = self.root / self.paths["template"]
        self.data_dir = str(self.root / data_cfg["paths"]["raw"])
        self.timeout = float(timeout_override
                             if timeout_override is not None
                             else agent_cfg["executor"]["timeout_seconds"])
        # Default to the interpreter running the harness, NOT a bare "python3": under
        # `uv run` (or any venv) those are different environments, and the pipeline would
        # silently lose access to lightgbm/pandas that the harness can see.
        self.python = agent_cfg["executor"].get("python") or sys.executable
        self.epsilon = float(agent_cfg["convergence"]["epsilon"])
        self.budget = Budget.from_config(agent_cfg.get("budget", {}))
        self.max_iterations = max_iterations

        self.client = LLMClient(
            model=agent_cfg["llm"]["model"],
            prompts_dir=self.root / self.paths["prompts"],
            max_output_tokens=agent_cfg["llm"]["max_output_tokens"],
            max_retries=agent_cfg["llm"].get("max_retries", 3),
            effort=agent_cfg["llm"].get("effort"),
        )
        self.policy = policy_mod.build(agent_cfg["policy"], self.client)
        self.generator: codegen.CodeGenerator = (
            codegen.StubCodeGenerator(seed=agent_cfg["policy"].get("seed", 0))
            if agent_cfg["policy"].get("kind") == "random"
            else codegen.LLMCodeGenerator(self.client))

    # ----------------------------- one iteration ------------------------------------

    def run_iteration(self, state: RunState, history: list[JournalEntry]) -> JournalEntry:
        """Execute one iteration and return the journal entry describing it.

        Never raises: every failure path produces an entry with error events and a
        recorded recovery, because an unjournaled failure is both a lost deliverable
        and a hole in the agent's own memory.
        """
        t0 = time.time()
        node_id = uuid.uuid4().hex[:8]
        errors: list[ErrorEvent] = []
        execution: Optional[ExecutionResult] = None
        metrics: Optional[dict] = None
        diff_path: Optional[str] = None
        stdout_tail: Optional[str] = None
        commit_sha: Optional[str] = None
        checkpoint_path: Optional[str] = None
        config: dict[str, Any] = {}
        generated: Optional[codegen.GeneratedCode] = None

        # The policy reads the code it is about to change.
        if isinstance(self.policy, policy_mod.LLMPolicy):
            try:
                self.policy.current_zone = codegen.extract_agent_zone(
                    self.workspace.read_file("main.py"))
            except Exception:
                self.policy.current_zone = "(pipeline source unavailable)"

        proposal = self.policy.propose(state, history)
        # A policy that failed still returns a usable fallback; its errors are journaled
        # here so a provider outage is visible rather than silently degrading the run.
        errors.extend(getattr(self.policy, "pending_errors", []))

        parent = state.tip()
        parent_id = proposal.parent_id or (parent.node_id if parent else None)

        # Backtracking: branch the workspace from an earlier accepted commit, so the
        # commit graph keeps matching the journal's parent links.
        if proposal.parent_id:
            target = state.nodes.get(proposal.parent_id)
            if target and target.commit_sha:
                try:
                    self.workspace.branch_from(target.commit_sha,
                                               f"node-{proposal.parent_id}")
                except Exception as exc:
                    errors.append(ErrorEvent("code_error",
                                             f"backtrack failed: {exc}"[:2000],
                                             "reroute", True))

        out_dir = self.root / "logs" / "_work" / node_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            current = self.workspace.read_file("main.py")
            gen_context = {
                "history": history,
                "journal_summary": memory.summarize_journal(history),
                "extra_context": (memory.error_context(history)
                                  if proposal.action_type == "debug" else ""),
            }
            generated = self.generator.generate(proposal, current, gen_context)
            config = dict(generated.config)

            problems = codegen.lint_generated_code(generated.source)
            if problems:
                # Caught before execution: cheaper than a failed run, and the reason is
                # specific enough for the debug action to act on next iteration.
                errors.append(ErrorEvent(
                    error_type=("code_error" if any("syntax" in p for p in problems)
                                else "bad_llm_output"),
                    message="; ".join(problems)[:2000],
                    recovery="revert", recovered=True))
            else:
                self.workspace.write_file("main.py", generated.source)
                execution = run_pipeline(
                    [self.python, "main.py", "--data-dir", self.data_dir,
                     "--split", "valid", "--out-dir", str(out_dir)],
                    cwd=self.workspace.path, timeout_seconds=self.timeout)

                # Captured whether the run succeeded or failed — a crashed run's partial
                # training log is often the most informative thing in the iteration.
                stdout_tail = _tail(execution.stdout)

                if execution.timed_out:
                    errors.append(ErrorEvent("timeout",
                                             execution.stderr[-2000:] or "hard timeout",
                                             "revert", True))
                elif not execution.ok:
                    errors.append(ErrorEvent("code_error",
                                             execution.stderr[-2000:] or "non-zero exit",
                                             "revert", True))
                else:
                    try:
                        metrics = scorer.score_submission(
                            out_dir / "predictions.csv", self.data_dir, split="valid")
                    except Exception as exc:  # misalignment, NaN, missing file, ...
                        errors.append(ErrorEvent("eval_error", str(exc)[:2000],
                                                 "revert", True))
        except codegen.CodeGenError as exc:
            errors.append(ErrorEvent("bad_llm_output", str(exc)[:2000], "revert", True))
        except LLMError as exc:
            errors.append(ErrorEvent("llm_api_error", str(exc)[:2000], "reroute", True))
        except Exception as exc:  # harness bug or unexpected state: never crash the run
            errors.append(ErrorEvent("code_error", f"{type(exc).__name__}: {exc}"[:2000],
                                     "revert", True))

        verdict = critic.judge(state, execution, metrics, epsilon=self.epsilon)

        if verdict.accepted:
            summary = generated.summary if generated else proposal.action_type
            commit_sha = self.workspace.commit(
                f"{proposal.action_type}: {summary} [{node_id}]")
            patch = self.workspace.export_patch(self.diffs_dir / f"{node_id}.patch")
            diff_path = str(patch.relative_to(self.root)) if patch else None
            dest = self.checkpoints_dir / node_id
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(out_dir, dest)
            checkpoint_path = str(dest.relative_to(self.root))
        else:
            self.workspace.revert_all()

        shutil.rmtree(out_dir, ignore_errors=True)
        tokens_in, tokens_out = self.client.take_iteration_usage()
        config.update({"verdict": verdict.reason, "reflection": verdict.reflection})

        return JournalEntry(
            node_id=node_id, parent_id=parent_id, iteration=state.iteration,
            timestamp=_now(), action_type=proposal.action_type,
            hypothesis=proposal.hypothesis, config=config,
            diff_path=diff_path, commit_sha=commit_sha, checkpoint_path=checkpoint_path,
            val_gauc=metrics["GAUC"] if metrics else None,
            val_ndcg5=metrics["nDCG@5"] if metrics else None,
            stdout_tail=stdout_tail,
            wall_seconds=round(time.time() - t0, 2),
            gpu_seconds=0.0,  # CPU-only by default; a torch escalation must measure this
            tokens_in=tokens_in, tokens_out=tokens_out,
            error_events=errors, accepted=verdict.accepted,
            intervention=False, intervention_note=None,
        )

    # -------------------------------- the run ---------------------------------------

    def run(self) -> RunState:
        """Iterate until the organisers' convergence rule fires."""
        self.workspace.path.parent.mkdir(parents=True, exist_ok=True)
        seeded = self.workspace.init_if_needed(self.template_dir)
        print(f"workspace: {'seeded from template' if seeded else 'resumed'} "
              f"at {self.workspace.path}")

        if self.cfg["policy"].get("kind") == "llm" and not _has_credentials():
            raise RuntimeError(
                "policy.kind is 'llm' but no Anthropic credentials were found.\n"
                "  export ANTHROPIC_API_KEY=sk-ant-...   (console.anthropic.com)\n"
                "  or run: ant auth login                 (stores a profile the SDK reads)\n"
                "  or run the harness proof with --policy random (no provider needed).")

        check = scorer.self_check(self.data_dir)
        print(f"eval self-check: {'PASS' if check['ok'] else 'FAIL'} "
              f"(random valid primary {check['got']:.4f}, expected ~{check['expected']})")
        if not check["ok"]:
            raise RuntimeError("evaluation self-check failed — fix the harness first")

        state = RunState.from_journal(self.journal_path)
        history = load_journal(self.journal_path) if self.journal_path.exists() else []
        if history:
            print(f"resumed from journal: {len(history)} prior iterations")

        while True:
            converged, reason = check_convergence(state, self.cfg["convergence"], self.budget)
            if converged:
                print(f"\nCONVERGED: {reason}")
                break
            if self.max_iterations is not None and state.iteration >= self.max_iterations:
                print(f"\nSTOPPED: iteration cap {self.max_iterations} reached "
                      f"(harness-proof mode, not the organisers' rule)")
                break

            entry = self.run_iteration(state, history)
            append_entry(self.journal_path, entry)
            history.append(entry)
            state.record_iteration(
                Node(entry.node_id, entry.parent_id, entry.commit_sha,
                     entry.val_gauc, entry.val_ndcg5, entry.accepted),
                tokens_in=entry.tokens_in, tokens_out=entry.tokens_out,
                gpu_seconds=entry.gpu_seconds, wall_seconds=entry.wall_seconds,
                intervention=entry.intervention)

            score = (f"primary {entry.val_primary:.4f}" if entry.val_primary is not None
                     else "no score")
            errs = f" | {len(entry.error_events)} error(s): " + \
                   ",".join(e.error_type for e in entry.error_events) if entry.error_events else ""
            print(f"  [{entry.iteration:02d}] {entry.action_type:8s} "
                  f"{'ACCEPT' if entry.accepted else 'reject'} | {score} "
                  f"| {entry.wall_seconds:.1f}s{errs}")

        best = state.best_node()
        if best:
            base = self.cfg["baseline"]["valid"]["primary"]
            print(f"\nvalidation-best: node {best.node_id} primary "
                  f"{best.selection_score:.4f} (baseline {base}, "
                  f"delta {best.selection_score - base:+.4f})")
        else:
            print("\nno scored iteration produced")
        print(f"tokens: {state.tokens_total:,d} | gpu_s: {state.gpu_seconds:.1f} "
              f"| wall_s: {state.wall_seconds:.1f} | interventions: {state.interventions}")
        return state


def run(agent_config_path: Path, data_config_path: Path,
        max_iterations: Optional[int] = None,
        timeout_override: Optional[float] = None,
        policy_override: Optional[str] = None,
        fresh: bool = False) -> RunState:
    """Execute one full agent run to convergence."""
    root = Path(agent_config_path).resolve().parent.parent
    agent_cfg = _load_yaml(agent_config_path)
    if fresh:
        archived = archive_run(root, agent_cfg)
        print(f"archived previous run to {archived}" if archived
              else "no previous run to archive; starting clean")
    return AgentRun(agent_cfg, _load_yaml(data_config_path),
                    root, max_iterations, timeout_override, policy_override).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the autonomous ML research agent.")
    parser.add_argument("--agent-config", type=Path, default=Path("configs/agent.yaml"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="hard cap for harness proofs; the real stop is convergence")
    parser.add_argument("--timeout", type=float, default=None,
                        help="override executor.timeout_seconds (harness proofs)")
    parser.add_argument("--policy", choices=["random", "llm"], default=None,
                        help="override policy.kind; 'random' is the no-provider harness proof")
    parser.add_argument("--fresh", action="store_true",
                        help="archive the current run to runs/<timestamp>/ and start a new "
                             "journal (without this, a run RESUMES the existing journal)")
    args = parser.parse_args()
    run(args.agent_config, args.data_config, args.max_iterations, args.timeout,
        args.policy, args.fresh)


if __name__ == "__main__":
    main()
