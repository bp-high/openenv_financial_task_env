#!/usr/bin/env python3
"""SFT corpus builder for the office-document task environment.

Reads teacher trajectories produced by ``inference.py``
(``runs/<run_dir>/trajectories/<task_id>.jsonl`` + ``summary.csv``) and emits
a ``messages``-formatted JSONL ready for ``trl.SFTTrainer``.

Filters applied (in order):

1. **error column non-empty**       — failed run, drop
2. **n_steps < --min-steps**         — too short, drop  (default 2)
3. **1-step submit-file**            — defense in depth against grader
                                       exploits where a model submits the
                                       source unchanged (Phase 7 issue).
                                       A real solve takes at least one
                                       code step.  Always dropped, even
                                       at high score.
4. **final_score < --score-threshold** — low quality, drop  (default 0.4)
5. **action_type not in (code, submit, submit_file)** — malformed, drop
6. **only the last step succeeded**  — agent never made progress, drop

Reconstructed messages format (TRL ``SFTTrainer`` ``messages`` column):

    [
      {"role": "system",    "content": <family-aware system prompt>},
      {"role": "user",      "content": <task instruction + source path + family>},
      {"role": "assistant", "content": <step-1 action: code block / SUBMIT_…>},
      {"role": "user",      "content": <step-1 feedback>},
      {"role": "assistant", "content": <step-2 action>},
      ...
    ]

Usage:
    python data_pipeline/build_sft_corpus.py \\
        --runs runs/teacher_kimi_k25_train \\
        --output data/sft_kimi_k25.jsonl \\
        --score-threshold 0.4
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Re-use the family-aware system prompts from inference.py so the SFT corpus
# matches what the model will see at deployment.
from inference import SYSTEM_PROMPTS  # noqa: E402
from tasks import TASKS               # noqa: E402


# ---------------------------------------------------------------------------
# Trajectory → messages
# ---------------------------------------------------------------------------

def _format_action_as_assistant(action_type: str, action_content: str) -> str:
    """Reconstruct the assistant turn the way inference.py expects to PARSE
    one (so the SFT corpus matches the action-extractor's input format)."""
    if action_type == "code":
        return f"```python\n{action_content}\n```"
    if action_type == "submit":
        return f"SUBMIT_ANSWER: {action_content}"
    if action_type == "submit_file":
        return f"SUBMIT_FILE: {action_content}"
    # Fallback — emit as code, this should be filtered upstream
    return action_content


def _format_step_feedback(feedback: str, step: int, max_steps: int, source_file: str) -> str:
    """Mirror the user message inference.py builds after each step."""
    return (
        f"Code execution result (step {step}/{max_steps}):\n"
        f"{feedback}\n\n"
        f"Source file: {source_file}"
    )


_PATH_RE = re.compile(r"['\"]([^'\"]*?(\.xlsx|\.docx|\.pptx))['\"]")


def _guess_source_file(trajectory: list[dict], fallback: str) -> str:
    """Try to pull the working-file path out of the agent's first code action.
    Falls back to the manifest's `source_file` (which is the data/ path —
    fine as a placeholder; the agent learns 'use whatever path is given')."""
    for entry in trajectory:
        if entry.get("action_type") == "code":
            content = entry.get("action_content", "")
            m = _PATH_RE.search(content)
            if m:
                return m.group(1)
            break
    return fallback


def _build_initial_user_message(task: dict, source_file: str) -> str:
    """Mirror the inference.py reset-time user message: instruction +
    constraints + source path + family + task type.  Skips the env's xlsx
    summary because we don't want to re-open the file at corpus-build time."""
    parts = [task.get("instruction", "").strip()]
    if task.get("constraints"):
        parts.append("Constraints:")
        parts.append(task["constraints"].strip())
    parts.extend([
        "",
        f"Source file path: {source_file}",
        f"File family: {task.get('family', 'xlsx')}",
        f"Task type: {task.get('task_type', 'MODIFY')}",
    ])
    return "\n".join(parts)


def reconstruct_messages(task: dict, trajectory: list[dict]) -> list[dict]:
    family = task.get("family", "xlsx")
    sys_prompt = SYSTEM_PROMPTS.get(family, SYSTEM_PROMPTS["xlsx"])
    source_file = _guess_source_file(trajectory, task.get("source_file", ""))

    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": _build_initial_user_message(task, source_file)},
    ]

    max_steps = task.get("max_steps", 15)
    n = len(trajectory)
    for i, entry in enumerate(trajectory):
        # Assistant turn
        messages.append({
            "role": "assistant",
            "content": _format_action_as_assistant(
                entry["action_type"], entry["action_content"],
            ),
        })
        # User turn (only if this isn't the terminating step)
        if i < n - 1 and not entry.get("done"):
            messages.append({
                "role": "user",
                "content": _format_step_feedback(
                    entry.get("feedback", ""), entry["step"], max_steps, source_file,
                ),
            })
    return messages


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _is_one_step_submit(trajectory: list[dict]) -> bool:
    """True iff the trajectory has exactly one action and it's submit_file.
    Defense against grader exploits like Phase 7 where submitting source
    unchanged scored high — a real solve takes at least one code step."""
    return len(trajectory) == 1 and trajectory[0].get("action_type") == "submit_file"


def _has_real_work(trajectory: list[dict]) -> bool:
    """At least one code step that actually executed (reward > 0.005 means
    exec_health credited the run, i.e. it didn't fail)."""
    return any(
        e.get("action_type") == "code" and float(e.get("reward") or 0) > 0.005
        for e in trajectory
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="path to the run dir produced by inference.py "
                         "(must contain summary.csv + trajectories/)")
    ap.add_argument("--output", required=True,
                    help="output JSONL path; rows look like "
                         "{task_id, family, score, n_steps, primary_tag, messages}")
    ap.add_argument("--score-threshold", type=float, default=0.4,
                    help="drop trajectories whose final_score < this (default 0.4)")
    ap.add_argument("--min-steps", type=int, default=2,
                    help="drop trajectories with fewer than this many steps "
                         "(default 2 — kills 1-step submit_file exploits even "
                         "if --score-threshold somehow lets them through)")
    ap.add_argument("--require-real-work", action="store_true", default=True,
                    help="drop trajectories with no successful code step "
                         "(default on)")
    ap.add_argument("--no-require-real-work", action="store_false",
                    dest="require_real_work")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    runs = Path(args.runs)
    summary_path = runs / "summary.csv"
    traj_dir = runs / "trajectories"
    if not summary_path.exists() or not traj_dir.exists():
        print(f"ERROR: {summary_path} or {traj_dir} not found", file=sys.stderr)
        return 1

    with open(summary_path) as f:
        rows = list(csv.DictReader(f))

    drops: Counter = Counter()
    accepted: list[dict] = []

    for r in rows:
        tid = r["task_id"]
        score = float(r["score"])
        n_steps = int(r["steps"])
        err = (r.get("error") or "").strip()

        # Filter 1: errors
        if err:
            drops["error"] += 1
            if args.verbose:
                print(f"  DROP {tid:55s} error: {err[:60]}")
            continue

        # Filter 2: too short
        if n_steps < args.min_steps:
            drops["too_short"] += 1
            if args.verbose:
                print(f"  DROP {tid:55s} only {n_steps} step(s)")
            continue

        # Load the trajectory file
        traj_path = traj_dir / f"{tid}.jsonl"
        if not traj_path.exists():
            drops["missing_trajectory"] += 1
            continue
        trajectory = [json.loads(l) for l in traj_path.read_text().splitlines() if l.strip()]
        if not trajectory:
            drops["empty_trajectory"] += 1
            continue

        # Filter 3: 1-step submit_file (defense in depth — even if score is high)
        if _is_one_step_submit(trajectory):
            drops["one_step_submit_file"] += 1
            if args.verbose:
                print(f"  DROP {tid:55s} 1-step submit_file (suspicious; score={score:.3f})")
            continue

        # Filter 4: low score
        if score < args.score_threshold:
            drops["low_score"] += 1
            if args.verbose:
                print(f"  DROP {tid:55s} score={score:.3f} < {args.score_threshold}")
            continue

        # Filter 5: malformed action types
        if any(e.get("action_type") not in ("code", "submit", "submit_file") for e in trajectory):
            drops["malformed_action"] += 1
            continue

        # Filter 6: no real work
        if args.require_real_work and not _has_real_work(trajectory):
            drops["no_real_work"] += 1
            if args.verbose:
                print(f"  DROP {tid:55s} no successful code step")
            continue

        task = TASKS.get(tid)
        if task is None:
            drops["task_not_in_manifest"] += 1
            continue

        messages = reconstruct_messages(task, trajectory)
        accepted.append({
            "task_id": tid,
            "family": r["family"],
            "primary_tag": r["primary_tag"],
            "split": r.get("split", "train"),
            "score": score,
            "n_steps": n_steps,
            "messages": messages,
        })

    # Write output
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for row in accepted:
            f.write(json.dumps(row) + "\n")

    # Report
    print(f"\n=== SFT corpus build summary ===")
    print(f"  Source run    : {runs}")
    print(f"  Output        : {out}")
    print(f"  Input rows    : {len(rows)}")
    print(f"  Accepted      : {len(accepted)}")
    print(f"  Drops:")
    for reason, n in drops.most_common():
        print(f"    {reason:25s}  {n:4d}")
    if accepted:
        by_family: Counter = Counter(r["family"] for r in accepted)
        avg_steps = sum(r["n_steps"] for r in accepted) / len(accepted)
        avg_score = sum(r["score"] for r in accepted) / len(accepted)
        print(f"\n  Accepted breakdown:")
        for fam, n in sorted(by_family.items()):
            print(f"    {fam:5s}  {n:4d}")
        print(f"  Avg steps   : {avg_steps:.1f}")
        print(f"  Avg score   : {avg_score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())