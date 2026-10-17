"""Assemble submission_v3.json for the Arena (terminal mode).

Usage: python scripts/build_submission_v3.py <image_ref> <schema.json> <task_ids_file> [submission_id] [out.json]
Per-task budgets copy the operator's own Terminal-Bench 2 smoke test (benchflow/openenv-smoke-fix-git): 4096
completion / 8192 context tokens, 32 tool calls, 60/min with a burst of 4. Training under the same budgets as the
private tasks is the point of v3 (docs/STRATEGY_v3.md). example_actions / finish_action is a bare `evaluate`
(terminal reward 0 on an untouched workdir; valid for every task).
"""

import json
import sys

image, schema_path, ids_path = sys.argv[1:4]
sub_id = sys.argv[4] if len(sys.argv) > 4 else "finance-desk-v3"
out = sys.argv[5] if len(sys.argv) > 5 else "submission_v3.json"
schema = json.load(open(schema_path))
ids = [t.strip() for t in open(ids_path).read().replace(",", "\n").split("\n") if t.strip()]
assert 1 <= len(ids) <= 50 and len(set(ids)) == len(ids), len(ids)

tasks = [{
    "task_id": t,
    "split": "train",
    "reset_wall_s": 120,
    "rollout_wall_s": 900,
    "verifier_wall_s": 120,
    "tool_wall_s": 110,
    "tool_calls_total": 32,
    "tool_calls_per_minute": 60,
    "tool_rate_burst": 4,
    "tool_calls_inflight": 1,
    "completion_tokens": 4096,
    "context_tokens": 8192,
} for t in ids]

submission = {
    "submission_id": sub_id,
    "name": "Finance Desk v3: terminal-mode software repair + finance (tbench2_env contract)",
    "image": image,
    "schema": {"action": schema["action"], "observation": schema["observation"]},
    "tasks": tasks,
    "example_actions": [{"action_type": "evaluate"}],
    "finish_action": {"action_type": "evaluate"},
    "dataset": "bpHigh/finance-desk",
    "source": "https://github.com/bp-high/openenv_financial_task_env/tree/dev/pensive-lovelace-bt5rsn/finance_desk",
}
json.dump(submission, open(out, "w"), indent=2)
print(f"wrote {out} ({sub_id}): {len(tasks)} tasks, image {image}")
