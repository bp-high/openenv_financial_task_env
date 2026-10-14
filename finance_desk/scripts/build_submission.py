"""Assemble submission.json for the Arena from the running server's /schema and the selected task list.

Usage: python scripts/build_submission.py <image_ref> [schema.json] [tasks.json]
Writes submission.json. example_actions is a single bare submit (terminal reward 0, valid under the schema and
admissible for every task); finish_action is the same.
"""

import json
import sys

image = sys.argv[1]
schema = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "schema.json"))
tasks = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "tasks.json"))

submission = {
    "submission_id": "finance-desk-v2",
    "name": "Finance Desk v2: single-step inline-files multi-domain environment",
    "image": image,
    "schema": {"action": schema["action"], "observation": schema["observation"]},
    "tasks": tasks,
    "example_actions": [{"tool": "submit", "args": {"answer": {}}}],
    "finish_action": {"tool": "submit", "args": {"answer": {}}},
    "dataset": "bpHigh/finance-desk",
    "source": "https://github.com/bp-high/openenv_financial_task_env/tree/dev/pensive-lovelace-bt5rsn/finance_desk",
}
json.dump(submission, open("submission.json", "w"), indent=2)
print(f"wrote submission.json: {len(tasks)} tasks, image {image}")
