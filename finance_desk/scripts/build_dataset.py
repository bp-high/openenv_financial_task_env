"""Build the public Hugging Face dataset: a tasks.jsonl preview, a README card, and a source snapshot.

Each line of tasks.jsonl is one concrete instance (prompt, files, answer, domain) for a selected task_id and seed,
so a visitor can see exactly what the agent reads and what counts as correct. No answer is hidden; the environment
regenerates fresh instances at train time, so the preview is illustrative, not the evaluated set.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, ".")
from fdesk.environment import FinanceDeskEnvironment  # noqa: E402
from fdesk.families import FAMILIES  # noqa: E402

OUT = "dataset"
os.makedirs(OUT, exist_ok=True)
tasks = json.load(open("tasks.json"))

rows = []
env = FinanceDeskEnvironment()
for t in tasks:
    tid = t["task_id"]
    name = tid.rpartition("-l")[0]
    for seed in range(3):
        env.reset(seed=seed, task_id=tid)
        inst = env._instance
        rows.append({
            "task_id": tid,
            "domain": FAMILIES[name].domain,
            "level": int(tid.rpartition("-l")[2]),
            "seed": seed,
            "prompt": inst.prompt,
            "files": inst.files,
            "answer": json.loads(json.dumps(inst.answer)),
        })
with open(os.path.join(OUT, "tasks.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

# source snapshot (pure-python package, small)
src = os.path.join(OUT, "source", "fdesk")
shutil.rmtree(os.path.join(OUT, "source"), ignore_errors=True)
shutil.copytree("fdesk", src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

print(f"wrote {len(rows)} preview instances across {len(tasks)} task ids")
