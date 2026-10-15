"""Build the v3 Hugging Face dataset preview: one row per (selected task id, seed) with what the agent sees.

Usage: python scripts/build_dataset_v3.py <task_ids_file>
Writes dataset/tasks.jsonl (v3, the default config), keeps the v2 preview at dataset/v2/tasks.jsonl, and
refreshes dataset/source/fdesk. Each row: task_id, domain, level, seed, instruction, files (the workdir), and the
reference solution (`oracle`: files that solve it). Hidden test cases are regenerated per seed at train time and
are not part of the preview. The environment regenerates fresh instances at train time; this is illustrative.
"""

import json
import os
import shutil
import sys

sys.path.insert(0, ".")
from fdesk.terminal.environment import TerminalDeskEnvironment  # noqa: E402
from fdesk.terminal.tasks import TASKS  # noqa: E402

OUT = "dataset"
ids = [t.strip() for t in open(sys.argv[1]).read().replace(",", "\n").split("\n") if t.strip()]

os.makedirs(os.path.join(OUT, "v2"), exist_ok=True)
if os.path.exists(os.path.join(OUT, "tasks.jsonl")) and not os.path.exists(os.path.join(OUT, "v2", "tasks.jsonl")):
    shutil.move(os.path.join(OUT, "tasks.jsonl"), os.path.join(OUT, "v2", "tasks.jsonl"))

env = TerminalDeskEnvironment()
rows = []
for tid in ids:
    name, _, lv = tid.rpartition("-l")
    for seed in range(3):
        obs = env.reset(seed=seed, task_id=tid)
        inst = env._inst
        rows.append({"task_id": tid, "domain": TASKS[name].domain, "level": int(lv), "seed": seed,
                     "instruction": obs.instruction, "files": inst.files, "oracle": inst.oracle,
                     "hidden_tests": len(inst.cases)})
        env.close()
with open(os.path.join(OUT, "tasks.jsonl"), "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

src = os.path.join(OUT, "source", "fdesk")
shutil.rmtree(os.path.join(OUT, "source"), ignore_errors=True)
shutil.copytree("fdesk", src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
print(f"wrote {len(rows)} v3 preview rows across {len(ids)} task ids")
