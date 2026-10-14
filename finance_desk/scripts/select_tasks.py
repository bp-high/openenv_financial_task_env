"""Pick a balanced <=50-task training subset covering all eight domains, and emit the submission's task list.

Strategy: finance is the deepest domain (it is the environment's theme), so it gets the most tasks; every other
domain gets a few. We prefer levels 2 and 3 (mixed-outcome, with caveats) over level 1, keeping a little level-1
coverage so not every group is hard. Output: tasks.json (submission `tasks`) and a domain summary.
"""

import collections
import json
import sys

sys.path.insert(0, ".")
from fdesk.families import FAMILIES  # noqa: E402

# target task count per domain (finance deepest)
QUOTA = {
    "finance and economics": 18,
    "office and white-collar work": 5,
    "industrial and physical systems": 5,
    "software engineering": 5,
    "natural science": 5,
    "math and formal reasoning": 4,
    "cybersecurity": 4,
    "media and content production": 4,
}

by_domain = collections.defaultdict(list)
for name, fam in FAMILIES.items():
    for lv in fam.levels:
        by_domain[fam.domain].append((name, lv))


def level_pref(item):
    # prefer level 2, then 3, then 1 (keep hard+mixed but some easy for gradient early)
    name, lv = item
    return {2: 0, 3: 1, 1: 2}[lv]


chosen = []
for domain, quota in QUOTA.items():
    items = sorted(by_domain[domain], key=lambda it: (level_pref(it), it[0], it[1]))
    chosen += items[:quota]

assert len(chosen) <= 50, len(chosen)

tasks = []
for name, lv in chosen:
    tasks.append({
        "task_id": f"{name}-l{lv}",
        "split": "train",
        # v2: tighter rollout/verifier so the trainer's "does another step fit" estimate packs ~2x more
        # optimizer steps into the 4h budget; episodes are single-step/short so this is comfortable.
        "reset_wall_s": 120,
        "rollout_wall_s": 300,
        "verifier_wall_s": 15,
        "tool_wall_s": 120,
        # inline files + optional reasoning; raise context so the first observation + reply always fit.
        "completion_tokens": 8192,
        "context_tokens": 16384,
    })

summary = collections.Counter(FAMILIES[name].domain for name, lv in chosen)
json.dump(tasks, open("tasks.json", "w"), indent=1)
print(f"selected {len(tasks)} tasks across {len(summary)} domains")
for d, c in sorted(summary.items(), key=lambda kv: -kv[1]):
    print(f"  {c:2d}  {d}")
print("task ids:", ", ".join(t["task_id"] for t in tasks))
