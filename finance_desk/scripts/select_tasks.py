"""Pick a balanced <=50-task training subset covering all eight domains, and emit the submission's task list.

Strategy (v2, see docs/STRATEGY.md): the board shows points live in office + industrial (the base model
already gets ~2 each) and that strong training tends to ERODE those base solves. So we (a) keep all eight
domains for the best-per-domain draw, (b) over-weight office + industrial so training reinforces the
competencies that actually carry points, and (c) prefer EASIER levels (1 then 2, avoid the hard level-3
hidden-caveat tasks) so training nudges "keep doing the sensible thing" rather than a narrow over-fit that
forgets it. Finance stays well represented as the theme. Output: tasks.json + a domain summary.
"""

import collections
import json
import sys

sys.path.insert(0, ".")
from fdesk.families import FAMILIES  # noqa: E402

# Over-weight the point-carrying domains (office, industrial); keep finance strong; cover the rest for the
# best-per-domain draw. Quotas are capped by how many unique family×level ids each domain has.
QUOTA = {
    "office and white-collar work": 6,       # base model's strongest cell — reinforce, don't erode
    "industrial and physical systems": 6,    # the point we trained away in v1 — recover it
    "finance and economics": 16,             # environment theme
    "media and content production": 6,
    "software engineering": 5,
    "natural science": 5,
    "math and formal reasoning": 3,
    "cybersecurity": 3,
}

by_domain = collections.defaultdict(list)
for name, fam in FAMILIES.items():
    for lv in fam.levels:
        by_domain[fam.domain].append((name, lv))


def level_pref(item):
    # prefer level 1, then 2, then 3: easier tasks keep the base model in its competent regime and avoid
    # over-hardening (the hard level-3 caveat tasks are the ones most likely to distort the policy).
    name, lv = item
    return {1: 0, 2: 1, 3: 2}[lv]


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
        # Short single-step episodes: tight rollout/verifier so the trainer packs more clean-JSON reps into
        # 4h (helps the one plausibly-transferable skill, output-format discipline; see docs/STRATEGY.md).
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
