"""Registry of task families. Task ids are '<family>-l<level>', e.g. 'loan_amort-l2'."""

from . import accounting, corporate, domains_a, domains_b, lending, media_music

FAMILIES = {f.name: f for mod in (lending, accounting, corporate, domains_a, domains_b, media_music) for f in mod.FAMILIES}


def task_ids() -> list[str]:
    return [f"{name}-l{lv}" for name, fam in FAMILIES.items() for lv in fam.levels]


def parse_task_id(task_id: str):
    name, _, lv = task_id.rpartition("-l")
    if name not in FAMILIES or not lv.isdigit() or int(lv) not in FAMILIES[name].levels:
        raise ValueError(f"unknown task_id {task_id!r}")
    return FAMILIES[name], int(lv)
