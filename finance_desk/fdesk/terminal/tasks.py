"""Terminal-mode task registry: software families plus the finance families from v2, adapted to the terminal.

Task ids keep the v2 form '<family>-l<level>'. A finance task writes its v2 workspace files to the workdir and
asks for the answer in answer.json (graded with the v2 field grader); a software task ships a small repo and is
graded by hidden test cases run against the agent's code.
"""

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any

from ..families import FAMILIES as V2_FAMILIES
from ..families.base import COMMON_RULES, grade
from . import software

ANSWER_FILE = "answer.json"


@dataclass
class TermInstance:
    instruction: str
    files: dict[str, str]
    oracle: dict[str, str]
    # software: hidden cases + file checks; finance: expected answer + specs (+ optional custom grader)
    cases: list[dict] = field(default_factory=list)
    file_checks: list[tuple[str, str]] = field(default_factory=list)
    answer: dict[str, Any] | None = None
    specs: dict = field(default_factory=dict)
    grader: Any = None
    domain: str = ""


class FinanceTask:
    def __init__(self, fam):
        self.fam = fam
        self.name = fam.name
        self.title = fam.title
        self.domain = fam.domain
        self.levels = fam.levels

    def generate(self, rng: random.Random, level: int) -> TermInstance:
        inst = self.fam.generate(rng, level)
        prompt = inst.prompt.replace(" " + COMMON_RULES, "").replace(COMMON_RULES, "")
        prompt = prompt.replace("Answer: ", f"Save the answer as a JSON object in {ANSWER_FILE}: ", 1)
        notes = "The input files are in the current directory."
        if any(n.startswith("notice") for n in inst.files):
            notes += " A notice file overrides the data it mentions."
        return TermInstance(instruction=f"{prompt} {notes}", files=dict(inst.files),
                            oracle={ANSWER_FILE: json.dumps(inst.answer)}, answer=inst.answer,
                            specs=inst.specs, grader=inst.grader, domain=self.domain)


class SoftwareTask:
    def __init__(self, fam):
        self.fam = fam
        self.name = fam.name
        self.title = fam.title
        self.domain = fam.domain
        self.levels = fam.levels

    def generate(self, rng: random.Random, level: int) -> TermInstance:
        s = self.fam.generate(rng, level)
        return TermInstance(instruction=s.instruction, files=s.files, oracle=s.oracle, cases=s.cases,
                            file_checks=s.file_checks, domain=self.domain)


FINANCE_NAMES = [n for n, f in V2_FAMILIES.items() if f.domain == "finance and economics"]
TASKS = {f.name: SoftwareTask(f) for f in software.FAMILIES}
TASKS.update({n: FinanceTask(V2_FAMILIES[n]) for n in FINANCE_NAMES})


def task_ids() -> list[str]:
    return [f"{name}-l{lv}" for name, t in TASKS.items() for lv in t.levels]


def parse_task_id(task_id: str):
    name, _, lv = task_id.rpartition("-l")
    if name not in TASKS or not lv.isdigit() or int(lv) not in TASKS[name].levels:
        raise ValueError(f"unknown task_id {task_id!r}")
    return TASKS[name], int(lv)


def grade_answer_file(inst: TermInstance, workdir: str) -> tuple[bool, str]:
    path = os.path.join(workdir, ANSWER_FILE)
    if not os.path.exists(path):
        return False, f"{ANSWER_FILE} not found"
    try:
        got = json.load(open(path))
    except Exception as e:
        return False, f"{ANSWER_FILE} is not valid JSON: {e}"
    ok = bool(inst.grader(got)) if inst.grader else grade(inst.answer, got, inst.specs)
    return ok, "all fields correct" if ok else "one or more fields are wrong"
