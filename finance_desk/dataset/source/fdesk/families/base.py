"""Shared pieces for task families: instances, CSV helpers, rounding and grading.

Every family generates workspace files only; its solver recomputes the answer from those files, so each answer
re-derives from what the agent can read. Levels: 1 states every rule, 2 adds a complication, 3 adds a notice file
that overrides part of the data (the caveat-ignoring answer is checked to differ).
"""

import csv
import io
import json
import random
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable

COMMON_RULES = (
    "Read every file before answering; a notice file overrides the data it mentions. "
    "Compute with the python tool rather than by hand. Submit once with exactly the documented keys."
)


@dataclass
class Spec:
    """How one answer key is graded."""
    kind: str = "num"          # num | ids (unordered) | seq (ordered) | str | bool
    tol: float = 0.01


@dataclass
class Instance:
    prompt: str
    files: dict[str, str]
    answer: dict[str, Any]
    specs: dict[str, Spec] = field(default_factory=dict)
    grader: Any = None  # optional callable(answer_dict) -> bool, overrides field grading


class Family:
    name: str = ""
    title: str = ""
    domain: str = "finance and economics"
    levels = (1, 2, 3)

    def files(self, rng: random.Random, level: int) -> tuple[str, dict[str, str]]:
        raise NotImplementedError

    def solve(self, files: dict[str, str], level: int) -> dict[str, Any]:
        raise NotImplementedError

    specs: dict[str, Spec] = {}

    def generate(self, rng: random.Random, level: int) -> Instance:
        for _ in range(50):
            prompt, files = self.files(rng, level)
            answer = self.solve(files, level)
            if answer is None:
                continue
            notices = [n for n in files if n.startswith("notice")]
            if level >= 3 and notices:
                blind = {k: v for k, v in files.items() if k not in notices}
                try:
                    blind_answer = self.solve(blind, level)
                except Exception:
                    blind_answer = None
                if blind_answer is not None and grade(answer, blind_answer, self.specs):
                    continue  # the notice must change the answer
            return Instance(prompt=prompt, files=files, answer=answer, specs=self.specs)
        raise RuntimeError(f"{self.name}: could not generate a valid level-{level} instance")


# ----------------------------------------------------------------------------- helpers

def q2(x: Any) -> float:
    return float(Decimal(repr(float(x)) if isinstance(x, float) else str(x)).quantize(Decimal("0.01"), ROUND_HALF_UP))


def qn(x: Any, n: int) -> float:
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-n), ROUND_HALF_UP))


def to_csv(header: list[str], rows: list[list[Any]]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def read_csv(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def dump(obj: Any) -> str:
    return json.dumps(obj, indent=1) + "\n"


def iso(d: date) -> str:
    return d.isoformat()


def pdate(s: str) -> date:
    return date.fromisoformat(s.strip())


def rand_date(rng: random.Random, start: date, span_days: int) -> date:
    return start + timedelta(days=rng.randrange(span_days))


def money(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 2)


def ids(prefix: str, n: int, rng: random.Random, start: int = 100) -> list[str]:
    base = rng.randrange(start, start + 800)
    return [f"{prefix}-{base + i * rng.randint(1, 3) + i}" for i in range(n)]


# ----------------------------------------------------------------------------- grading

def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace(",", "").replace("$", "").replace("%", "").replace(" ", "")
        if s.startswith("(") and s.endswith(")"):
            s = "-" + s[1:-1]
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _norm(v: Any) -> str:
    s = str(v).strip().lower()
    if len(s) >= 16 and s[4:5] == "-" and s[10:11] in ("t", " "):  # datetime: YYYY-MM-DDTHH:MM[:00][Z]
        s = s.replace(" ", "t").rstrip("z")
        if s.endswith("+00:00"):
            s = s[:-6]
        if len(s) == 19 and s.endswith(":00"):
            s = s[:16]
    return s


def match(expected: Any, got: Any, spec: Spec) -> bool:
    if spec.kind == "num":
        g = _num(got)
        return g is not None and abs(g - float(expected)) <= spec.tol + 1e-9
    if spec.kind == "ids":
        if isinstance(got, str):
            got = [p for p in got.replace(";", ",").split(",") if p.strip()]
        if not isinstance(got, list):
            return False
        return sorted(_norm(x) for x in got) == sorted(_norm(x) for x in expected)
    if spec.kind == "seq":
        if isinstance(got, str):
            got = [p for p in got.replace(";", ",").replace("->", ",").split(",") if p.strip()]
        if not isinstance(got, list):
            return False
        return [_norm(x) for x in got] == [_norm(x) for x in expected]
    if spec.kind == "bool":
        if isinstance(got, str):
            got = {"true": True, "yes": True, "false": False, "no": False}.get(_norm(got), got)
        return got is expected
    return _norm(got) == _norm(expected)


def infer(value: Any) -> Spec:
    if isinstance(value, bool):
        return Spec(kind="bool")
    if isinstance(value, (int, float)):
        return Spec(tol=0.0 if isinstance(value, int) else 0.01)
    if isinstance(value, list):
        return Spec(kind="ids")
    return Spec(kind="str")


def grade(expected: dict[str, Any], got: Any, specs: dict[str, Spec]) -> bool:
    if not isinstance(got, dict):
        return False
    lowered = {str(k).strip().lower(): v for k, v in got.items()}
    for key, value in expected.items():
        if key.lower() not in lowered:
            return False
        if not match(value, lowered[key.lower()], specs.get(key) or infer(value)):
            return False
    return True


SolveFn = Callable[[dict[str, str], int], dict[str, Any]]
