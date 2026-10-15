"""Software-engineering task families for terminal mode: fix seeded bugs, implement to a spec, repair a script.

Each family builds a small repo in the workdir and a set of hidden test cases. At generation time the shipped
code is run against the hidden cases in-process and must FAIL, and the reference must PASS, so every instance
needs a real code change. Levels: 1 = a failing visible test points at the bug; 2 = visible tests pass and a bug
report (ISSUE.md) describes the symptom; 3 = two problems (or a NOTICE.md that changes the spec).
"""

import random
from dataclasses import dataclass, field
from typing import Any

from .runner import check_sources

HIDDEN_NOTE = "Hidden tests will check every function, so fix the cause rather than special-casing the examples."


@dataclass
class SoftwareInstance:
    instruction: str
    files: dict[str, str]
    cases: list[dict]
    oracle: dict[str, str]                      # file path -> content that solves the task
    file_checks: list[tuple[str, str]] = field(default_factory=list)  # (workdir path, expected text)


def _py(v: Any) -> str:
    return repr(v)


def _call_str(fn: str, args: list) -> str:
    return f"{fn}({', '.join(_py(a) for a in args)})"


def _approx(v: Any) -> str:
    if isinstance(v, float):
        return f"pytest.approx({v!r})"
    if isinstance(v, list) and any(isinstance(x, float) for x in v):
        return f"pytest.approx({v!r})"
    return repr(v)


# =============================================================================== module bug-fix families

class ModuleBugfix:
    """A module of small documented functions; one or two get a seeded bug."""
    name = ""
    title = ""
    domain = "software engineering"
    levels = (1, 2, 3)
    module_names: tuple[str, ...] = ()
    header = ""
    # fn -> {"ref": source, "bugs": {key: source}}
    functions: dict[str, dict[str, Any]] = {}

    def cases_for(self, fn: str, rng: random.Random) -> list[list]:
        """Hidden argument lists for fn (designed so every seeded bug fails at least one)."""
        raise NotImplementedError

    def visible_args(self, fn: str, rng: random.Random) -> list[list]:
        return self.cases_for(fn, rng)[:2]

    def _source(self, chosen: dict[str, str]) -> str:
        parts = [self.header.strip() + "\n"]
        for fn in self.functions:
            parts.append(chosen[fn].strip() + "\n")
        return "\n\n".join(parts)

    def generate(self, rng: random.Random, level: int) -> SoftwareInstance:
        mod = rng.choice(self.module_names)
        fns = list(self.functions)
        for _ in range(60):
            n_bugs = 2 if level == 3 else 1
            buggy_fns = rng.sample([f for f in fns if self.functions[f]["bugs"]], n_bugs)
            bugs = {f: rng.choice(sorted(self.functions[f]["bugs"])) for f in buggy_fns}
            ref_src = self._source({f: self.functions[f]["ref"] for f in fns})
            bug_src = self._source({f: (self.functions[f]["bugs"][bugs[f]] if f in bugs else self.functions[f]["ref"])
                                    for f in fns})
            # expected values come from the reference
            def expect(fn, args):
                res = check_sources({f"{mod}.py": ref_src}, [{"module": mod, "fn": fn, "args": args, "expect": None}])
                return res

            ref_mod: dict = {}
            exec(compile(ref_src, mod, "exec"), ref_mod)
            cases = []
            for fn in fns:
                for args in self.cases_for(fn, rng):
                    cases.append({"module": mod, "fn": fn, "args": args,
                                  "expect": _jsonable(ref_mod[fn](*args)), "tol": 1e-9})
            # every bug must be caught, the reference must pass
            if not all(ok for ok, _ in check_sources({f"{mod}.py": ref_src}, cases)):
                continue
            ok_bug = True
            for f in buggy_fns:
                fcases = [c for c in cases if c["fn"] == f]
                if all(ok for ok, _ in check_sources({f"{mod}.py": bug_src}, fcases)):
                    ok_bug = False
            if not ok_bug:
                continue
            # which bug is shown by a failing visible test, which by ISSUE.md
            visible_bug = buggy_fns[0] if level in (1, 3) else None
            issue_bug = buggy_fns[-1] if level in (2, 3) else None
            tests, issue = self._visible(mod, fns, ref_mod, bug_src, visible_bug, issue_bug, rng)
            if tests is None:
                continue
            files = {f"{mod}.py": bug_src, f"tests/test_{mod}.py": tests}
            if issue:
                files["ISSUE.md"] = issue
            instr = self._instruction(mod, level)
            return SoftwareInstance(instruction=instr, files=files, cases=cases, oracle={f"{mod}.py": ref_src})
        raise RuntimeError(f"{self.name}: could not build a level-{level} instance")

    def _visible(self, mod, fns, ref_mod, bug_src, visible_bug, issue_bug, rng):
        bug_mod: dict = {}
        exec(compile(bug_src, mod, "exec"), bug_mod)
        lines = ["import pytest", "", f"import {mod} as m", ""]
        for fn in fns:
            arg_lists = self.visible_args(fn, rng)
            if fn == visible_bug:
                # pick a visible case the bug actually fails
                arg_lists = [a for a in self.cases_for(fn, rng) if _fails(bug_mod[fn], ref_mod[fn], a)][:1]
                if not arg_lists:
                    return None, None
            elif fn == issue_bug:
                # visible tests must not reveal the reported bug
                arg_lists = [a for a in arg_lists if not _fails(bug_mod[fn], ref_mod[fn], a)][:1]
            else:
                arg_lists = arg_lists[:1]
            for i, args in enumerate(arg_lists):
                want = _jsonable(ref_mod[fn](*args))
                lines += [f"def test_{fn}_{i}():", f"    assert m.{_call_str(fn, args)} == {_approx(want)}", ""]
        issue = ""
        if issue_bug:
            for args in self.cases_for(issue_bug, rng):
                if _fails(bug_mod[issue_bug], ref_mod[issue_bug], args):
                    want = _jsonable(ref_mod[issue_bug](*args))
                    try:
                        got = f"returns `{_jsonable(bug_mod[issue_bug](*args))!r}`"
                    except Exception as e:
                        got = f"raises `{type(e).__name__}`"
                    issue = (f"# Bug report: {issue_bug} is wrong\n\n`{mod}.{_call_str(issue_bug, args)}` {got}, "
                             f"but according to its docstring it should return `{want!r}`.\n")
                    break
            if not issue:
                return None, None
        return "\n".join(lines), issue

    def _instruction(self, mod: str, level: int) -> str:
        if level == 1:
            what = f"`pytest tests/` fails for the `{mod}` module in the current directory. Find and fix the bug in {mod}.py."
        elif level == 2:
            what = (f"A user reported a bug in the `{mod}` module in the current directory (see ISSUE.md); the existing tests "
                    f"in tests/ still pass. Fix {mod}.py.")
        else:
            what = (f"The `{mod}` module in the current directory has two bugs: `pytest tests/` fails, and ISSUE.md "
                    f"reports another problem. Fix both in {mod}.py.")
        return (f"{what} Every function must behave exactly as its docstring describes. Keep the function names and "
                f"signatures, and do not edit the tests. {HIDDEN_NOTE}")


def _jsonable(v):
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def _fails(bug_fn, ref_fn, args) -> bool:
    want = _jsonable(ref_fn(*args))
    try:
        got = _jsonable(bug_fn(*args))
    except Exception:
        return True
    from .runner import _ns
    return not _ns["_same"](want, got, 1e-9)


# ------------------------------------------------------------------------------- stats

class StatsBugfix(ModuleBugfix):
    name = "sw_stats_fix"
    title = "Fix a statistics module"
    module_names = ("stats", "metrics", "numutil")
    header = '"""Small descriptive-statistics helpers used by the reporting jobs."""'
    functions = {
        "mean": {
            "ref": '''
def mean(xs):
    """Arithmetic mean of a non-empty list of numbers."""
    return sum(xs) / len(xs)''',
            "bugs": {"int_div": '''
def mean(xs):
    """Arithmetic mean of a non-empty list of numbers."""
    return sum(xs) // len(xs)'''},
        },
        "median": {
            "ref": '''
def median(xs):
    """Median of a non-empty list; for an even count, the mean of the two middle values. xs is not modified."""
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2''',
            "bugs": {"even_upper": '''
def median(xs):
    """Median of a non-empty list; for an even count, the mean of the two middle values. xs is not modified."""
    s = sorted(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return s[m]''', "unsorted": '''
def median(xs):
    """Median of a non-empty list; for an even count, the mean of the two middle values. xs is not modified."""
    s = list(xs)
    m = len(s) // 2
    if len(s) % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2'''},
        },
        "percentile": {
            "ref": '''
def percentile(xs, p):
    """Linear-interpolation percentile of a non-empty list, p in [0, 100] (numpy's default method)."""
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)''',
            "bugs": {"no_clamp": '''
def percentile(xs, p):
    """Linear-interpolation percentile of a non-empty list, p in [0, 100] (numpy's default method)."""
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1
    return s[f] + (s[c] - s[f]) * (k - f)''', "n_scale": '''
def percentile(xs, p):
    """Linear-interpolation percentile of a non-empty list, p in [0, 100] (numpy's default method)."""
    s = sorted(xs)
    k = min(len(s) * p / 100, len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)'''},
        },
        "moving_average": {
            "ref": '''
def moving_average(xs, window):
    """Trailing moving averages: one value per full window, so len(xs) - window + 1 values."""
    return [sum(xs[i:i + window]) / window for i in range(len(xs) - window + 1)]''',
            "bugs": {"drops_last": '''
def moving_average(xs, window):
    """Trailing moving averages: one value per full window, so len(xs) - window + 1 values."""
    return [sum(xs[i:i + window]) / window for i in range(len(xs) - window)]''', "int_div": '''
def moving_average(xs, window):
    """Trailing moving averages: one value per full window, so len(xs) - window + 1 values."""
    return [sum(xs[i:i + window]) // window for i in range(len(xs) - window + 1)]'''},
        },
        "zscores": {
            "ref": '''
def zscores(xs):
    """(x - mean) / population standard deviation for each x; all 0.0 when the deviation is 0."""
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
    if sd == 0:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]''',
            "bugs": {"sample_sd": '''
def zscores(xs):
    """(x - mean) / population standard deviation for each x; all 0.0 when the deviation is 0."""
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
    if sd == 0:
        return [0.0 for _ in xs]
    return [(x - mu) / sd for x in xs]''', "no_guard": '''
def zscores(xs):
    """(x - mean) / population standard deviation for each x; all 0.0 when the deviation is 0."""
    mu = sum(xs) / len(xs)
    sd = (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
    return [(x - mu) / sd for x in xs]'''},
        },
        "mode": {
            "ref": '''
def mode(xs):
    """Most frequent value of a non-empty list; ties go to the smallest value."""
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return min(x for x, c in counts.items() if c == best)''',
            "bugs": {"tie_largest": '''
def mode(xs):
    """Most frequent value of a non-empty list; ties go to the smallest value."""
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    return max(x for x, c in counts.items() if c == best)''', "first_seen": '''
def mode(xs):
    """Most frequent value of a non-empty list; ties go to the smallest value."""
    counts = {}
    for x in xs:
        counts[x] = counts.get(x, 0) + 1
    best = max(counts.values())
    for x, c in counts.items():
        if c == best:
            return x'''},
        },
    }

    def cases_for(self, fn, rng):
        r = random.Random(rng.random())
        ints = lambda n, lo=1, hi=60: [r.randint(lo, hi) for _ in range(n)]
        if fn == "mean":
            return [[ints(r.randint(3, 7))] for _ in range(3)] + [[[1, 2]], [[7, 8, 10]]]
        if fn == "median":
            return [[ints(5)], [[9, 2, 7, 4]], [ints(6)], [[3, 1, 2]], [[10, 1, 4, 7]]]
        if fn == "percentile":
            xs = ints(r.randint(5, 9))
            return [[xs, 50], [xs, 90], [xs, 100], [ints(4), 25], [xs, 0], [ints(7), 75]]
        if fn == "moving_average":
            return [[ints(r.randint(5, 8)), r.randint(2, 3)], [[1, 2, 4, 7, 11], 2], [ints(6), 4], [[3, 5, 6], 3]]
        if fn == "zscores":
            return [[ints(5)], [[4, 4, 4]], [ints(6, 1, 9)], [[2, 4, 4, 4, 5, 5, 7, 9]]]
        if fn == "mode":
            a = r.randint(2, 5)
            b = a + r.randint(2, 6)
            tie = [b, a, b, a, r.randint(10, 20)]
            r.shuffle(tie)
            tie2 = [9, 3, 9, 3, 1]
            return [[tie], [ints(8, 1, 5)], [tie2], [[5, 1, 5, 2]], [[8, 2, 2, 8, 6]]]
        raise KeyError(fn)


# ------------------------------------------------------------------------------- money

class MoneyBugfix(ModuleBugfix):
    name = "sw_money_fix"
    title = "Fix a money-handling module"
    module_names = ("money", "cents", "ledger_math")
    header = '"""Money helpers for the billing service. Amounts are integer cents."""\n\nfrom decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN'
    functions = {
        "parse_amount": {
            "ref": '''
def parse_amount(text):
    """Parse '$1,234.50', '1234.5', '-$3' or '(12.00)' (parentheses mean negative) into integer cents.
    More than two decimals round half-up to the cent."""
    s = text.strip().replace(",", "").replace("$", "")
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    if s.startswith("-"):
        neg, s = not neg, s[1:]
    cents = int((Decimal(s) * 100).quantize(Decimal(1), ROUND_HALF_UP))
    return -cents if neg else cents''',
            "bugs": {"no_parens": '''
def parse_amount(text):
    """Parse '$1,234.50', '1234.5', '-$3' or '(12.00)' (parentheses mean negative) into integer cents.
    More than two decimals round half-up to the cent."""
    s = text.strip().replace(",", "").replace("$", "").replace("(", "").replace(")", "")
    neg = False
    if s.startswith("-"):
        neg, s = not neg, s[1:]
    cents = int((Decimal(s) * 100).quantize(Decimal(1), ROUND_HALF_UP))
    return -cents if neg else cents''', "truncates": '''
def parse_amount(text):
    """Parse '$1,234.50', '1234.5', '-$3' or '(12.00)' (parentheses mean negative) into integer cents.
    More than two decimals round half-up to the cent."""
    s = text.strip().replace(",", "").replace("$", "")
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    if s.startswith("-"):
        neg, s = not neg, s[1:]
    cents = int((Decimal(s) * 100).quantize(Decimal(1), ROUND_DOWN))
    return -cents if neg else cents'''},
        },
        "format_cents": {
            "ref": '''
def format_cents(cents):
    """Format integer cents: -123456 -> '-$1,234.56', 5 -> '$0.05'."""
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}${c // 100:,}.{c % 100:02d}"''',
            "bugs": {"no_pad": '''
def format_cents(cents):
    """Format integer cents: -123456 -> '-$1,234.56', 5 -> '$0.05'."""
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}${c // 100:,}.{c % 100}"''', "no_commas": '''
def format_cents(cents):
    """Format integer cents: -123456 -> '-$1,234.56', 5 -> '$0.05'."""
    sign = "-" if cents < 0 else ""
    c = abs(cents)
    return f"{sign}${c // 100}.{c % 100:02d}"'''},
        },
        "allocate": {
            "ref": '''
def allocate(total_cents, weights):
    """Split non-negative total_cents across weights (non-negative, sum > 0) in proportion. Each share is
    floored, then the leftover cents go one at a time to the largest remainders (ties: lower index first),
    so the shares always sum to total_cents."""
    tw = sum(weights)
    base = [total_cents * w // tw for w in weights]
    rem = [total_cents * w % tw for w in weights]
    left = total_cents - sum(base)
    for i in sorted(range(len(weights)), key=lambda i: (-rem[i], i))[:left]:
        base[i] += 1
    return base''',
            "bugs": {"largest_weight": '''
def allocate(total_cents, weights):
    """Split non-negative total_cents across weights (non-negative, sum > 0) in proportion. Each share is
    floored, then the leftover cents go one at a time to the largest remainders (ties: lower index first),
    so the shares always sum to total_cents."""
    tw = sum(weights)
    base = [total_cents * w // tw for w in weights]
    left = total_cents - sum(base)
    for i in sorted(range(len(weights)), key=lambda i: (-weights[i], i))[:left]:
        base[i] += 1
    return base''', "tie_high": '''
def allocate(total_cents, weights):
    """Split non-negative total_cents across weights (non-negative, sum > 0) in proportion. Each share is
    floored, then the leftover cents go one at a time to the largest remainders (ties: lower index first),
    so the shares always sum to total_cents."""
    tw = sum(weights)
    base = [total_cents * w // tw for w in weights]
    rem = [total_cents * w % tw for w in weights]
    left = total_cents - sum(base)
    for i in sorted(range(len(weights)), key=lambda i: (-rem[i], -i))[:left]:
        base[i] += 1
    return base'''},
        },
        "apply_rate": {
            "ref": '''
def apply_rate(cents, rate_bps):
    """cents * rate_bps / 10000, rounded half-up to a whole cent (half away from zero for negatives)."""
    q, r = divmod(abs(cents) * rate_bps, 10000)
    if r * 2 >= 10000:
        q += 1
    return q if cents >= 0 else -q''',
            "bugs": {"truncates": '''
def apply_rate(cents, rate_bps):
    """cents * rate_bps / 10000, rounded half-up to a whole cent (half away from zero for negatives)."""
    q, r = divmod(abs(cents) * rate_bps, 10000)
    return q if cents >= 0 else -q''', "half_down": '''
def apply_rate(cents, rate_bps):
    """cents * rate_bps / 10000, rounded half-up to a whole cent (half away from zero for negatives)."""
    q, r = divmod(abs(cents) * rate_bps, 10000)
    if r * 2 > 10000:
        q += 1
    return q if cents >= 0 else -q'''},
        },
        "compound": {
            "ref": '''
def compound(cents, annual_rate_bps, years):
    """Balance after compounding once a year, rounding half-up to the cent at the end of every year."""
    for _ in range(years):
        cents += apply_rate(cents, annual_rate_bps)
    return cents''',
            "bugs": {"round_at_end": '''
def compound(cents, annual_rate_bps, years):
    """Balance after compounding once a year, rounding half-up to the cent at the end of every year."""
    return int(Decimal(cents * (1 + annual_rate_bps / 10000) ** years).quantize(Decimal(1), ROUND_HALF_UP))'''},
        },
    }

    def cases_for(self, fn, rng):
        r = random.Random(rng.random())
        if fn == "parse_amount":
            a = r.randint(1000, 999999)
            return [[f"${a // 100:,}.{a % 100:02d}"], [f"({r.randint(1, 900)}.{r.randint(10, 99)})"], ["12.345"],
                    [f"-${r.randint(1, 99)}"], ["(1,000.00)"], [f"{r.randint(1, 99)}.{r.randint(1, 9)}"], ["0.125"]]
        if fn == "format_cents":
            return [[r.randint(100000, 9999999)], [r.randint(1, 9)], [-r.randint(100000, 999999)], [1205], [r.randint(100, 999) * 100 + 7]]
        if fn == "allocate":
            return [[r.randint(100, 999), [1, 1, 1]], [100, [1, 1, 1]], [r.randint(1000, 5000), [r.randint(1, 9) for _ in range(4)]],
                    [10, [3, 3, 1]], [7, [1, 2, 2, 2]], [5, [1, 1, 1, 1, 1, 1]]]
        if fn == "apply_rate":
            return [[r.randint(1000, 99999), r.randint(50, 900)], [150, 5000], [-150, 5000], [50, 100], [999, 1250],
                    [r.randint(1000, 9999), 333]]
        if fn == "compound":
            return [[r.randint(10000, 99999), r.randint(100, 900), r.randint(2, 6)], [10001, 333, 7],
                    [99999, 125, 10], [r.randint(1000, 9999), 5000, 4]]
        raise KeyError(fn)


# ------------------------------------------------------------------------------- dates

class DatesBugfix(ModuleBugfix):
    name = "sw_dates_fix"
    title = "Fix a date-utilities module"
    module_names = ("dates", "calendar_utils", "billing_dates")
    header = '"""Date helpers for invoicing. Dates are ISO strings (YYYY-MM-DD)."""\n\nimport calendar\nfrom datetime import date, timedelta'
    functions = {
        "add_months": {
            "ref": '''
def add_months(day, n):
    """Same day-of-month n months later (n may be negative), clamped to the last day of the target month."""
    d = date.fromisoformat(day)
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    last = calendar.monthrange(y, m)[1]
    return date(y, m, min(d.day, last)).isoformat()''',
            "bugs": {"no_clamp": '''
def add_months(day, n):
    """Same day-of-month n months later (n may be negative), clamped to the last day of the target month."""
    d = date.fromisoformat(day)
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, d.day).isoformat()''', "thirty_days": '''
def add_months(day, n):
    """Same day-of-month n months later (n may be negative), clamped to the last day of the target month."""
    return (date.fromisoformat(day) + timedelta(days=30 * n)).isoformat()'''},
        },
        "month_end": {
            "ref": '''
def month_end(day):
    """Last day of the month containing day."""
    d = date.fromisoformat(day)
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1]).isoformat()''',
            "bugs": {"no_leap": '''
def month_end(day):
    """Last day of the month containing day."""
    d = date.fromisoformat(day)
    days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(d.year, d.month, days[d.month - 1]).isoformat()'''},
        },
        "business_days": {
            "ref": '''
def business_days(start, end):
    """Number of Monday-Friday days d with start <= d < end (0 if end <= start)."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    n = 0
    while a < b:
        if a.weekday() < 5:
            n += 1
        a += timedelta(days=1)
    return n''',
            "bugs": {"inclusive": '''
def business_days(start, end):
    """Number of Monday-Friday days d with start <= d < end (0 if end <= start)."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    n = 0
    while a <= b:
        if a.weekday() < 5:
            n += 1
        a += timedelta(days=1)
    return n''', "weekend_wrong": '''
def business_days(start, end):
    """Number of Monday-Friday days d with start <= d < end (0 if end <= start)."""
    a, b = date.fromisoformat(start), date.fromisoformat(end)
    n = 0
    while a < b:
        if a.weekday() < 6:
            n += 1
        a += timedelta(days=1)
    return n'''},
        },
        "iso_week": {
            "ref": '''
def iso_week(day):
    """ISO-8601 week label, e.g. '2026-W01' (the ISO year can differ from the calendar year)."""
    y, w, _ = date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"''',
            "bugs": {"calendar_year": '''
def iso_week(day):
    """ISO-8601 week label, e.g. '2026-W01' (the ISO year can differ from the calendar year)."""
    d = date.fromisoformat(day)
    return f"{d.year}-W{d.isocalendar()[1]:02d}"''', "sunday_weeks": '''
def iso_week(day):
    """ISO-8601 week label, e.g. '2026-W01' (the ISO year can differ from the calendar year)."""
    d = date.fromisoformat(day)
    return f"{d.year}-W{int(d.strftime('%U')):02d}"'''},
        },
        "days_late": {
            "ref": '''
def days_late(due, paid):
    """Calendar days a payment was late: paid - due in days, never below 0."""
    return max(0, (date.fromisoformat(paid) - date.fromisoformat(due)).days)''',
            "bugs": {"negative": '''
def days_late(due, paid):
    """Calendar days a payment was late: paid - due in days, never below 0."""
    return (date.fromisoformat(paid) - date.fromisoformat(due)).days'''},
        },
    }

    def cases_for(self, fn, rng):
        r = random.Random(rng.random())
        from datetime import date as _d, timedelta as _td
        rd = lambda: (_d(2024, 1, 1) + _td(days=r.randrange(900))).isoformat()
        if fn == "add_months":
            return [[rd(), r.randint(1, 14)], ["2025-01-31", 1], ["2024-01-31", 1], ["2026-03-31", -1], [rd(), -r.randint(1, 9)],
                    ["2026-05-15", 3]]
        if fn == "month_end":
            return [["2024-02-10"], [rd()], ["2025-02-03"], ["2026-04-30"], ["2028-02-01"]]
        if fn == "business_days":
            a = rd()
            b = (_d.fromisoformat(a) + _td(days=r.randint(3, 40))).isoformat()
            return [[a, b], ["2026-10-05", "2026-10-12"], ["2026-10-12", "2026-10-05"], ["2026-10-09", "2026-10-13"],
                    ["2026-10-10", "2026-10-11"]]
        if fn == "iso_week":
            return [[rd()], ["2026-01-01"], ["2024-12-30"], ["2027-01-03"], ["2026-06-15"]]
        if fn == "days_late":
            return [[rd(), rd()], ["2026-03-01", "2026-03-15"], ["2026-03-15", "2026-03-01"], ["2026-02-27", "2026-03-02"]]
        raise KeyError(fn)


# =============================================================================== implement to a spec

_LOG_LEVELS = ("DEBUG", "INFO", "WARN", "ERROR")
_LOG_PATHS = ("/api/orders", "/api/users", "/api/cart", "/health", "/api/search", "/api/pay")

_LOG_REF = '''
"""Summarize request logs."""

import math
import re
from collections import Counter

_LEGACY = re.compile(r"^\\[(\\d\\d:\\d\\d:\\d\\d)\\] (DEBUG|INFO|WARN|ERROR) (GET|POST|PUT|DELETE) (\\S+) -> (\\d{3}) \\((\\d+)ms\\)$")


def _parse(line, legacy):
    parts = line.split()
    if parts and parts[0].endswith("Z") and "T" in parts[0]:
        kv = {}
        for tok in parts[1:]:
            k, sep, v = tok.partition("=")
            if not sep:
                return None
            kv[k] = v
        if not {"level", "path", "status", "ms"} <= kv.keys():
            return None
        if not (kv["status"].isdigit() and kv["ms"].isdigit()):
            return None
        return kv["level"], kv["path"], int(kv["status"]), int(kv["ms"])
    if legacy:
        m = _LEGACY.match(line)
        if m:
            return m.group(2), m.group(4), int(m.group(5)), int(m.group(6))
    return None


def summarize(path):
    total = skipped = errors = 0
    levels, paths, ms = Counter(), Counter(), []
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            rec = _parse(line, LEGACY)
            if rec is None:
                skipped += 1
                continue
            level, p, status, t = rec
            total += 1
            levels[level] += 1
            paths[p] += 1
            ms.append(t)
            if status >= 500:
                errors += 1
    s = sorted(ms)
    return {
        "total": total,
        "skipped": skipped,
        "by_level": dict(levels),
        "error_rate": round(errors / total, 4) if total else 0.0,
        "p95_ms": s[math.ceil(0.95 * len(s)) - 1] if s else 0,
        "top_path": min(paths, key=lambda q: (-paths[q], q)) if paths else "",
    }
'''

_LOG_DOC = '''"""Summarize request logs.

Each line of a log looks like

    2026-10-03T14:22:05Z level=INFO path=/api/orders status=200 ms=123

i.e. an ISO timestamp ending in Z, then space-separated key=value fields. {order_note}Blank lines are
ignored. {skip_note}

summarize(path) reads the log file at `path` and returns a dict:

    total       number of request lines parsed
    skipped     number of non-blank lines that could not be parsed{skip_zero}
    by_level    {{level: count}} for the levels that occur
    error_rate  share of parsed requests with status >= 500, rounded to 4 decimals (0.0 if total is 0)
    p95_ms      nearest-rank 95th percentile of ms: the ceil(0.95 * n)-th smallest value (0 if total is 0)
    top_path    the most frequent path; ties go to the alphabetically smallest ("" if total is 0)
"""


def summarize(path):
    raise NotImplementedError
'''


class LogSummarizer:
    name = "sw_logstats_impl"
    title = "Implement a log summarizer"
    domain = "software engineering"
    levels = (1, 2, 3)

    def _log(self, r: random.Random, level: int, n: int) -> str:
        lines = []
        for _ in range(n):
            hh, mm, ss = r.randint(0, 23), r.randint(0, 59), r.randint(0, 59)
            lvl = r.choices(_LOG_LEVELS, weights=(1, 6, 2, 1))[0]
            p = r.choice(_LOG_PATHS)
            status = r.choices((200, 201, 404, 500, 503), weights=(10, 2, 2, 1, 1))[0]
            t = r.randint(3, 900)
            fields = [f"level={lvl}", f"path={p}", f"status={status}", f"ms={t}"]
            if level >= 2:
                r.shuffle(fields)
            if level >= 3 and r.random() < 0.3:
                verb = r.choice(("GET", "POST", "PUT", "DELETE"))
                lines.append(f"[{hh:02d}:{mm:02d}:{ss:02d}] {lvl} {verb} {p} -> {status} ({t}ms)")
            else:
                lines.append(f"2026-10-{r.randint(1, 28):02d}T{hh:02d}:{mm:02d}:{ss:02d}Z " + " ".join(fields))
            if level >= 2 and r.random() < 0.12:
                lines.append(r.choice([
                    f"2026-10-03T{hh:02d}:00:00Z level=WARN path={p}",
                    f"2026-10-03T{hh:02d}:00:01Z level=INFO path={p} status=ok ms=12",
                    "--- log rotated ---",
                    f"2026-10-03T{hh:02d}:00:02Z level=INFO path={p} status=200 ms=-4",
                ]))
            if level >= 2 and r.random() < 0.05:
                lines.append("")
        return "\n".join(lines) + "\n"

    def generate(self, rng: random.Random, level: int) -> SoftwareInstance:
        mod = rng.choice(("logstats", "access_summary", "reqlog"))
        ref = _LOG_REF.replace("LEGACY)", f"{level >= 3})")
        doc = _LOG_DOC.format(
            order_note="The fields can appear in any order and every field is required. " if level >= 2 else "",
            skip_note=("Any other line (missing a field, a non-integer status or ms, or not a log line at all) is "
                       "skipped and counted." if level >= 2 else "Every non-blank line is a valid log line."),
            skip_zero="" if level >= 2 else " (always 0 for valid logs)",
        )
        r = random.Random(rng.random())
        sample = self._log(r, level, r.randint(25, 40))
        cases = []
        for k in range(4):
            name = f"h{k}.log"
            cases.append({"module": mod, "fn": "summarize", "files_in": {name: self._log(r, level, r.randint(30, 80))},
                          "args": [f"@{name}"]})
        # expectations from the reference
        ns: dict = {}
        exec(compile(ref, mod, "exec"), ns)
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for c in cases:
                (name, text), = c["files_in"].items()
                with open(os.path.join(d, name), "w") as f:
                    f.write(text)
                c["expect"] = ns["summarize"](os.path.join(d, name))
            with open(os.path.join(d, "sample.log"), "w") as f:
                f.write(sample)
            sample_out = ns["summarize"](os.path.join(d, "sample.log"))
        tests = (f"import json\n\nimport {mod}\n\n\ndef test_sample():\n"
                 f"    assert {mod}.summarize('sample.log') == json.loads({__import__('json').dumps(sample_out)!r})\n")
        files = {f"{mod}.py": doc, "sample.log": sample, f"tests/test_{mod}.py": tests}
        if level >= 3:
            files["NOTICE.md"] = (
                "# Notice from the platform team\n\nOlder services still write the legacy format, and those lines "
                "must be parsed too, not skipped:\n\n    [14:22:05] INFO GET /api/orders -> 200 (123ms)\n\n"
                "i.e. [HH:MM:SS] LEVEL METHOD PATH -> STATUS (MSms). Count them exactly like the new format.\n")
        instr = (f"Implement `summarize(path)` in {mod}.py in the current directory. The docstring at the top of {mod}.py is the "
                 f"specification{' (NOTICE.md amends it)' if level >= 3 else ''}; sample.log is an example input and "
                 f"tests/ has one example test. Keep the function name and signature. Hidden tests run summarize on "
                 f"other logs.")
        oracle_src = doc.split('def summarize(path):')[0].rstrip() + "\n" + ref.split('"""Summarize request logs."""', 1)[1]
        return SoftwareInstance(instruction=instr, files=files, cases=cases, oracle={f"{mod}.py": oracle_src})


# =============================================================================== repair a report script

_CATS = ("Grocery", "Fuel", "Software", "Travel", "Office", "Meals")


def _etl_source(bugs: set, void_rule: bool) -> str:
    strip = "" if "no_strip" in bugs else ".strip()"
    refund = "" if "refund_sign" in bugs else (
        "            if row[\"type\"].strip().lower() == \"refund\":\n                amount = -amount\n")
    void = ("            if row.get(\"status\", \"\").strip().lower() == \"void\":\n                continue\n"
            if void_rule else "")
    count_inc = "(1 if row[\"type\"].strip().lower() == \"sale\" else 0)" if "count_sales" in bugs else "1"
    sort_key = "(kv[0][0], kv[1][0], kv[0][1])" if "sort_asc" in bugs else "(kv[0][0], -kv[1][0], kv[0][1])"
    month = "row[\"date\"][:4] + \"-\" + row[\"date\"][8:10]" if "month_day" in bugs else "row[\"date\"][:7]"
    return f'''"""Monthly net sales by category. See README.md for the expected output."""

import csv
import os
from decimal import Decimal


def build(in_path, out_path):
    groups = {{}}
    with open(in_path, newline="") as f:
        for row in csv.DictReader(f):
{void}            month = {month}
            amount = Decimal(row["amount"])
{refund}            key = (month, row["category"]{strip})
            net, count = groups.get(key, (Decimal(0), 0))
            groups[key] = (net + amount, count + {count_inc})
    rows = sorted(groups.items(), key=lambda kv: {sort_key})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\\n")
        w.writerow(["month", "category", "net", "count"])
        for (month, category), (net, count) in rows:
            w.writerow([month, category, f"{{net:.2f}}", count])


if __name__ == "__main__":
    build("transactions.csv", os.path.join("out", "summary.csv"))
'''


_ETL_README = """# Monthly category report

`python report.py` reads `transactions.csv` and writes `out/summary.csv` (creating `out/` if needed).
`build(in_path, out_path)` does the work and is also called directly by other jobs.

Input columns: `date` (YYYY-MM-DD), `category`, `amount` (positive decimal), `type` (`sale` or `refund`){status_col}.

Output: header `month,category,net,count`, one row per (month, category) where

- `month` is YYYY-MM of the transaction date,
- `category` is the category with surrounding whitespace removed,
- `net` is the sum of sale amounts minus the sum of refund amounts, with exactly 2 decimals,
- `count` is the number of transactions (sales and refunds) in the group;

rows sorted by month ascending, then net descending, then category ascending.
"""

_ETL_BUGS = ("refund_sign", "no_strip", "count_sales", "sort_asc", "month_day")


class ReportRepair:
    name = "sw_report_fix"
    title = "Repair a report script"
    domain = "software engineering"
    levels = (1, 2, 3)

    def _data(self, r: random.Random, level: int, n: int) -> str:
        rows = ["date,category,amount,type" + (",status" if level >= 3 else "")]
        months = r.sample(range(1, 13), 3)
        for _ in range(n):
            m = r.choice(months)
            d = r.randint(13, 28)  # day > 12 so a month/day mix-up shows
            cat = r.choice(_CATS[:4])
            cat = cat + (" " if r.random() < 0.25 else "")
            typ = "refund" if r.random() < 0.25 else "sale"
            line = f"2026-{m:02d}-{d:02d},{cat},{r.randint(100, 50000) / 100:.2f},{typ}"
            if level >= 3:
                line += "," + ("void" if r.random() < 0.15 else "posted")
            rows.append(line)
        return "\n".join(rows) + "\n"

    def generate(self, rng: random.Random, level: int) -> SoftwareInstance:
        for _ in range(60):
            nb = {1: 1, 2: 2, 3: 2}[level]
            bugs = set(rng.sample(_ETL_BUGS, nb))
            void_rule = level >= 3
            ref = _etl_source(set(), void_rule)
            shipped = _etl_source(bugs, False)
            r = random.Random(rng.random())
            data = self._data(r, level, r.randint(18, 30))
            hidden = [self._data(r, level, r.randint(25, 45)) for _ in range(3)]
            ns: dict = {}
            exec(compile(ref, "report", "exec"), ns)
            import os
            import tempfile
            cases = []
            with tempfile.TemporaryDirectory() as d:
                def run_ref(text):
                    with open(os.path.join(d, "in.csv"), "w") as f:
                        f.write(text)
                    ns["build"](os.path.join(d, "in.csv"), os.path.join(d, "o.csv"))
                    return open(os.path.join(d, "o.csv")).read()
                expected_main = run_ref(data)
                for text in hidden:
                    cases.append({"module": "report", "fn": "build", "files_in": {"in.csv": text},
                                  "args": ["@in.csv", "@out.csv"], "expect_file": ["@out.csv", run_ref(text)]})
            if not all(ok for ok, _ in check_sources({"report.py": ref}, cases)):
                continue
            if all(ok for ok, _ in check_sources({"report.py": shipped}, cases)):
                continue
            readme = _ETL_README.format(status_col=", `status` (`posted` or `void`)" if level >= 3 else "")
            files = {"report.py": shipped, "README.md": readme, "transactions.csv": data}
            if level >= 3:
                files["NOTICE.md"] = ("# Finance ops notice\n\nVoided transactions (`status` = `void`) must be left out of "
                                      "the report entirely: they count toward neither `net` nor `count`.\n")
            instr = ("report.py in the current directory produces out/summary.csv from transactions.csv, but the numbers "
                     "are wrong. Fix report.py so it follows README.md"
                     + (" and NOTICE.md" if level >= 3 else "")
                     + ", then run it to regenerate out/summary.csv. Keep `build(in_path, out_path)`; hidden tests also "
                       "call it on other inputs.")
            return SoftwareInstance(instruction=instr, files=files, cases=cases, oracle={"report.py": ref, "out/summary.csv": expected_main},
                                    file_checks=[("out/summary.csv", expected_main)])
        raise RuntimeError(f"{self.name}: could not build a level-{level} instance")


FAMILIES = [StatsBugfix(), MoneyBugfix(), DatesBugfix(), LogSummarizer(), ReportRepair()]
