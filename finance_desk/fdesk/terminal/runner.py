"""Hidden-test runner for software tasks.

RUNNER_SRC is a self-contained script (no fdesk imports: the agent's user cannot read the server's code). At
evaluate time the server writes it plus a cases file into a private temp dir and runs it as the agent's user,
with the workdir on sys.path, so the agent's module is imported exactly as the agent left it. The same source is
exec'd in-process at generation time to prove the shipped (buggy/stub) code fails and the reference passes.

A case is {"module": "stats", "fn": "median", "args": [...], "kwargs": {...}, "expect": <json>, "tol": 1e-9}
or, for file-producing functions, {"module": "report", "fn": "build", "files_in": {"in.csv": "..."},
"args": ["@in.csv", "@out.csv"], "expect_file": ["@out.csv", "<expected text>"]} where "@name" is a path in a
fresh temp dir. "expect": {"raises": true} is not used: every case expects a value.
"""

RUNNER_SRC = r'''
import importlib, json, math, os, sys, tempfile, traceback

def _same(a, b, tol):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b if isinstance(a, bool) and isinstance(b, bool) else False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if isinstance(b, float) and (math.isnan(b) or math.isinf(b)):
            return False
        return abs(a - b) <= tol * max(1.0, abs(a))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(map(str, a)) == set(map(str, b)) and all(
            _same(a[k], {str(kk): vv for kk, vv in b.items()}[str(k)], tol) for k in a)
    return a == b

def _norm_text(t):
    return "\n".join(line.rstrip() for line in t.replace("\r\n", "\n").strip().split("\n"))

def run_cases(load, cases):
    """load(module_name) -> module. Returns list of (ok, message)."""
    results = []
    for c in cases:
        tmp = tempfile.mkdtemp(prefix="case_")
        sub = lambda v: os.path.join(tmp, v[1:]) if isinstance(v, str) and v.startswith("@") else v
        try:
            for name, text in (c.get("files_in") or {}).items():
                with open(os.path.join(tmp, name), "w") as f:
                    f.write(text)
            mod = load(c["module"])
            fn = getattr(mod, c["fn"])
            args = [sub(a) for a in c.get("args", [])]
            got = fn(*args, **(c.get("kwargs") or {}))
            if "expect_file" in c:
                path, want = c["expect_file"]
                path = sub(path)
                if not os.path.exists(path):
                    results.append((False, f"{c['fn']} did not write {os.path.basename(path)}"))
                    continue
                ok = _norm_text(open(path).read()) == _norm_text(want)
                results.append((ok, "" if ok else f"{c['fn']}: output file differs"))
            else:
                got = json.loads(json.dumps(got))  # tuples -> lists, like the expectation
                ok = _same(c["expect"], got, c.get("tol", 1e-9))
                results.append((ok, "" if ok else f"{c['fn']}{tuple(c.get('args', []))!r:.120} -> {got!r:.120}"))
        except Exception as e:
            results.append((False, f"{c['fn']}: {type(e).__name__}: {e}"[:200]))
    return results

if __name__ == "__main__":
    workdir, cases_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, workdir)
    os.chdir(workdir)
    cases = json.load(open(cases_path))
    def load(name):
        if name in sys.modules:
            return sys.modules[name]
        return importlib.import_module(name)
    res = run_cases(load, cases)
    print("@@RESULT@@" + json.dumps({"passed": sum(ok for ok, _ in res), "total": len(res),
                                     "failures": [m for ok, m in res if not ok][:5]}))
'''

_ns: dict = {}
exec(compile(RUNNER_SRC, "runner", "exec"), _ns)
run_cases = _ns["run_cases"]


def check_sources(sources: dict[str, str], cases: list[dict]) -> list[tuple[bool, str]]:
    """In-process: exec each module source in a fresh namespace and run the cases against it."""
    import types

    mods: dict[str, types.ModuleType] = {}

    def load(name):
        if name not in mods:
            m = types.ModuleType(name)
            exec(compile(sources[name + ".py"], name + ".py", "exec"), m.__dict__)
            mods[name] = m
        return mods[name]

    return run_cases(load, cases)
