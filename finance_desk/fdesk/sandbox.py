"""Run agent Python in the episode workspace as an unprivileged user with time and memory limits."""

import os
import signal
import subprocess

NOBODY = 65534
TIMEOUT_S = 10
MAX_OUTPUT = 2000

# Runs inside the child before the agent's code: resource limits, then the code itself.
PRELUDE = (
    "import resource as _r\n"
    "_r.setrlimit(_r.RLIMIT_AS, (1 << 30, 1 << 30))\n"
    "_r.setrlimit(_r.RLIMIT_CPU, (12, 12))\n"
    "_r.setrlimit(_r.RLIMIT_FSIZE, (1 << 24, 1 << 24))\n"
    "_r.setrlimit(_r.RLIMIT_NPROC, (64, 64))\n"
    "del _r\n"
    "exec(compile(open('main.py').read(), 'main.py', 'exec'))\n"
)


def prepare_workspace(path: str) -> None:
    if os.geteuid() == 0:
        os.chown(path, NOBODY, NOBODY)
        for name in os.listdir(path):
            os.chown(os.path.join(path, name), NOBODY, NOBODY)


def run_python(workspace: str, code: str) -> str:
    main = os.path.join(workspace, "main.py")
    with open(main, "w") as f:
        f.write(code)
    kwargs = {}
    if os.geteuid() == 0:
        os.chown(main, NOBODY, NOBODY)
        kwargs = {"user": NOBODY, "group": NOBODY, "extra_groups": []}
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": workspace, "PYTHONDONTWRITEBYTECODE": "1",
           "MPLCONFIGDIR": workspace, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    try:
        proc = subprocess.Popen(
            ["python3", "-I", "-c", PRELUDE], cwd=workspace, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True, **kwargs,
        )
    except OSError as e:
        return f"error: could not start python: {e}"
    try:
        out, _ = proc.communicate(timeout=TIMEOUT_S)
        status = "" if proc.returncode == 0 else f"\n[exit code {proc.returncode}]"
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, _ = proc.communicate()
        status = f"\n[killed after {TIMEOUT_S}s]"
    finally:
        _kill(proc)
    text = out.decode("utf-8", "replace")
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n[output truncated, {len(text)} chars]"
    return (text + status) if (text or status) else "(no output; use print)"


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
