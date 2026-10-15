"""Run the agent's shell commands in the episode workdir with time, memory and output limits.

Commands run as `nobody` when the server is root and the sandbox allows switching users. Some arena nodes drop
SETUID/SETGID (board #212/#213): there every switched exec fails, so we probe once and fall back to the server's
own user rather than failing every call.
"""

import os
import signal
import subprocess

NOBODY = 65534
CMD_TIMEOUT_S = 60
MAX_OUTPUT = 6000  # head + tail kept; long dumps cost the episode its token budget anyway

# rlimits, then exec bash; runs as the child's first program so limits apply to everything the command starts.
_PRELUDE = (
    "import os,resource as r,sys\n"
    "r.setrlimit(r.RLIMIT_AS,(2<<30,2<<30))\n"
    "r.setrlimit(r.RLIMIT_CPU,(90,90))\n"
    "r.setrlimit(r.RLIMIT_FSIZE,(64<<20,64<<20))\n"
    "r.setrlimit(r.RLIMIT_NPROC,(256,256))\n"
    "os.execvp('bash',['bash','-c',sys.argv[1]])\n"
)

_switch_ok: bool | None = None


def _can_switch_user() -> bool:
    global _switch_ok
    if _switch_ok is None:
        _switch_ok = False
        if os.geteuid() == 0:
            try:
                subprocess.run(["true"], user=NOBODY, group=NOBODY, extra_groups=[], timeout=10, check=True)
                _switch_ok = True
            except Exception:
                _switch_ok = False
    return _switch_ok


def agent_user_kwargs() -> dict:
    return {"user": NOBODY, "group": NOBODY, "extra_groups": []} if _can_switch_user() else {}


def give_to_agent(path: str) -> None:
    """Make a workdir tree writable by the user commands run as."""
    if not _can_switch_user():
        return
    for root, dirs, files in os.walk(path):
        os.chown(root, NOBODY, NOBODY)
        for n in files:
            os.chown(os.path.join(root, n), NOBODY, NOBODY)


def agent_env(workdir: str) -> dict:
    home = os.path.join(workdir, ".home") if not _can_switch_user() else "/tmp"
    path = os.environ.get("FDESK_AGENT_PATH", "/usr/local/bin:/usr/bin:/bin")
    return {"PATH": path, "HOME": home, "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": "/tmp", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "LANG": "C.UTF-8",
            "TERM": "dumb", "PYTHONPATH": workdir}


def run(workdir: str, command: str, timeout_s: float = CMD_TIMEOUT_S) -> tuple[str, int | None]:
    """Run one shell command; returns (combined output, exit code or None if killed)."""
    try:
        proc = subprocess.Popen(
            ["python3", "-I", "-c", _PRELUDE, command], cwd=workdir, env=agent_env(workdir),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
            **agent_user_kwargs(),
        )
    except OSError as e:
        return f"error: could not start shell: {e}", 127
    code: int | None
    try:
        out, _ = proc.communicate(timeout=timeout_s)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        _kill(proc)
        out, _ = proc.communicate()
        out += f"\n[command killed after {int(timeout_s)}s]".encode()
        code = None
    finally:
        _kill(proc)
    return clip(out.decode("utf-8", "replace")), code


def clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    head, tail = text[: limit * 2 // 3], text[-limit // 3:]
    return f"{head}\n[... {len(text) - len(head) - len(tail)} chars omitted ...]\n{tail}"


def _kill(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
