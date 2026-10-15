"""Finance Desk v3, terminal mode: the tbench2_env contract (exec / write_file / evaluate) over procedural tasks.

Each reset builds a fresh workdir (/app in the image) with the task's files. `exec` runs a shell command there,
`write_file` writes a file, `evaluate` grades the workdir and ends the episode with reward 1 or 0. Like
tbench2_env, every observation carries the instruction. An episode the arena ends without `evaluate` (a
{"finish": true} reply or a spent budget) is scored through finish_action, which is `evaluate`.
"""

import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from . import shell
from .models import Tbench2Action, Tbench2Observation, Tbench2State
from .runner import RUNNER_SRC
from .tasks import grade_answer_file, parse_task_id, task_ids

MAX_STEPS = 40          # the arena's tool_calls_total (32) ends episodes first; this is a server-side backstop
SEED_WINDOW_S = 600     # unseeded resets in one 10-minute window share an instance (one GRPO group, one problem)
VERIFY_TIMEOUT_S = 60

_WORKDIR = os.environ.get("FDESK_WORKDIR", "").strip()
_lock = threading.Lock()
_in_use: set[str] = set()


def _claim_workdir() -> str:
    """The configured workdir (/app in the image) for the first live session; a temp dir for any other."""
    with _lock:
        if _WORKDIR and _WORKDIR not in _in_use:
            os.makedirs(_WORKDIR, exist_ok=True)
            for n in os.listdir(_WORKDIR):
                p = os.path.join(_WORKDIR, n)
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) and not os.path.islink(p) else os.unlink(p)
            _in_use.add(_WORKDIR)
            return _WORKDIR
    return tempfile.mkdtemp(prefix="app_")


def _release_workdir(path: str) -> None:
    with _lock:
        if path == _WORKDIR:
            _in_use.discard(path)
            return
    shutil.rmtree(path, ignore_errors=True)


class TerminalDeskEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = Tbench2State(episode_id=str(uuid4()), step_count=0)
        self._workdir: Optional[str] = None
        self._inst = None
        self._instruction = ""
        self._done = True

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None,
              **kwargs: Any) -> Tbench2Observation:
        task_id = task_id or kwargs.get("task_name") or kwargs.get("task") or task_ids()[0]
        task, level = parse_task_id(task_id)
        if seed is None:
            seed = int(time.time() // SEED_WINDOW_S)
        self._cleanup()
        self._inst = task.generate(random.Random(f"{task_id}:{seed}"), level)
        self._workdir = _claim_workdir()
        for rel, text in self._inst.files.items():
            path = os.path.join(self._workdir, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
        shell.give_to_agent(self._workdir)
        self._instruction = self._inst.instruction
        self._done = False
        self._state = Tbench2State(episode_id=episode_id or str(uuid4()), step_count=0, task_id=task_id,
                                   task_path=self._workdir, session_id="tb2-session", terminal_ready=True)
        return self._obs(output="", action_type="reset", reward=0.0, done=False)

    def step(self, action: Tbench2Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Tbench2Observation:  # type: ignore[override]
        if self._done or self._inst is None:
            return self._obs(output="", error="Episode is over; call reset.", success=False, action_type=action.action_type,
                             reward=0.0, done=True)
        self._state.step_count += 1
        kind = (action.action_type or "exec").strip().lower()
        self._state.last_action_type = kind
        self._state.last_command = action.command
        if kind in ("evaluate", "close"):
            ok, msg = self._evaluate()
            return self._finish(msg, 1.0 if ok else 0.0, kind)
        output, error, success = "", "", True
        if kind == "exec":
            if not action.command.strip():
                success, error = False, "exec needs a command"
            else:
                output, code = shell.run(self._workdir, action.command)
                if code not in (0, None):
                    output += f"\n[exit code {code}]"
        elif kind == "write_file":
            output, error, success = self._write_file(action.file_path, action.content)
        elif kind in ("write", "view", "wait", "kill"):
            success, error = False, "interactive sessions are not available here; use exec with a command that finishes"
        else:
            success, error = False, f"Unsupported action_type: {action.action_type}"
        self._state.last_output = output
        if self._state.step_count >= MAX_STEPS:
            ok, msg = self._evaluate()
            return self._finish(output + "\n[step limit reached; evaluated]", 1.0 if ok else 0.0, kind)
        return self._obs(output=output, error=error, success=success, action_type=kind, reward=None, done=False)

    # ------------------------------------------------------------------ actions

    def _write_file(self, file_path: str, content: str) -> tuple[str, str, bool]:
        if not file_path.strip():
            return "", "write_file needs file_path", False
        p = file_path.strip()
        for prefix in ("/app/", "./"):
            if p.startswith(prefix):
                p = p[len(prefix):]
        target = os.path.realpath(os.path.join(self._workdir, p))
        root = os.path.realpath(self._workdir)
        if not (target == root or target.startswith(root + os.sep)):
            return "", f"file_path must be inside the working directory ({self._display_dir()})", False
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w") as f:
                f.write(content)
            shell.give_to_agent(os.path.dirname(target))
        except OSError as e:
            return "", str(e), False
        return f"Wrote content to {file_path}", "", True

    def _evaluate(self) -> tuple[bool, str]:
        inst = self._inst
        if inst.answer is not None:
            return grade_answer_file(inst, self._workdir)
        for rel, want in inst.file_checks:
            path = os.path.join(self._workdir, rel)
            if not os.path.exists(path):
                return False, f"FAILED: {rel} was not produced"
            norm = lambda t: "\n".join(x.rstrip() for x in t.replace("\r\n", "\n").strip().split("\n"))
            if norm(open(path).read()) != norm(want):
                return False, f"FAILED: {rel} has the wrong contents"
        return self._run_hidden_tests(inst.cases)

    def _run_hidden_tests(self, cases: list[dict]) -> tuple[bool, str]:
        tmp = tempfile.mkdtemp(prefix="verify_")
        try:
            with open(os.path.join(tmp, "runner.py"), "w") as f:
                f.write(RUNNER_SRC)
            with open(os.path.join(tmp, "cases.json"), "w") as f:
                json.dump(cases, f)
            shell.give_to_agent(tmp)
            env = shell.agent_env(self._workdir)
            try:
                proc = subprocess.run(["python3", "-I", os.path.join(tmp, "runner.py"), self._workdir,
                                       os.path.join(tmp, "cases.json")], cwd=self._workdir, env=env,
                                      capture_output=True, text=True, timeout=VERIFY_TIMEOUT_S,
                                      **shell.agent_user_kwargs())
            except subprocess.TimeoutExpired:
                return False, "FAILED: hidden tests timed out"
            out = proc.stdout
            if "@@RESULT@@" not in out:
                return False, "FAILED: hidden tests could not run: " + (proc.stderr or out)[-300:]
            res = json.loads(out.rsplit("@@RESULT@@", 1)[1])
            ok = res["total"] > 0 and res["passed"] == res["total"]
            msg = f"{res['passed']}/{res['total']} hidden tests passed"
            return ok, ("PASSED: " + msg) if ok else ("FAILED: " + msg)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    # ------------------------------------------------------------------ helpers

    def _display_dir(self) -> str:
        return self._workdir or ""

    def _obs(self, output: str, action_type: str, reward, done: bool, error: str = "", success: bool = True,
             info: Optional[dict] = None) -> Tbench2Observation:
        return Tbench2Observation(instruction=self._instruction, output=output, success=success, error=error,
                                  task_id=self._state.task_id, task_path=self._state.task_path,
                                  session_id=None if action_type == "reset" else "tb2-session",
                                  action_type=action_type, info=info or {}, reward=reward, done=done)

    def _finish(self, msg: str, reward: float, kind: str) -> Tbench2Observation:
        obs = self._obs(output=msg, action_type=kind, reward=reward, done=True,
                        info={"tests_passed": reward == 1.0})
        self._done = True
        self._cleanup()
        return obs

    def _cleanup(self) -> None:
        if self._workdir:
            _release_workdir(self._workdir)
            self._workdir = None

    def close(self) -> None:
        self._cleanup()

    @property
    def state(self) -> Tbench2State:
        return self._state
