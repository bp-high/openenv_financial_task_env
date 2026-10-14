"""Finance Desk environment: one procedural workspace per episode, four tools, all-or-nothing reward."""

import os
import random
import shutil
import tempfile
import time
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from .families import FAMILIES, parse_task_id, task_ids
from .families.base import grade
from .models import DeskAction, DeskObservation
from .sandbox import prepare_workspace, run_python

MAX_STEPS = 8
MAX_READ = 6000
# Unseeded resets in the same 10-minute window get the same instance, so the parallel episodes of one GRPO
# group (separate sandboxes started together) compare answers to one problem.
SEED_WINDOW_S = 600
# v2: embed every file's contents in the first observation (like the top MetaHarness env) so a task can be
# answered in one step with no reading turns. The python tool stays available for heavy arithmetic.
INLINE_FILES = True
MAX_INLINE = 6000  # per-file cap when inlining (keeps the observation inside the context budget)

TOOLS_HELP = (
    'Reply with ONE JSON object. To answer: {"tool": "submit", "args": {"answer": {...the documented keys...}}} '
    "— graded once, all-or-nothing (reward 1 only if every field is exactly right, else 0). "
    'If you need to compute, first call {"tool": "python", "args": {"code": "..."}} (the files are in the current '
    "directory; numpy and pandas available; print what you need; 10 s limit). You may reason first, but your reply "
    f"must end with exactly one JSON object and nothing after it. You have {MAX_STEPS} replies."
)


class FinanceDeskEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._workspace: Optional[str] = None
        self._instance = None
        self._done = True
        self._task_id = ""

    # ------------------------------------------------------------------ lifecycle

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: Optional[str] = None,
              **kwargs: Any) -> DeskObservation:
        task_id = task_id or kwargs.get("task") or task_ids()[0]
        family, level = parse_task_id(task_id)
        if seed is None:
            seed = int(time.time() // SEED_WINDOW_S)
        self._cleanup()
        self._instance = family.generate(random.Random(f"{task_id}:{seed}"), level)
        self._task_id = task_id
        self._workspace = tempfile.mkdtemp(prefix="desk_")
        for name, text in self._instance.files.items():
            with open(os.path.join(self._workspace, name), "w") as f:
                f.write(text)
        prepare_workspace(self._workspace)
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._done = False
        if INLINE_FILES:
            blocks = []
            for n, t in sorted(self._instance.files.items()):
                body = t if len(t) <= MAX_INLINE else t[:MAX_INLINE] + f"\n[truncated; read {n} with python for the rest]"
                blocks.append(f"=== {n} ===\n{body}")
            files_section = "FILES (full contents):\n" + "\n".join(blocks)
        else:
            files_section = "Workspace files: " + ", ".join(
                f"{n} ({len(t)} chars)" for n, t in sorted(self._instance.files.items()))
        brief = f"Task [{family.title}]: {self._instance.prompt}\n\n{files_section}\n\n{TOOLS_HELP}"
        return DeskObservation(output=brief, steps_left=MAX_STEPS, done=False, reward=0.0)

    def step(self, action: DeskAction, timeout_s: Optional[float] = None, **kwargs: Any) -> DeskObservation:  # type: ignore[override]
        if self._done or self._instance is None:
            return DeskObservation(output="Episode is over; call reset.", steps_left=0, done=True, reward=0.0)
        self._state.step_count += 1
        left = MAX_STEPS - self._state.step_count
        tool = (action.tool or "").strip().lower()
        args = action.args or {}
        if tool == "submit":
            answer = args.get("answer", {k: v for k, v in args.items()})
            if getattr(self._instance, "grader", None):
                ok = bool(self._instance.grader(answer))
            else:
                ok = grade(self._instance.answer, answer, self._instance.specs)
            return self._finish("Submitted. " + ("Correct." if ok else "Not correct."), 1.0 if ok else 0.0)
        if tool == "list":
            out = "\n".join(f"{n}\t{len(t)} chars" for n, t in sorted(self._instance.files.items()))
        elif tool == "read":
            paths = args.get("paths") or [args.get("path") or args.get("file") or ""]
            if isinstance(paths, str):
                paths = [paths]
            out = "\n".join(self._read(str(p)) for p in paths[:6])
        elif tool in ("python", "py", "run_python"):
            code = args.get("code") or args.get("source") or ""
            out = run_python(self._workspace, str(code)) if code else "error: python needs {\"code\": \"...\"}"
        else:
            out = f"error: unknown tool {tool!r}. Use list, read, python or submit."
        if left <= 0:
            return self._finish(out + "\nNo tool calls left; episode ended without a submission.", 0.0)
        return DeskObservation(output=out, steps_left=left, done=False, reward=0.0)

    def _read(self, path: str) -> str:
        name = os.path.basename(path.strip())
        if name not in self._instance.files:
            return f"== {path}: no such file (files: {', '.join(sorted(self._instance.files))})"
        text = self._instance.files[name]
        if len(text) > MAX_READ:
            text = text[:MAX_READ] + f"\n[truncated at {MAX_READ} chars; use python to read the rest]"
        return f"== {name}\n{text}"

    def _finish(self, msg: str, reward: float) -> DeskObservation:
        self._done = True
        self._cleanup()
        return DeskObservation(output=msg, steps_left=0, done=True, reward=reward)

    def _cleanup(self) -> None:
        if self._workspace:
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    def close(self) -> None:
        self._cleanup()

    @property
    def state(self) -> State:
        return self._state
