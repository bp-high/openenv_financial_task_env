"""Financial Task Environment — core environment logic.

A code-execution environment where the agent writes Python code (using openpyxl)
to read, analyze, and modify real Excel workbooks from enterprise finance workflows.

For QA tasks: the agent reads the xlsx and submits a text answer.
For MODIFY tasks: the agent writes code that modifies the xlsx, then the result
is compared cell-by-cell against a reference workbook.
"""

from __future__ import annotations

import io
import openpyxl
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Observation, State

from models import FinancialAction, FinancialObservation
from tasks import TASKS, TASK_IDS, get_task
from graders import grade_task


class FinancialEnvironment(Environment):
    """OpenEnv environment for financial spreadsheet tasks with code execution.

    Episode flow
    ────────────
    1. ``reset(task_id="task_1")`` → observation with task info + xlsx summary.
    2. ``step(action_type="code", content="import openpyxl; ...")`` → execute code, get stdout.
    3. ``step(action_type="submit", content="answer text")`` → grade and end episode.
       *or* for MODIFY tasks:
       ``step(action_type="submit_file", content="<path>")`` → grade xlsx and end.

    The episode also ends when *max_steps* is reached.
    """

    MAX_STEPS = 15

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_task: dict[str, Any] | None = None
        self._done = False
        self._cumulative_reward = 0.0
        self._workdir: str | None = None

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> FinancialObservation:
        task_id: str = kwargs.get("task_id", "task_1")
        self._current_task = get_task(task_id)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._done = False
        self._cumulative_reward = 0.0

        # Create a working directory and copy the source xlsx into it
        self._workdir = tempfile.mkdtemp(prefix=f"financial_env_{task_id}_")
        src = self._current_task.get("source_file", "")
        if src and Path(src).exists():
            shutil.copy2(src, self._workdir)
            work_file = str(Path(self._workdir) / Path(src).name)
        else:
            work_file = ""

        # Generate an xlsx summary to include in the observation
        xlsx_summary = self._summarize_xlsx(work_file) if work_file else "No source file."

        task = self._current_task
        task_info = (
            f"Task: {task['title']}\n"
            f"Difficulty: {task['difficulty']}\n"
            f"Type: {task['task_type']} ({task['category']})\n\n"
            f"Instruction:\n{task['instruction']}\n"
        )
        if task.get("constraints"):
            task_info += f"\nConstraints:\n{task['constraints']}\n"
        task_info += (
            f"\nSource file: {work_file}\n"
            f"\nSpreadsheet Summary:\n{xlsx_summary}\n\n"
            "Actions:\n"
            "  action_type='code'    → Execute Python code (openpyxl available).\n"
            "                          The working file path is in the source_file field.\n"
            "  action_type='submit'  → Submit a text answer (QA tasks).\n"
            "  action_type='submit_file' → Submit a modified xlsx path (MODIFY tasks).\n"
        )

        return FinancialObservation(
            done=False,
            reward=0.0,
            task_id=task["id"],
            task_description=task_info,
            financial_data=xlsx_summary,
            difficulty=task["difficulty"],
            task_type=task["task_type"],
            feedback="Environment reset. Read the spreadsheet and task instructions carefully.",
            current_step=0,
            max_steps=self.MAX_STEPS,
            available_tasks=",".join(TASK_IDS),
            source_file=work_file,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(
        self,
        action: FinancialAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> FinancialObservation:
        self._state.step_count += 1

        if self._current_task is None:
            return self._obs(feedback="No task loaded. Call reset() first.", reward=0.001, done=True)

        if self._done:
            return self._obs(feedback="Episode already finished. Call reset().", reward=0.001, done=True)

        action_type = action.action_type.strip().lower()

        if action_type == "code":
            return self._handle_code(action.content)
        elif action_type == "submit":
            return self._handle_submit_text(action.content)
        elif action_type == "submit_file":
            return self._handle_submit_file(action.content)
        else:
            return self._obs(
                feedback=f"Unknown action_type '{action.action_type}'. Use 'code', 'submit', or 'submit_file'.",
                reward=0.001, done=False,
            )

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Code execution
    # ------------------------------------------------------------------
    def _compute_code_reward(self, code: str, succeeded: bool, stdout: str) -> float:
        """Compute a step reward for code execution based on quality signals."""
        if not succeeded:
            return 0.005  # Failed code gets minimal reward

        # Count substantive lines (not imports, blanks, comments)
        lines = code.strip().splitlines()
        substantive = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("import ") or stripped.startswith("from "):
                continue
            substantive += 1

        # Base reward for successful execution
        reward = 0.02

        # Bonus for substantive code (up to +0.03)
        reward += min(substantive * 0.002, 0.03)

        # Bonus for producing output (agent is exploring data)
        if stdout.strip():
            output_lines = len(stdout.strip().splitlines())
            reward += min(output_lines * 0.001, 0.02)

        # Bonus for modification actions (save, wb.save, etc.)
        if "save(" in code or ".save(" in code:
            reward += 0.03

        return min(reward, 0.10)  # Cap at 0.10 per code step

    def _handle_code(self, code: str) -> FinancialObservation:
        """Execute Python code in a subprocess and return stdout/stderr."""
        if not self._workdir:
            return self._obs(feedback="No working directory. Call reset() first.", reward=0.001, done=False)

        succeeded = False
        stdout = ""
        stderr = ""

        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self._workdir,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            stdout = result.stdout[:4000] if result.stdout else ""
            stderr = result.stderr[:2000] if result.stderr else ""

            if result.returncode == 0:
                succeeded = True
                feedback = f"Code executed successfully.\n\nSTDOUT:\n{stdout}"
                if stderr:
                    feedback += f"\n\nSTDERR:\n{stderr}"
            else:
                feedback = f"Code execution failed (exit code {result.returncode}).\n\nSTDERR:\n{stderr}"
                if stdout:
                    feedback += f"\n\nSTDOUT:\n{stdout}"
        except subprocess.TimeoutExpired:
            feedback = "Code execution timed out (30s limit)."
        except Exception as e:
            feedback = f"Code execution error: {e}"

        reward = self._compute_code_reward(code, succeeded, stdout)
        self._cumulative_reward += reward

        at_limit = self._state.step_count >= self.MAX_STEPS
        if at_limit:
            self._done = True
            feedback += "\n\n⚠ Maximum steps reached — episode ending."

        return self._obs(feedback=feedback, reward=reward, done=at_limit)

    # ------------------------------------------------------------------
    # Submit handlers
    # ------------------------------------------------------------------
    def _handle_submit_text(self, answer: str) -> FinancialObservation:
        """Grade a text answer (for QA tasks)."""
        task = self._current_task
        assert task is not None

        score = grade_task(task, answer=answer)
        self._done = True
        self._cumulative_reward += score

        quality = "Excellent" if score >= 0.9 else "Good" if score >= 0.7 else "Partial" if score >= 0.4 else "Needs improvement"
        return self._obs(
            feedback=f"Answer graded. Score: {score:.2f}/1.00 — {quality}.\nCumulative reward: {self._cumulative_reward:.2f}",
            reward=score, done=True,
        )

    def _handle_submit_file(self, file_path: str) -> FinancialObservation:
        """Grade a modified xlsx file (for MODIFY tasks)."""
        task = self._current_task
        assert task is not None

        # Resolve relative paths against workdir
        p = Path(file_path)
        if not p.is_absolute() and self._workdir:
            p = Path(self._workdir) / p

        if not p.exists():
            self._done = True
            return self._obs(
                feedback=f"File not found: {p}. Score: 0.001",
                reward=0.001, done=True,
            )

        score = grade_task(task, output_path=str(p))
        self._done = True
        self._cumulative_reward += score

        quality = "Excellent" if score >= 0.9 else "Good" if score >= 0.7 else "Partial" if score >= 0.4 else "Needs improvement"
        return self._obs(
            feedback=f"File graded. Score: {score:.2f}/1.00 — {quality}.\nCumulative reward: {self._cumulative_reward:.2f}",
            reward=score, done=True,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _summarize_xlsx(self, path: str) -> str:
        """Return a text summary of an xlsx file (sheet names, dimensions, sample data)."""
        try:
            wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
            lines = [f"Workbook: {Path(path).name}", f"Sheets: {wb.sheetnames}", ""]
            for name in wb.sheetnames[:5]:  # Limit to 5 sheets
                ws = wb[name]
                lines.append(f"--- Sheet: {name} (rows≈{ws.max_row}, cols≈{ws.max_column}) ---")
                # Show first 8 rows
                row_count = 0
                for row in ws.iter_rows(max_row=8, values_only=True):
                    vals = [str(v)[:30] if v is not None else "" for v in row[:12]]
                    lines.append("  " + " | ".join(vals))
                    row_count += 1
                if ws.max_row and ws.max_row > 8:
                    lines.append(f"  ... ({ws.max_row - 8} more rows)")
                lines.append("")
            wb.close()
            return "\n".join(lines)
        except Exception as e:
            return f"Could not read xlsx: {e}"

    def _obs(self, *, feedback: str, reward: float, done: bool) -> FinancialObservation:
        task = self._current_task or {}
        work_file = ""
        if self._workdir and task.get("source_file"):
            work_file = str(Path(self._workdir) / Path(task["source_file"]).name)
        return FinancialObservation(
            done=done,
            reward=reward,
            task_id=task.get("id", ""),
            task_description=task.get("instruction", ""),
            financial_data="",
            difficulty=task.get("difficulty", ""),
            task_type=task.get("task_type", ""),
            feedback=feedback,
            current_step=self._state.step_count,
            max_steps=self.MAX_STEPS,
            available_tasks=",".join(TASK_IDS),
            source_file=work_file,
        )

    def close(self) -> None:
        """Clean up the temporary working directory."""
        if self._workdir and Path(self._workdir).exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
