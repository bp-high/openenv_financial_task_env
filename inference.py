#!/usr/bin/env python3
"""Baseline inference script for the Financial Task Environment.

Runs an LLM agent against all 10 tasks.  The agent generates Python code
to read/modify Excel workbooks, then submits answers or modified files.

Uses WebSocket for persistent sessions (HTTP endpoints are stateless).

Environment variables
─────────────────────
  API_BASE_URL   LLM API endpoint  (required)
  MODEL_NAME     Model identifier  (required)
  HF_TOKEN       Hugging Face / API key  (required)
  ENV_URL        Environment server URL (default: http://localhost:8000)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "MiniMaxAI/MiniMax-M2.1")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("API_KEY")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

BENCHMARK = "financial_task_env"
MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 12000

TASK_IDS = [
    "task_1", "task_2", "task_3",  # easy (QA)
    "task_5", "task_8",              # medium + hard (MODIFY)
]

SYSTEM_PROMPT = textwrap.dedent("""\
You are an expert financial analyst and Python programmer.
You are working with a real Excel workbook. The file path is given to you.

CRITICAL RULES:
1. Do NOT call reset(). Just write plain Python code.
2. Use the EXACT file path provided. Do not guess paths.
3. Each code block runs in a FRESH subprocess — you must re-import and re-open
   the workbook every time. Variables do NOT persist between steps.
4. Use print() liberally to see data. Read the output carefully before your next step.
5. You have limited steps. Be efficient — explore in step 1, solve in step 2-3, submit.

RESPONSE FORMAT — use EXACTLY one of:

To run Python code:
```python
your code here
```

To submit a text answer (QA tasks):
SUBMIT_ANSWER: your answer here

To submit a modified file (MODIFY tasks):
SUBMIT_FILE: /path/to/saved.xlsx

STRATEGY:
- Step 1: Run code to explore the spreadsheet structure and data
- Step 2-3: Run code to compute the answer or make modifications
- Then SUBMIT immediately. Do not waste steps.

For MODIFY tasks: load the workbook, make changes, save it back to the SAME path,
then use SUBMIT_FILE with that path.
""")


# ---------------------------------------------------------------------------
# Logging helpers (strict hackathon format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = str(error).lower() if error else "none"
    short_action = action[:500].replace("\n", " ")
    print(
        f"[STEP] step={step} action={short_action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# WebSocket environment interaction
# ---------------------------------------------------------------------------

async def ws_send_recv(ws, message: dict) -> dict:
    """Send a message and receive a response over WebSocket."""
    await ws.send(json.dumps(message))
    resp = json.loads(await ws.recv())
    if resp.get("type") == "error":
        raise RuntimeError(f"Server error: {resp.get('data', {}).get('message', 'unknown')}")
    return resp


async def ws_reset(ws, task_id: str) -> dict:
    """Reset the environment via WebSocket."""
    resp = await ws_send_recv(ws, {"type": "reset", "data": {"task_id": task_id}})
    data = resp.get("data", {})
    obs = data.get("observation", data)
    return {
        "observation": obs,
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False),
    }


async def ws_step(ws, action_type: str, content: str) -> dict:
    """Execute a step via WebSocket."""
    resp = await ws_send_recv(ws, {
        "type": "step",
        "data": {"action_type": action_type, "content": content},
    })
    data = resp.get("data", {})
    obs = data.get("observation", data)
    return {
        "observation": obs,
        "reward": data.get("reward", 0.0),
        "done": data.get("done", False),
    }


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------

def get_model_response(client: OpenAI, messages: List[Dict[str, str]]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""


def extract_action(response: str):
    """Parse model response into (action_type, content)."""
    if "SUBMIT_ANSWER:" in response:
        answer = response.split("SUBMIT_ANSWER:", 1)[1].strip()
        # Strip trailing markdown artifacts
        answer = re.sub(r'```\s*$', '', answer).strip()
        return "submit", answer
    if "SUBMIT_FILE:" in response:
        path = response.split("SUBMIT_FILE:", 1)[1].strip()
        # Strip trailing backticks, quotes, whitespace
        path = re.sub(r'[`\s"\']+$', '', path).strip()
        # Also strip leading backticks/quotes
        path = re.sub(r'^[`"\']+', '', path).strip()
        return "submit_file", path

    # Extract code block
    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return "code", m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "import" in code or "openpyxl" in code or "print" in code:
            return "code", code

    # Fallback: if it looks like code, treat as code
    if response.strip().startswith("import ") or "openpyxl" in response:
        return "code", response.strip()

    # Otherwise treat as text answer
    return "submit", response.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _to_ws_url(http_url: str) -> str:
    """Convert http(s):// URL to ws(s):// URL."""
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


TASK_TIMEOUT = 240  # 4 minutes per task (5 tasks × 4 min = 20 min max)


async def run_task(client: OpenAI, ws_url: str, task_id: str) -> float:
    import websockets
    import time

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    task_start = time.time()

    try:
        async with websockets.connect(
            f"{ws_url}/ws",
            open_timeout=30,
            close_timeout=10,
            max_size=100 * 1024 * 1024,
            ping_interval=60,
            ping_timeout=60,
        ) as ws:
            # Reset
            reset_data = await ws_reset(ws, task_id)
            obs = reset_data["observation"]
            task_desc = obs.get("task_description", "")
            feedback = obs.get("feedback", "")
            source_file = obs.get("source_file", "")
            task_type = obs.get("task_type", "QA")

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"{task_desc}\n\n"
                    f"Source file path: {source_file}\n"
                    f"Task type: {task_type}\n\n"
                    f"{feedback}"
                )},
            ]

            for step_num in range(1, MAX_STEPS + 1):
                # Check per-task timeout
                elapsed = time.time() - task_start
                if elapsed > TASK_TIMEOUT:
                    print(f"[DEBUG] Task {task_id} timeout after {elapsed:.0f}s (limit {TASK_TIMEOUT}s)", flush=True)
                    break

                response = get_model_response(client, messages)
                if not response:
                    break

                action_type, content = extract_action(response)
                messages.append({"role": "assistant", "content": response})

                step_data = await ws_step(ws, action_type, content)
                step_obs = step_data["observation"]
                reward = float(step_data.get("reward") or 0)
                done = step_data.get("done", False)
                step_feedback = step_obs.get("feedback", "")

                rewards.append(reward)
                steps_taken = step_num

                log_step(
                    step=step_num,
                    action=f"[{action_type}] {content}",
                    reward=reward,
                    done=done,
                    error=None,
                )

                if done:
                    final_score = reward
                    success = final_score >= 0.5
                    break

                # Feed the execution result back to the LLM
                remaining = MAX_STEPS - step_num
                urgency = ""
                if remaining <= 2:
                    urgency = f"\n\n⚠ Only {remaining} step(s) remaining! You MUST submit now."
                    if task_type == "QA":
                        urgency += " Use: SUBMIT_ANSWER: <your answer>"
                    else:
                        urgency += f" Save the file and use: SUBMIT_FILE: {source_file}"

                messages.append({"role": "user", "content": (
                    f"Code execution result (step {step_num}/{MAX_STEPS}):\n"
                    f"{step_feedback}\n\n"
                    f"Source file: {source_file}{urgency}"
                )})

            # Send close
            try:
                await ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} error: {exc}", flush=True)
        log_step(step=steps_taken + 1, action="error", reward=0.001, done=True, error=str(exc))

    # Clamp final score to (0.001, 0.999) — evaluator rejects exact 0.0 and 1.0
    final_score = max(0.001, min(0.999, final_score))
    rewards = [max(0.001, min(0.999, r)) for r in rewards]
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
    return final_score


async def async_main() -> None:
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL not set.", file=sys.stderr)
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME not set.", file=sys.stderr)
        sys.exit(1)
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN environment variable not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    ws_url = _to_ws_url(ENV_URL)
    all_scores: List[float] = []

    for task_id in TASK_IDS:
        print(f"\n{'='*60}\nRunning {task_id}...\n{'='*60}", flush=True)
        score = await run_task(client, ws_url, task_id)
        all_scores.append(score)
        print(f"  -> {task_id} score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(
        f"\n{'='*60}\nOVERALL AVERAGE SCORE: {avg:.3f}\n"
        f"Per-task: {[f'{s:.3f}' for s in all_scores]}\n{'='*60}",
        flush=True,
    )


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
