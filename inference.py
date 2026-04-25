#!/usr/bin/env python3
"""Baseline inference script for the Office Document Task Environment.

Runs an LLM agent against a manifest-defined subset of tasks across the
xlsx / docx / pptx families.  The agent generates Python code to read or
modify the source file, then submits a text answer or modified file.

Outputs a `runs/<timestamp>_<model_slug>/` directory containing:
  - results.json         — summary + per-task scores
  - summary.csv          — flat table for plotting
  - trajectories/<id>.jsonl — full step-by-step trace per task
  - log.txt              — mirrors stdout

Environment variables
─────────────────────
  API_BASE_URL   LLM API endpoint  (required)
  MODEL_NAME     Model identifier  (required, can override with --model)
  HF_TOKEN       Hugging Face / API key  (required)
  ENV_URL        Environment server URL (default: http://localhost:8000)

CLI examples
────────────
  python inference.py --split eval                       # all 22 eval tasks
  python inference.py --family docx --split eval         # 4 docx eval tasks
  python inference.py --task-ids finch_10,osworld_0a0faba3
  python inference.py --limit 5 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Defaults — overridable via env or CLI
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "https://router.huggingface.co/v1"
DEFAULT_MODEL = "MiniMaxAI/MiniMax-M2.1"
DEFAULT_ENV_URL = "http://localhost:8000"
DEFAULT_MAX_STEPS = 15        # matches env's MAX_STEPS
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 12000
DEFAULT_TASK_TIMEOUT = 360    # 6 min per task; pptx decks need more steps

BENCHMARK = "office_document_task_env"

REPO_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = REPO_ROOT / "data" / "manifest.jsonl"

# ---------------------------------------------------------------------------
# Family-specific system prompts
# ---------------------------------------------------------------------------

_BASE_RULES = """\
CRITICAL RULES:
1. Do NOT call reset(). Just write plain Python code.
2. Use the EXACT file path provided. Do not guess paths.
3. Each code block runs in a FRESH subprocess — you must re-import and re-open
   the file every time. Variables do NOT persist between steps.
4. Use print() liberally to see data. Read the output carefully before your next step.
5. You have a limited number of steps. Be efficient — explore in step 1, solve in
   step 2-3, submit.
6. **You MUST execute at least one code step before submitting.** The
   environment will reject SUBMIT_ANSWER and SUBMIT_FILE on step 1 — you
   need to read or modify the file with code first. Submitting the source
   file unchanged is never a correct solve and will be rejected.

RESPONSE FORMAT — use EXACTLY one of:

To run Python code:
```python
your code here
```

To submit a text answer (QA tasks):
SUBMIT_ANSWER: your answer here

To submit a modified file (MODIFY tasks):
SUBMIT_FILE: /path/to/saved.<ext>
"""

SYSTEM_PROMPTS = {
    "xlsx": textwrap.dedent(f"""\
You are an expert financial analyst and Python programmer.
You are working with a real Excel workbook (.xlsx) using `openpyxl`.

{_BASE_RULES}

For MODIFY tasks: load with `openpyxl.load_workbook(path)`, make changes,
save with `wb.save(path)` to the SAME path, then SUBMIT_FILE that path.
"""),
    "docx": textwrap.dedent(f"""\
You are an expert document editor and Python programmer.
You are working with a real Word document (.docx) using `python-docx`.

{_BASE_RULES}

Common imports: `from docx import Document`, `from docx.shared import Pt, RGBColor`,
`from docx.enum.text import WD_PARAGRAPH_ALIGNMENT`.

For MODIFY tasks: load with `Document(path)`, make changes, save with
`doc.save(path)` to the SAME path, then SUBMIT_FILE that path.
"""),
    "pptx": textwrap.dedent(f"""\
You are an expert presentation editor and Python programmer.
You are working with a real PowerPoint deck (.pptx) using `python-pptx`.

{_BASE_RULES}

Common imports: `from pptx import Presentation`, `from pptx.util import Pt, Inches`,
`from pptx.dml.color import RGBColor`.

For MODIFY tasks: load with `Presentation(path)`, mutate slides/shapes,
save with `prs.save(path)` to the SAME path, then SUBMIT_FILE that path.
"""),
}


# ---------------------------------------------------------------------------
# Manifest-driven task selection
# ---------------------------------------------------------------------------

def load_tasks() -> List[Dict[str, Any]]:
    """Read data/manifest.jsonl + the original 10 hand-curated xlsx tasks."""
    tasks: List[Dict[str, Any]] = []

    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))

    # The 10 hand-curated tasks (task_1..task_10) live in tasks.py, not the
    # manifest.  Inject lightweight metadata for them so they can be selected
    # via --task-ids or --split (they don't have a split field — treat as train).
    hand_curated_ids = [f"task_{i}" for i in range(1, 11)]
    seen = {t["id"] for t in tasks}
    for tid in hand_curated_ids:
        if tid in seen:
            continue
        tasks.append({
            "id": tid,
            "family": "xlsx",
            "origin": "hand_curated",
            "split": "train",
            "primary_tag": "hand_curated",
        })
    return tasks


def select_tasks(args, all_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if args.task_ids:
        wanted = {tid.strip() for tid in args.task_ids.split(",") if tid.strip()}
        return [t for t in all_tasks if t["id"] in wanted]

    out = list(all_tasks)
    if args.split != "all":
        out = [t for t in out if t.get("split", "train") == args.split]
    if args.family != "all":
        out = [t for t in out if t.get("family", "xlsx") == args.family]
    # Sort deterministically: family, primary_tag, id
    out.sort(key=lambda t: (t.get("family", ""), t.get("primary_tag", ""), t["id"]))
    if args.limit:
        out = out[: args.limit]
    return out


# ---------------------------------------------------------------------------
# Logging — mirrors stdout to log.txt and structured trajectory file
# ---------------------------------------------------------------------------

class Tee:
    """File-or-stdout dual writer; flushes both."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def log_start(task: str, family: str, model: str) -> None:
    print(f"[START] task={task} family={family} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = str(error).lower() if error else "none"
    short_action = action[:500].replace("\n", " ")
    print(
        f"[STEP] step={step} action={short_action} reward={reward:.3f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.3f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ---------------------------------------------------------------------------
# WebSocket plumbing
# ---------------------------------------------------------------------------

async def ws_send_recv(ws, message: dict) -> dict:
    await ws.send(json.dumps(message))
    resp = json.loads(await ws.recv())
    if resp.get("type") == "error":
        raise RuntimeError(f"Server error: {resp.get('data', {}).get('message', 'unknown')}")
    return resp


async def ws_reset(ws, task_id: str) -> dict:
    resp = await ws_send_recv(ws, {"type": "reset", "data": {"task_id": task_id}})
    data = resp.get("data", {})
    obs = data.get("observation", data)
    return {"observation": obs, "reward": data.get("reward", 0.0), "done": data.get("done", False)}


async def ws_step(ws, action_type: str, content: str) -> dict:
    resp = await ws_send_recv(ws, {
        "type": "step",
        "data": {"action_type": action_type, "content": content},
    })
    data = resp.get("data", {})
    obs = data.get("observation", data)
    return {"observation": obs, "reward": data.get("reward", 0.0), "done": data.get("done", False)}


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_model_response(client: OpenAI, model_name: str, messages: List[Dict[str, str]],
                      temperature: float, max_tokens: int) -> str:
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return ""


def _extract_kimi_tool_call_code(response: str) -> Optional[str]:
    """Kimi K2/K2.5 emits its native tool-call wire format even when the
    chat API is called without `tools=`.  Examples we've seen in the wild:

        <|tool_calls_section_begin|>
        <|tool_call_begin|> functions.python:0
        <|tool_call_argument_begin|>
        {"code": "import openpyxl\n..."}     # may or may not be terminated;
                                              # responses often hit max_tokens
                                              # mid-string

    Strategy: locate the marker, then try (in order):
      1. Strict JSON parse of a `{...}` block
      2. Regex pull of an `"code|source|script|python": "..."` value, even
         if the closing `"` and `}` are missing (truncation case)
    """
    if "<|tool_call_begin|>" not in response and "<|tool_calls_section_begin|>" not in response:
        return None

    # Slice everything after the argument-begin marker (or the call-begin
    # marker as a fallback) — that's where the JSON arg lives.
    body = response
    for marker in ("<|tool_call_argument_begin|>", "<|tool_call_begin|>"):
        if marker in response:
            body = response.split(marker, 1)[1]
            break

    # Strict JSON first (works on well-formed, untruncated responses)
    m = re.search(r"\{.*?\}", body, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            for key in ("code", "source", "script", "python", "command"):
                if key in obj and isinstance(obj[key], str):
                    return obj[key]
        except Exception:
            pass

    # Truncation-tolerant extraction: find `"code": "...` and take everything
    # to the end of the body OR the last unescaped `"` we can find.  If the
    # response was cut mid-string, we end up with a partial-but-runnable code
    # snippet, which is still better than dropping the action entirely.
    for key in ("code", "source", "script", "python", "command"):
        m = re.search(rf'"{key}"\s*:\s*"', body)
        if not m:
            continue
        rest = body[m.end():]
        # Try to find the closing unescaped quote
        out_chars = []
        i = 0
        while i < len(rest):
            c = rest[i]
            if c == "\\" and i + 1 < len(rest):
                out_chars.append(c)
                out_chars.append(rest[i + 1])
                i += 2
                continue
            if c == '"':
                break
            out_chars.append(c)
            i += 1
        raw = "".join(out_chars)
        try:
            return raw.encode().decode("unicode_escape")
        except Exception:
            return raw

    return None


def extract_action(response: str):
    """Parse model response into (action_type, content)."""
    if "SUBMIT_ANSWER:" in response:
        answer = response.split("SUBMIT_ANSWER:", 1)[1].strip()
        answer = re.sub(r"```\s*$", "", answer).strip()
        return "submit", answer
    if "SUBMIT_FILE:" in response:
        path = response.split("SUBMIT_FILE:", 1)[1].strip()
        path = re.sub(r"[`\s\"']+$", "", path).strip()
        path = re.sub(r"^[`\"']+", "", path).strip()
        return "submit_file", path

    # Kimi K2/K2.5 native tool-call format
    tool_code = _extract_kimi_tool_call_code(response)
    if tool_code:
        return "code", tool_code

    m = re.search(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if m:
        return "code", m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", response, re.DOTALL)
    if m:
        code = m.group(1).strip()
        if "import" in code or "openpyxl" in code or "docx" in code or "pptx" in code or "print" in code:
            return "code", code

    if response.strip().startswith("import "):
        return "code", response.strip()

    return "submit", response.strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _to_ws_url(http_url: str) -> str:
    return http_url.replace("https://", "wss://").replace("http://", "ws://")


async def run_task(
    client: OpenAI,
    ws_url: str,
    task: Dict[str, Any],
    *,
    model_name: str,
    max_steps: int,
    task_timeout: float,
    temperature: float,
    max_tokens: int,
    traj_dir: Path,
) -> Dict[str, Any]:
    import websockets

    task_id = task["id"]
    family = task.get("family", "xlsx")
    log_start(task=task_id, family=family, model=model_name)

    rewards: List[float] = []
    trajectory: List[Dict[str, Any]] = []  # serialized step-by-step
    steps_taken = 0
    final_score = 0.0
    success = False
    task_start = time.time()
    error_msg: Optional[str] = None

    try:
        async with websockets.connect(
            f"{ws_url}/ws",
            open_timeout=30,
            close_timeout=10,
            max_size=100 * 1024 * 1024,
            # Disable application-level pings entirely.  The OpenAI client call
            # is synchronous and blocks the asyncio loop while a thinking model
            # reasons for 60–180s — pings can't flow, the WS dies with
            # "1011 keepalive ping timeout".  Rely on TCP keepalive instead.
            ping_interval=None,
        ) as ws:
            reset_data = await ws_reset(ws, task_id)
            obs = reset_data["observation"]
            task_desc = obs.get("task_description", "")
            feedback = obs.get("feedback", "")
            source_file = obs.get("source_file", "")
            task_type = obs.get("task_type", "QA")
            obs_family = obs.get("family") or family  # env may emit family in obs

            sys_prompt = SYSTEM_PROMPTS.get(obs_family, SYSTEM_PROMPTS["xlsx"])
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": (
                    f"{task_desc}\n\n"
                    f"Source file path: {source_file}\n"
                    f"File family: {obs_family}\n"
                    f"Task type: {task_type}\n\n"
                    f"{feedback}"
                )},
            ]

            for step_num in range(1, max_steps + 1):
                elapsed = time.time() - task_start
                if elapsed > task_timeout:
                    print(f"[DEBUG] {task_id} timeout after {elapsed:.0f}s "
                          f"(limit {task_timeout:.0f}s)", flush=True)
                    error_msg = "task_timeout"
                    break

                response = get_model_response(client, model_name, messages, temperature, max_tokens)
                if not response:
                    error_msg = "empty_response"
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

                trajectory.append({
                    "step": step_num,
                    "action_type": action_type,
                    "action_content": content[:4000],
                    "reward": reward,
                    "done": done,
                    "feedback": step_feedback[:4000],
                })

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

                remaining = max_steps - step_num
                urgency = ""
                if remaining <= 2:
                    urgency = f"\n\n⚠ Only {remaining} step(s) remaining! You MUST submit now."
                    if task_type == "QA":
                        urgency += " Use: SUBMIT_ANSWER: <your answer>"
                    else:
                        urgency += f" Save the file and use: SUBMIT_FILE: {source_file}"

                messages.append({"role": "user", "content": (
                    f"Code execution result (step {step_num}/{max_steps}):\n"
                    f"{step_feedback}\n\n"
                    f"Source file: {source_file}{urgency}"
                )})

            try:
                await ws.send(json.dumps({"type": "close"}))
            except Exception:
                pass

    except Exception as exc:
        print(f"[DEBUG] {task_id} error: {exc}", flush=True)
        error_msg = str(exc)
        log_step(step=steps_taken + 1, action="error", reward=0.001, done=True, error=error_msg)

    final_score = max(0.001, min(0.999, final_score))
    rewards = [max(0.001, min(0.999, r)) for r in rewards]
    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)

    # Persist trajectory
    traj_path = traj_dir / f"{task_id}.jsonl"
    with open(traj_path, "w") as f:
        for entry in trajectory:
            f.write(json.dumps(entry) + "\n")

    return {
        "task_id": task_id,
        "family": family,
        "primary_tag": task.get("primary_tag", ""),
        "split": task.get("split", "train"),
        "score": final_score,
        "success": success,
        "steps": steps_taken,
        "elapsed_s": round(time.time() - task_start, 2),
        "step_rewards": rewards,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline inference for the office-document env")
    p.add_argument("--split", choices=["train", "eval", "all"], default="eval",
                   help="manifest split to run (default: eval)")
    p.add_argument("--family", choices=["xlsx", "docx", "pptx", "all"], default="all",
                   help="task family to filter (default: all)")
    p.add_argument("--limit", type=int, default=0,
                   help="cap number of tasks (0 = no cap)")
    p.add_argument("--task-ids", default="",
                   help="comma-separated task IDs to run (overrides --split/--family)")
    p.add_argument("--output-dir", default="",
                   help="results directory (default: runs/<timestamp>_<model_slug>/)")
    p.add_argument("--resume", action="store_true",
                   help="merge new task results into an existing --output-dir "
                        "(replaces any prior entries for the same task_ids; "
                        "leaves all other task results and trajectories intact)")
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    p.add_argument("--api-base", default=os.environ.get("API_BASE_URL", DEFAULT_API_BASE))
    p.add_argument("--env-url", default=os.environ.get("ENV_URL", DEFAULT_ENV_URL))
    p.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    p.add_argument("--task-timeout", type=float, default=DEFAULT_TASK_TIMEOUT)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    return p.parse_args(argv)


def model_slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


async def async_main(args: argparse.Namespace) -> None:
    # Pick the API key based on the api-base URL so you don't have to alias
    # env vars when switching providers.  Provider-specific env wins; falls back
    # to a generic chain if nothing matches.
    if "nebius" in args.api_base:
        _envs = ("NEBIUS_API_KEY", "API_KEY", "HF_TOKEN")
    elif "huggingface" in args.api_base or "hf.co" in args.api_base:
        _envs = ("HF_TOKEN", "API_KEY", "NEBIUS_API_KEY")
    elif "openai" in args.api_base:
        _envs = ("OPENAI_API_KEY", "API_KEY", "HF_TOKEN")
    else:
        _envs = ("API_KEY", "HF_TOKEN", "NEBIUS_API_KEY", "OPENAI_API_KEY")
    api_key = next((os.environ[k] for k in _envs if os.environ.get(k)), None)
    if not api_key:
        print(f"ERROR: none of {_envs} are set for api_base={args.api_base!r}", file=sys.stderr)
        sys.exit(1)

    # Pick tasks
    all_tasks = load_tasks()
    tasks = select_tasks(args, all_tasks)
    if not tasks:
        print("ERROR: no tasks selected.  Check --split / --family / --task-ids", file=sys.stderr)
        sys.exit(1)

    # Output dir
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "runs" / f"{ts}_{model_slug(args.model)}"

    # Resume mode: load any existing results.json so we can merge new entries
    # back in afterward.  Old trajectories are preserved unless overwritten by
    # this run's task IDs.
    prior_results: List[Dict[str, Any]] = []
    if args.resume and (out_dir / "results.json").exists():
        try:
            prior = json.loads((out_dir / "results.json").read_text())
            prior_results = list(prior.get("results", []))
        except Exception as e:
            print(f"WARNING: --resume passed but couldn't load prior results.json: {e}",
                  file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to log.txt — append in resume mode, fresh otherwise
    log_mode = "a" if (args.resume and (out_dir / "log.txt").exists()) else "w"
    log_file = open(out_dir / "log.txt", log_mode)
    sys.stdout = Tee(sys.__stdout__, log_file)
    if args.resume and prior_results:
        print(f"\n# RESUME: loaded {len(prior_results)} prior task results from {out_dir}/results.json")

    print(f"# Run config")
    print(f"  model       : {args.model}")
    print(f"  api_base    : {args.api_base}")
    print(f"  env_url     : {args.env_url}")
    print(f"  split       : {args.split}")
    print(f"  family      : {args.family}")
    print(f"  task count  : {len(tasks)}")
    print(f"  max_steps   : {args.max_steps}")
    print(f"  task_timeout: {args.task_timeout}s")
    print(f"  output_dir  : {out_dir}")
    print()

    client = OpenAI(base_url=args.api_base, api_key=api_key)
    ws_url = _to_ws_url(args.env_url)

    results: List[Dict[str, Any]] = []
    overall_start = time.time()

    for i, task in enumerate(tasks, 1):
        print(f"\n{'='*70}\n[{i}/{len(tasks)}] {task['id']}  "
              f"({task.get('family')}, {task.get('primary_tag', '')[:40]})\n{'='*70}", flush=True)
        result = await run_task(
            client, ws_url, task,
            model_name=args.model,
            max_steps=args.max_steps,
            task_timeout=args.task_timeout,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            traj_dir=traj_dir,
        )
        results.append(result)
        print(f"  -> {task['id']} score={result['score']:.3f} steps={result['steps']} "
              f"elapsed={result['elapsed_s']:.1f}s", flush=True)

    # Merge with prior results if --resume was used (new entries replace old
    # entries with the same task_id; everything else is preserved).
    if args.resume and prior_results:
        new_ids = {r["task_id"] for r in results}
        kept = [r for r in prior_results if r["task_id"] not in new_ids]
        merged = kept + results
        print(f"\n# RESUME merge: {len(kept)} prior + {len(results)} new = {len(merged)} total")
    else:
        merged = results

    # Aggregate over the MERGED set so the summary covers the full eval
    total_elapsed = time.time() - overall_start
    if merged:
        avg = sum(r["score"] for r in merged) / len(merged)
        success_rate = sum(1 for r in merged if r["success"]) / len(merged)
    else:
        avg = success_rate = 0.0

    by_family: Dict[str, List[float]] = {}
    for r in merged:
        by_family.setdefault(r["family"], []).append(r["score"])

    summary = {
        "model": args.model,
        "split": args.split,
        "family": args.family,
        "n_tasks": len(merged),
        "avg_score": round(avg, 4),
        "success_rate": round(success_rate, 4),
        "total_elapsed_s": round(total_elapsed, 2),
        "by_family": {fam: {
            "n": len(scores),
            "avg": round(sum(scores) / len(scores), 4),
        } for fam, scores in by_family.items()},
        "results": merged,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Flat CSV for plotting
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "family", "primary_tag", "split", "score", "success", "steps", "elapsed_s", "error"])
        for r in merged:
            w.writerow([r["task_id"], r["family"], r["primary_tag"], r["split"],
                        r["score"], r["success"], r["steps"], r["elapsed_s"], r.get("error") or ""])

    family_lines = []
    for fam in sorted(by_family):
        scores = by_family[fam]
        fam_avg = sum(scores) / len(scores) if scores else 0.0
        family_lines.append(f"  {fam}: avg={fam_avg:.3f}  n={len(scores)}")

    print(
        f"\n{'='*70}\n"
        f"OVERALL  avg_score={avg:.3f}  success_rate={success_rate:.3f}  "
        f"n={len(results)}  elapsed={total_elapsed:.1f}s\n"
        + "\n".join(family_lines)
        + f"\nResults written to: {out_dir}\n"
        + "="*70,
        flush=True,
    )

    log_file.close()


def main() -> None:
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
