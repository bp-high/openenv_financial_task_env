#!/usr/bin/env python3
"""In-process eval for LoRA-adapter models against the office-document env.

Mirror of inference.py but with two key differences:

  1. **Loads base model + LoRA via transformers/peft** instead of hitting an
     external API.  Lets us eval models that no Inference Provider hosts
     (i.e., our own SFT'd Qwen2.5-Coder-3B + LoRA adapters).
  2. **Instantiates `FinancialEnvironment` directly** instead of connecting
     over WebSocket.  Cuts WS overhead and is the same code path GRPO will
     use later (rollouts in-process).

Multi-adapter mode is supported — pass a comma-separated list to
`--adapters` and the script evals each in turn (loading base once,
wrapping/unwrapping the LoRA between iterations).  Pass `none` as an
adapter to evaluate the unmodified base model.

Output structure (mirrors inference.py):
    runs/eval_lora_<timestamp>/<adapter_slug>/
        results.json         summary + per-task records
        summary.csv          flat table for plotting
        trajectories/<id>.jsonl
        log.txt              mirrored stdout

Designed for HF Jobs (1× L40S 48 GB, ~$1.80/hr, ~15-20 min for 22 eval
tasks × 2 adapters = ~$0.50).

Example:
    # Local (CUDA box):
    python eval_lora.py \\
        --adapters bpHigh/qwen3b-office-sft-kimi,bpHigh/qwen3b-office-sft-kimi-long \\
        --split eval --output-dir runs/sft_eval

    # HF Jobs (cleanest for users without GPUs):
    hf jobs run --flavor l40sx1 --timeout 4h --secrets HF_TOKEN \\
        pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \\
        bash -c "<git clone + pip install + python eval_lora.py ...>"
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "server"))


# ---------------------------------------------------------------------------
# Re-use helpers from inference.py so the eval surface is identical
# ---------------------------------------------------------------------------

from inference import (  # noqa: E402
    SYSTEM_PROMPTS,
    extract_action,
    load_tasks,
    select_tasks,
    log_start,
    log_step,
    log_end,
    Tee,
    model_slug,
)
from server.financial_environment import FinancialEnvironment  # noqa: E402
from models import FinancialAction  # noqa: E402


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_base_and_tokenizer(base_model_id: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16_ok else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print(f"Loading base model: {base_model_id}")
    print(f"  precision: {str(dtype).split('.')[-1]}  cuda={torch.cuda.is_available()}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    model.eval()
    return tokenizer, model


def attach_lora(base_model, adapter_id_or_path: str):
    """Wrap base in a PeftModel with the given LoRA adapter."""
    from peft import PeftModel
    print(f"  Attaching LoRA adapter: {adapter_id_or_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_id_or_path)
    peft_model.eval()
    return peft_model


def detach_lora(peft_model):
    """Return the underlying base model and free LoRA-side memory.

    `PeftModel.unload()` returns the unwrapped base model with LoRA modules
    removed, so we can immediately wrap the next adapter on top.
    """
    try:
        base = peft_model.unload()
    except Exception:
        base = getattr(peft_model, "base_model", None) or peft_model
        if hasattr(base, "model"):
            base = base.model
    del peft_model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return base


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(tokenizer, model, messages: List[Dict[str, str]],
                       max_new_tokens: int, temperature: float,
                       max_input_tokens: int = 12000) -> str:
    """Tokenize chat-template-formatted messages, generate, decode."""
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 0.01),
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Per-task eval (in-process — direct env, no WebSocket)
# ---------------------------------------------------------------------------

def run_task_inproc(
    tokenizer, model, task: Dict[str, Any],
    *, max_steps: int, max_new_tokens: int, temperature: float,
    traj_dir: Path, model_name: str,
) -> Dict[str, Any]:
    task_id = task["id"]
    family = task.get("family", "xlsx")
    log_start(task=task_id, family=family, model=model_name)

    rewards: List[float] = []
    trajectory: List[Dict[str, Any]] = []
    final_score = 0.0
    success = False
    error_msg: Optional[str] = None
    task_start = time.time()

    env = FinancialEnvironment()
    try:
        obs = env.reset(task_id=task_id)
        sys_prompt = SYSTEM_PROMPTS.get(family, SYSTEM_PROMPTS["xlsx"])
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": (
                f"{obs.task_description}\n\n"
                f"Source file path: {obs.source_file}\n"
                f"File family: {family}\n"
                f"Task type: {obs.task_type}\n\n"
                f"{obs.feedback}"
            )},
        ]

        for step_num in range(1, max_steps + 1):
            response = generate_response(
                tokenizer, model, messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            if not response:
                error_msg = "empty_response"
                break

            action_type, content = extract_action(response)
            messages.append({"role": "assistant", "content": response})

            try:
                action = FinancialAction(action_type=action_type, content=content)
                obs = env.step(action)
            except Exception as e:
                error_msg = f"env.step failed: {e}"
                break

            reward = float(obs.reward or 0)
            done = bool(obs.done)
            step_feedback = obs.feedback or ""

            rewards.append(reward)
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
                reward=reward, done=done, error=None,
            )

            if done:
                final_score = reward
                success = final_score >= 0.5
                break

            remaining = max_steps - step_num
            urgency = ""
            if remaining <= 2:
                urgency = f"\n\n⚠ Only {remaining} step(s) remaining! You MUST submit now."
                if obs.task_type == "QA":
                    urgency += " Use: SUBMIT_ANSWER: <your answer>"
                else:
                    urgency += f" Save the file and use: SUBMIT_FILE: {obs.source_file}"

            messages.append({"role": "user", "content": (
                f"Code execution result (step {step_num}/{max_steps}):\n"
                f"{step_feedback}\n\n"
                f"Source file: {obs.source_file}{urgency}"
            )})

    except Exception as exc:
        error_msg = str(exc)
        print(f"[DEBUG] {task_id} crashed: {exc}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    final_score = max(0.001, min(0.999, final_score))
    rewards = [max(0.001, min(0.999, r)) for r in rewards]
    log_end(success=success, steps=len(trajectory), score=final_score, rewards=rewards)

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
        "steps": len(trajectory),
        "elapsed_s": round(time.time() - task_start, 2),
        "step_rewards": rewards,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Per-adapter eval
# ---------------------------------------------------------------------------

def eval_one_adapter(
    *, tokenizer, model, adapter_label: str, tasks: List[dict],
    out_dir: Path, max_steps: int, max_new_tokens: int, temperature: float,
) -> Dict[str, Any]:
    """Run all tasks against the given (already-loaded) model.  Writes
    results.json + summary.csv + trajectories/ inside out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_dir = out_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# Evaluating: {adapter_label}")
    print(f"# Output    : {out_dir}")
    print(f"# Tasks     : {len(tasks)}")
    print(f"{'#' * 70}\n")

    results: List[Dict[str, Any]] = []
    overall_start = time.time()
    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(tasks)}] {task['id']}  "
              f"({task.get('family')}, {task.get('primary_tag', '')[:40]})")
        print(f"{'=' * 70}")
        result = run_task_inproc(
            tokenizer, model, task,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            traj_dir=traj_dir,
            model_name=adapter_label,
        )
        results.append(result)
        print(f"  -> {task['id']} score={result['score']:.3f} "
              f"steps={result['steps']} elapsed={result['elapsed_s']:.1f}s")

    total_elapsed = time.time() - overall_start
    if results:
        avg = sum(r["score"] for r in results) / len(results)
        success_rate = sum(1 for r in results if r["success"]) / len(results)
    else:
        avg = success_rate = 0.0

    by_family: Dict[str, List[float]] = {}
    for r in results:
        by_family.setdefault(r["family"], []).append(r["score"])

    summary = {
        "model": adapter_label,
        "n_tasks": len(results),
        "avg_score": round(avg, 4),
        "success_rate": round(success_rate, 4),
        "total_elapsed_s": round(total_elapsed, 2),
        "by_family": {fam: {
            "n": len(scores), "avg": round(sum(scores) / len(scores), 4),
        } for fam, scores in by_family.items()},
        "results": results,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task_id", "family", "primary_tag", "split",
                    "score", "success", "steps", "elapsed_s", "error"])
        for r in results:
            w.writerow([r["task_id"], r["family"], r["primary_tag"], r["split"],
                        r["score"], r["success"], r["steps"], r["elapsed_s"],
                        r.get("error") or ""])

    print(f"\n{'=' * 70}")
    print(f"OVERALL [{adapter_label}]  avg={avg:.3f}  success_rate={success_rate:.0%}  "
          f"n={len(results)}  elapsed={total_elapsed:.0f}s")
    for fam in sorted(by_family):
        scores = by_family[fam]
        print(f"  {fam}: avg={sum(scores) / len(scores):.3f}  n={len(scores)}")
    print(f"{'=' * 70}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-3B-Instruct")
    p.add_argument("--adapters", required=True,
                   help="comma-separated list of LoRA adapters (HF repo IDs "
                        "or local paths).  Pass 'none' as an entry to also "
                        "evaluate the bare base model.")
    p.add_argument("--split", choices=["train", "eval", "all"], default="eval")
    p.add_argument("--family", choices=["xlsx", "docx", "pptx", "all"], default="all")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--task-ids", default="")
    p.add_argument("--output-dir", default="",
                   help="parent dir; per-adapter subdirs created underneath. "
                        "Default: runs/eval_lora_<timestamp>/")
    p.add_argument("--max-steps", type=int, default=15)
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="generation budget per assistant turn")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Output parent dir
    if args.output_dir:
        parent_out = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_out = REPO_ROOT / "runs" / f"eval_lora_{ts}"
    parent_out.mkdir(parents=True, exist_ok=True)

    # Tokenizer + base model — loaded ONCE, reused across adapters
    tokenizer, base_model = load_base_and_tokenizer(args.base_model)

    # Tasks selected ONCE
    all_tasks = load_tasks()
    tasks = select_tasks(args, all_tasks)
    if not tasks:
        print("ERROR: no tasks selected (check --split / --family / --task-ids)",
              file=sys.stderr)
        return 1
    print(f"\nSelected {len(tasks)} tasks")

    adapters = [a.strip() for a in args.adapters.split(",") if a.strip()]
    print(f"Will evaluate {len(adapters)} adapter(s): {adapters}")

    overall_summaries: Dict[str, Dict[str, Any]] = {}

    for i, adapter in enumerate(adapters):
        # Each adapter gets its own subdir + log file
        adapter_lower = adapter.lower()
        is_base = adapter_lower in ("none", "base", "")
        adapter_label = args.base_model if is_base else adapter
        adapter_tag = "base" if is_base else model_slug(adapter)
        out_dir = parent_out / adapter_tag

        # Tee stdout to per-adapter log
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(out_dir / "log.txt", "w")
        sys.stdout = Tee(sys.__stdout__, log_file)

        # Wrap base in PeftModel (or use base directly)
        if is_base:
            eval_model = base_model
        else:
            eval_model = attach_lora(base_model, adapter)

        # Run eval
        summary = eval_one_adapter(
            tokenizer=tokenizer,
            model=eval_model,
            adapter_label=adapter_label,
            tasks=tasks,
            out_dir=out_dir,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        overall_summaries[adapter_tag] = {
            "label": adapter_label,
            "avg_score": summary["avg_score"],
            "success_rate": summary["success_rate"],
            "by_family": summary["by_family"],
        }

        # Detach + free GPU memory for next adapter
        if not is_base:
            base_model = detach_lora(eval_model)

        log_file.close()
        sys.stdout = sys.__stdout__

    # Cross-adapter comparison (printed + saved)
    print(f"\n{'=' * 70}")
    print("CROSS-ADAPTER COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'adapter':40s}  avg     succ%   xlsx    docx    pptx")
    for tag, info in overall_summaries.items():
        bf = info["by_family"]
        print(f"  {tag:38s}  {info['avg_score']:.3f}   "
              f"{info['success_rate']:.0%}    "
              f"{bf.get('xlsx', {}).get('avg', 0):.3f}   "
              f"{bf.get('docx', {}).get('avg', 0):.3f}   "
              f"{bf.get('pptx', {}).get('avg', 0):.3f}")
    with open(parent_out / "cross_summary.json", "w") as f:
        json.dump(overall_summaries, f, indent=2)
    print(f"\nResults written to: {parent_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())