# ruff: noqa: E402
"""
GRPO training notebook for the office-document task env.

Continues from `bpHigh/qwen3b-office-sft-kimi` (the SFT'd Qwen2.5-Coder-3B
LoRA) with GRPO on the 97 train tasks, using TRL's `environment_factory`
integration.  The env is loaded as a remote OpenEnv Space via WebSocket —
no env code runs in the training process.

Designed for **Modal notebooks** on a single A100 40GB ($2.50/hr).
Designed budget: ~$25–30 (3-5h training + 1-2h eval).

Format: each `# %%` is a cell — opens in Modal, Jupyter, or runs as a
plain script with `python train_grpo.py`.

References
----------
* TRL OpenEnv guide:    https://huggingface.co/docs/trl/openenv
* TRL Trackio guide:    https://huggingface.co/docs/trl/trackio_integration
* SFT base model:       https://huggingface.co/bpHigh/qwen3b-office-sft-kimi
* Env Space:            https://huggingface.co/spaces/bpHigh/financial-task-env

Pipeline
--------
1. Install deps + clone repo (for the env client + manifest)
2. Imports + auth check
3. Duplicate the Space to your account, set FINANCIAL_ENV_GOLD_STASH=copy
4. Define the OpenEnv tool wrapper (one tool per env action_type)
5. Build the train dataset (one prompt per train task)
6. Reward function (env.reward → list[float])
7. Load model + SFT adapter for trainable continuation
8. GRPO config + trainer (with Trackio logging)
9. Train (~3-5 hr) — live reward / loss curves on Trackio Space
10. Eval on 22-task held-out split (~1-2 hr)
11. Save + summarize results

Required env vars / Modal secrets
---------------------------------
* HF_TOKEN          — pull SFT adapter, push GRPO adapter to your Hub repo,
                      and host Trackio logs on a Space
* TRACKIO_SPACE_ID  — (recommended) HF Space ID where Trackio runs are
                      hosted live, e.g. 'bpHigh/trackio-office-grpo'.
                      If unset, logs land locally only (still inspectable
                      after the run from /tmp/grpo_qwen3b_office/).
* TRACKIO_PROJECT   — project group name (default 'office-doc-grpo')

IMPORTANT: before running, duplicate the env Space to your account and set
the env-var `FINANCIAL_ENV_GOLD_STASH=copy` in its Settings → Variables.
This switches gold-file stashing to per-session COPY mode so concurrent
GRPO rollouts don't race on the same source's rename.
"""

# %% [markdown]
# # GRPO training: SFT'd Qwen3B → GRPO on the office-document env
#
# **Starting point:** [bpHigh/qwen3b-office-sft-kimi](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi)
# (SFT'd on 53 Kimi-K2.5 trajectories, train_loss 0.196).
#
# **Goal:** continue with GRPO on the 97 train tasks via TRL's
# `environment_factory`, evaluate on the held-out 22-task split, push
# the trained adapter to HF Hub.
#
# **Budget:** ~$25–30 on Modal A100 40GB.

# %% [markdown]
# ## 1. Install deps + clone the env repo

# %%
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/bp-high/openenv_financial_task_env.git"
REPO_DIR = Path("/work/openenv_financial_task_env")
RUN_START = time.time()

if not REPO_DIR.exists():
    REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth=1", REPO_URL, str(REPO_DIR)], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "fetch", "origin", "main"], check=True)
subprocess.run(["git", "-C", str(REPO_DIR), "reset", "--hard", "origin/main"], check=True)

# Pip install the stack.  vLLM is required for `use_vllm=True` in colocate mode.
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-U",
    # typing_extensions>=4.15 is required by recent pydantic_core — the
    # Sentinel symbol pydantic_core imports was first added in 4.15.0.
    # Pin both first so the rest of the install resolves cleanly.
    "typing_extensions>=4.15",
    "pydantic>=2.9", "pydantic_core>=2.23",
    "trl>=0.11", "peft>=0.13", "accelerate>=1.0",
    "datasets>=3.0", "bitsandbytes>=0.43",
    "vllm>=0.6.0",
    "openenv-core>=0.2.0",
    "trackio",
], check=True)

# Add repo to path so we can import the env CLIENT + manifest helpers
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "server"))
os.chdir(REPO_DIR)

print(f"✓ Repo + deps ready at {REPO_DIR}")

# %% [markdown]
# ## 2. Imports + auth check

# %%
import json
from typing import List, Optional

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import GRPOConfig, GRPOTrainer

# Env client + helpers (from the cloned repo)
from client import FinancialTaskEnv
from models import FinancialAction
from tasks import TASKS, split_ids

assert os.environ.get("HF_TOKEN"), \
    "Set HF_TOKEN env var (Modal: add as a Secret) — needed for SFT-adapter pull and GRPO-adapter push"

# Trackio config — TRL picks these up via report_to='trackio'.
# TRACKIO_SPACE_ID is optional; if unset, runs are logged locally only.
os.environ.setdefault("TRACKIO_PROJECT", "office-doc-grpo")
if os.environ.get("TRACKIO_SPACE_ID"):
    print(f"✓ Trackio Space:  {os.environ['TRACKIO_SPACE_ID']}  (project='{os.environ['TRACKIO_PROJECT']}')")
else:
    print(f"⚠ TRACKIO_SPACE_ID unset — runs will log locally only "
          f"(project='{os.environ['TRACKIO_PROJECT']}')")

print(f"✓ Imports OK · CUDA: {torch.cuda.is_available()} · "
      f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## 3. Configure the env Space URL
#
# **Important:** before training, duplicate `bpHigh/financial-task-env`
# to your own HF account so you have a private Space with concurrency
# capacity for your training run.  Set the env-var
# `FINANCIAL_ENV_GOLD_STASH=copy` in the Space's Settings → Variables
# so concurrent GRPO rollouts don't race on the same gold file's rename.
#
# Then update `ENV_URL` below to your duplicate.

# %%
# Default points at the public Space — duplicate this for training.
ENV_URL = os.environ.get("ENV_URL", "https://bphigh-financial-task-env.hf.space")
print(f"Using env Space: {ENV_URL}")
# Quick health check — fail fast if the Space is sleeping or wrong URL
import urllib.request
try:
    with urllib.request.urlopen(f"{ENV_URL}/health", timeout=15) as r:
        if r.status != 200:
            raise RuntimeError(f"Space /health returned {r.status}")
    print("✓ Env Space is alive")
except Exception as e:
    print(f"⚠ Env Space health check failed: {e}")
    print(f"  Wake it up by visiting {ENV_URL} once, then re-run.")
    raise


# %% [markdown]
# ## 4. The OpenEnv tool wrapper
#
# This class is what TRL's `environment_factory` instantiates per generation.
# Each public method becomes a tool the model can call.  We expose three
# tools matching the env's three action types (`code`, `submit`, `submit_file`).
#
# The model sees these as function-callable tools with typed args + docstrings;
# TRL handles parsing + multi-turn rollout automatically.

# %%
class OfficeDocumentEnv:
    """OpenEnv wrapper for the cross-format office-document env.

    Exposes three tools matching the env's three action types:
      - run_python_code(code)    → action_type='code'
      - submit_text_answer(answer) → action_type='submit'  (QA tasks)
      - submit_file(path)        → action_type='submit_file' (MODIFY tasks)

    The model picks tools based on task_type (returned in reset).  Tool
    methods raise ValueError when called after the episode is done; the
    trainer catches these and feeds the message back to the model.
    """

    def __init__(self):
        # FinancialTaskEnv is async-by-default; wrap with .sync() and open
        # the WebSocket connection up-front so per-call latency is just RPC.
        self.client = FinancialTaskEnv(base_url=ENV_URL).sync()
        self.client.connect()
        self.reward = 0.0
        self.done = False
        self._task_type = "MODIFY"

    def __del__(self):
        # Best-effort WS cleanup — TRL doesn't manage env lifecycle explicitly.
        try:
            self.client.close()
        except Exception:
            pass

    def reset(self, task_id: Optional[str] = None, **kwargs) -> Optional[str]:
        """Receives task_id (and any other dataset columns) as kwargs."""
        result = self.client.reset(task_id=task_id) if task_id else self.client.reset()
        obs = result.observation
        self.reward = 0.0
        self.done = False
        self._task_type = obs.task_type
        # Initial observation as a single string the model sees
        return (
            f"{obs.task_description}\n\n"
            f"Source file: {obs.source_file}\n"
            f"Family: {getattr(obs, 'task_type', 'MODIFY')}\n\n"
            f"{obs.feedback}"
        )

    def run_python_code(self, code: str) -> str:
        """Execute Python code in the env's sandbox.

        Use this to read or modify the source file. Variables do NOT
        persist between calls — each call runs in a fresh subprocess.
        Available libs: openpyxl, python-docx, python-pptx, Pillow.

        Args:
            code: Python source to execute.

        Returns:
            stdout/stderr from the code, plus per-step reward decomposition.
        """
        if self.done:
            raise ValueError("Episode already finished — submit your answer.")
        result = self.client.step(FinancialAction(action_type="code", content=code))
        self.reward = result.reward
        self.done = result.done
        return result.observation.feedback

    def submit_file(self, path: str) -> str:
        """Submit the modified file as the final answer (MODIFY tasks).

        Args:
            path: Absolute filesystem path to the modified file. Use the
                  source_file path from the initial observation.

        Returns:
            Grading result. The episode ends after this call.
        """
        if self.done:
            raise ValueError("Episode already finished.")
        result = self.client.step(FinancialAction(action_type="submit_file", content=path))
        self.reward = result.reward
        self.done = True
        return result.observation.feedback

    def submit_text_answer(self, answer: str) -> str:
        """Submit a text answer (QA tasks like 'How many plants?').

        Args:
            answer: The text answer to submit. May include numbers or
                    descriptive text — the grader extracts numbers and
                    matches keywords.

        Returns:
            Grading result. The episode ends after this call.
        """
        if self.done:
            raise ValueError("Episode already finished.")
        result = self.client.step(FinancialAction(action_type="submit", content=answer))
        self.reward = result.reward
        self.done = True
        return result.observation.feedback


# Smoke test the wrapper
print("Smoke testing the env wrapper on a single task...")
_smoke = OfficeDocumentEnv()
obs = _smoke.reset(task_id="finch_10")
print(f"  reset OK: obs is {len(obs)} chars")
fb = _smoke.run_python_code("print('hello')")
print(f"  run_python_code OK: reward={_smoke.reward:.3f}, done={_smoke.done}")
del _smoke


# %% [markdown]
# ## 5. Build the train dataset
#
# One row per train task. The `task_id` column is passed through to
# `reset()` via kwargs.  TRL handles the rollout loop — we don't need to
# generate prompts; the user message is the system prompt only, the env's
# initial observation comes from `reset()`.

# %%
# This system prompt mirrors the one in inference.py that produced the
# Kimi-K2.5 / MiniMax baseline scores AND the SFT corpus.  Markdown
# ```python``` blocks for code, plain SUBMIT_FILE: / SUBMIT_ANSWER: text
# directives for finalization.  This is the format the SFT'd model already
# emits fluently from training - matching it removes the format mismatch
# that was causing 0 rewards under TRL's <tool_call>-XML expectation.
SYSTEM_PROMPT = """You are an expert at editing office documents (Excel, Word, PowerPoint) with Python.
You are working with a real .xlsx, .docx, or .pptx file.  Pick the right
library based on the source_file path shown in the initial observation:
  - .xlsx -> openpyxl  (load_workbook / wb.save)
  - .docx -> python-docx  (Document / doc.save)
  - .pptx -> python-pptx  (Presentation / prs.save)

CRITICAL RULES:
1. Do NOT call reset(). Just write plain Python code in a ```python``` block.
2. Use the EXACT file path provided in the observation. Do not guess paths.
3. Each code block runs in a FRESH subprocess - you must re-import and re-open
   the file every time. Variables do NOT persist between steps.
4. Use print() liberally to see data. Read the output carefully before your
   next step.
5. You have at most 12 steps per episode. Be efficient: explore in step 1,
   solve in step 2-3, submit.
6. **You MUST execute at least one code step before submitting.** The env
   will reject SUBMIT_ANSWER and SUBMIT_FILE on step 1 - you need to read
   or modify the file with code first. Submitting the source file unchanged
   is never a correct solve.

RESPONSE FORMAT - use EXACTLY one of these three forms per response:

To run Python code:
```python
your code here
```

To submit a text answer (QA tasks like 'How many plants?'):
SUBMIT_ANSWER: your answer here

To submit a modified file (MODIFY tasks):
SUBMIT_FILE: /absolute/path/to/saved.<ext>

For MODIFY tasks: load with the right library, mutate, save to the SAME path,
then SUBMIT_FILE that path."""

train_ids = split_ids("train")
# Drop hand-curated tasks (task_*) to focus GRPO on the larger Round-2 pool;
# the SFT was already exposed to the hand-curated set via Kimi trajectories.
train_ids = [tid for tid in train_ids if not tid.startswith("task_")]
print(f"Train tasks for GRPO: {len(train_ids)}")

train_data = [
    {"prompt": [{"role": "user", "content": SYSTEM_PROMPT}], "task_id": tid}
    for tid in train_ids
]
train_ds = Dataset.from_list(train_data)


# %% [markdown]
# ## 6. Reward function — read env.reward after each episode

# %%
def env_reward(environments, **kwargs) -> List[float]:
    """TRL's environment_factory passes the list of OfficeDocumentEnv instances
    after the multi-turn rollout ends.  We read `env.reward` (set by the last
    tool call before done=True, OR the per-step reward if the episode ran out
    of steps)."""
    return [float(env.reward) for env in environments]


# %% [markdown]
# ## 7. Load base model + SFT adapter (trainable continuation)

# %%
BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
SFT_ADAPTER = "bpHigh/qwen3b-office-sft-kimi"
GRPO_HUB_ID = "bpHigh/qwen3b-office-grpo"

print(f"Loading tokenizer: {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = "left"

# TRL's environment_factory path runs add_response_schema(tokenizer) to parse
# tool calls from completions.  Auto-detection fails for Qwen2.5-Coder's chat
# template (it only knows qwen3, qwen3.5, llama3, glm4, gptoss), so we attach
# qwen3_schema manually — it parses <tool_call>...</tool_call> blocks via
# regex and is template-agnostic, so it works fine for Qwen2.5 too.
from trl.chat_template_utils import qwen3_schema
tokenizer.response_schema = qwen3_schema

print(f"Loading base model: {BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)

print(f"Attaching SFT adapter (trainable): {SFT_ADAPTER}")
model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, is_trainable=True)
model.print_trainable_parameters()


# %% [markdown]
# ## 8. GRPO config + trainer

# %%
config = GRPOConfig(
    output_dir="/tmp/grpo_qwen3b_office",
    num_train_epochs=1,
    learning_rate=1e-5,                 # gentler than SFT's 2e-4
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,      # 4 prompts × 2 gens = 8 rollouts/step (fits 40GB w/ vLLM colocate)
    num_generations=2,
    max_completion_length=2048,         # cumulative assistant tokens across all tool-call turns
    vllm_max_model_length=10240,        # prompt + completion + env-feedback growth across ≤12 turns
    vllm_gpu_memory_utilization=0.22,   # leave more VRAM for trainer forward pass
    max_tool_calling_iterations=12,     # cap multi-turn rollouts (env's MAX_STEPS=15 is the hard ceiling)
    temperature=0.8,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=1,
    save_steps=20,
    save_total_limit=2,
    warmup_ratio=0.05,
    use_vllm=True,                      # 2-3× faster generation
    vllm_mode="colocate",               # single GPU
    chat_template_kwargs={"enable_thinking": False},
    log_completions=True,
    push_to_hub=True,
    hub_model_id=GRPO_HUB_ID,
    hub_strategy="end",
    hub_private_repo=False,
    report_to="trackio",
    run_name="grpo_qwen3b_office_sft-kimi",
    seed=42,
)

print("Creating GRPOTrainer with environment_factory=OfficeDocumentEnv...")
trainer = GRPOTrainer(
    model=model,
    args=config,
    train_dataset=train_ds,
    reward_funcs=env_reward,
    environment_factory=OfficeDocumentEnv,    # ← TRL handles the multi-turn loop + XML tool-call parsing
    processing_class=tokenizer,
)


# %% [markdown]
# ## 9. Train

# %%
print("Starting GRPO training...")
train_start = time.time()
trainer.train()
train_dur = time.time() - train_start
print(f"✓ Training complete in {train_dur / 60:.1f} min")

trainer.save_model("/tmp/grpo_qwen3b_office")
tokenizer.save_pretrained("/tmp/grpo_qwen3b_office")
trainer.push_to_hub()
print(f"✓ Pushed to HF Hub: {GRPO_HUB_ID}")


# %% [markdown]
# ## 10. Evaluate on the 22-task held-out eval split
#
# Run greedy multi-step rollouts via the same env wrapper, return per-task
# scores in the same format as `runs/sft_eval_v2/*/results.json` so the
# comparison plot can pick them up directly.

# %%
@torch.inference_mode()
def run_eval_episode(task_id: str, max_steps: int = 15) -> dict:
    """Greedy rollout via the env wrapper; return final score + step count."""
    env = OfficeDocumentEnv()
    actions = []
    obs = env.reset(task_id=task_id)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs},
    ]

    final_reward = 0.0
    n_steps = 0
    for step in range(1, max_steps + 1):
        n_steps = step
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=24000)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=1500,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        # Naive tool-call parser for eval (we don't use TRL's loop here).
        # The model should emit either:
        #   ```python ... ``` for code
        #   SUBMIT_FILE: <path> for file submit
        #   SUBMIT_ANSWER: <text> for text submit
        # If the trained policy uses TRL tool-calling format instead, this
        # naive parser will fail and the eval score will be low — see
        # eval_lora.py for a more robust parser.
        import re
        if "SUBMIT_FILE:" in response:
            path = response.split("SUBMIT_FILE:", 1)[1].strip().splitlines()[0].strip()
            try:
                env.submit_file(path)
            except Exception:
                pass
            final_reward = env.reward
            break
        if "SUBMIT_ANSWER:" in response:
            ans = response.split("SUBMIT_ANSWER:", 1)[1].strip()
            try:
                env.submit_text_answer(ans)
            except Exception:
                pass
            final_reward = env.reward
            break
        m = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
        code = m.group(1).strip() if m else response.strip()
        try:
            fb = env.run_python_code(code)
        except ValueError:
            final_reward = env.reward
            break
        actions.append("code")
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": (
            f"Code execution result (step {step}/{max_steps}):\n{fb}"
        )})
        if env.done:
            final_reward = env.reward
            break

    return {
        "task_id": task_id,
        "family": TASKS[task_id].get("family", "?") if task_id in TASKS else "?",
        "score": max(0.001, min(0.999, final_reward)),
        "steps": n_steps,
    }


eval_ids = split_ids("eval")
print(f"\nEvaluating GRPO model on {len(eval_ids)} held-out tasks...")
eval_start = time.time()
eval_results = []
for tid in eval_ids:
    r = run_eval_episode(tid)
    eval_results.append(r)
    print(f"  {r['task_id']:55s} ({r['family']:4s}) score={r['score']:.3f} steps={r['steps']}")
eval_dur = time.time() - eval_start
print(f"\n✓ Eval done in {eval_dur / 60:.1f} min")


# %% [markdown]
# ## 11. Save + summarize

# %%
avg_score = sum(r["score"] for r in eval_results) / len(eval_results)
success_rate = sum(1 for r in eval_results if r["score"] >= 0.5) / len(eval_results)

by_family = {}
for r in eval_results:
    by_family.setdefault(r["family"], []).append(r["score"])
fam_summary = {
    fam: {"n": len(scores), "avg": round(sum(scores) / len(scores), 4)}
    for fam, scores in by_family.items()
}

total_dur_s = time.time() - RUN_START
output = {
    "model": GRPO_HUB_ID,
    "base": SFT_ADAPTER,
    "n_tasks": len(eval_results),
    "avg_score": round(avg_score, 4),
    "success_rate": round(success_rate, 4),
    "by_family": fam_summary,
    "results": eval_results,
    "wall_clock": {
        "total_min": round(total_dur_s / 60, 1),
        "training_min": round(train_dur / 60, 1),
        "eval_min": round(eval_dur / 60, 1),
        "estimated_cost_usd": round((total_dur_s / 3600) * 2.50, 2),
    },
}

with open("/tmp/grpo_eval_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\n" + "=" * 70)
print(f"GRPO training + eval complete")
print(f"  model:        {GRPO_HUB_ID}")
print(f"  avg score:    {avg_score:.3f}")
print(f"  success rate: {success_rate:.0%}")
print(f"  by family:")
for fam, info in fam_summary.items():
    print(f"    {fam}: avg={info['avg']:.3f}  n={info['n']}")
print(f"  wall-clock:   {output['wall_clock']['total_min']:.1f} min")
print(f"  est cost:     ${output['wall_clock']['estimated_cost_usd']:.2f}")
print("=" * 70)
print(f"\nResults: /tmp/grpo_eval_results.json")
print(f"Adapter: https://huggingface.co/{GRPO_HUB_ID}")