# Building a cross-format office-document RL environment

## What this is

[`bpHigh/financial-task-env`](https://huggingface.co/spaces/bpHigh/financial-task-env)
is an **OpenEnv** environment for training agents to edit office documents
with code. It started as a the online submission with 10 hand-picked Finch
spreadsheet tasks. For Round 2/ Offline hack phase it expanded to **119 tasks across three
file formats** ,  Excel (xlsx), Word (docx), PowerPoint (pptx) and with a
multi-layer reward function designed to be hard to hack and a four-defense
anti-exploit stack hardened against agents that try to game it.

The environment is live as a Hugging Face Space at
`https://bphigh-financial-task-env.hf.space`. The TRL/OpenEnv-compatible
client lives in [`client.py`](https://huggingface.co/spaces/bpHigh/financial-task-env/blob/main/client.py); the dashboard shows a leaderboard, training plots, anti-hacking design, and step-by-step replays of the teacher solving tasks. Everything in the
repo is reproducible.

## The design that still holds up

### The task split

| Family | Source | Train | Eval | Total | What it tests |
|---|---|---|---|---|---|
| `xlsx` | Finch (FinWorkBench) - hand-curated (Round 1) | 10 | 0 | 10 | Diverse Financial workbook tasks for the original submission |
| `xlsx` | Finch - stratified pull (Round 2) | 40 | 10 | 50 | Stratified across 7 task-type tags |
| `docx` | OSWorld-Verified `libreoffice_writer` | 17 | 4 | 21 | 16 distinct evaluator functions ported from OSWorld |
| `pptx` | PPTArena | 30 | 8 | 38 | 16 distinct edit_types, including singletons (transitions, animations, A/V) |
| **Total** | | **97** | **22** | **119** | |

The eval split is **stratified** so that at least one task per tag bucket per
family so the environment doesn't get gamed by overfitting to one task
type.

### The 6-component reward

Each step the agent runs code and gets back stdout/stderr **plus an
inline reward decomposition**. Six components, all clamped, totalling
0.10/step max:

| Component | Range | Fires when |
|---|---|---|
| `exec_health` | 0-0.020 | Subprocess exit code 0; bonus if stdout non-empty |
| `lib_engagement` | 0-0.015 | Code uses the family's expected library (openpyxl / python-docx / python-pptx) |
| `mutation` | 0-0.025 | Working file's SHA-256 changed since last step |
| `validity` | 0-0.015 | Mutated file still parses with its loader |
| `progress` | 0-0.015 | Structural distance to the gold reference *decreased* |
| `eval_check` | 0-0.010 | Per-task evaluator score went up (docx-only) |

The final reward at `submit` is **not** the sum of step rewards, it's
the grader's score on the submitted file (or text answer). The step
rewards exist to give shaped signal for mid-episode learning. The
clamps matter: an agent can't farm tiny rewards for 1000 steps and
beat someone who actually solves the task.

### Anti-hacking - 4 independent defenses

This part I'm proud of. After watching teacher trajectories, we found
the model had figured out it could "submit `submit_file` with the
*source* file path on step 1" and getting some reward because the source
parses, has the right size, etc. and without making any edits. So we
built four overlapping defenses, any of which independently kills that
exploit:

1. **Phase 4 - Per-episode gold stash**: at task reset, gold files are
   moved to a uuid4-randomized tmp directory. An agent globbing `data/`
   finds nothing.
2. **Phase 7 - Grader byte-equality check**: if the submitted file is
   byte-equal to the source, the grader returns the floor reward.
3. **Phase 9 - Env early-submit gate**: the env refuses `submit` /
   `submit_file` until at least one `code` step has run. Configured via
   `FINANCIAL_ENV_MIN_CODE_STEPS=1` (default).
4. **Phase 8 - SFT corpus filter**: any teacher trajectory that did
   `submit_file` on step 1 was dropped before SFT, so the student
   never learns the cheat from the teacher.
   
Each layer covers a slightly different attack surface.

---

## SFT training (cheap, converged, unhelpful)

Two SFT runs on 1× L40S 48GB at \$1.80/hr. Same recipe both times:
LoRA r=32 on all-linear targets, bf16, assistant-only loss masking,
TRL `SFTTrainer`. The only difference: 4K vs 8K context length.

| | 4K context | 8K context |
|---|---|---|
| Hardware | L40S 48GB | L40S 48GB |
| Runtime | 198s | 354s |
| Loss start → end | 0.412 → 0.069 | 0.384 → 0.103 |
| Final train_loss | **0.196** | **0.193** |
| Cost | ~\$0.50 | ~\$0.80 |

Loss converges nicely. The 8K run sees the *long* debugging
trajectories from Kimi (5-8 of 53 episodes get truncated at 4K), so
it's the more interesting comparison point.


![image](https://cdn-uploads.huggingface.co/production/uploads/623f2f5828672458f74879b3/PNjl8xGOpIfS5baDDV0T5.png)

### Then we evaluated

| Model | Avg score | Success rate | xlsx (n=10) | docx (n=4) | pptx (n=8) |
|---|---|---|---|---|---|
| MiniMaxAI/MiniMax-M2.1 (frontier baseline) | 0.390 | 41% | 0.293 | 0.445 | 0.485 |
| moonshotai/Kimi-K2.5 (teacher) | **0.481** | 52% | 0.370 | 0.472 | 0.673 |
| Qwen/Qwen2.5-Coder-3B-Instruct (vanilla student) | 0.001 | 0% | 0.001 | 0.001 | 0.001 |
| Qwen2.5-Coder-3B + LoRA SFT (4K) | 0.006 | 0% | 0.007 | 0.005 | 0.005 |
| Qwen2.5-Coder-3B + LoRA SFT (8K) | **0.011** | 0% | 0.018 | 0.005 | 0.005 |

The SFT lift is real - **6×-11× over vanilla** - but every episode
still bottoms out at the env's 0.005 reward floor. The model
*produces parseable code*, but it doesn't mutate files in ways the
grader rewards. SFT loss is well-converged on the training
distribution; the gap is **generalization**, not under-training. We
trained on 53 trajectories; the eval has 22 tasks all OOD; that's a
brutal regime for a 3B model with LoRA.

---

## Pointers

- **Code**: <https://github.com/bp-high/openenv_financial_task_env>
- **Env Space (live)**: <https://huggingface.co/spaces/bpHigh/financial-task-env>
- **SFT-4K adapter**: <https://huggingface.co/bpHigh/qwen3b-office-sft-kimi>
- **SFT-8K adapter**: <https://huggingface.co/bpHigh/qwen3b-office-sft-kimi-long>
- **Trackio**: <https://huggingface.co/spaces/bpHigh/trackio-office-grpo>
- **Edits log** (build journal, all 13 phases): [`edits.md`](https://huggingface.co/spaces/bpHigh/financial-task-env/blob/main/edits.md)
- **Re-deploy checklist**: [`edits.md`](https://huggingface.co/spaces/bpHigh/financial-task-env/blob/main/edits.md) - bottom section

If you're a judge reading this and want to see the env do something
in 30 seconds, open the dashboard's "Replay" widget and watch the
docx task, it's the shortest end-to-end demonstration of the
6-component reward in motion.


![image](https://cdn-uploads.huggingface.co/production/uploads/623f2f5828672458f74879b3/7O5Vbqdbdh7k9u-4PYRjU.png)



![image](https://cdn-uploads.huggingface.co/production/uploads/623f2f5828672458f74879b3/lOC0L9aLkOrXhk3B1IFrp.png)

![image](https://cdn-uploads.huggingface.co/production/uploads/623f2f5828672458f74879b3/aSMgluM3dthKSHDFt_qCC.png)

The headers in [`edits.md`](https://huggingface.co/spaces/bpHigh/financial-task-env/blob/main/edits.md) - Phases 1 through 13 are the long version of this blog.

---

## GRPO - what didn't work, what we tried, where we are now

This section is the truth: GRPO has not had a clean training run yet.
Three failure modes deep, still iterating. I'm leaving it here because
the failure modes are **interesting** and pretending we cleared them
in one shot would be dishonest.

### Setup

- **Hardware**: A100 40GB on Modal Notebooks (\$2.50/hr).
- **Stack**: TRL `GRPOTrainer` + vLLM colocate + PEFT (continuing the SFT'd LoRA).
- **Multi-turn**: 12 turns per episode, agent uses three tools
  (`run_python_code`, `submit_file`, `submit_text_answer`).
- **Env**: same Space as eval, but with
  `FINANCIAL_ENV_GOLD_STASH=copy` (concurrent rollouts can't race on
  the gold-file rename) and `SUPPORTS_CONCURRENT_SESSIONS=True` on the
  `FinancialEnvironment` class.
- **Tracking**: Trackio Space at
  [`bpHigh/trackio-office-grpo`](https://huggingface.co/spaces/bpHigh/trackio-office-grpo).

### Failure 1 - Reward stuck at 0 ("the format mismatch")

First run started with `environment_factory=OfficeDocumentEnv` (TRL's
managed multi-turn rollout). Trackio showed reward = 0.0 across every
step. Captured a sample completion:

````
```json
{"name": "run_python_code", "arguments": {"code": "..."}}
```
````

The SFT'd model emits **markdown JSON blocks** (because the SFT teacher
Kimi-K2.5 emits markdown JSON), but TRL's tool-call parser only
accepts `<tool_call>...</tool_call>` **XML** (it has hard-coded
schemas for qwen3, qwen3.5, llama3, glm4, gptoss - Qwen2.5-Coder
isn't on that list). Parser found 0 tool calls, env never executed
code, reward stayed 0, advantage was 0, gradients were 0. ~5 min of
A100 time burned learning nothing.

To rule out "SFT broke it", I tried the same setup with the **base
model + fresh LoRA** (no SFT). Same failure. So the issue isn't
SFT-induced - Qwen2.5-Coder's chat template just defaults to markdown
JSON when tools are bound, regardless of SFT.

### Attempt at fix #1 - Custom `rollout_func`

TRL has two rollout paths: `environment_factory` (managed, XML-only)
and `rollout_func` (you control everything). I wrote ~150 LOC of
custom rollout that:
- Spawned an `OfficeDocumentEnv` per generation
- Called vLLM directly per turn
- Parsed completions for ` ```json ... ``` ` blocks (with fallbacks
  for ` ```python ... ``` ` and Kimi's `<|tool_call_begin|>` markers)
- Dispatched to env tools, tokenized feedback as a chat-template diff,
  appended with `env_mask=0` so loss only flowed through model tokens

It **worked**. First batch printed:
```
[rollout_func] Done: 16/16 rollouts got non-zero reward
```

The format problem was solved. Then…

### Failure 2 - OOM during gradient computation

```
OutOfMemoryError: CUDA out of memory. Tried to allocate 12.73 GiB.
GPU 0 has 39.49 GiB total, 8.80 GiB free, 30.69 GiB in use.
```

`selective_log_softmax` trying to compute logits over 16 rollouts ×
~10-20K tokens each. The completions had grown enormous because each
rollout was 4-8 turns × `max_completion_length=4096` per turn. Add
vLLM colocate's ~10GB footprint and there's no headroom.

Fix: lower `max_completion_length` (4096 → 1500), drop
`gradient_accumulation_steps` (8 → 4), enable `vllm_enable_sleep_mode=True`,
lower `vllm_gpu_memory_utilization` (0.3 → 0.22).

### Failure 3 - CUDA index-out-of-bounds (twice)

```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111: operator():
block: [0,0,0], thread: [2,0,0]
Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

Async-reported, so the stack trace pointed at a benign downstream op
(`shuffle_sequence_dict`'s `v[permutation]`). The real cause was
*earlier* in the gradient pass - likely a token ID outside the
model's embedding range slipping into `completion_ids`.

Hypothesis: `vllm_enable_sleep_mode=True` + LoRA is buggy. vLLM wakes
from sleep by reloading weights from disk - but the on-disk weights
are the **base model**, not the LoRA-adapted one. So vLLM generates
from base while the trainer expects LoRA-adapted distribution; some
edge token mismatches and `torch.gather` blows up.

Disabled sleep mode, also disabled `vllm_importance_sampling_correction`
(which is what calls `selective_log_softmax`). **Still failed** - same
async assert, this time deeper in `_prepare_inputs`. CUDA context
becomes poisoned after a device-side assert; even subsequent
`torch.cuda.manual_seed_all()` raises the same error. **Forced kernel
restart**, lose the loaded model.

### Where we are now

Three full GRPO runs in, no successful gradient step yet, ~30 min of
A100 time burned across debug iterations. Two paths forward:

**Path A - Revert to `environment_factory`, force XML via prompt
engineering.** Restore the original setup but rewrite the SYSTEM_PROMPT
with explicit "DO NOT use ```json or ```python" warnings and three
concrete `<tool_call>` few-shot examples. The SFT-bias toward markdown
will fight the prompt, but maybe 30-50% compliance is enough to start
producing gradients. Cheapest to try.

**Path B - Custom rollout_func with `CUDA_LAUNCH_BLOCKING=1`.** Re-run
the rollout_func setup with synchronous CUDA so the next assert
points at the actual line. Probably reveals an OOV-token issue we can
clip defensively in the rollout_func itself.

I'm currently on **Path A** - already reverted the script, prompt now
hard-pushes XML format, env_factory back in place. Restart kernel,
push the revert, see what happens.

If Path A also fails, the fallback is to skip Qwen2.5-Coder entirely
and use a smaller Qwen3 model (which TRL's parser knows natively) for
the GRPO comparison. Less interesting story but faster path to a
working number.

---

## What I'd do differently

- **Pick a model TRL natively supports for the tool-calling format.**
  Qwen3 (any size) instead of Qwen2.5-Coder. The format mismatch ate
  hours.
- **Rule out vLLM `sleep_mode` + LoRA from the start.** This combo is
  fragile. Either disable sleep mode and pay the memory cost, or pick
  a smaller model that fits comfortably without sleeping.
- **Test the SFT model on the env *before* spending eval budget.**
  53 trajectories was always going to be a tight regime; an
  early in-process smoke test with `eval_lora.py` on 4-5 tasks
  would have set expectations correctly.
- **Trackio rename, not delete, for failed runs.** The `office-doc-grpo`
  project on Trackio has a 0-reward run that's evidence of the
  format-mismatch issue, not waste. It stays as `-attempt1`.

---

## What still needs to happen

- A successful GRPO run end-to-end (any path) with non-zero
  reward curve in Trackio.
- Eval the GRPO'd model on the 22-task held-out split. Fill in
  the leaderboard's "GRPO" row.
- Fair comparison: GRPO from base + fresh LoRA vs GRPO from SFT'd
  adapter. Both runs need to use the same prompt format and rollout
  path so the comparison is real.
- Tighten the dashboard - add a Trackio embed in the dashboard once
  GRPO is producing curves worth showing.

---

