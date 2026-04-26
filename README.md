---
title: Office Document Task Environment
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /dashboard/
tags:
  - openenv
  - agent-environment
  - rl-environment
  - office
  - excel
  - word
  - powerpoint
---

# Office Document Task Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) **code-execution
environment** for training and evaluating LLM agents on **real-world office
document work** — Excel spreadsheets, Word documents, and PowerPoint decks.
The agent writes Python code (`openpyxl` / `python-docx` / `python-pptx`)
to read or modify authentic enterprise files and gets graded by a
**multi-layer, gaming-resistant** scoring stack.

> 119 tasks across 3 file formats · 22-task eval split · Real artifacts from
> [Finch (FinWorkBench)](https://huggingface.co/datasets/FinWorkBench/Finch),
> [OSWorld-Verified](https://github.com/xlang-ai/OSWorld), and
> [PPTArena](https://github.com/michaelofengend/PPTArena) · Multi-layer
> grading + four independent reward-hacking defenses.

---

## The story (~30 sec read)

**Problem.** Office workers spend hundreds of hours/year on spreadsheet, doc,
and slide work. Current LLMs are tested on each format in isolation, in
synthetic settings, with diff-based graders that an agent can game by
copying the gold file. Nobody trains end-to-end across the three formats
on real artifacts with proper anti-hacking defenses.

**Environment.** The agent gets a real `.xlsx`/`.docx`/`.pptx`, an
instruction in natural language, and a Python sandbox. It has 15 steps to
read, modify, and submit the file. Per-step rewards measure *real file
state* — did your code actually mutate the file? Did the file stay valid?
Did its structural distance to the gold reference actually decrease? Final
grade is a 2- or 3-layer composition: validity gate + structural diff +
(for docx) the per-task evaluator from OSWorld.

**Training pipeline.** Kimi-K2.5 (the teacher) ran on the 97 training tasks
to produce trajectories; we filtered them with a defense-in-depth pipeline
(score ≥ 0.4, more than 1 step, never `1-step submit_file`) into a 53-trajectory
SFT corpus. Qwen2.5-Coder-3B-Instruct (the student) was warm-started with
LoRA on this corpus across two configs (4K and 8K context) — both runs
logged on HF Jobs L40S, ~$0.50–0.80 each. The 8K run is online at
[bpHigh/qwen3b-office-sft-kimi-long](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi-long).

---

## Hero results — 22-task eval split

| Model | Avg score | Success rate | xlsx (n=10) | docx (n=4) | pptx (n=8) |
|---|---|---|---|---|---|
| **MiniMaxAI/MiniMax-M2.1** (frontier baseline) | 0.390 | 41% | 0.293 | 0.445 | 0.485 |
| **moonshotai/Kimi-K2.5** (teacher) | 0.481 | 52% | 0.370 | 0.472 | 0.673 |
| **Qwen/Qwen2.5-Coder-3B-Instruct** (student baseline) | **0.001** | 0% | 0.001 | 0.001 | 0.001 |
| **Qwen2.5-Coder-3B + LoRA SFT (4K)** | 0.006 | 0% | 0.007 | 0.005 | 0.005 |
| **Qwen2.5-Coder-3B + LoRA SFT (8K)** | 0.011 | 0% | 0.018 | 0.005 | 0.005 |

> **Reading the SFT rows.** Both adapters lift the vanilla baseline ~6–11×
> on the eval set, but every episode still bottoms out at the env's reward
> floor (0.005) — the model produces *parseable* code but it doesn't mutate
> the source file in ways the grader rewards. The SFT loss is well-
> converged (0.19 on the training distribution), so the gap is a
> generalization-from-Kimi-trajectories problem, not an under-training one.
> The *next* step — GRPO continuation directly against the env's reward
> signal — is what's expected to close this. See [`train_grpo.py`](train_grpo.py)
> and the rollout-format note in [`edits.md`](edits.md) Phase 13.

Reproduce any row:

```bash
# Hosted models via HF Router
python inference.py --split eval --model MiniMaxAI/MiniMax-M2.1 \
  --output-dir runs/baseline_minimax_m21_eval

# In-process LoRA eval (no hosted endpoint needed)
python eval_lora.py \
  --adapters bpHigh/qwen3b-office-sft-kimi,bpHigh/qwen3b-office-sft-kimi-long \
  --split eval --output-dir runs/sft_eval
```

Per-task breakdown lives in `runs/<dir>/summary.csv`; full step-by-step
trajectories in `runs/<dir>/trajectories/<task_id>.jsonl`.

---

## SFT training run — what the student saw

Student model: `Qwen/Qwen2.5-Coder-3B-Instruct`. LoRA r=32 on all-linear
targets, bf16, assistant-only loss masking. Two runs on 1× L40S 48GB
($1.80/hr).

![SFT loss — 4K vs 8K context length ablation](runs/sft_plots/comparison_4k_vs_8k.png)

| | 4K context | 8K context |
|---|---|---|
| Hardware | L40S 48GB | L40S 48GB |
| Runtime | 198s | 354s |
| Loss start → end | 0.412 → 0.069 | 0.384 → 0.103 |
| Final train_loss | 0.196 | 0.193 |
| Cost | ~$0.50 | ~$0.80 |
| Adapter | [bpHigh/qwen3b-office-sft-kimi](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi) | [bpHigh/qwen3b-office-sft-kimi-long](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi-long) |

The 8K curve has slightly higher end-loss because it sees the *long*
debugging trajectories from Kimi (5–8 of 53 episodes get truncated at 4K).
Same convergence target, harder distribution → which configuration generalizes
better is what the eval will tell us.

**Training artifacts — every run is reproducible from these:**

| Run | Adapter on Hub | Raw stdout log | HF Job page | Loss curve |
|---|---|---|---|---|
| 4K context | [bpHigh/qwen3b-office-sft-kimi](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi) | [raw_logs.txt](https://raw.githubusercontent.com/bp-high/openenv_financial_task_env/main/runs/sft_plots/qwen3b_kimi/raw_logs.txt) ([repo path](runs/sft_plots/qwen3b_kimi/raw_logs.txt)) | [Job 69ed74ae…4fc](https://huggingface.co/jobs/bpHigh/69ed74aed70108f37acdf4fc) | [PNG](runs/sft_plots/qwen3b_kimi/sft_loss_curve.png) |
| 8K context | [bpHigh/qwen3b-office-sft-kimi-long](https://huggingface.co/bpHigh/qwen3b-office-sft-kimi-long) | [raw_logs.txt](https://raw.githubusercontent.com/bp-high/openenv_financial_task_env/main/runs/sft_plots/qwen3b_kimi_long/raw_logs.txt) ([repo path](runs/sft_plots/qwen3b_kimi_long/raw_logs.txt)) | [Job 69ed7b51…ef4](https://huggingface.co/jobs/bpHigh/69ed7b51d2c8bd8662bceef4) | [PNG](runs/sft_plots/qwen3b_kimi_long/sft_loss_curve.png) |

Re-parse any HF Job's stdout into clean metrics + a loss curve PNG with
[`data_pipeline/analyze_sft_logs.py`](data_pipeline/analyze_sft_logs.py)
— takes a `--job-id` and emits `training_metrics.jsonl`, `summary.json`,
and `sft_loss_curve.png`. Both runs above were generated this way.

**Eval artifacts — both SFT adapters scored against the 22-task held-out split:**

| Run | Eval results.json | Raw stdout log | HF Job page |
|---|---|---|---|
| 4K context | [results.json](runs/sft_eval_v2/bpHigh_qwen3b-office-sft-kimi/results.json) | [raw_logs.txt](runs/sft_eval_v2/raw_logs.txt) | [Job 69ed97e5…2ad](https://huggingface.co/jobs/bpHigh/69ed97e5d2c8bd8662bcf2ad) |
| 8K context | [results.json](runs/sft_eval_v2/bpHigh_qwen3b-office-sft-kimi-long/results.json) | [raw_logs.txt](runs/sft_eval_v2/raw_logs.txt) | [Job 69ed97e5…2ad](https://huggingface.co/jobs/bpHigh/69ed97e5d2c8bd8662bcf2ad) |

Both adapters were evaluated in a single HF Jobs run (L40S, ~30 min, ~$1)
via [`eval_lora.py --adapters A,B`](eval_lora.py) — the base model loads
once and each adapter is detached/reattached without reloading.

---

## Task inventory (119 total)

| Family | Source | Train | Eval | Total | What it tests |
|---|---|---|---|---|---|
| `xlsx` | [Finch](https://huggingface.co/datasets/FinWorkBench/Finch) — hand-curated (Round 1) | 10 | 0 | 10 | Diverse Finch tasks hand-picked for the original submission (QA + MODIFY mix) |
| `xlsx` | [Finch](https://huggingface.co/datasets/FinWorkBench/Finch) — stratified pull (Round 2) | 40 | 10 | 50 | Stratified across 7 task-type tags |
| `docx` | [OSWorld-Verified](https://github.com/xlang-ai/OSWorld) (libreoffice_writer) | 17 | 4 | 21 | 16 distinct evaluator functions ported from `desktop_env/evaluators/metrics/docs.py` |
| `pptx` | [PPTArena](https://github.com/michaelofengend/PPTArena) | 30 | 8 | 38 | 16 distinct edit_types, including singletons (transitions, animations, A/V) |
| **Total** | | **97** | **22** | **119** | |

All 60 xlsx tasks come from **Finch (FinWorkBench)** — the 10 hand-curated
Round-1 picks plus the 50 stratified Round-2 pull. The 22-task eval set is
stratified (at least 1 task per tag bucket per family) so the benchmark
isn't biased toward one task type.

---

## How an episode works

```
reset(task_id="finch_10")
  ↓ obs.task_description = "Per the headers and established formula logic, populate
                            formulas for columns X through AH so the timing model's
                            performance statistics for 2013–2025 are complete..."
    obs.source_file = "/tmp/financial_env_finch_10_xxx/10_src_0.xlsx"
    obs.family = "xlsx"

step(action_type="code", content="...")     # 0–15 of these
  ↓ subprocess runs the code, returns stdout/stderr
  ↓ env measures: did the file change? is it still valid? did it move toward gold?
  ↓ reward = 0.005–0.10 (dense process reward)

step(action_type="submit_file", content="<path>")   # ends episode
  ↓ multi-layer grading
  ↓ reward = 0.001–0.999 (final grade)
```

Three action types: `"code"` (Python), `"submit"` (text answer for QA tasks),
`"submit_file"` (path to a modified file). **Submit is rejected on step 1** —
the agent must execute code at least once before any submission is accepted
(see *Defenses* below).

---

## Reward design

Two layers, both designed for *spec-aligned* signal.

### Per-step process reward (6 components, capped at 0.10/step)

| Signal | Range | Measured from |
|---|---|---|
| `exec_health` | 0–0.020 | Subprocess exit code; bonus if stdout non-empty |
| `lib_engagement` | 0–0.010 | Code uses the family's expected library |
| `mutation` | 0–0.030 | SHA-256 of working file changed |
| `validity` | 0–0.020 | Mutated file still parses with the family's loader |
| `progress` | 0–0.040 | Structural distance to gold *decreased* |
| `eval_check` | 0–0.020 | Per-task evaluator score *increased* (docx-only) |

`progress` and `eval_check` give RL a dense gradient *toward correctness*,
not just "code ran". They're disabled at eval time
(`FINANCIAL_ENV_PROGRESS=0`) to keep the benchmark honest.

### Final grade (per family)

| Family | Layer 1 (gate) | Layer 2 | Layer 3 |
|---|---|---|---|
| `xlsx` | – | 30% sheet-name match | 70% cell-level diff (2% numeric tolerance) |
| `docx` | python-docx parse + byte-equality refusal | 40% paragraph diff | 60% per-task OSWorld evaluator (`compare_docx_files`, `check_tabstops`, etc.) |
| `pptx` | python-pptx parse + byte-equality refusal | 20% slide-count | 80% avg per-shape composite: 40% text + 20% style + 20% position + 20% size |

The `docx` 3rd layer is a port of OSWorld's
[`metrics/docs.py`](https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/evaluators/metrics/docs.py)
(Apache-2.0). 16 evaluator functions, including compound `or` (multi-gold)
and `and` (all-must-pass) checks.

### Anti-hacking defenses (4 independent layers)

A model attempting the [Kimi-K2.5 exploit](edits.md#phase-7--live-discovered-exploit--anti-exploit-fix)
(submit unmodified source on step 1, score 0.998) hits **all four** of:

| Layer | Phase | What it does |
|---|---|---|
| **Env action gate** | 9 | Refuses `submit_file` if no code step has been taken — episode stays open for recovery |
| **Per-episode gold stash** | 4 | Gold files moved to `/tmp/oe_gold_<random>/` at episode start; restored on close. Defeats `glob('data/**/*Gold*')` searches |
| **Grader byte-equality refusal** | 7 | If submit's bytes match source bytes → score=0.001 (unless task is OSWorld `infeasible`) |
| **SFT corpus filter** | 8 | Builder drops `n_steps==1 + submit_file` trajectories even at high score |

See [`edits.md`](edits.md) for the live story of how Kimi found the exploit
during eval and the 3 fixes that followed.

---

## Action & Observation spaces

### `FinancialAction`

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `"code"`, `"submit"`, `"submit_file"` |
| `content` | `str` | Code, answer text, or absolute file path |

### `FinancialObservation`

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | e.g. `finch_10`, `osworld_0a0faba3`, `pptarena_case_60_fix_text_placement` |
| `task_description` | `str` | Instruction + constraints + source-file summary |
| `source_file` | `str` | Path to the working file (per-episode tmpdir copy) |
| `task_type` | `str` | `"QA"` or `"MODIFY"` |
| `feedback` | `str` | Stdout/stderr of code, or grading explanation. Includes the per-step reward decomposition. |
| `current_step` / `max_steps` | `int` | 0–15 |
| `done` | `bool` | Episode finished |
| `reward` | `float` | Step or final reward in (0.001, 0.999) |

---

## Setup & usage

### Prerequisites

- Python 3.10+
- Docker (for HF Space deployment)
- LLM API key (for `inference.py`) or HF Jobs subscription (for training)

### Local dev

```bash
pip install -e ".[dev]"
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000 \
  --ws-ping-interval 600 --ws-ping-timeout 600 --reload
```

### Run a baseline against a hosted model

```bash
export HF_TOKEN="hf_..."
python inference.py \
  --split eval \
  --model MiniMaxAI/MiniMax-M2.1 \
  --api-base https://router.huggingface.co/v1 \
  --env-url http://localhost:8000 \
  --task-timeout 900
```

### Run an in-process LoRA eval (no hosted endpoint needed)

```bash
python eval_lora.py \
  --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
  --adapters bpHigh/qwen3b-office-sft-kimi,bpHigh/qwen3b-office-sft-kimi-long \
  --split eval \
  --output-dir runs/sft_eval
```

CLI flags worth knowing on `inference.py`:
- `--split {train,eval,all}`
- `--family {xlsx,docx,pptx,all}`
- `--task-ids id1,id2,…` (overrides split/family)
- `--limit N`
- `--resume` — merge new task results into an existing `--output-dir`
- `--skip-completed` — re-run only failed/exploit tasks (paired with `--resume`)

### Re-pull data from upstream sources

```bash
python data_pipeline/finch_pull.py
python data_pipeline/osworld_writer_pull.py
python data_pipeline/pptarena_pull.py --root /path/to/PPTArena-main
```

### Build the SFT corpus

```bash
python data_pipeline/build_sft_corpus.py \
  --runs runs/teacher_kimi_k25_train \
  --output data/sft_kimi_k25.jsonl \
  --score-threshold 0.4
```

### Train the student (HF Jobs)

```bash
hf jobs run --flavor l40sx1 --timeout 8h --secrets HF_TOKEN \
  pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
  bash -c "apt-get update -qq && apt-get install -y -qq git && \
           git clone https://github.com/bp-high/openenv_financial_task_env.git /work && \
           cd /work && \
           pip install -U 'trl>=0.11' peft accelerate bitsandbytes && \
           python train_sft.py \
             --dataset data/sft_kimi_k25.jsonl \
             --base-model Qwen/Qwen2.5-Coder-3B-Instruct \
             --output-dir /tmp/qwen3b-sft \
             --epochs 4 --gradient-accumulation 2 --lora-r 32 \
             --max-seq-len 8192 \
             --push-to-hub bpHigh/qwen3b-office-sft-kimi-long"
```

### Docker

```bash
docker build -t office-task-env:latest .
docker run -p 8000:8000 office-task-env:latest
```

---

## Project structure

```
.
├── data/
│   ├── manifest.jsonl                 # 109 rows: 50 Finch + 21 OSWorld + 38 PPTArena
│   ├── sft_kimi_k25.jsonl             # 53 filtered teacher trajectories
│   ├── 0/, 21/, …                     # 10 hand-curated xlsx tasks
│   ├── finch_50/<id>/{src,ref}.xlsx
│   ├── osworld_writer/<uuid>/<src + N gold>.docx
│   └── pptarena/<slug>/{src,ref}.pptx
├── data_pipeline/
│   ├── finch_pull.py                  # Phase 1 — Finch xlsx tasks
│   ├── osworld_writer_pull.py         # Phase 3 — OSWorld docx tasks
│   ├── pptarena_pull.py               # Phase 5 — PPTArena pptx tasks
│   ├── build_sft_corpus.py            # Phase 8 — trajectories → SFT JSONL
│   ├── analyze_sft_logs.py            # Phase 10.1 — HF Job logs → metrics + PNG
│   └── compare_sft_runs.py            # overlay multiple SFT runs
├── graders/
│   ├── __init__.py                    # grade_xlsx + grade_docx + grade_pptx
│   └── docx_metrics.py                # 16 ported OSWorld evaluators
├── server/
│   ├── financial_environment.py       # OpenEnv environment + gold-stash + early-submit gate
│   └── app.py                         # FastAPI + WebSocket
├── rewards.py                         # 6-component RewardTracker
├── tasks.py                           # Manifest loader
├── inference.py                       # API-based eval (Router/Nebius/OpenAI)
├── eval_lora.py                       # In-process LoRA eval (no API needed)
├── train_sft.py                       # SFT trainer (TRL + PEFT, HF Jobs)
├── runs/                              # Baseline + teacher + SFT artifacts (incl. plots)
└── edits.md                           # Full Round-1 → Round-2 change log (10 phases)
```

---

## Round-1 → Round-2 change log

The journey from the original 10-task xlsx-only env to today's 3-format /
119-task / multi-layer-graded env is documented phase-by-phase in
[`edits.md`](edits.md): manifest loader, RewardTracker, OSWorld docx port,
PPTArena ingest, layout+style-aware pptx grader, gold-stash hardening,
inference v2, anti-exploit defenses, SFT corpus builder, training script,
log analyzer, in-process LoRA eval.

---

## Acknowledgments

- **Finch / FinWorkBench** ([dataset](https://huggingface.co/datasets/FinWorkBench/Finch),
  [paper](https://arxiv.org/abs/2512.13168)) — the xlsx tasks
- **OSWorld-Verified** ([repo](https://github.com/xlang-ai/OSWorld)) — the
  docx tasks and the evaluator functions in `graders/docx_metrics.py` (Apache-2.0)
- **PPTArena** ([repo](https://github.com/michaelofengend/PPTArena)) — the
  pptx tasks and the `evaluation_pairs_refined.json` schema
- **Kimi-K2.5** (Moonshot AI, served via Nebius) — the SFT teacher
- **Qwen2.5-Coder-3B-Instruct** (Alibaba Qwen team) — the student model
- **TRL + PEFT + Unsloth + transformers** — training stack
- **OpenEnv / Meta PyTorch** ([repo](https://github.com/meta-pytorch/OpenEnv)) — host framework
- **Hugging Face Jobs** — compute for SFT runs

If you use this environment in research, please cite the upstream datasets:

```bibtex
@article{dong2025finch,
  title={Finch: Benchmarking Finance \& Accounting across Spreadsheet-Centric Enterprise Workflows},
  author={Dong, Haoyu and Zhang, Pengkun and Gao, Yan and Dong, Xuanyu and Cheng, Yilin and Lu, Mingzhe and Yakefu, Adina and Zheng, Shuxin},
  journal={arXiv preprint arXiv:2512.13168},
  year={2025}
}
```
