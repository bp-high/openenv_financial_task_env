---
title: Office Document Task Environment
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
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
to read or modify authentic enterprise files, and gets graded by a
**multi-layer, gaming-resistant** scoring stack.

> 119 tasks across 3 file formats. 22-task eval split. Real artifacts from
> [Finch (FinWorkBench)](https://huggingface.co/datasets/FinWorkBench/Finch),
> [OSWorld-Verified](https://github.com/xlang-ai/OSWorld), and
> [PPTArena](https://github.com/michaelofengend/PPTArena). Multi-layer
> grading: validity gate → structural diff → spec-aligned per-task evaluator.

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

**Result you can reproduce today.** A frontier 270B-class model
(MiniMax-M2.1) gets **0.390 avg / 41% success rate** on the 22-task eval.
A small 3B trainable target (Qwen2.5-Coder-3B-Instruct) gets **0.002 avg
/ 0% success**. That gap is the RL training story.

---

## Hero results — 22-task eval split

| Model | Avg score | Success rate | xlsx (n=10) | docx (n=4) | pptx (n=8) |
|---|---|---|---|---|---|
| **MiniMaxAI/MiniMax-M2.1** (frontier baseline) | **0.390** | 41% | 0.293 | 0.445 | 0.485 |
| **Qwen/Qwen2.5-Coder-3B-Instruct** (training target) | **0.002** | 0% | 0.003 | 0.001 | 0.003 |
| **Qwen3-Coder-3B-RL** *(after SFT + GRPO — TBD)* | *coming* | *coming* | *coming* | *coming* | *coming* |

Reproduce:

```bash
# MiniMax baseline
python inference.py --split eval --model MiniMaxAI/MiniMax-M2.1 \
  --output-dir runs/baseline_minimax_m21_eval

# Qwen baseline
python inference.py --split eval --model Qwen/Qwen2.5-Coder-3B-Instruct \
  --output-dir runs/baseline_qwen25coder3b_eval
```

Per-task breakdown lives in `runs/<dir>/summary.csv` and full step-by-step
trajectories in `runs/<dir>/trajectories/<task_id>.jsonl`.

---

## Task inventory (119 total)

| Family | Source | Train | Eval | Total | What it tests |
|---|---|---|---|---|---|
| `xlsx` | Hand-curated (Round 1) | 10 | 0 | 10 | Diverse Finch tasks (QA + MODIFY) |
| `xlsx` | [Finch](https://huggingface.co/datasets/FinWorkBench/Finch) | 40 | 10 | 50 | Stratified across 7 task-type tags |
| `docx` | [OSWorld-Verified](https://github.com/xlang-ai/OSWorld) (libreoffice_writer) | 17 | 4 | 21 | 16 distinct evaluator functions ported from `desktop_env/evaluators/metrics/docs.py` |
| `pptx` | [PPTArena](https://github.com/michaelofengend/PPTArena) | 30 | 8 | 38 | 16 distinct edit_types, including singletons (transitions, animations, A/V) |
| **Total** | | **97** | **22** | **119** | |

The 22-task eval set is stratified — at least 1 task per tag bucket — so the
benchmark isn't biased toward one task type.

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
  ↓ reward = 0.005–0.10 (dense process reward, see below)

step(action_type="submit_file", content="<path>")   # ends episode
  ↓ multi-layer grading
  ↓ reward = 0.001–0.999 (final grade)
```

Three action types: `"code"` (Python), `"submit"` (text answer for QA tasks),
`"submit_file"` (path to a modified file).

---

## Reward design

This is the most opinionated part of the env, because the [judging guide](https://docs.google.com/document/d/1Odznuzwtb1ecDOm2t6ToZd4MuMXXfO6vWUGcxbC6mFs/edit)
explicitly calls out reward hacking as a top failure mode. Two layers, both
designed for *spec-aligned* signal.

### Per-step process reward (6 components, capped at 0.10/step)

Every code step gets scored across six independent signals, all measured
from real file state — not regex on the agent's code:

| Signal | Range | What it actually checks |
|---|---|---|
| `exec_health` | 0–0.020 | Subprocess exited 0; bonus if stdout non-empty |
| `lib_engagement` | 0–0.010 | Code uses the family's expected library (`openpyxl` / `python-docx` / `python-pptx`) |
| `mutation` | 0–0.030 | SHA-256 of the working file changed since last step |
| `validity` | 0–0.020 | Mutated file still parses with the family's loader (no corruption) |
| `progress` | 0–0.040 | Structural distance to gold *decreased* this step |
| `eval_check` | 0–0.020 | Per-task evaluator score *increased* (docx-only currently) |

`progress` and `eval_check` give RL a dense gradient *toward correctness*,
not just "code ran". They're disabled at eval time (`FINANCIAL_ENV_PROGRESS=0`)
to keep the benchmark honest.

### Final grade (per family)

| Family | Layer 1 (gate) | Layer 2 | Layer 3 |
|---|---|---|---|
| `xlsx` | — | 30% sheet-name match | 70% cell-level diff (2% numeric tolerance) |
| `docx` | python-docx parse | 40% paragraph diff | 60% per-task OSWorld evaluator (`compare_docx_files`, `check_tabstops`, `is_first_line_centered`, `compare_line_spacing`, …) |
| `pptx` | python-pptx parse | 20% slide-count | 80% avg per-shape composite: 40% text + 20% style + 20% position + 20% size |

The `docx` 3rd layer is a port of OSWorld's [`metrics/docs.py`](https://github.com/xlang-ai/OSWorld/blob/main/desktop_env/evaluators/metrics/docs.py)
(Apache-2.0). 16 evaluator functions, including compound `or` (multi-gold)
and `and` (all-must-pass) checks. Single + compound normalized into a
uniform `{conj, checks: [...]}` schema.

### Anti-hacking defenses

The env is built on the assumption that an agent will try to game the
reward. Defenses (per [edits.md](edits.md) Phase 4):

| Vector | Defense |
|---|---|
| Persistent globals | Each step is a fresh `subprocess.run` |
| Time runaway | 30s subprocess timeout per step |
| **Read the gold file from `data/`** | **At episode start, `move()` every gold file to `/tmp/oe_gold_<random>/` with generic names; restore on `close()`. The agent can't `glob('data/**/*Gold*')` for it.** |
| Generic-distance gaming | `eval_check` rewards *spec-aligned* progress, not just diff-shrinkage |
| `lib_engagement` regex gaming | Capped at 0.010/step — trivially bounded |
| `mutation` spam (save garbage) | Capped, and the `progress`/`eval_check` signals dwarf it on real edits |

Caveat: full sandbox isolation (bwrap / seccomp / read-only mount) is the
right long-term answer; we ship the path-stash defense as a pragmatic v1.
See [edits.md](edits.md) for the full audit.

---

## Action & Observation spaces

### `FinancialAction`

| Field | Type | Description |
|---|---|---|
| `action_type` | `str` | `"code"` (Python), `"submit"` (text), `"submit_file"` (path) |
| `content` | `str` | Code, answer text, or absolute file path |

### `FinancialObservation`

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | e.g. `finch_10`, `osworld_0a0faba3`, `pptarena_case_60_fix_text_placement` |
| `task_description` | `str` | Instruction + constraints + source-file summary |
| `source_file` | `str` | Path to the working file (already copied into a per-episode tmpdir) |
| `task_type` | `str` | `"QA"` or `"MODIFY"` |
| `feedback` | `str` | Stdout/stderr of code, or grading explanation. Includes the per-step reward decomposition for debugging. |
| `current_step` / `max_steps` | `int` | 0–15 |
| `done` | `bool` | Episode finished |
| `reward` | `float` | Step or final reward, in (0.001, 0.999) |

---

## Setup & usage

### Prerequisites

- Python 3.10+
- Docker (for HF Space deployment)
- LLM API key (for `inference.py`)

### Local dev

```bash
pip install -e ".[dev]"
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000 \
  --ws-ping-interval 600 --ws-ping-timeout 600 --reload
```

### Run a baseline

```bash
export HF_TOKEN="hf_..."
python inference.py \
  --split eval \
  --model MiniMaxAI/MiniMax-M2.1 \
  --api-base https://router.huggingface.co/v1 \
  --env-url http://localhost:8000 \
  --task-timeout 900
```

CLI flags worth knowing:
- `--split {train,eval,all}` — manifest split
- `--family {xlsx,docx,pptx,all}` — filter to one family
- `--task-ids id1,id2,…` — explicit list (overrides split/family)
- `--limit N` — cap number of tasks
- `--resume` — merge new task results into an existing `--output-dir`
  (useful for retrying flaky tasks without losing prior trajectories)

Output lands at `runs/<timestamp>_<model_slug>/` with `results.json`,
`summary.csv`, per-task `trajectories/*.jsonl`, and a mirrored `log.txt`.

### Re-pull data from upstream sources

```bash
python data_pipeline/finch_pull.py            # 50 Finch xlsx tasks
python data_pipeline/osworld_writer_pull.py   # 21 OSWorld docx tasks
python data_pipeline/pptarena_pull.py --root /path/to/PPTArena-main   # 38 PPTArena pptx tasks
```

### Docker

```bash
docker build -t office-task-env:latest .
docker run -p 8000:8000 office-task-env:latest
```

The provided [`Dockerfile`](Dockerfile) installs `openpyxl`, `python-docx`,
`python-pptx`, `rapidfuzz`, and `Pillow`.

---

## Project structure

```
.
├── data/
│   ├── manifest.jsonl                 # 109 rows: 50 Finch + 21 OSWorld + 38 PPTArena
│   ├── 0/, 21/, …                     # 10 hand-curated xlsx tasks
│   ├── finch_50/<id>/{src,ref}.xlsx
│   ├── osworld_writer/<uuid>/<src + N gold>.docx
│   └── pptarena/<slug>/{src,ref}.pptx
├── data_pipeline/                     # Pullers for each upstream dataset
├── graders/
│   ├── __init__.py                    # grade_xlsx + grade_docx + grade_pptx
│   └── docx_metrics.py                # 16 ported OSWorld evaluators
├── server/
│   ├── financial_environment.py       # OpenEnv environment + gold-stash
│   └── app.py                         # FastAPI + WebSocket
├── rewards.py                         # 6-component RewardTracker
├── tasks.py                           # Manifest loader + helpers
├── inference.py                       # Baseline runner with --split / --family / --resume
├── runs/                              # Baseline & training-run results
└── edits.md                           # Full Round-1 → Round-2 change log
```

---

## What's next (training pipeline — in progress)

Per the budget plan in [edits.md](edits.md) (~$45 on HF Jobs):

1. Run a teacher (Claude Haiku 4.5) on the 97 train tasks, filter by score
2. SFT-warm-start `Qwen2.5-Coder-3B-Instruct` with LoRA on filtered trajectories (Unsloth, ~$10 on 1× A100 80GB)
3. GRPO continued training with rollouts hitting this env in-process (~$30, 12h on the same GPU)
4. Re-eval on the 22-task split → before/after plot

The trajectory persistence in `runs/<dir>/trajectories/*.jsonl` doubles as
the SFT corpus format — `(messages, completion)` pairs ready for
`SFTTrainer`.

---

## Round-1 → Round-2 change log

The full journey from the original 10-task xlsx-only env to today's
3-format / 119-task / multi-layer-graded env is documented in
[`edits.md`](edits.md): manifest loader, RewardTracker, OSWorld docx port,
PPTArena ingest, layout+style-aware pptx grader, gold-stash hardening,
inference v2.

---

## Acknowledgments

- **Finch / FinWorkBench** ([dataset](https://huggingface.co/datasets/FinWorkBench/Finch),
  [paper](https://arxiv.org/abs/2512.13168)) — the xlsx tasks
- **OSWorld-Verified** ([repo](https://github.com/xlang-ai/OSWorld)) — the
  docx tasks and the evaluator functions in `graders/docx_metrics.py` (Apache-2.0)
- **PPTArena** ([repo](https://github.com/michaelofengend/PPTArena)) — the
  pptx tasks and the `evaluation_pairs_refined.json` schema
- **OpenEnv / Meta PyTorch** ([repo](https://github.com/meta-pytorch/OpenEnv)) — the host framework

If you use this environment in research, please cite the upstream datasets:

```bibtex
@article{dong2025finch,
  title={Finch: Benchmarking Finance \& Accounting across Spreadsheet-Centric Enterprise Workflows},
  author={Dong, Haoyu and Zhang, Pengkun and Gao, Yan and Dong, Xuanyu and Cheng, Yilin and Lu, Mingzhe and Yakefu, Adina and Zheng, Shuxin},
  journal={arXiv preprint arXiv:2512.13168},
  year={2025}
}
```