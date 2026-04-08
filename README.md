---
title: Financial Task Environment
emoji: 📊
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Financial Task Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) **code-execution
environment** for training and evaluating AI agents on **real-world finance
& accounting spreadsheet tasks**.  Agents write Python code (using
`openpyxl`) to read, analyze, and modify authentic Excel workbooks from
enterprise workflows.

## Motivation

Finance professionals spend hundreds of hours on spreadsheet-centric tasks —
extracting values, computing ratios, auditing formulas, entering data, building
scenarios, and consolidating reports.  This environment provides 10 diverse
tasks backed by real `.xlsx` files so agents can be trained and evaluated on
the same kind of work.

## How It Works

1. **Reset** with a `task_id` → receive task instructions + xlsx file path + a
   summary of the spreadsheet contents.
2. **Execute code** (`action_type="code"`) → run Python code that reads or
   modifies the xlsx.  The environment returns stdout/stderr.
3. **Submit** a text answer (`action_type="submit"` for QA tasks) or a modified
   file (`action_type="submit_file"` for MODIFY tasks).
4. The environment **grades** the submission: QA answers are scored by numeric
   matching + keyword overlap; MODIFY tasks are scored by cell-level comparison
   against a reference workbook.

## Tasks (10 total)

| # | Task ID | Title | Difficulty | Type | Category |
|---|---------|-------|------------|------|----------|
| 1 | `task_1` | Count Plants in Spreadsheet | Easy | QA | Calculation |
| 2 | `task_2` | Retrieve TW EOL Charge | Easy | QA | Cross-sheet Retrieval |
| 3 | `task_3` | Portfolio Mark-to-Market Change | Easy | QA | Calculation |
| 4 | `task_4` | Summarize Pipeline Imbalances | Medium | MODIFY | Calculation |
| 5 | `task_5` | Audit and Correct Formula Errors | Medium | MODIFY | Validation / Review |
| 6 | `task_6` | Create Table and Apply Filter | Medium | MODIFY | Structuring / Formatting |
| 7 | `task_7` | Add Weekday Row and Data Entry | Medium | MODIFY | Data Entry / Import |
| 8 | `task_8` | Balance Sheet Validation & Indicators | Hard | MODIFY | Validation, Calculation |
| 9 | `task_9` | Create Scenario3 Worksheet | Hard | MODIFY | Financial Modeling |
| 10 | `task_10` | Consolidate by Type and Area | Hard | MODIFY | Multi-type |

### Difficulty Progression

- **Easy (3 tasks):** QA — read the spreadsheet and answer a question.
- **Medium (4 tasks):** MODIFY — edit/augment the workbook (summaries, audits, formatting, data entry).
- **Hard (3 tasks):** MODIFY — complex multi-sheet operations (validation, new scenario sheets, consolidation).

## Action & Observation Spaces

### Action — `FinancialAction`

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `str` | `"code"` to execute Python, `"submit"` for text answer, `"submit_file"` for xlsx |
| `content` | `str` | Python code, text answer, or file path |

### Observation — `FinancialObservation`

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | `str` | Current task identifier |
| `task_description` | `str` | Full task instructions + xlsx summary |
| `source_file` | `str` | Path to the working xlsx copy |
| `difficulty` | `str` | `easy`, `medium`, or `hard` |
| `task_type` | `str` | `QA` or `MODIFY` |
| `feedback` | `str` | Code output or grading result |
| `current_step` | `int` | Current step (max 15) |
| `done` | `bool` | Whether the episode is finished |
| `reward` | `float` | Reward for this step (0.0–1.0) |

## Reward Design

| Action | Reward | Signal |
|--------|--------|--------|
| `code` (failed) | 0.005 | Penalized — syntax/runtime error |
| `code` (simple) | ~0.02 | Minimal — just imports and a print |
| `code` (exploration) | ~0.05 | Good — reading data, producing output |
| `code` (modification + save) | ~0.06–0.10 | Best — actively editing the workbook |
| `submit` / `submit_file` | 0.001–0.999 | Full grading against reference |
| Max steps (15) | Episode ends | |

Code step rewards are computed from:
- **Execution success** — failed code gets only 0.005
- **Substantive lines** — lines beyond imports/comments earn +0.002 each (up to +0.03)
- **Output produced** — printing data earns +0.001 per line (up to +0.02)
- **Save operations** — calling `.save()` earns +0.03 (agent is modifying the workbook)

**QA grading:** Numeric extraction with 5% tolerance + keyword overlap.
**MODIFY grading:** 30% sheet-name match + 70% cell-level comparison (2% numeric tolerance).

All scores are clamped to the open interval (0.001, 0.999).

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- `pip install openenv-core openpyxl`

### Local Development

```bash
pip install -e ".[dev]"
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

```bash
docker build -t financial-task-env:latest .
docker run -p 8000:8000 financial-task-env:latest
```

### Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:8000"
python inference.py
```

## Baseline Scores

The environment includes 10 tasks, but the baseline inference runs 5 representative
tasks (3 easy + 1 medium + 1 hard) to stay within the 20-minute runtime constraint.

**Model:** `MiniMaxAI/MiniMax-M2.5` via HuggingFace Router

| Task | Difficulty | Type | Score | Step Rewards |
|------|------------|------|-------|-------------|
| task_1 — Count Plants | Easy | QA | 0.001 | 0.05, 0.06, 0.06, 0.06, 0.00 |
| task_2 — Retrieve EOL Charge | Easy | QA | 0.001 | 0.04, 0.01, 0.07, 0.06, 0.02, 0.00 |
| task_3 — Portfolio MTM Change | Easy | QA | 0.367 | 0.06, 0.01, 0.07, ..., 0.37 |
| task_5 — Audit Formulas | Medium | MODIFY | **0.958** | 0.07, 0.01, 0.07, ..., 0.96 |
| task_8 — Balance Sheet Validation | Hard | MODIFY | 0.001 | 0.06, 0.01, 0.06, ..., 0.05 |
| **Average** | | | **0.266** | |

**Runtime:** 12 min 10 sec (limit: 20 min) · **Server memory:** ~40 MB (limit: 8 GB)

Note: Step rewards vary based on code quality — failed code gets 0.005, exploration
~0.05, modification+save ~0.06–0.10.

### Run 2 — `google/gemma-4-26B-A4B-it`

| Task | Difficulty | Type | Score |
|------|------------|------|-------|
| task_1 — Count Plants | Easy | QA | 0.001 |
| task_2 — Retrieve EOL Charge | Easy | QA | **0.999** |
| task_3 — Portfolio MTM Change | Easy | QA | 0.001 |
| task_5 — Audit Formulas | Medium | MODIFY | 0.001 |
| task_8 — Balance Sheet Validation | Hard | MODIFY | 0.001 |
| **Average** | | | **0.201** |

**Runtime:** 19 min 27 sec (limit: 20 min) · **Server memory:** ~40 MB

Gemma 4 26B solved task_2 perfectly in just 2 steps but timed out on more
complex tasks due to longer generation times.

### Run 3 — `Qwen/Qwen3.5-122B-A10B`

| Task | Difficulty | Type | Score |
|------|------------|------|-------|
| task_1 — Count Plants | Easy | QA | 0.001 |
| task_2 — Retrieve EOL Charge | Easy | QA | **0.999** |
| task_3 — Portfolio MTM Change | Easy | QA | 0.001 |
| task_5 — Audit Formulas | Medium | MODIFY | 0.001 |
| task_8 — Balance Sheet Validation | Hard | MODIFY | 0.001 |
| **Average** | | | **0.201** |

**Runtime:** 2 min 11 sec · Fast inference but hit per-task timeout on complex tasks.

### Run 4 — `deepseek-ai/DeepSeek-R1`

| Task | Difficulty | Type | Score |
|------|------------|------|-------|
| task_1 — Count Plants | Easy | QA | 0.001 |
| task_2 — Retrieve EOL Charge | Easy | QA | 0.001 |
| task_3 — Portfolio MTM Change | Easy | QA | 0.001 |
| task_5 — Audit Formulas | Medium | MODIFY | 0.001 |
| task_8 — Balance Sheet Validation | Hard | MODIFY | 0.001 |
| **Average** | | | **0.001** |

**Runtime:** 11 min 57 sec · DeepSeek-R1's long chain-of-thought reasoning consumed
most of the output tokens, leaving answers that didn't parse correctly.

### Run 5 — `MiniMaxAI/MiniMax-M2.1` (Best)

| Task | Difficulty | Type | Score | Steps |
|------|------------|------|-------|-------|
| task_1 — Count Plants | Easy | QA | 0.001 | 5 |
| task_2 — Retrieve EOL Charge | Easy | QA | **0.999** | 4 |
| task_3 — Portfolio MTM Change | Easy | QA | 0.001 | 10 |
| task_5 — Audit Formulas | Medium | MODIFY | **0.958** | 4 |
| task_8 — Balance Sheet Validation | Hard | MODIFY | **0.733** | 10 |
| **Average** | | | **0.538** | |

**Runtime:** 3 min 18 sec · Best overall performance — solved 3/5 tasks with high
scores including the hard MODIFY task (0.733). Fast and efficient.

### Model Comparison Summary

| Model | Avg Score | Runtime | Best Task |
|-------|-----------|---------|-----------|
| **MiniMax-M2.1** | **0.538** | **3m 18s** | task_5: 0.958, task_8: 0.733 |
| MiniMax-M2.5 | 0.266 | 12m 10s | task_5: 0.958 |
| Gemma 4 26B | 0.201 | 19m 27s | task_2: 0.999 |
| Qwen 3.5 122B | 0.201 | 2m 11s | task_2: 0.999 |
| DeepSeek-R1 | 0.001 | 11m 57s | — |

## Project Structure

```
financial_task_env/
├── __init__.py              # Module exports
├── models.py                # FinancialAction & FinancialObservation
├── tasks.py                 # 10 task definitions + xlsx paths
├── graders.py               # QA grading + xlsx cell comparison
├── client.py                # FinancialTaskEnv (EnvClient)
├── inference.py             # Baseline inference script
├── openenv.yaml             # OpenEnv manifest
├── pyproject.toml           # Dependencies
├── Dockerfile               # Container image
├── data/                    # xlsx source & reference files
│   ├── 0/                   # Balance sheet validation
│   ├── 21/                  # Data entry
│   ├── 24/                  # Scenario modeling
│   ├── 34/                  # Portfolio calculation
│   ├── 35/                  # Pipeline imbalances
│   ├── 40/                  # Formula audit
│   ├── 60/                  # Table formatting
│   ├── 67/                  # Consolidation
│   ├── 118/                 # Value retrieval
│   └── 119/                 # Plant counting
└── server/
    ├── __init__.py
    ├── financial_environment.py  # Code-execution environment
    ├── app.py                    # FastAPI application
    └── Dockerfile
```

## Environment Description

This environment models real financial spreadsheet work:

- **Data extraction** — read values from complex multi-sheet workbooks
- **Calculation** — compute portfolio changes, imbalances, indicators
- **Validation** — audit and fix formula errors in workbooks
- **Data entry** — add rows, enter values, format new columns
- **Structuring** — create tables, apply filters, build new worksheets
- **Financial modeling** — replicate scenario sheets with new parameters
- **Consolidation** — aggregate data across sheets into summary views

Each task uses a genuine enterprise Excel workbook.  MODIFY tasks are graded
by spreadsheet properties comparison against a reference workbook.

## Acknowledgments

The spreadsheet tasks and reference workbooks used in this environment are
sourced from the **FinWorkBench (Finch)** dataset. If you use this environment
in your research, please cite:

```bibtex
@article{dong2025finch,
  title={Finch: Benchmarking Finance \& Accounting across Spreadsheet-Centric Enterprise Workflows},
  author={Dong, Haoyu and Zhang, Pengkun and Gao, Yan and Dong, Xuanyu and Cheng, Yilin and Lu, Mingzhe and Yakefu, Adina and Zheng, Shuxin},
  journal={arXiv preprint arXiv:2512.13168},
  year={2025}
}
```
