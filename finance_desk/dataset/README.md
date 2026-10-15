---
license: apache-2.0
task_categories:
  - reinforcement-learning
language:
  - en
tags:
  - openenv
  - rl-environment
  - agent-environment
  - terminal-bench
  - software-engineering
  - finance
pretty_name: Finance Desk
configs:
  - config_name: default
    data_files: tasks.jsonl
  - config_name: v2
    data_files: v2/tasks.jsonl
---

# Finance Desk: OpenEnv environment tasks

Public task preview and source for the **Finance Desk** OpenEnv Arena environment.

## v3: terminal mode (current)

- **Image (public, linux/amd64):** `ghcr.io/bp-high/finance-desk:v3`
- **Source:** https://github.com/bp-high/openenv_financial_task_env/tree/dev/pensive-lovelace-bt5rsn/finance_desk
- **Contract:** OpenEnv's `tbench2_env` (Terminal-Bench 2). The action and observation models are copied field for
  field, and `/schema` matches the operator's TB2 smoke test exactly. Actions: `exec` (run a shell command
  in the workdir), `write_file`, `evaluate` (grade and end the episode). Every observation carries the instruction.
- **Budgets per task:** 4096 completion / 8192 context tokens, 32 tool calls. These are the same as the TB2 smoke test.

Each episode builds a fresh workdir (`/app`) holding a small repo or a set of data files, and gives a prose instruction.

- **Software (11 task ids):**
  - fix seeded bugs in a stats, money or dates module (a failing visible test, a bug report, or both);
  - implement a log summarizer to a docstring spec;
  - repair a report script and regenerate its output.

  Hidden tests run against the agent's code. At generation time the shipped code is proven to fail them and the
  reference to pass.
- **Finance (8 task ids):** procedural finance problems (bank reconciliation, bond pricing, cap table, covenants, FX
  consolidation, payroll, PE waterfall, portfolio return). The answer goes into `answer.json` and is graded field by
  field.

Reward is all-or-nothing: `1.0` if the hidden tests or every answer field pass, else `0.0`.

Every task id was calibrated on live Qwen3.8-27B with an arena-style loop. Only mixed-outcome ids are submitted. The
per-id table is in `calibration_v3.md`, and why terminal mode is explained in `STRATEGY_v3.md` (both in this dataset).

`tasks.jsonl` has three illustrative instances per submitted task id: `instruction`, the workdir `files`, the
reference solution `oracle`, `domain`, `level`, `seed`. Hidden test cases are regenerated per instance at train
time. The container makes fresh instances during training, so this preview is illustrative, not the trained set.

## v2 (previous)

`v2/tasks.jsonl` is the v2 preview: single-step, with the files inlined in the first observation, across all eight
domains. Image `ghcr.io/bp-high/finance-desk:v2`. The ABC-notation (music) family in v2 is derived from
[`XiaomiMiMo/MiMo-V2.6-RL-oss`](https://huggingface.co/datasets/XiaomiMiMo/MiMo-V2.6-RL-oss) (Apache-2.0).

`source/fdesk/` is the exact environment package (v2 under `fdesk/`, v3 under `fdesk/terminal/`).
